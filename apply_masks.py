import torch
import math
from icecream import ic
import random
from blosum62 import p_j_given_i as P_JGI

ic.configureOutput(includeContext=True)

def sample_blosum_mutations(seq, p_blosum=0.9, p_uni=0.1, p_mask=0):
    """
    Given a sequence,
    """
    assert len(seq.shape) == 1
    assert math.isclose(sum([p_blosum, p_uni, p_mask]), 1)
    L = len(seq)

    # uniform prob
    U = torch.full((L,20), .05)
    U = torch.cat((U,torch.zeros(L,1)), dim=-1)
    U = U*p_uni

    # mask prob
    M = torch.full((L,21),0)
    M[:,-1] = 1
    M = M*p_mask

    # blosum probs
    B = torch.from_numpy( P_JGI[seq] ) # slice out the transition probabilities from blossom
    B = torch.cat((B,torch.zeros(L,1)), dim=-1)
    B = B*p_blosum

    # build transition probabilities for each residue
    P = U+M+B

    C = torch.distributions.categorical.Categorical(probs=P)

    sampled_seq = C.sample()

    return sampled_seq


def mask_inputs(seq, 
                msa_masked, 
                msa_full, 
                xyz_t, 
                t1d, 
                mask_msa, 
                atom_mask,
                input_seq_mask=None, 
                input_str_mask=None, 
                input_floating_mask=None, 
                input_t1dconf_mask=None, 
                loss_seq_mask=None, 
                loss_str_mask=None, 
                loss_str_mask_2d=None,
                diffuser=None,
                predict_previous=False,
                true_crds_in=None,
                decode_mask_frac=0.,
                corrupt_blosum=0.9,
                corrupt_uniform=0.1):
    """
    Parameters:
        seq (torch.tensor, required): (I,L) integer sequence 

        msa_masked (torch.tensor, required): (I,N_short,L,48)

        msa_full  (torch,.tensor, required): (I,N_long,L,25)
        
        xyz_t (torch,tensor): (T,L,14,3) template crds BEFORE they go into get_init_xyz 
        
        t1d (torch.tensor, required): (I,L,22) this is the t1d before tacking on the chi angles 
        
        str_mask_1D (torch.tensor, required): Shape (L) rank 1 tensor where structure is masked at False positions 

        seq_mask_1D (torch.tensor, required): Shape (L) rank 1 tensor where seq is masked at False positions 

        ...
        
        decode_mask_frac (float, optional): Fraction of decoded residues which are to be corrupted 

        corrupt_blosum (float, optional): Probability that a decoded residue selected for corruption will transition according to BLOSUM62 probs 

        corrupt_unifom (float, optional): Probability that ... according to uniform probs 


    NOTE: in the MSA, the order is 20aa, 1x unknown, 1x mask token. We set the masked region to 22 (masked).
        For the t1d, this has 20aa, 1x unkown, and 1x template conf. Here, we set the masked region to 21 (unknown).
        This, we think, makes sense, as the template in normal RF training does not perfectly correspond to the MSA.
    """
    # print('Made it into mask inputs')
    ### Perform diffusion, pick a random t and then let that be the input template and xyz_prev
    if (not diffuser is None) :


        # NOTE: assert that xyz_t is the TRUE coordinates! Should come from fixbb loader 
        #       also assumes all 4 of seq are identical 

        # pick t uniformly 
        t = random.randint(1,diffuser.T)
        t_list = [t]
        if predict_previous:
            # grab previous t. if t is 0 force a prediction of x_t=0
            tprev = t-1 if t > 0 else t
            t_list.append(tprev)

        kwargs = {'xyz'             :xyz_t[0],
                  'seq'             :seq[0],
                  'atom_mask'       :atom_mask.squeeze(),
                  'diffusion_mask'  :input_str_mask.squeeze(),
                  't_list':t_list}

        diffused_fullatoms, aa_masks, true_crds = diffuser.diffuse_pose(**kwargs)
        

        # seq_mask - True-->revealed, False-->masked 
        seq_mask = torch.ones_like(seq.squeeze()[0]).to(dtype=bool) # all revealed 

        # grab noised inputs / create masks based on time t
        aa_mask_raw = aa_masks[t-1] 
    
        # mark False where aa_mask_raw is False -- assumes everybody is potentially diffused
        seq_mask[~aa_mask_raw] = False
        
        # reset to True any positions which aren't being diffused 
        seq_mask[input_seq_mask.squeeze()] = True

        ###  DJ new - make mutations in the decoded sequence 
        sampled_blosum = sample_blosum_mutations(seq.squeeze()[0])

        # find decoded residues and select them with 21% probability 
        decoded_non_motif = torch.ones_like(seq.squeeze()[0]).to(dtype=bool)

        decoded_non_motif[~aa_mask_raw] = False                  # mark False where residues are masked via diffusion
        decoded_non_motif[input_seq_mask.squeeze()] = False      # mark False where motif exists 

        # set (1-decode_mask_frac) proportion to False, keeping <decode_mask_frac> proportion still available 
        tmp_mask = torch.rand(decoded_non_motif.shape) < (1-decode_mask_frac)
        decoded_non_motif[tmp_mask] = False 

        # Anything left as True, replace with blosum sample
        blosum_replacement = sampled_blosum[decoded_non_motif]

        xyz_t       = diffused_fullatoms[0][None]
        if predict_previous:
            true_crds = diffused_fullatoms[1][None]


        # scale confidence wrt t 
        # multiplicitavely applied to a default conf mask of 1.0 everywhwere 
        input_t1dconf_mask[~input_str_mask] = 1 - t/diffuser.T 

        mask_msa[:,:,seq_mask] = False # don't score revealed positions 
    else:
        print('WARNING: Diffuser not being used in apply masks')




    ###########
    assert len(blosum_replacement) == decoded_non_motif.sum()

    #seq_mask = input_seq_mask[0] # DJ - old, commenting out bc using seq mask from diffuser 
    seq[:,~seq_mask] = 21 # mask token categorical value
    seq[:,decoded_non_motif] = blosum_replacement 

    ### msa_masked ###
    ################## 
    msa_masked[:,:,~seq_mask,:20] = 0
    msa_masked[:,:,~seq_mask,21]  = 1 #set to mask token
    msa_masked[:,:,~seq_mask,20]  = 0    
    
    # index 44/45 is insertion/deletion
    # index 43 is the masked token NOTE check this
    # index 42 is the unknown token 
    msa_masked[:,:,~seq_mask,22:42] = 0
    msa_masked[:,:,~seq_mask,43] = 1 
    msa_masked[:,:,~seq_mask,42] = 0

    # insertion/deletion stuff 
    msa_masked[:,:,~seq_mask,44:46] = 0

    # blosum mutations 
    msa_masked[:,:,decoded_non_motif,:] = 0
    msa_masked[:,:,decoded_non_motif,blosum_replacement]  = 1
    msa_masked[:,:,decoded_non_motif,22:44] = 0                  
    msa_masked[:,:,decoded_non_motif,22+blosum_replacement] = 1
    

    ### msa_full ### 
    ################
    msa_full[:,:,~seq_mask,:21] = 0
    msa_full[:,:,~seq_mask,21]  = 1
    msa_full[:,:,~seq_mask,-3]  = 0   #NOTE: double check this is insertions/deletions and 0 makes sense 

    # blosum mutations 
    msa_full[:,:,decoded_non_motif,:] = 0
    msa_full[:,:,decoded_non_motif,blosum_replacement]  = 1


    ### t1d ###
    ########### 
    # NOTE: Not adjusting t1d last dim (confidence) from sequence mask
    t1d[:,~seq_mask,:20] = 0 
    t1d[:,~seq_mask,20]  = 1 # unknown
    
    t1d[:,decoded_non_motif,:] = 0
    t1d[:,decoded_non_motif,blosum_replacement] = 1

    t1d[:,:,21] *= input_t1dconf_mask
    
    # TRYING NOT PROVIDING diffused SIDECHAINS TO THE MODEL
    xyz_t[:,~input_str_mask.squeeze(),3:,:] = float('nan')
    #xyz_t[:,:,~seq_mask,3:,:] = float('nan') # don't know sidechain information for masked seq 

    # Structure masking
    # str_mask = input_str_mask[0]
    # xyz_t[:,:,~str_mask,:,:] = float('nan') # NOTE: not using this because diffusion is effectively the mask 
    
    ### mask_msa ###
    ################
    # this is for loss scoring
    mask_msa[:,:,~loss_seq_mask[0]] = False
  
    assert torch.sum(torch.isnan(xyz_t[:,:,:3]))==0
    return seq, msa_masked, msa_full, xyz_t, t1d, mask_msa, t, true_crds 
