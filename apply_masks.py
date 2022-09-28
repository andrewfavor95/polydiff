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
                input_t1d_str_conf_mask=None, 
                input_t1d_seq_conf_mask=None, 
                loss_seq_mask=None, 
                loss_str_mask=None, 
                loss_str_mask_2d=None,
                diffuser=None,
                seq_diffuser=None,
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
        
        xyz_t (torch,tensor): (T,L,27,3) template crds BEFORE they go into get_init_xyz 
        
        t1d (torch.tensor, required): (T,L,22) this is the t1d before tacking on the chi angles 
        
        input_seq_mask (torch.tensor, required): Shape (L) rank 1 tensor where sequence is masked at False positions 

        input_str_mask (torch.tensor, required): Shape (L) rank 1 tensor where structure is masked at False positions 

        input_t1d_str_conf_mask (torch.tensor, required): Shape (L) rank 1 tensor with entries for str confidence

        input_t1d_seq_conf_mask (torch.tensor, required): Shape (L) rank 1 tensor with entries for seq confidence

        loss_seq_mask (torch.tensor, required): Shape (L)

        loss_str_mask (torch.tensor, required): Shape (L)

        loss_str_mask_2d (torch.tensor, required): Shape (L,L) 

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
        if t == diffuser.T: t_list = [t,t]
        else: t_list = [t+1,t]
        if predict_previous:
            # grab previous t. if t is 0 force a prediction of x_t=0
            tprev = t-1 if t > 0 else t
            t_list.append(tprev)
        
        # Ensures that all I dimensions are size 1 - NRB
        seq        = seq[:1] 
        msa_masked = msa_masked[:1]
        msa_full   = msa_full[:1]

        assert(seq.shape[0] == 1), "Number of repeats of seq must be 1"
        L = seq.shape[-1]

        kwargs = {'xyz'             :xyz_t.squeeze(),
                  'seq'             :seq.squeeze(0),
                  'atom_mask'       :atom_mask.squeeze(),
                  'diffusion_mask'  :input_str_mask.squeeze(),
                  't_list':t_list}
        
        diffused_fullatoms, aa_masks, true_crds = diffuser.diffuse_pose(**kwargs)

        if not seq_diffuser is None:
            seq_args = {
                        'seq'            : seq.squeeze(0),
                        'diffusion_mask' : input_seq_mask.squeeze(),
                        't_list'         : t_list
                       }
            diffused_seq, true_seq = seq_diffuser.diffuse_sequence(**seq_args)

            if seq_diffuser.continuous_seq():
                diffused_seq_bits = diffused_seq

                assert(diffused_seq.shape[-1] == 20) # Must be probabilities
                if predict_previous: raise NotImplementedError() # This involves changing the shape of true crds - NRB
            else:
                diffused_seq_bits = torch.nn.functional.one_hot(diffused_seq, num_classes=20).float()

        if predict_previous: assert(diffused_fullatoms.shape[0] == 3)
        else: assert(diffused_fullatoms.shape[0] == 2)

        # seq_mask - True-->revealed, False-->masked 
        seq_mask = torch.ones(2,L).to(dtype=bool) # all revealed [t,L]

        # grab noised inputs / create masks based on time t
        #aa_mask_raw = aa_masks # [t,L]
    
        if not seq_diffuser is None:
            seq_mask[:,~input_seq_mask.squeeze()] = False # All non-fixed positions are diffused in sequence diffusion
        else:
            # mark False where aa_mask_raw is False -- assumes everybody is potentially diffused
            seq_mask[0,~aa_masks[0]] = False
            seq_mask[1,~aa_masks[1]] = False

            # reset to True any positions which aren't being diffused 
            seq_mask[:,input_seq_mask.squeeze()] = True

            ###  DJ new - make mutations in the decoded sequence 
            sampled_blosum = torch.stack([sample_blosum_mutations(seq.squeeze(0)), sample_blosum_mutations(seq.squeeze(0))], dim=0) # [n,L]

            # find decoded residues and select them with 21% probability 
            decoded_non_motif = torch.ones_like(sampled_blosum).to(dtype=bool) # [n,L]

            # mark False where residues are masked via diffusion 
            decoded_non_motif[0,~aa_masks[0]] = False
            decoded_non_motif[1,~aa_masks[1]] = False

            decoded_non_motif[:,input_seq_mask.squeeze()] = False      # mark False where motif exists 

            # set (1-decode_mask_frac) proportion to False, keeping <decode_mask_frac> proportion still available 
            tmp_mask = torch.rand(decoded_non_motif.shape) < (1-decode_mask_frac) # [n,L]

            decoded_non_motif[0,tmp_mask[0]] = False 
            decoded_non_motif[1,tmp_mask[1]] = False 

            # Anything left as True, replace with blosum sample
            # These may be different lengths so cannot convert to a Tensor - NRB
            blosum_replacement = []

            blosum_replacement.append(sampled_blosum[0,decoded_non_motif[0]])
            blosum_replacement.append(sampled_blosum[1,decoded_non_motif[1]])

            onehot_blosum_rep = [torch.nn.functional.one_hot(i, num_classes=20).float() for i in blosum_replacement] # [n,dim_replace,20]

            ### End DJ new
       
        xyz_t       = diffused_fullatoms[:2].unsqueeze(1) # [n,T,L,27,3]

        if predict_previous:
            true_crds = diffused_fullatoms[-1][None]
            true_seq  = diffused_seq[-1][None]

        # Scale str confidence wrt t 
        # multiplicitavely applied to a default conf mask of 1.0 everywhwere 
        input_t1d_str_conf_mask = torch.stack([input_t1d_str_conf_mask,input_t1d_str_conf_mask], dim=0) # [n,L]
        input_t1d_str_conf_mask[0,~input_str_mask.squeeze()] = 1 - t_list[0]/diffuser.T
        input_t1d_str_conf_mask[1,~input_str_mask.squeeze()] = 1 - t_list[1]/diffuser.T

        # Scale seq confidence wrt t
        input_t1d_seq_conf_mask = torch.stack([input_t1d_seq_conf_mask,input_t1d_seq_conf_mask], dim=0) # [n,L]
        input_t1d_seq_conf_mask[0,~input_seq_mask.squeeze()] = 1 - t_list[0]/diffuser.T
        input_t1d_seq_conf_mask[1,~input_seq_mask.squeeze()] = 1 - t_list[1]/diffuser.T

        mask_msa = torch.stack([mask_msa,mask_msa], dim=0) # [n,I,N_long,L,25]

        if not seq_diffuser is None:
            mask_msa[:,:,:,~input_seq_mask.squeeze()] = False # don't score non-diffused positions
        else:
            mask_msa[0,:,:,seq_mask[0]] = False # don't score revealed positions 
            mask_msa[1,:,:,seq_mask[1]] = False # don't score revealed positions 

    else:
        print('WARNING: Diffuser not being used in apply masks')

    ###########

    #seq_mask = input_seq_mask[0] # DJ - old, commenting out bc using seq mask from diffuser 

    seq = torch.stack([seq,seq], dim=0) # [n,I,L]
    if not seq_diffuser is None:

        alldim_diffused_seq = diffused_seq_bits[:,None,:,:] # [n,I,L,20]
        zeros = torch.zeros(2,1,L,2)
        seq   = torch.cat((alldim_diffused_seq, zeros), dim=-1) # [n,I,L,22]

    else:
        assert len(blosum_replacement[0]) == decoded_non_motif[0].sum() and len(blosum_replacement[1]) == decoded_non_motif[1].sum()
        seq = torch.nn.functional.one_hot(seq, num_classes=22).float() # [n,I,L,22]
        seq[0,:,~seq_mask[0],:21] = 0
        seq[1,:,~seq_mask[1],:21] = 0 

        seq[0,:,~seq_mask[0],21] = 1 # mask token categorical value
        seq[1,:,~seq_mask[1],21] = 1 # mask token categorical value

        seq[0,:,decoded_non_motif[0],:20] = onehot_blosum_rep[0]
        seq[1,:,decoded_non_motif[1],:20] = onehot_blosum_rep[1] 

    ### msa_masked ###
    ################## 
    msa_masked = torch.stack([msa_masked,msa_masked], dim=0) # [n,I,N_short,L,48]
    if not seq_diffuser is None:
        msa_masked[...,:20]   = diffused_seq_bits[:,None,None,:,:]
        msa_masked[...,22:42] = diffused_seq_bits[:,None,None,:,:]

        # These dimensions are gap and mask - NRB
        msa_masked[...,20:22] = 0
        msa_masked[...,42:44] = 0

        # insertion/deletion stuff 
        msa_masked[:,:,:,~input_seq_mask.squeeze(),44:46] = 0
        
    else:
        # Standard autoregressive masking
        msa_masked[0,:,:,~seq_mask[0],:21] = 0
        msa_masked[1,:,:,~seq_mask[0],:21] = 0

        msa_masked[0,:,:,~seq_mask[0],21]  = 1 # set to mask token
        msa_masked[1,:,:,~seq_mask[1],21]  = 1 # set to mask token

        # index 44/45 is insertion/deletion
        # index 43 is the masked token NOTE check this
        # index 42 is the unknown token 
        msa_masked[0,:,:,~seq_mask[0],22:43] = 0
        msa_masked[1,:,:,~seq_mask[1],22:43] = 0

        msa_masked[0,:,:,~seq_mask[0],43]    = 1
        msa_masked[1,:,:,~seq_mask[1],43]    = 1 

        # insertion/deletion stuff 
        msa_masked[0,:,:,~seq_mask[0],44:46] = 0
        msa_masked[1,:,:,~seq_mask[1],44:46] = 0

        # blosum mutations 
        msa_masked[0,:,:,decoded_non_motif[0],:] = 0
        msa_masked[1,:,:,decoded_non_motif[1],:] = 0

        msa_masked[0,:,:,decoded_non_motif[0],blosum_replacement[0]]  = 1
        msa_masked[1,:,:,decoded_non_motif[1],blosum_replacement[1]]  = 1

        msa_masked[0,:,:,decoded_non_motif[0],22:44] = 0                  
        msa_masked[1,:,:,decoded_non_motif[1],22:44] = 0                  

        msa_masked[0,:,:,decoded_non_motif[0],22+blosum_replacement[0]] = 1
        msa_masked[1,:,:,decoded_non_motif[1],22+blosum_replacement[1]] = 1

    ### msa_full ### 
    ################
    msa_full = torch.stack([msa_full,msa_full], dim=0) # [n,I,N_long,L,25]

    if not seq_diffuser is None:
        # These sequences will only go up to 20
        msa_full[...,:20]   = diffused_seq_bits[:,None,None,:,:]

        # These dimensions are gap and mask - NRB
        msa_full[...,20:22] = 0

        msa_full[:,:,:,~input_seq_mask.squeeze(),-3:]  = 0   #NOTE: double check this is insertions/deletions and 0 makes sense 
        
    else:
        # Standard autoregressive masking
        msa_full[0,:,:,~seq_mask[0],:21] = 0
        msa_full[1,:,:,~seq_mask[1],:21] = 0

        msa_full[0,:,:,~seq_mask[0],21]  = 1
        msa_full[1,:,:,~seq_mask[1],21]  = 1

        msa_full[0,:,:,~seq_mask[0],-3:] = 0   
        msa_full[1,:,:,~seq_mask[1],-3:] = 0   #NOTE: double check this is insertions/deletions and 0 makes sense 

        # blosum mutations 
        msa_full[0,:,:,decoded_non_motif[0],:] = 0
        msa_full[1,:,:,decoded_non_motif[1],:] = 0

        msa_full[0,:,:,decoded_non_motif[0],blosum_replacement[0]]  = 1
        msa_full[1,:,:,decoded_non_motif[1],blosum_replacement[1]]  = 1


    ### t1d ###
    ########### 
    # NOTE: Not adjusting t1d last dim (confidence) from sequence mask

    t1d = torch.stack([t1d,t1d], dim=0) # [n,I,L,22]
    if not seq_diffuser is None:
        t1d[...,:20] = diffused_seq_bits[:,None,:,:]
        
        t1d[...,20]  = 0 # No unknown characters in seq diffusion

    else:
        t1d[0,:,~seq_mask[0],:20] = 0 
        t1d[1,:,~seq_mask[1],:20] = 0 

        t1d[0,:,~seq_mask[0],20]  = 1
        t1d[1,:,~seq_mask[1],20]  = 1 # unknown

        t1d[0,:,decoded_non_motif[0],:]  = 0
        t1d[1,:,decoded_non_motif[1],:]  = 0

        t1d[0,:,decoded_non_motif[0],blosum_replacement[0]] = 1
        t1d[1,:,decoded_non_motif[1],blosum_replacement[1]] = 1

    # input_t1d_conf_mask is shape [n,L]
    t1d[:,:,:,21] *= input_t1d_str_conf_mask[:,None,:]
    t1d[:,:,:,22] *= input_t1d_seq_conf_mask[:,None,:]

    # Try providing no sidechains to the model - NRB
    #xyz_t[:,:,:,3:,:] = float('nan') # [n,I,L,27,3]

    # TRYING NOT PROVIDING diffused SIDECHAINS TO THE MODEL
    xyz_t[:,:,~input_str_mask.squeeze(),3:,:] = float('nan') # [n,I,L,27,3]

    #xyz_t[:,:,~seq_mask,3:,:] = float('nan') # don't know sidechain information for masked seq 

    # Structure masking
    # str_mask = input_str_mask[0]
    # xyz_t[:,:,~str_mask,:,:] = float('nan') # NOTE: not using this because diffusion is effectively the mask 
    
    ### mask_msa ###
    ################
    # NOTE: this is for loss scoring
    mask_msa[:,:,:,~loss_seq_mask[0]] = False # n dimension is accounted for - NRB
    
    return seq, msa_masked, msa_full, xyz_t, t1d, mask_msa, t_list[:2], true_crds 
