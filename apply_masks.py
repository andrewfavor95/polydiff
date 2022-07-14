import torch
from icecream import ic
import random

ic.configureOutput(includeContext=True)

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
                true_crds_in=None):
    """
    Parameters:
        seq (torch.tensor, required): (B,I,L) integer sequence 

        msa_masked (torch.tensor, required): (B,I,N_short,L,48)

        msa_full  (torch,.tensor, required): (B,I,N_long,L,25)
        
        xyz_t (torch,tensor): (B,T,L,14,3) template crds BEFORE they go into get_init_xyz 
        
        t1d (torch.tensor, required): (B,I,L,22) this is the t1d before tacking on the chi angles 
        
        str_mask_1D (torch.tensor, required): Shape (L) rank 1 tensor where structure is masked at False positions 

        seq_mask_1D (torch.tensor, required): Shape (L) rank 1 tensor where seq is masked at False positions 

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
        t = random.randint(0,diffuser.T-1)
        t_list = [t]
        if predict_previous:
            # grab previous t. if t is 0 force a prediction of x_t=0
            tprev = t-1 if t > 0 else t
            t_list.append(tprev)

        kwargs = {'xyz'             :xyz_t.squeeze(),
                  'seq'             :seq.squeeze()[0],
                  'atom_mask'       :atom_mask.squeeze(),
                  'diffusion_mask'  :input_str_mask.squeeze(),
                  't_list':t_list}

        _,_,_,_,_,diffused_fullatoms, aa_masks = diffuser.diffuse_pose(**kwargs)


        seq_mask = torch.ones_like(seq.squeeze()[0]).to(dtype=bool) # all revealed 

        # grab noised inputs / create masks based on time t
        aa_mask_raw = aa_masks[t] 
    
        # mark False where aa_mask_raw is False -- assumes everybody is potentially diffused
        seq_mask[~aa_mask_raw] = False
        
        # reset to True any positions which aren't being diffused 
        seq_mask[input_seq_mask.squeeze()] = True
       
        xyz_t       = diffused_fullatoms[0][None,None]
        if predict_previous:
            true_crds = diffused_fullatoms[1][None]
        else:
            true_crds = true_crds_in 


        # scale confidence wrt t 
        # multiplicitavely applied to a default conf mask of 1.0 everywhwere 
        input_t1dconf_mask[~input_str_mask] = 1 - t/diffuser.T 

        mask_msa[:,:,:,seq_mask] = False # don't score revealed positions 
    else:
        print('WARNING: Diffuser not being used in apply masks')




    ###########
    B,_,_ = seq.shape
    assert B == 1, 'batch sizes > 1 not supported'
    #seq_mask = input_seq_mask[0] # DJ - old, commenting out bc using seq mask from diffuser 
    seq[:,:,~seq_mask] = 21 # mask token categorical value

    ### msa_masked ###
    ################## 
    msa_masked[:,:,:,~seq_mask,:20] = 0
    msa_masked[:,:,:,~seq_mask,21]  = 1 #set to mask token
    msa_masked[:,:,:,~seq_mask,20]  = 0    
    
    # index 44/45 is insertion/deletion
    # index 43 is the masked token NOTE check this
    # index 42 is the unknown token 
    msa_masked[:,:,:,~seq_mask,22:42] = 0
    msa_masked[:,:,:,~seq_mask,43] = 1 
    msa_masked[:,:,:,~seq_mask,42] = 0

    # insertion/deletion stuff 
    msa_masked[:,:,:,~seq_mask,44:46] = 0

    ### msa_full ### 
    ################
    msa_full[:,:,:,~seq_mask,:21] = 0
    msa_full[:,:,:,~seq_mask,21]  = 1
    msa_full[:,:,:,~seq_mask,-3]  = 0   #NOTE: double check this is insertions/deletions and 0 makes sense 

    ### t1d ###
    ########### 
    # NOTE: Not adjusting t1d last dim (confidence) from sequence mask
    t1d[:,:,~seq_mask,:20] = 0 
    t1d[:,:,~seq_mask,20]  = 1 # unknown

    t1d[:,:,:,21] *= input_t1dconf_mask

    xyz_t[:,:,~seq_mask,3:,:] = float('nan') # don't know sidechain information for masked seq 

    # Structure masking
    # str_mask = input_str_mask[0]
    # xyz_t[:,:,~str_mask,:,:] = float('nan')
    
    ### mask_msa ###
    ################
    # NOTE: this is for loss scoring
    mask_msa[:,:,:,~loss_seq_mask[0]] = False
    
    return seq, msa_masked, msa_full, xyz_t, t1d, mask_msa, t, true_crds 
