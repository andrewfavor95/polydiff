import torch
from icecream import ic

ic.configureOutput(includeContext=True)

def mask_inputs(seq,
    msa_masked,
    msa_full,
    xyz_t,
    t1d,
    atom_mask, 
    input_seq_mask=None,
    input_str_mask=None,
    input_floating_mask=None,
    input_t1dconf_mask=None,
    loss_seq_mask=None,
    loss_str_mask=None,
    diffuser=None,
    diffusion_mask=None):
    """
    Parameters:
        seq (torch.tensor, required): (B,I,L) integer sequence 

        msa_masked (torch.tensor, required): (B,I,N_short,L,48)

        msa_full  (torch,.tensor, required): (B,I,N_long,L,25)
        
        xyz_t (torch,tensor): (B,T,L,14,3) template crds BEFORE they go into get_init_xyz 
        
        t1d (torch.tensor, required): (B,I,L,22) this is the t1d before tacking on the chi angles 
        
        str_mask_1D (torch.tensor, required): Shape (L) rank 1 tensor where structure is masked at False positions 

        seq_mask_1D (torch.tensor, required): Shape (L) rank 1 tensor where seq is masked at False positions 
    """

    ### Perform diffusion, pick a random t and then let that be 
    if (not diffuser is None) and (not diffusion_mask is None):
        kwargs = {'xyz':xyz_t,
                  'seq':seq,
                  'atom_mask':atom_mask,
                  'diffusion_mask':diffusion_mask}

        _,_,_,_,_, diffused_fullatoms, aa_masks = diffuser.diffuse_pose(**kwargs)

        # now pick t 
        t = random.randint(0,T-1)

        seq_mask = aa_masks[t] 
        xyz_t    = diffused_fullatoms[t]



    ###########
    B,_,_ = seq.shape
    assert B == 1, 'batch sizes > 1 not supported'
    seq_mask = input_seq_mask[0]
    seq[:,:,~seq_mask] = 20 # mask token categorical value

    ### msa_masked ###
    ################## 
    msa_masked[:,:,:,~seq_mask,:20] = 0
    msa_masked[:,:,:,~seq_mask,21]  = 0
    msa_masked[:,:,:,~seq_mask,20]  = 1     # set to the unkown char
    
    # index 44/45 is insertion/deletion
    # index 43 is the unknown token
    # index 42 is the masked token 
    msa_masked[:,:,:,~seq_mask,22:42] = 0
    msa_masked[:,:,:,~seq_mask,43] = 0 
    msa_masked[:,:,:,~seq_mask,42] = 1

    # insertion/deletion stuff 
    msa_masked[:,:,:,~seq_mask,44:46] = 0

    ### msa_full ### 
    ################
    msa_full[:,:,:,~seq_mask,:20] = 0
    msa_full[:,:,:,~seq_mask,21]  = 0
    msa_full[:,:,:,~seq_mask,20]  = 1 
    msa_full[:,:,:,~seq_mask,-3]  = 0   #NOTE: double check this is insertions/deletions and 0 makes sense 

    ### t1d ###
    ########### 
    # NOTE: Not adjusting t1d last dim (confidence) from sequence mask
    t1d[:,:,~seq_mask,:20] = 0 
    t1d[:,:,~seq_mask,20]  = 1 # unknown

    t1d[:,:,:,21] *= input_t1dconf_mask

    xyz_t[:,:,~seq_mask,3:,:] = float('nan')

    # Structure masking
    str_mask = input_str_mask[0]
    xyz_t[:,:,~str_mask,:,:] = float('nan')
    
    return seq, msa_masked, msa_full, xyz_t, t1d

def get_loss_masks(mask_msa, loss_seq_mask, loss_str_mask, loss_str_mask_2d):

    mask_2d = loss_str_mask_2d
    mask_crds = loss_str_mask

    B, I, M, L = mask_msa.shape
    assert B==1, 'batch sizes > 1 are not currently supported'
    mask_msa = mask_msa * loss_seq_mask

    return mask_crds, mask_2d, mask_msa




