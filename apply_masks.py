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
        t = random.randint(1,diffuser.T)
        if t == diffuser.T: t_list = [t,t]
        else: t_list = [t+1,t]
        if predict_previous:
            # grab previous t. if t is 0 force a prediction of x_t=0
            tprev = t-1 if t > 0 else t
            t_list.append(tprev)

        assert(seq.shape[1] == 1), "Number of repeats of seq must be 1"

        kwargs = {'xyz'             :xyz_t.squeeze(),
                  'seq'             :seq.squeeze(0)[0],
                  'atom_mask'       :atom_mask.squeeze(),
                  'diffusion_mask'  :input_str_mask.squeeze(),
                  't_list':t_list}

        diffused_fullatoms, aa_masks, true_crds = diffuser.diffuse_pose(**kwargs)
        

        if predict_previous: assert(diffused_fullatoms.shape[0] == 3)
        else: assert(diffused_fullatoms.shape[0] == 2)

        L = seq.shape[-1]

        seq_mask = torch.ones(2,L).to(dtype=bool) # all revealed [t,L]

        # grab noised inputs / create masks based on time t
        #aa_mask_raw = aa_masks # [t,L]
    
        # mark False where aa_mask_raw is False -- assumes everybody is potentially diffused
        seq_mask[0,~aa_masks[0]] = False
        seq_mask[1,~aa_masks[1]] = False

        # reset to True any positions which aren't being diffused 
        seq_mask[:,input_seq_mask.squeeze()] = True
       
        xyz_t       = diffused_fullatoms[:2].unsqueeze(0).unsqueeze(2) # [B,n,T,L,27,3]
        if predict_previous:
            true_crds = diffused_fullatoms[-1][None]


        # scale confidence wrt t 
        # multiplicitavely applied to a default conf mask of 1.0 everywhwere 
        input_t1dconf_mask = torch.stack([input_t1dconf_mask,input_t1dconf_mask], dim=0) # [n,I,L]
        input_t1dconf_mask[0,~input_str_mask] = 1 - t_list[0]/diffuser.T
        input_t1dconf_mask[1,~input_str_mask] = 1 - t_list[1]/diffuser.T

        mask_msa = torch.stack([mask_msa,mask_msa], dim=3) # [B,I,N_long,n,L,25]
        mask_msa[:,:,:,seq_mask] = False # don't score revealed positions 

        mask_msa = mask_msa.transpose(1,3) # [B,n,I,N_long,L,25]
    else:
        print('WARNING: Diffuser not being used in apply masks')


    ###########
    B,_,_ = seq.shape
    assert B == 1, 'batch sizes > 1 not supported'
    #seq_mask = input_seq_mask[0] # DJ - old, commenting out bc using seq mask from diffuser 

    seq = torch.stack([seq,seq], dim=2) # [B,I,n,L]
    seq[:,:,~seq_mask] = 21 # mask token categorical value
    seq = seq.transpose(1,2) # [B,n,I,L]

    ### msa_masked ###
    ################## 
    msa_masked = torch.stack([msa_masked,msa_masked], dim=1) # [B,n,I,N_short,L,48]
    msa_masked[:,0,:,:,~seq_mask[0],:21] = 0
    msa_masked[:,1,:,:,~seq_mask[0],:21] = 0

    msa_masked[:,0,:,:,~seq_mask[0],21]  = 1 # set to mask token
    msa_masked[:,1,:,:,~seq_mask[1],21]  = 1 # set to mask token

    # index 44/45 is insertion/deletion
    # index 43 is the masked token NOTE check this
    # index 42 is the unknown token 
    msa_masked[:,0,:,:,~seq_mask[0],22:43] = 0
    msa_masked[:,1,:,:,~seq_mask[1],22:43] = 0

    msa_masked[:,0,:,:,~seq_mask[0],43]    = 1
    msa_masked[:,1,:,:,~seq_mask[1],43]    = 1 

    # insertion/deletion stuff 
    msa_masked[:,0,:,:,~seq_mask[0],44:46] = 0
    msa_masked[:,1,:,:,~seq_mask[1],44:46] = 0

    ### msa_full ### 
    ################
    msa_full = torch.stack([msa_full,msa_full], dim=1) # [B,n,I,N_long,L,25]
    msa_full[:,0,:,:,~seq_mask[0],:21] = 0
    msa_full[:,1,:,:,~seq_mask[1],:21] = 0

    msa_full[:,0,:,:,~seq_mask[0],21]  = 1
    msa_full[:,1,:,:,~seq_mask[1],21]  = 1

    msa_full[:,0,:,:,~seq_mask[0],-3]  = 0   
    msa_full[:,1,:,:,~seq_mask[1],-3]  = 0   #NOTE: double check this is insertions/deletions and 0 makes sense 

    ### t1d ###
    ########### 
    # NOTE: Not adjusting t1d last dim (confidence) from sequence mask

    t1d = torch.stack([t1d,t1d], dim=1) # [B,n,I,L,22]
    t1d[:,0,:,~seq_mask[0],:20] = 0 
    t1d[:,1,:,~seq_mask[1],:20] = 0 

    t1d[:,0,:,~seq_mask[0],20]  = 1
    t1d[:,1,:,~seq_mask[1],20]  = 1 # unknown

    # input_t1d_conf_mask is shape [n,I,L]
    t1d[:,:,:,:,21] *= input_t1dconf_mask[None]

    # TRYING NOT PROVIDING diffused SIDECHAINS TO THE MODEL
    # xyz_t[:,:,:,~input_str_mask.squeeze(),3:,:] = float('nan')

    # Try providing no sidechains to the model - NRB
    xyz_t[:,:,:,:,3:,:] = float('nan')

    #xyz_t[:,:,~seq_mask,3:,:] = float('nan') # don't know sidechain information for masked seq 

    # Structure masking
    # str_mask = input_str_mask[0]
    # xyz_t[:,:,~str_mask,:,:] = float('nan')
    
    ### mask_msa ###
    ################
    # NOTE: this is for loss scoring
    mask_msa[:,:,:,:,~loss_seq_mask[0]] = False # n dimension is accounted for - NRB
    
    return seq, msa_masked, msa_full, xyz_t, t1d, mask_msa, t_list[:2], true_crds 
