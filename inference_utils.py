from kinematics import xyz_to_t2d
import torch
import torch.nn.functional as nn
from util import get_torsions
from icecream import ic
from util import torsion_indices as ti_dev
from util import torsion_can_flip as ti_flip
from util import reference_angles as ang_ref
def preprocess(seq, xyz_t, t, device):
    """
    Function to prepare inputs to diffusion model
        
        seq (torch.tensor, required): (L) integer sequence 

        msa_masked (torch.tensor, required): (1,1,L,48)

        msa_full  (torch,.tensor, required): (1,1, L,25)
        
        xyz_t (torch,tensor): (L,14,3) template crds (diffused) 
        
        t1d (torch.tensor, required): (1,L,22) this is the t1d before tacking on the chi angles. Last plane is 1/t (conf hacked as timestep)
    """
    L = seq.shape[-1]
    ### msa_masked ###
    ##################
    msa_masked = torch.zeros((1,1,L,48))
    msa_masked[:,:,:,:22] = nn.one_hot(seq, num_classes=22)[None, None]
    msa_masked[:,:,:,22:44] = nn.one_hot(seq, num_classes=22)[None, None]

    ### msa_full ###
    ################
    msa_full = torch.zeros((1,1,L,25))
    msa_full[:,:,:,:22] = nn.one_hot(seq, num_classes=22)[None, None]

    ### t1d ###
    ########### 
    # NOTE: Not adjusting t1d last dim (confidence) from sequence mask
    t1d = torch.zeros((1,1,L,21))
    t1d[:,:,:,:21] = nn.one_hot(torch.where(seq == 21, 20, seq), num_classes=21)[None,None]
    
    conf = torch.where(seq == 21, 1/t, 1.)[None,None,...,None]
    t1d = torch.cat((t1d, conf), dim=-1)

    ### xyz_t ###
    #############
    xyz_t[torch.where(seq == 21, True, False),3:,:] = float('nan')
    xyz_t=xyz_t[None, None]
    xyz_t = torch.cat((xyz_t, torch.full((1,1,L,13,3), float('nan'))), dim=3)
    
    ### t2d ###
    ###########
    t2d = xyz_to_t2d(xyz_t)
  
    ### idx ###
    ###########
    idx = torch.arange(L)[None]

    ### alpha_t ###
    ###############
    seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,L)
    alpha, _, alpha_mask, _ = get_torsions(xyz_t.reshape(-1,L,27,3), seq_tmp, ti_dev, ti_flip, ang_ref)
    alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
    alpha[torch.isnan(alpha)] = 0.0
    alpha = alpha.reshape(1,-1,L,10,2)
    alpha_mask = alpha_mask.reshape(1,-1,L,10,1)
    alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 30)
    
    #put tensors on device
    msa_masked = msa_masked.to(device)
    msa_full = msa_full.to(device)
    seq = seq.to(device)
    xyz_t = xyz_t.to(device)
    idx = idx.to(device)
    t1d = t1d.to(device)
    t2d = t2d.to(device)
    alpha_t = alpha_t.to(device)
    return msa_masked, msa_full, seq[None], torch.squeeze(xyz_t, dim=0), idx, t1d, t2d, xyz_t, alpha_t

