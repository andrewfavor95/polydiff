import sys
import warnings
import assertpy

import numpy as np
import torch
import warnings
from assertpy import assert_that

import scipy.sparse
import scipy.sparse.csgraph
import networkx as nx
import itertools
from itertools import combinations
from collections import OrderedDict
from openbabel import openbabel
from scipy.spatial.transform import Rotation
from icecream import ic
import math

import rf2aa.chemical as chemical
from rf2aa.chemical import *
from rf2aa.kinematics import get_atomize_protein_chirals
from rf2aa.scoring import *
from pdb import set_trace

DEFAULT_SILENT_HEADER='SEQUENCE: VRGGALVRAIAALTGKSKRRIYGAIGGRARRWGISRRTAAALILAYLLAHGSWFGKSPREIAAAYAAAVAgcagatctgcacatcatgtgcagatctgc\nSCORE:     score     fa_atr     fa_rep     fa_sol    fa_intra_rep    fa_intra_sol_xover4    lk_ball_wtd    fa_elec    pro_close    hbond_sr_bb    hbond_lr_bb    hbond_bb_sc    hbond_sc    dslf_fa13      omega     fa_dun    p_aa_pp    yhh_planarity        ref    rama_prepro       time                description\nREMARK BINARY SILENTFILE\n'


def random_rot_trans(xyz, random_noise=20.0):
    # xyz: (N, L, 27, 3)
    N, L = xyz.shape[:2]

    # pick random rotation axis
    R_mat = torch.tensor(Rotation.random(N).as_matrix(), dtype=xyz.dtype).to(xyz.device)
    xyz = torch.einsum('nij,nlaj->nlai', R_mat, xyz) + torch.rand(N,1,1,3, device=xyz.device)*random_noise
    return xyz

def get_prot_sm_mask(atom_mask, seq):
    """
    Parameters
    ----------
    atom_mask : (..., L, Natoms) 
    seq : (L) 

    Returns
    -------
    mask : (..., L) 
    """
    sm_mask = is_atom(seq).to(atom_mask.device) # (L)
    # Asserting that atom_mask is full for masked regions of proteins [should be]
    has_backbone = atom_mask[...,:3].all(dim=-1)
    # has_backbone_prot = has_backbone[...,~sm_mask]
    # n_protein_with_backbone = has_backbone.sum()
    # n_protein = (~sm_mask).sum()
    #assert_that((n_protein/n_protein_with_backbone).item()).is_greater_than(0.8)
    mask_prot = has_backbone & ~sm_mask # valid protein/NA residues (L)
    mask_ca_sm = atom_mask[...,1] & sm_mask # valid sm mol positions (L)

    mask = mask_prot | mask_ca_sm # valid positions
    return mask

def center_and_realign_missing(xyz, mask_t, seq=None, same_chain=None, should_center: bool = True):
    """
    Moves center of mass of xyz to origin, then moves positions with missing
    coordinates to nearest existing residue on same chain.

    Parameters
    ----------
    seq : (L)
    xyz : (L, Natms, 3)
    mask_t : (L, Natms)
    same_chain : (L, L)

    Returns
    -------
    xyz : (L, Natms, 3)
    
    """
    L = xyz.shape[0]

    if same_chain is None:
        same_chain = torch.full((L,L), True)

    # valid protein/NA/small mol. positions
    if seq is None:
        mask = torch.full((L,), True)
    else:
        mask = get_prot_sm_mask(mask_t, seq)

    # center c.o.m of existing residues at the origin
    if should_center:
        center_CA = xyz[mask,1].mean(dim=0) # (3)
        xyz = torch.where(mask.view(L,1,1), xyz - center_CA.view(1, 1, 3), xyz)

    # move missing residues to the closest valid residues on same chain
    exist_in_xyz = torch.where(mask)[0] # (L_sub)
    same_chain_in_xyz = same_chain[:,mask].bool() # (L, L_sub)
    seqmap = (torch.arange(L, device=xyz.device)[:,None] - exist_in_xyz[None,:]).abs() # (L, L_sub)
    seqmap[~same_chain_in_xyz] += 99999
    seqmap = torch.argmin(seqmap, dim=-1) # (L)
    idx = torch.gather(exist_in_xyz, 0, seqmap) # (L)
    offset_CA = torch.gather(xyz[:,1], 0, idx.reshape(L,1).expand(-1,3))
    has_neighbor = same_chain_in_xyz.all(-1) 
    offset_CA[~has_neighbor] = 0 # stay at origin if nothing on same chain has coords
    xyz = torch.where(mask.view(L, 1, 1), xyz, xyz + offset_CA.reshape(L,1,3))

    return xyz


def th_ang_v(ab,bc,eps:float=1e-8):
    def th_norm(x,eps:float=1e-8):
        return x.square().sum(-1,keepdim=True).add(eps).sqrt()
    def th_N(x,alpha:float=0):
        return x/th_norm(x).add(alpha)
    ab, bc = th_N(ab),th_N(bc)
    cos_angle = torch.clamp( (ab*bc).sum(-1), -1, 1)
    sin_angle = torch.sqrt(1-cos_angle.square() + eps)
    dih = torch.stack((cos_angle,sin_angle),-1)
    return dih

def th_dih_v(ab,bc,cd):
    def th_cross(a,b):
        a,b = torch.broadcast_tensors(a,b)
        return torch.cross(a,b, dim=-1)
    def th_norm(x,eps:float=1e-8):
        return x.square().sum(-1,keepdim=True).add(eps).sqrt()
    def th_N(x,alpha:float=0):
        return x/th_norm(x).add(alpha)

    ab, bc, cd = th_N(ab),th_N(bc),th_N(cd)
    n1 = th_N( th_cross(ab,bc) )
    n2 = th_N( th_cross(bc,cd) )
    sin_angle = (th_cross(n1,bc)*n2).sum(-1)
    cos_angle = (n1*n2).sum(-1)
    dih = torch.stack((cos_angle,sin_angle),-1)
    return dih

def th_dih(a,b,c,d):
    return th_dih_v(a-b,b-c,c-d)

# build a frame from 3 points
#fd  -  more complicated version splits angle deviations between CA-N and CA-C (giving more accurate CB position)
#fd  -  makes no assumptions about input dims (other than last 1 is xyz)
def rigid_from_3_points(N, Ca, C, is_na=None, non_ideal=False, eps=1e-4):
    dims = N.shape[:-1]

    v1 = C-Ca
    v2 = N-Ca
    e1 = v1/(torch.norm(v1, dim=-1, keepdim=True)+eps)
    u2 = v2-(torch.einsum('...li, ...li -> ...l', e1, v2)[...,None]*e1)
    e2 = u2/(torch.norm(u2, dim=-1, keepdim=True)+eps)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.cat([e1[...,None], e2[...,None], e3[...,None]], axis=-1) #[B,L,3,3] - rotation matrix

    v2 = v2/(torch.norm(v2, dim=-1, keepdim=True)+eps)
    cosref = torch.sum(e1*v2, dim=-1)

    costgt = torch.full(dims, -0.3616, device=N.device)
    
    if is_na is not None:
        # costgt[is_na] = -0.2744
        costgt[..., is_na] = -0.2744



    cos2del = torch.clamp( cosref*costgt + torch.sqrt((1-cosref*cosref)*(1-costgt*costgt)+eps), min=-1.0, max=1.0 )

    cosdel = torch.sqrt(0.5*(1+cos2del)+eps)

    sindel = torch.sign(costgt-cosref) * torch.sqrt(1-0.5*(1+cos2del)+eps)

    Rp = torch.eye(3, device=N.device).repeat(*dims,1,1)
    Rp[...,0,0] = cosdel
    Rp[...,0,1] = -sindel
    Rp[...,1,0] = sindel
    Rp[...,1,1] = cosdel
    R = torch.einsum('...ij,...jk->...ik', R,Rp)

    # af: not sure if this is left over from some bad old crap, but let's see...
    if non_ideal:
        v2 = v2/(torch.norm(v2, dim=-1, keepdim=True)+eps)
        cosref = torch.sum(e1*v2, dim=-1) # cosine of current N-CA-C bond angle
        costgt = cos_ideal_NCAC.item()
        cos2del = torch.clamp( cosref*costgt + torch.sqrt((1-cosref*cosref)*(1-costgt*costgt)+eps), min=-1.0, max=1.0 )
        cosdel = torch.sqrt(0.5*(1+cos2del)+eps)
        sindel = torch.sign(costgt-cosref) * torch.sqrt(1-0.5*(1+cos2del)+eps)
        Rp = torch.eye(3, device=N.device).repeat(*dims,1,1)
        Rp[:,:,0,0] = cosdel
        Rp[:,:,0,1] = -sindel
        Rp[:,:,1,0] = sindel
        Rp[:,:,1,1] = cosdel
    
        R = torch.einsum('blij,bljk->blik', R,Rp)

    return R, Ca

# note: needs consistency with chemical.py
def is_protein(seq):
    return seq < NPROTAAS

def is_nucleic(seq):
    return (seq>=NPROTAAS) * (seq <= NNAPROTAAS)

# fd hacky
def is_DNA(seq):
    return (seq>=NPROTAAS) * (seq < NPROTAAS+5)

# fd hacky
def is_RNA(seq):
    return (seq>=NPROTAAS+5) * (seq < NNAPROTAAS)


def is_atom(seq):
    return seq > NNAPROTAAS


def idealize_reference_frame(seq, xyz_in):
    xyz = xyz_in.clone()

    namask = is_nucleic(seq)
    Rs, Ts = rigid_from_3_points(xyz[...,0,:],xyz[...,1,:],xyz[...,2,:], namask)

    protmask = ~namask

    pmask_bs,pmask_rs = protmask.nonzero(as_tuple=True)
    nmask_bs,nmask_rs = namask.nonzero(as_tuple=True)
    xyz[pmask_bs,pmask_rs,0,:] = torch.einsum('...ij,j->...i', Rs[pmask_bs,pmask_rs], chemical.init_N.to(device=xyz_in.device) ) + Ts[pmask_bs,pmask_rs]
    xyz[pmask_bs,pmask_rs,2,:] = torch.einsum('...ij,j->...i', Rs[pmask_bs,pmask_rs], chemical.init_C.to(device=xyz_in.device) ) + Ts[pmask_bs,pmask_rs]
    xyz[nmask_bs,nmask_rs,0,:] = torch.einsum('...ij,j->...i', Rs[nmask_bs,nmask_rs], chemical.init_O4p.to(device=xyz_in.device) ) + Ts[nmask_bs,nmask_rs]
    xyz[nmask_bs,nmask_rs,2,:] = torch.einsum('...ij,j->...i', Rs[nmask_bs,nmask_rs], chemical.init_C2p.to(device=xyz_in.device)) + Ts[nmask_bs,nmask_rs]

    return xyz

def xyz_to_frame_xyz(xyz, seq_unmasked, atom_frames):
    """
    xyz (1, L, natoms, 3)
    seq_unmasked (1, L)
    atom_frames (1, L, 3, 2)
    """ 
    xyz_frame = xyz.clone()
    atoms = is_atom(seq_unmasked)
    if torch.all(~atoms):
        return xyz_frame

    atom_crds = xyz_frame[atoms]
    atom_L, natoms, _ = atom_crds.shape
    frames_reindex = torch.zeros(atom_frames.shape[:-1])
    
    for i in range(atom_L):
        frames_reindex[:, i, :] = (i+atom_frames[..., i, :, 0])*natoms + atom_frames[..., i, :, 1]
    frames_reindex = frames_reindex.long()

    xyz_frame[atoms, :, :3] = atom_crds.reshape(atom_L*natoms, 3)[frames_reindex]
    return xyz_frame

def xyz_frame_from_rotation_mask(xyz,rotation_mask, atom_frames):
    """
    function to get xyz_frame for l1 feature in Structure module
    xyz (1, L, natoms, 3)
    rotation_mask (1, L)
    atom_frames (1, L, 3, 2)
    """
    xyz_frame = xyz.clone()
    if torch.all(~rotation_mask):
        return xyz_frame

    atom_crds = xyz_frame[rotation_mask]
    atom_L, natoms, _ = atom_crds.shape
    frames_reindex = torch.zeros(atom_frames.shape[:-1])
    
    for i in range(atom_L):
        frames_reindex[:, i, :] = (i+atom_frames[..., i, :, 0])*natoms + atom_frames[..., i, :, 1]
    frames_reindex = frames_reindex.long()
    xyz_frame[rotation_mask, :, :3] = atom_crds.reshape(atom_L*natoms, 3)[frames_reindex]
    return xyz_frame

def xyz_t_to_frame_xyz(xyz_t, seq_unmasked, atom_frames):
    """
    Parameters:
        xyz_t (1, T, L, natoms, 3)
        seq_unmasked (B, L)
        atom_frames (1, A, 3, 2)
    Returns:
        xyz_t_frame (B, T, L, natoms, 3)
    """
    is_sm = is_atom(seq_unmasked[0])
    return xyz_t_to_frame_xyz_sm_mask(xyz_t, is_sm, atom_frames)

def xyz_t_to_frame_xyz_sm_mask(xyz_t, is_sm, atom_frames):
    """
    Parameters:
        xyz_t (1, T, L, natoms, 3)
        is_sm (L)
        atom_frames (1, A, 3, 2)
    Returns:
    xyz_t_frame (B, T, L, natoms, 3)
    """
    # ic(xyz_t.shape, is_sm.shape, atom_frames.shape)
    # xyz_t.shape: torch.Size([1, 1, 194, 36, 3]) 
    # is_sm.shape: torch.Size([194])
    # atom_frames.shape: torch.Size([1, 29, 3, 2])
    xyz_t_frame = xyz_t.clone()
    atoms = is_sm
    if torch.all(~atoms):
        return xyz_t_frame
    atom_crds_t = xyz_t_frame[:, :, atoms]

    B, T, atom_L, natoms, _ = atom_crds_t.shape
    frames_reindex = torch.zeros(atom_frames.shape[:-1])
    for i in range(atom_L):
        frames_reindex[:, i, :] = (i+atom_frames[..., i, :, 0])*natoms + atom_frames[..., i, :, 1]
    frames_reindex = frames_reindex.long()
    xyz_t_frame[:, :, atoms, :3] = atom_crds_t.reshape(T, atom_L*natoms, 3)[:, frames_reindex.squeeze(0)]
    return xyz_t_frame


def str_doesnt_have(mystr, frags):
    """
    detects if any fragment in frags is in mystr
    """
    for frag in frags:
        if frag in mystr:
            return False
    return True

def resolve_loss_summation(tot_loss, 
                           loss_dict, 
                           loss_weights, 
                           had_sm, 
                           had_prot_motif,
                           had_ss_shown,
                           below_fa_size_limit,
                           ):
    """
    Conditional loss summation depending on content of training example
    """

    for k, loss in loss_dict.items():


            # ensuring rmsd calc doesn't turn total loss to nan
            if str_doesnt_have(k, ['rmsd']):
                
                
                # ensure had a ligand for ligand FAPE, else nan
                if ('lig' in k):
                    if had_sm:
                        
                        if k != 'prot_lig_inter_fape':
                            weight = loss_weights[k]
                            tot_loss += loss*weight
                            assert not math.isnan(loss), f'{k} is nan'
                        
                        
                        else:
                            if had_prot_motif:
                                weight = loss_weights[k]
                                tot_loss += loss*weight
                                assert not math.isnan(loss), f'{k} is nan'
                            else:
                                assert math.isnan(loss) # no prot motif, prot-sm-inter must be nan

                    else:
                        assert math.isnan(loss)
                        pass

                
                # ensure we had motif for motif fape 
                elif (k == 'motif_fape'): 
                    if had_prot_motif: 
                        weight = loss_weights[k]
                        tot_loss += loss*weight
                        assert not math.isnan(loss), f'{k} is nan'
                    else:
                        assert math.isnan(loss)
                        pass 

                # ensure input is small enough for fa fape (memory issues)
                elif k in ['fa_fape','inter_fape','atom_bond_loss','skip_bond_loss','rigid_loss']: 
                    if below_fa_size_limit: 
                        weight = loss_weights[k]
                        tot_loss += loss*weight
                        assert not math.isnan(loss), f'{k} is nan'
                    else:
                        assert math.isnan(loss)
                        pass 
                        

                # ensure we had ss shown for ss fape
                elif (k == 'ss_fape'): 
                    if had_ss_shown: 
                        weight = loss_weights[k]
                        tot_loss += loss*weight
                        assert not math.isnan(loss), f'{k} is nan'
                    else:
                        assert math.isnan(loss)
                        pass 


                else: 
                    weight = loss_weights[k]
                    tot_loss += loss*weight
                    assert not math.isnan(loss), f'{k} is nan'



    
    return tot_loss



def get_frames(xyz_in, xyz_mask, seq, frame_indices, atom_frames=None):
    B,L,natoms = xyz_in.shape[:3]
    frames = frame_indices[seq]
    atoms = is_atom(seq)
    if torch.any(atoms):
        frames[:,atoms[0].nonzero().flatten(), 0] = atom_frames

    frame_mask = ~torch.all(frames[...,0, :] == frames[...,1, :], axis=-1)

    # frame_mask *= torch.all(
    #     torch.gather(xyz_mask,2,frames.reshape(B,L,-1)).reshape(B,L,-1,3),
    #     axis=-1)

    return frames, frame_mask

def get_tips(xyz, seq):
    B,L = xyz.shape[:2]

    xyz_tips = torch.gather(xyz, 2, tip_indices.to(xyz.device)[seq][:,:,None,None].expand(-1,-1,-1,3)).reshape(B, L, 3)
    if torch.isnan(xyz_tips).any(): # replace NaN tip atom with virtual Cb atom
        # three anchor atoms
        N  = xyz[:,:,0]
        Ca = xyz[:,:,1]
        C  = xyz[:,:,2]

        # recreate Cb given N,Ca,C
        b = Ca - N
        c = C - Ca
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca    

        xyz_tips = torch.where(torch.isnan(xyz_tips), Cb, xyz_tips)
    return xyz_tips

def superimpose(pred, true, atom_mask):
    
    def centroid(X):
        return X.mean(dim=-2, keepdim=True)
    
    B, L, natoms = pred.shape[:3]

    # center to centroid
    pred_allatom = pred[atom_mask][None]
    true_allatom = true[atom_mask][None]

    cp = centroid(pred_allatom)
    ct = centroid(true_allatom)
    
    pred_allatom_origin = pred_allatom - cp
    true_allatom_origin = true_allatom - ct

    # Computation of the covariance matrix
    C = torch.matmul(pred_allatom_origin.permute(0,2,1), true_allatom_origin)

    # Compute optimal rotation matrix using SVD
    V, S, W = torch.svd(C)

    # get sign to ensure right-handedness
    d = torch.ones([B,3,3], device=pred.device)
    d[:,:,-1] = torch.sign(torch.det(V)*torch.det(W)).unsqueeze(1)

    # Rotation matrix U
    U = torch.matmul(d*V, W.permute(0,2,1)) # (IB, 3, 3)
    pred_rms = pred - cp
    true_rms = true - ct
    
    # Rotate pred
    rP = torch.matmul(pred_rms, U) # (IB, L*3, 3)
    
    return rP+ct



def writepdb(filename, *args, file_mode='w', **kwargs, ):
    f = open(filename, file_mode)
    writepdb_file(f, *args, **kwargs)

def writepdb_file(f, atoms, seq, modelnum=None, chain="A", idx_pdb=None, bfacts=None, 
             bond_feats=None, file_mode="w",atom_mask=None, atom_idx_offset=0, chain_Ls=None,
             remap_atomtype=True, lig_name='LG1', atom_names=None, chain_letters=None, rna_oneletter=False, natoms=23):
             # remap_atomtype=True, lig_name='LG1', atom_names=None, chain_letters=None, rna_oneletter=False,):
    #ic(atoms.shape, seq.shape, bond_feats.shape)
    #ic(chain_Ls)

    # ic(seq)

    def _get_atom_type(atom_name):
        atype = ''
        if atom_name[0].isalpha():
            atype += atom_name[0]
        atype += atom_name[1]
        return atype

    def _get_std_alphabet(num2aa_in):
        """
        This should only be used when saving pdbs.
        The purpose is to switch the RNA 3-character code from RF conventions
        to the standard conventions which are used by pymol, mpnn, etc.
        In the future, we can extend this to all other pdb features that RF handles with irregular codes.
        """
        num2aa_out = num2aa_in.copy()

        if num2aa_out[27]==' RA':
            num2aa_out[27] = '  A'
        if num2aa_out[28]==' RC':
            num2aa_out[28] = '  C'
        if num2aa_out[29]==' RG':
            num2aa_out[29] = '  G'
        if num2aa_out[30]==' RU':
            num2aa_out[30] = '  U'
        if num2aa_out[31]==' RX':
            num2aa_out[31] = '  X'

        return num2aa_out

    # if needed, correct mistake in atomic number assignment in RF2-allatom (fold&dock 3 & earlier)
    atom_names_ = [
        "F",  "Cl", "Br", "I",  "O",  "S",  "Se", "Te", "N",  "P",  "As", "Sb",
        "C",  "Si", "Ge", "Sn", "Pb", "B",  "Al", "Zn", "Hg", "Cu", "Au", "Ni", 
        "Pd", "Pt", "Co", "Rh", "Ir", "Pr", "Fe", "Ru", "Os", "Mn", "Re", "Cr", 
        "Mo", "W",  "V",  "U",  "Tb", "Y",  "Be", "Mg", "Ca", "Li", "K",  "ATM"]
    atom_num = [
        9,    17,   35,   53,   8,    16,   34,   52,   7,    15,   33,   51,
        6,    14,   32,   50,   82,   5,    13,   30,   80,   29,   79,   28,
        46,   78,   27,   45,   77,   59,   26,   44,   76,   25,   75,   24,   
        42,   74,   23,   92,   65,   39,   4,    12,   20,   3,    19,   0] 
    atomnum2atomtype_ = dict(zip(atom_num,atom_names_))
    if remap_atomtype:
        atomtype_map = {v:atomnum2atomtype_[k] for k,v in chemical.atomnum2atomtype.items()}
    else:
        atomtype_map = {v:v for k,v in chemical.atomnum2atomtype.items()} # no change
        
    ctr = 1+atom_idx_offset
    scpu = seq.cpu().squeeze(0)
    atomscpu = atoms.cpu().squeeze(0)
    natoms_max = atomscpu.shape[-2]

    if bfacts is None:
        bfacts = torch.zeros(atomscpu.shape[0])
    if idx_pdb is None:
        idx_pdb = 1 + torch.arange(atomscpu.shape[0])

    alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')
    assert not (chain_Ls and chain_letters)
    if chain_letters is None:
        if chain_Ls is not None:
            chain_letters = np.concatenate([np.full(L, alphabet[i]) for i,L in enumerate(chain_Ls)])
        else:
            chain_letters = [chain]*len(scpu)
        
    if modelnum is not None:
        f.write(f"MODEL        {modelnum}\n")

    Bfacts = torch.clamp( bfacts.cpu(), 0, 1)
    atom_idxs = {}
    i_res_lig = 0

    # Make list of how many atoms to save:
    natoms_list = []
    for resi_i in seq:
        try:
            natoms_i = chemical.aa2long[resi_i].index(None)
        except ValueError:
            # If None is not in the list, set num_atoms_i to the length of the list
            # natoms_i = natoms
            natoms_i = min(natoms,natoms_max)

        # natoms_list.append(min(natoms,natoms_i))
        natoms_list.append(natoms_i)

    # set_trace()

    # Option to switch alphabet if requested by user
    if rna_oneletter: # Current option: RNA 3-letter alphabet to the standard 1-letter convention used by basically all non-RF software:
        _num2aa = _get_std_alphabet(num2aa)

    else: # I hate that I have to do this...
        _num2aa = num2aa.copy() # But I swear, it's the only way to avoid an error with the local variables.


    assert len(natoms_list)==len(scpu), "every element in seq must have an assigned number of atoms to save."
    
    for i_res,s,ch in zip(range(len(scpu)), scpu, chain_letters):
        # natoms = atomscpu.shape[-2]
        #if (natoms!=NHEAVY and natoms!=NTOTAL and natoms!=3):
        #    print ('bad size!', natoms, NHEAVY, NTOTAL, atoms.shape)
        #    assert(False)

        if s >= len(aa2long):
            atom_idxs[i_res] = ctr

            # hack to make sure H's are output properly (they are not in RFAA alphabet)
            if atom_names is not None:
                atom_type = _get_atom_type(atom_names[i_res_lig])
                atom_name = atom_names[i_res_lig]
            else:
                atom_type = atomtype_map[_num2aa[s]]
                atom_name = atom_type

            f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %+2s\n"%(
                    "HETATM", ctr, atom_name, lig_name,
                    ch, idx_pdb.max()+10, atomscpu[i_res,1,0], atomscpu[i_res,1,1], atomscpu[i_res,1,2],
                    1.0, Bfacts[i_res],  atom_type) )
            i_res_lig += 1
            ctr += 1
            continue

        atms = aa2long[s]
        # for i_atm,atm in enumerate(atms):
        prev_coords = torch.tensor([torch.nan, torch.nan, torch.nan])

        for i_atm,atm in enumerate(atms[:natoms_list[i_res]]):
            if atom_mask is not None and not atom_mask[i_res,i_atm]: continue # skip missing atoms

            if i_atm<natoms_max and torch.equal(atomscpu[i_res,i_atm,:], prev_coords): continue # skip atoms with duplicate coordinates

            if (i_atm<natoms_max and atm is not None and not torch.isnan(atomscpu[i_res,i_atm,:]).any()):
                f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                    "ATOM", ctr, atm, _num2aa[s],
                    ch, idx_pdb[i_res], atomscpu[i_res,i_atm,0], atomscpu[i_res,i_atm,1], atomscpu[i_res,i_atm,2],
                    1.0, Bfacts[i_res] ) )
                ctr += 1
                prev_coords = atomscpu[i_res,i_atm,:] # Updata the "previous coords" so we can compare

    if bond_feats != None:
        atom_bonds = (bond_feats > 0) * (bond_feats <5)
        atom_bonds = atom_bonds.cpu()
        b, i, j = atom_bonds.nonzero(as_tuple=True)
        for start, end in zip(i,j):
            f.write(f"CONECT{atom_idxs[int(start.cpu().numpy())]:5d}{atom_idxs[int(end.cpu().numpy())]:5d}\n")
    if modelnum is not None:
        f.write("ENDMDL\n")




# process ideal frames
def make_frame(X, Y):
    Xn = X / torch.linalg.norm(X)
    Y = Y - torch.dot(Y, Xn) * Xn
    Yn = Y / torch.linalg.norm(Y)
    Z = torch.cross(Xn,Yn)
    Zn =  Z / torch.linalg.norm(Z)
    return torch.stack((Xn,Yn,Zn), dim=-1)


# resolve tip atom indices
tip_indices = torch.full((NAATOKENS,), 0)
for i in range(NAATOKENS):
    if i > NNAPROTAAS-1:
        # all atoms are at index 1 in the atom array 
        tip_indices[i] = 1
    else:
        tip_atm = aa2tip[i]
        atm_long = aa2long[i]
        tip_indices[i] = atm_long.index(tip_atm)


# resolve torsion indices
#  a negative index indicates the previous residue
# order:
#    omega/phi/psi: 0-2
#    chi_1-4(prot): 3-6
#    cb/cg bend: 7-9
#    eps(p)/zeta(p): 10-11
#    alpha/beta/gamma/delta: 12-15
#    nu2/nu1/nu0: 16-18
#    chi_1(na): 19
torsion_indices = torch.full((NAATOKENS,NTOTALDOFS,4),0)
torsion_can_flip = torch.full((NAATOKENS,NTOTALDOFS),False,dtype=torch.bool)
for i in range(NPROTAAS):
    i_l, i_a = aa2long[i], aa2longalt[i]

    # protein omega/phi/psi
    torsion_indices[i,0,:] = torch.tensor([-1,-2,0,1]) # omega
    torsion_indices[i,1,:] = torch.tensor([-2,0,1,2]) # phi
    torsion_indices[i,2,:] = torch.tensor([0,1,2,3]) # psi (+pi)

    # protein chis
    for j in range(4):
        if torsions[i][j] is None:
            continue
        for k in range(4):
            a = torsions[i][j][k]
            torsion_indices[i,3+j,k] = i_l.index(a)
            if (i_l.index(a) != i_a.index(a)):
                torsion_can_flip[i,3+j] = True ##bb tors never flip

    # CB/CG angles (only masking uses these indices)
    torsion_indices[i,7,:] = torch.tensor([0,2,1,4]) # CB ang1
    torsion_indices[i,8,:] = torch.tensor([0,2,1,4]) # CB ang2
    torsion_indices[i,9,:] = torch.tensor([0,2,4,5]) # CG ang (arg 1 ignored)

# HIS is a special case for flip
torsion_can_flip[8,4]=False

# DNA/RNA
for i in range(NPROTAAS,NNAPROTAAS):
    torsion_indices[i,10,:] = torch.tensor([-2,-9,-10,4])  # epsilon_prev
    torsion_indices[i,11,:] = torch.tensor([-9,-10,4,6])   # zeta_prev
    torsion_indices[i,12,:] = torch.tensor([7,6,4,3])     # alpha c5'-o5'-p-op1
    torsion_indices[i,13,:] = torch.tensor([8,7,6,4])     # beta c4'-c5'-o5'-p
    torsion_indices[i,14,:] = torch.tensor([9,8,7,6])     # gamma c3'-c4'-c5'-o5'
    torsion_indices[i,15,:] = torch.tensor([2,9,8,7])     # delta c2'-c3'-c4'-c5'

    torsion_indices[i,16,:] = torch.tensor([1,2,9,8])     # nu2
    torsion_indices[i,17,:] = torch.tensor([0,1,2,9])     # nu1
    torsion_indices[i,18,:] = torch.tensor([2,1,0,8])     # nu0

    # NA chi
    if torsions[i][0] is not None:
        i_l = aa2long[i]
        for k in range(4):
            a = torsions[i][0][k]
            torsion_indices[i,19,k] = i_l.index(a) # chi

# build the mapping from atoms in the full rep (Nx27) to the "alternate" rep
allatom_mask = torch.zeros((NAATOKENS,NTOTAL), dtype=torch.bool)
long2alt = torch.zeros((NAATOKENS,NTOTAL), dtype=torch.long)
for i in range(NNAPROTAAS):
    i_l, i_lalt = aa2long[i],  aa2longalt[i]
    for j,a in enumerate(i_l):
        if (a is None):
            long2alt[i,j] = j
        else:
            long2alt[i,j] = i_lalt.index(a)
            allatom_mask[i,j] = True
for i in range(NNAPROTAAS, NAATOKENS):
    for j in range(NTOTAL):
        long2alt[i, j] = j
allatom_mask[NNAPROTAAS:,1] = True

# bond graph traversal
num_bonds = torch.zeros((NAATOKENS,NTOTAL,NTOTAL), dtype=torch.long)
for i in range(NNAPROTAAS):
    num_bonds_i = np.zeros((NTOTAL,NTOTAL))
    for (bnamei,bnamej) in aabonds[i]:
        bi,bj = aa2long[i].index(bnamei),aa2long[i].index(bnamej)
        num_bonds_i[bi,bj] = 1
    num_bonds_i = scipy.sparse.csgraph.shortest_path (num_bonds_i,directed=False)
    num_bonds_i[num_bonds_i>=4] = 4
    num_bonds[i,...] = torch.tensor(num_bonds_i)


# atom type indices
idx2aatype = []
for x in aa2type:
    for y in x:
        if y and y not in idx2aatype:
            idx2aatype.append(y)
aatype2idx = {x:i for i,x in enumerate(idx2aatype)}

# element indices
idx2elt = []
for x in aa2elt:
    for y in x:
        if y and y not in idx2elt:
            idx2elt.append(y)
elt2idx = {x:i for i,x in enumerate(idx2elt)}

# LJ/LK scoring parameters
atom_type_index = torch.zeros((NAATOKENS,NTOTAL), dtype=torch.long)
element_index = torch.zeros((NAATOKENS,NTOTAL), dtype=torch.long)

ljlk_parameters = torch.zeros((NAATOKENS,NTOTAL,5), dtype=torch.float)
lj_correction_parameters = torch.zeros((NAATOKENS,NTOTAL,4), dtype=bool) # donor/acceptor/hpol/disulf
for i in range(NNAPROTAAS):
    for j,a in enumerate(aa2type[i]):
        if (a is not None):
            atom_type_index[i,j] = aatype2idx[a]
            ljlk_parameters[i,j,:] = torch.tensor( type2ljlk[a] )
            lj_correction_parameters[i,j,0] = (type2hb[a]==HbAtom.DO)+(type2hb[a]==HbAtom.DA)
            lj_correction_parameters[i,j,1] = (type2hb[a]==HbAtom.AC)+(type2hb[a]==HbAtom.DA)
            lj_correction_parameters[i,j,2] = (type2hb[a]==HbAtom.HP)
            lj_correction_parameters[i,j,3] = (a=="SH1" or a=="HS")
    for j,a in enumerate(aa2elt[i]):
        if (a is not None):
            element_index[i,j] = elt2idx[a]

# hbond scoring parameters
def donorHs(D,bonds,atoms):
    dHs = []
    for (i,j) in bonds:
        if (i==D):
            idx_j = atoms.index(j)
            if (idx_j>=NHEAVY):  # if atom j is a hydrogen
                dHs.append(idx_j)
        if (j==D):
            idx_i = atoms.index(i)
            if (idx_i>=NHEAVY):  # if atom j is a hydrogen
                dHs.append(idx_i)
    assert (len(dHs)>0)
    return dHs

def acceptorBB0(A,hyb,bonds,atoms):
    if (hyb == HbHybType.SP2):
        for (i,j) in bonds:
            if (i==A):
                B = atoms.index(j)
                if (B<NHEAVY):
                    break
            if (j==A):
                B = atoms.index(i)
                if (B<NHEAVY):
                    break
        for (i,j) in bonds:
            if (i==atoms[B]):
                B0 = atoms.index(j)
                if (B0<NHEAVY):
                    break
            if (j==atoms[B]):
                B0 = atoms.index(i)
                if (B0<NHEAVY):
                    break
    elif (hyb == HbHybType.SP3 or hyb == HbHybType.RING):
        for (i,j) in bonds:
            if (i==A):
                B = atoms.index(j)
                if (B<NHEAVY):
                    break
            if (j==A):
                B = atoms.index(i)
                if (B<NHEAVY):
                    break
        for (i,j) in bonds:
            if (i==A and j!=atoms[B]):
                B0 = atoms.index(j)
                break
            if (j==A and i!=atoms[B]):
                B0 = atoms.index(i)
                break

    return B,B0


hbtypes = torch.full((NAATOKENS,NTOTAL,3),-1, dtype=torch.long) # (donortype, acceptortype, acchybtype)
hbbaseatoms = torch.full((NAATOKENS,NTOTAL,2),-1, dtype=torch.long) # (B,B0) for acc; (D,-1) for don
hbpolys = torch.zeros((HbDonType.NTYPES,HbAccType.NTYPES,3,15)) # weight,xmin,xmax,ymin,ymax,c9,...,c0

for i in range(NNAPROTAAS):
    for j,a in enumerate(aa2type[i]):
        if (a in type2dontype):
            j_hs = donorHs(aa2long[i][j],aabonds[i],aa2long[i])
            for j_h in j_hs:
                hbtypes[i,j_h,0] = type2dontype[a]
                hbbaseatoms[i,j_h,0] = j
        if (a in type2acctype):
            j_b, j_b0 = acceptorBB0(aa2long[i][j],type2hybtype[a],aabonds[i],aa2long[i])
            hbtypes[i,j,1] = type2acctype[a]
            hbtypes[i,j,2] = type2hybtype[a]
            hbbaseatoms[i,j,0] = j_b
            hbbaseatoms[i,j,1] = j_b0

for i in range(HbDonType.NTYPES):
    for j in range(HbAccType.NTYPES):
        weight = dontype2wt[i]*acctype2wt[j]

        pdist,pbah,pahd = hbtypepair2poly[(i,j)]
        xrange,yrange,coeffs = hbpolytype2coeffs[pdist]
        hbpolys[i,j,0,0] = weight
        hbpolys[i,j,0,1:3] = torch.tensor(xrange)
        hbpolys[i,j,0,3:5] = torch.tensor(yrange)
        hbpolys[i,j,0,5:] = torch.tensor(coeffs)
        xrange,yrange,coeffs = hbpolytype2coeffs[pahd]
        hbpolys[i,j,1,0] = weight
        hbpolys[i,j,1,1:3] = torch.tensor(xrange)
        hbpolys[i,j,1,3:5] = torch.tensor(yrange)
        hbpolys[i,j,1,5:] = torch.tensor(coeffs)
        xrange,yrange,coeffs = hbpolytype2coeffs[pbah]
        hbpolys[i,j,2,0] = weight
        hbpolys[i,j,2,1:3] = torch.tensor(xrange)
        hbpolys[i,j,2,3:5] = torch.tensor(yrange)
        hbpolys[i,j,2,5:] = torch.tensor(coeffs)

# cartbonded scoring parameters
# (0) inter-res
cb_lengths_CN = (1.32868, 369.445)
cb_angles_CACN = (2.02807,160)
cb_angles_CNCA = (2.12407,96.53)
cb_torsions_CACNH = (0.0,41.830) # also used for proline CACNCD
cb_torsions_CANCO = (0.0,38.668)

# note for the below, the extra amino acid corrsponds to cb params for HIS_D
# (1) intra-res lengths
cb_lengths = [[] for i in range(NAATOKENS+1)]
for cst in cartbonded_data_raw['lengths']:
    res_idx = aa2num[ cst['res'] ]
    cb_lengths[res_idx].append( (
        aa2long[res_idx].index(cst['atm1']),
        aa2long[res_idx].index(cst['atm2']),
        cst['x0'],cst['K']
    ) )
ncst_per_res=max([len(i) for i in cb_lengths])
cb_length_t = torch.zeros(NAATOKENS+1,ncst_per_res,4)
for i in range(NNAPROTAAS+1):
    src = i
    if (num2aa[i]=='UNK' or num2aa[i]=='MAS'):
        src=aa2num['ALA']
    if (len(cb_lengths[src])>0):
        cb_length_t[i,:len(cb_lengths[src]),:] = torch.tensor(cb_lengths[src])

# (2) intra-res angles
cb_angles = [[] for i in range(NAATOKENS+1)]
for cst in cartbonded_data_raw['angles']:
    res_idx = aa2num[ cst['res'] ]
    cb_angles[res_idx].append( (
        aa2long[res_idx].index(cst['atm1']),
        aa2long[res_idx].index(cst['atm2']),
        aa2long[res_idx].index(cst['atm3']),
        cst['x0'],cst['K']
    ) )
ncst_per_res=max([len(i) for i in cb_angles])
cb_angle_t = torch.zeros(NAATOKENS+1,ncst_per_res,5)
for i in range(NNAPROTAAS+1):
    src = i
    if (num2aa[i]=='UNK' or num2aa[i]=='MAS'):
        src=aa2num['ALA']

    if (len(cb_angles[src])>0):
        cb_angle_t[i,:len(cb_angles[src]),:] = torch.tensor(cb_angles[src])

# (3) intra-res torsions
cb_torsions = [[] for i in range(NAATOKENS+1)]
for cst in cartbonded_data_raw['torsions']:
    res_idx = aa2num[ cst['res'] ]
    cb_torsions[res_idx].append( (
        aa2long[res_idx].index(cst['atm1']),
        aa2long[res_idx].index(cst['atm2']),
        aa2long[res_idx].index(cst['atm3']),
        aa2long[res_idx].index(cst['atm4']),
        cst['x0'],cst['K'],cst['period']
    ) )
ncst_per_res=max([len(i) for i in cb_torsions])
cb_torsion_t = torch.zeros(NAATOKENS+1,ncst_per_res,7)
cb_torsion_t[...,6]=1.0 # periodicity
for i in range(NNAPROTAAS):
    src = i
    if (num2aa[i]=='UNK' or num2aa[i]=='MAS'):
        src=aa2num['ALA']

    if (len(cb_torsions[src])>0):
        cb_torsion_t[i,:len(cb_torsions[src]),:] = torch.tensor(cb_torsions[src])

# kinematic parameters
base_indices = torch.full((NAATOKENS,NTOTAL),0, dtype=torch.long) # base frame that builds each atom
xyzs_in_base_frame = torch.ones((NAATOKENS,NTOTAL,4)) # coords of each atom in the base frame
RTs_by_torsion = torch.eye(4).repeat(NAATOKENS,NTOTALTORS,1,1) # torsion frames
reference_angles = torch.ones((NAATOKENS,NPROTANGS,2)) # reference values for bendable angles

## PROTEIN
for i in range(NPROTAAS):
    i_l = aa2long[i]
    for name, base, coords in ideal_coords[i]:
        idx = i_l.index(name)
        base_indices[i,idx] = base
        xyzs_in_base_frame[i,idx,:3] = torch.tensor(coords)

    # omega frame
    RTs_by_torsion[i,0,:3,:3] = torch.eye(3)
    RTs_by_torsion[i,0,:3,3] = torch.zeros(3)

    # phi frame
    RTs_by_torsion[i,1,:3,:3] = make_frame(
        xyzs_in_base_frame[i,0,:3] - xyzs_in_base_frame[i,1,:3],
        torch.tensor([1.,0.,0.])
    )
    RTs_by_torsion[i,1,:3,3] = xyzs_in_base_frame[i,0,:3]

    # psi frame
    RTs_by_torsion[i,2,:3,:3] = make_frame(
        xyzs_in_base_frame[i,2,:3] - xyzs_in_base_frame[i,1,:3],
        xyzs_in_base_frame[i,1,:3] - xyzs_in_base_frame[i,0,:3]
    )
    RTs_by_torsion[i,2,:3,3] = xyzs_in_base_frame[i,2,:3]

    # chi1 frame
    if torsions[i][0] is not None:
        a0,a1,a2 = torsion_indices[i,3,0:3]
        RTs_by_torsion[i,3,:3,:3] = make_frame(
            xyzs_in_base_frame[i,a2,:3]-xyzs_in_base_frame[i,a1,:3],
            xyzs_in_base_frame[i,a0,:3]-xyzs_in_base_frame[i,a1,:3],
        )
        RTs_by_torsion[i,3,:3,3] = xyzs_in_base_frame[i,a2,:3]

    # chi2/3/4 frame
    for j in range(1,4):
        if torsions[i][j] is not None:
            a2 = torsion_indices[i,3+j,2]
            if ((i==18 and j==2) or (i==8 and j==2)):  # TYR CZ-OH & HIS CE1-HE1 a special case
                a0,a1 = torsion_indices[i,3+j,0:2]
                RTs_by_torsion[i,3+j,:3,:3] = make_frame(
                    xyzs_in_base_frame[i,a2,:3]-xyzs_in_base_frame[i,a1,:3],
                    xyzs_in_base_frame[i,a0,:3]-xyzs_in_base_frame[i,a1,:3] )
            else:
                RTs_by_torsion[i,3+j,:3,:3] = make_frame(
                    xyzs_in_base_frame[i,a2,:3],
                    torch.tensor([-1.,0.,0.]), )
            RTs_by_torsion[i,3+j,:3,3] = xyzs_in_base_frame[i,a2,:3]

    # CB/CG angles
    NCr = 0.5*(xyzs_in_base_frame[i,0,:3]+xyzs_in_base_frame[i,2,:3])
    CAr = xyzs_in_base_frame[i,1,:3]
    CBr = xyzs_in_base_frame[i,4,:3]
    CGr = xyzs_in_base_frame[i,5,:3]
    reference_angles[i,0,:]=th_ang_v(CBr-CAr,NCr-CAr)
    NCp = xyzs_in_base_frame[i,2,:3]-xyzs_in_base_frame[i,0,:3]
    NCpp = NCp - torch.dot(NCp,NCr)/ torch.dot(NCr,NCr) * NCr
    reference_angles[i,1,:]=th_ang_v(CBr-CAr,NCpp)
    reference_angles[i,2,:]=th_ang_v(CGr,torch.tensor([-1.,0.,0.]))

## NUCLEIC ACIDS
for i in range(NPROTAAS, NNAPROTAAS):
    i_l = aa2long[i]

    for name, base, coords in ideal_coords[i]:
        idx = i_l.index(name)
        base_indices[i,idx] = base
        xyzs_in_base_frame[i,idx,:3] = torch.tensor(coords)

    # epsilon(p)/zeta(p) - like omega in protein, not used to build atoms
    #                    - keep as identity
    RTs_by_torsion[i,NPROTTORS+0,:3,:3] = torch.eye(3)
    RTs_by_torsion[i,NPROTTORS+0,:3,3] = torch.zeros(3)
    RTs_by_torsion[i,NPROTTORS+1,:3,:3] = torch.eye(3)
    RTs_by_torsion[i,NPROTTORS+1,:3,3] = torch.zeros(3)

    # nu1
    RTs_by_torsion[i,NPROTTORS+7,:3,:3] = make_frame(
        xyzs_in_base_frame[i,2,:3] , xyzs_in_base_frame[i,0,:3]
    )
    RTs_by_torsion[i,NPROTTORS+7,:3,3] = xyzs_in_base_frame[i,2,:3]
    
    # nu0 - currently not used for atom generation
    RTs_by_torsion[i,NPROTTORS+8,:3,:3] = make_frame(
        xyzs_in_base_frame[i,0,:3] , xyzs_in_base_frame[i,2,:3]
    )
    RTs_by_torsion[i,NPROTTORS+8,:3,3] = xyzs_in_base_frame[i,0,:3] # C2'
    
    # NA chi
    if torsions[i][0] is not None:
        a0,a1,a2 = torsion_indices[i,19,0:3]
        RTs_by_torsion[i,NPROTTORS+9,:3,:3] = make_frame(
            xyzs_in_base_frame[i,a2,:3], xyzs_in_base_frame[i,a0,:3]
        )
        RTs_by_torsion[i,NPROTTORS+9,:3,3] = xyzs_in_base_frame[i,a2,:3]

    # nu2
    RTs_by_torsion[i,NPROTTORS+6,:3,:3] = make_frame(
        xyzs_in_base_frame[i,9,:3] , torch.tensor([-1.,0.,0.])
    )
    RTs_by_torsion[i,NPROTTORS+6,:3,3] = xyzs_in_base_frame[i,6,:3]
    

    # alpha
    RTs_by_torsion[i,NPROTTORS+2,:3,:3] = make_frame(
        xyzs_in_base_frame[i,4,:3], torch.tensor([-1.,0.,0.])
    )
    RTs_by_torsion[i,NPROTTORS+2,:3,3] = xyzs_in_base_frame[i,4,:3]
    
    # beta
    RTs_by_torsion[i,NPROTTORS+3,:3,:3] = make_frame(
        xyzs_in_base_frame[i,6,:3] , torch.tensor([-1.,0.,0.])
    )
    RTs_by_torsion[i,NPROTTORS+3,:3,3] = xyzs_in_base_frame[i,6,:3]
    
    # gamma
    RTs_by_torsion[i,NPROTTORS+4,:3,:3] = make_frame(
        xyzs_in_base_frame[i,7,:3] , torch.tensor([-1.,0.,0.])
    )
    RTs_by_torsion[i,NPROTTORS+4,:3,3] = xyzs_in_base_frame[i,7,:3]
    
    # delta
    RTs_by_torsion[i,NPROTTORS+5,:3,:3] = make_frame(
        xyzs_in_base_frame[i,8,:3] , torch.tensor([-1.,0.,0.])
    )
    RTs_by_torsion[i,NPROTTORS+5,:3,3] = xyzs_in_base_frame[i,8,:3]


#Small molecules
xyzs_in_base_frame[NNAPROTAAS:,1, :3] = 0
# general FAPE parameters
frame_indices = torch.full((NAATOKENS,NFRAMES,3,2),0, dtype=torch.long)
for i in range(NNAPROTAAS):
    i_l = aa2long[i]
    for j,x in enumerate(frames[i]):
        if x is not None:
            # frames are stored as (residue offset, atom position)
            frame_indices[i,j,0] = torch.tensor((0, i_l.index(x[0])))
            frame_indices[i,j,1] = torch.tensor((0, i_l.index(x[1])))
            frame_indices[i,j,2] = torch.tensor((0, i_l.index(x[2])))


### Create atom frames for FAPE loss calculation ###
def get_nxgraph(mol):
    '''build NetworkX graph from openbabel's OBMol'''

    N = mol.NumAtoms()

    # pairs of bonded atoms, openbabel indexes from 1 so readjust to indexing from 0
    bonds = [(bond.GetBeginAtomIdx()-1, bond.GetEndAtomIdx()-1) for bond in openbabel.OBMolBondIter(mol)]

    # connectivity graph
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(bonds)

    return G

def find_all_rigid_groups(bond_feats):
    """
    remove all single bonds from the graph and find connected components
    """
    rigid_atom_bonds = (bond_feats>1)*(bond_feats<5)
    rigid_atom_bonds_np = rigid_atom_bonds[0].cpu().numpy()
    G = nx.from_numpy_matrix(rigid_atom_bonds_np)
    connected_components = nx.connected_components(G)
    connected_components = [cc for cc in connected_components if len(cc)>2]
    connected_components = [torch.tensor(list(combinations(cc,2))) for cc in connected_components]
    if connected_components:
        connected_components = torch.cat(connected_components, dim=0)
    else:
        connected_components = None
    return connected_components

def find_all_paths_of_length_n(G : nx.Graph,
                               n : int) -> torch.Tensor:
    '''find all paths of length N in a networkx graph
    https://stackoverflow.com/questions/28095646/finding-all-paths-walks-of-given-length-in-a-networkx-graph'''

    def findPaths(G,u,n):
        if n==0:
            return [[u]]
        paths = [[u]+path for neighbor in G.neighbors(u) for path in findPaths(G,neighbor,n-1) if u not in path]
        return paths

    # all paths of length n
    allpaths = [tuple(p) if p[0]<p[-1] else tuple(reversed(p))
                for node in G for p in findPaths(G,node,n)]

    # unique paths
    allpaths = list(set(allpaths))

    #return torch.tensor(allpaths)
    return allpaths

def get_atom_frames(msa, G):
    """choose a frame of 3 bonded atoms for each atom in the molecule, rule based system that chooses frame based on atom priorities"""

    query_seq = msa
    frames = find_all_paths_of_length_n(G, 2)
    selected_frames = []
    for n in range(msa.shape[0]):
        frames_with_n = [frame for frame in frames if n == frame[1]]

        # some chemical groups don't have two bonded heavy atoms; so choose a frame with an atom 2 bonds away
        if not frames_with_n:
            frames_with_n = [frame for frame in frames if n in frame]
        # if the atom isn't in a 3 atom frame, it should be ignored in loss calc, set all the atoms to n
        if not frames_with_n:
            selected_frames.append([(0,1),(0,1),(0, 1)])
            continue
        frame_priorities = []
        for frame in frames_with_n:
            # hacky but uses the "query_seq" to convert index of the atom into an "atom type" and converts that into a priority
            indices = [index for index in frame if index!=n]
            aas = [num2aa[int(query_seq[index].numpy())] for index in indices]
            frame_priorities.append(sorted([atom2frame_priority[aa] for aa in aas]))
            
        # np.argsort doesn't sort tuples correctly so just sort a list of indices using a key
        sorted_indices = sorted(range(len(frame_priorities)), key=lambda i: frame_priorities[i])
        # calculate residue offset for frame
        frame = [(frame-n, 1) for frame in frames_with_n[sorted_indices[0]]]
        selected_frames.append(frame)
    assert msa.shape[0] == len(selected_frames)
    return torch.tensor(selected_frames).long()

def get_atomized_protein_frames(msa, ra):
    """ returns a unique frame for each atom in a n_res_atomize of residues """
    r,a = ra.T
    residue_frames = atomized_protein_frames[msa]
    # handle out of bounds at the termini, terminal N and C inherit Calpha frame, offset dimension needs to be updated
    offset = torch.zeros(3,2)
    offset[:,0] = 1
    residue_frames[r[0], 0] = residue_frames[r[0],1] + offset
    residue_frames[r[-1], 2] = residue_frames[r[-1],1] - offset
    # terminal O needs to have O-C-Calpha frame
    residue_frames[r[-1],3,2] = torch.tensor([-2,1])
    frames = residue_frames[r,a]

    # make sure no masked atoms are in frames
    atom_idx = torch.arange(frames.shape[0]).unsqueeze(1).repeat(1,3)
    masked_atoms = ~torch.all((frames[...,0] + atom_idx) < frames.shape[0], dim=-1)
    reframe_indices = masked_atoms.nonzero()
    for i in reframe_indices:
        frames[i] = frames[i-1]-offset
    return frames.long() 
    

    
### Generate bond features for small molecules ###
def get_bond_feats(mol):                                                                                 
    """creates 2d bond graph for small molecules"""
    N = mol.NumAtoms()
    bond_feats = torch.zeros((N, N)).long()

    for bond in openbabel.OBMolBondIter(mol):
        i,j = (bond.GetBeginAtomIdx()-1, bond.GetEndAtomIdx()-1)
        bond_feats[i,j] = bond.GetBondOrder() if not bond.IsAromatic() else 4
        bond_feats[j,i] = bond_feats[i,j]

    return bond_feats.long()

def get_protein_bond_feats(protein_L):
    """ creates protein residue connectivity graphs """
    bond_feats = torch.zeros((protein_L, protein_L))
    residues = torch.arange(protein_L-1)
    bond_feats[residues, residues+1] = 5
    bond_feats[residues+1, residues] = 5
    return bond_feats

def get_protein_bond_feats_from_idx(protein_L, idx_protein):
    """ creates protein residue connectivity graphs """
    bond_feats = torch.zeros((protein_L, protein_L))
    residues = torch.arange(protein_L-1)
    mask = idx_protein[:,None] == idx_protein[None,:]+1
    bond_feats[mask] = 5
    bond_feats[mask.T] = 5
    return bond_feats

def get_atomize_protein_bond_feats(i_start, msa, ra, n_res_atomize=5):
    """ 
    generate atom bond features for atomized residues 
    currently ignores long-range bonds like disulfides
    """
    ra2ind = {}
    for i, two_d in enumerate(ra):
        ra2ind[tuple(two_d.numpy())] = i
    N = len(ra2ind.keys())
    bond_feats = torch.zeros((N, N))
    for i, res in enumerate(msa[0, i_start:i_start+n_res_atomize]):
        for j, bond in enumerate(aabonds[res]):
            start_idx = aa2long[res].index(bond[0])
            end_idx = aa2long[res].index(bond[1])
            if (i, start_idx) not in ra2ind or (i, end_idx) not in ra2ind:
                #skip bonds with atoms that aren't observed in the structure
                continue
            start_idx = ra2ind[(i, start_idx)]
            end_idx = ra2ind[(i, end_idx)]

            # maps the 2d index of the start and end indices to btype
            bond_feats[start_idx, end_idx] = aabtypes[res][j]
            bond_feats[end_idx, start_idx] = aabtypes[res][j]
        #accounting for peptide bonds
        if i > 0:
            if (i-1, 2) not in ra2ind or (i, 0) not in ra2ind:
                #skip bonds with atoms that aren't observed in the structure
                continue
            start_idx = ra2ind[(i-1, 2)]
            end_idx = ra2ind[(i, 0)]
            bond_feats[start_idx, end_idx] = SINGLE_BOND
            bond_feats[end_idx, start_idx] = SINGLE_BOND
    return bond_feats


# given a bond graph, get the path lengths
def get_path_lengths(bond_feats):
    atom_bonds = (bond_feats > 0)*(bond_feats<5)
    atom_bonds = atom_bonds + atom_bonds.permute((0,2,1))
    dist_matrix = scipy.sparse.csgraph.shortest_path(atom_bonds[0].long().detach().cpu().numpy(), directed=False)
    dist_matrix = torch.tensor(np.nan_to_num(dist_matrix, posinf=4.0), device=mask.device) # protein portion is inf and you don't want to mask it out
    return dist_matrix


### Generate atom features for proteins ###
def atomize_protein(i_start, msa, xyz, mask, n_res_atomize=5):
    """ given an index i_start, make the following flank residues into "atom" nodes """
    residues_atomize = msa[0, i_start:i_start+n_res_atomize]
    residues_atom_types = [aa2elt[num][:14] for num in residues_atomize]
    residue_atomize_mask = mask[i_start:i_start+n_res_atomize].float() # mask of resolved atoms in the sidechain
    residue_atomize_allatom_mask = allatom_mask[residues_atomize][:, :14] # the indices that have heavy atoms in that sidechain
    xyz_atomize = xyz[i_start:i_start+n_res_atomize]

    # handle symmetries
    xyz_alt = torch.zeros_like(xyz.unsqueeze(0))
    xyz_alt.scatter_(2, long2alt[msa[0],:,None].repeat(1,1,1,3), xyz.unsqueeze(0))
    xyz_alt_atomize = xyz_alt[0, i_start:i_start+n_res_atomize]

    coords_stack = torch.stack((xyz_atomize, xyz_alt_atomize), dim=0)
    swaps = (coords_stack[0] == coords_stack[1]).all(dim=1).all(dim=1).squeeze() #checks whether theres a swap at each position
    swaps = torch.nonzero(~swaps).squeeze() # indices with a swap eg. [2,3]
    if swaps.numel() != 0:
        # if there are residues with alternate numbering scheme, create a stack of coordinate with each combo of swaps
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=UserWarning)
            combs = torch.combinations(torch.tensor([0,1]), r=swaps.numel(), with_replacement=True) #[[0,0], [0,1], [1,1]]
        stack = torch.stack((combs, swaps.repeat(swaps.numel()+1,1)), dim=-1).squeeze()
        coords_stack = coords_stack.repeat(swaps.numel()+1,1,1,1)
        nat_symm = coords_stack[0].repeat(swaps.numel()+1,1,1,1) # (N_symm, num_atomize_residues, natoms, 3)
        swapped_coords = coords_stack[stack[...,0], stack[...,1]].squeeze(1) #
        nat_symm[:,swaps] = swapped_coords
    else:
        nat_symm = xyz_atomize.unsqueeze(0)
    # every heavy atom that is in the sidechain is modelled but losses only applied to resolved atoms
    ra = residue_atomize_allatom_mask.nonzero()
    lig_seq = torch.tensor([aa2num[residues_atom_types[r][a]] if residues_atom_types[r][a] in aa2num else aa2num["ATM"] for r,a in ra])
    ins = torch.zeros_like(lig_seq)

    r,a = ra.T
    lig_xyz = torch.zeros((len(ra), 3))
    lig_xyz = nat_symm[:, r, a]
    lig_mask = residue_atomize_mask[r, a].repeat(nat_symm.shape[0], 1)
    bond_feats = get_atomize_protein_bond_feats(i_start, msa, ra, n_res_atomize=n_res_atomize)
    #HACK: use networkx graph to make the atom frames, correct implementation will include frames with "residue atoms"
    G = nx.from_numpy_matrix(bond_feats.numpy())
        
    frames = get_atom_frames(lig_seq, G)
    chirals = get_atomize_protein_chirals(residues_atomize, lig_xyz[0], residue_atomize_allatom_mask, bond_feats)
    return lig_seq, ins, lig_xyz, lig_mask, frames, bond_feats, ra, chirals


def reindex_protein_feats_after_atomize(
    residues_to_atomize,
    prot_partners,
    msa, 
    ins,
    xyz,
    mask,
    bond_feats,
    idx,
    xyz_t,
    f1d_t,
    mask_t,
    same_chain,
    ch_label,
    Ls_prot,
    Ls_sm, 
    akeys_sm
):
    """
    Removes residues that have been atomized from protein features.
    """
    Ls = Ls_prot + Ls_sm
    chain_bins = [sum(Ls[:i]) for i in range(len(Ls)+1)]
    akeys_sm = list(itertools.chain.from_iterable(akeys_sm)) # list of list of tuples get flattened to a list of tuples

    # get tensor indices of atomized residues
    residue_chain_nums = []
    residue_indices = []
    for residue in residues_to_atomize:
        # residue object is a list of tuples:
        #   ((chain_letter, res_number, res_name), (chain_letter, xform_index))

        #### Need to identify what chain you're in to get correct res idx
        residue_chid_xf = residue[1]
        residue_chain_num = [p[:2] for p in prot_partners].index(residue_chid_xf)
        residue_index = (int(residue[0][1]) - 1) + sum(Ls_prot[:residue_chain_num])  # residues are 1 indexed in the cif files

        # skip residues with all backbone atoms masked
        if torch.sum(mask[0, residue_index, :3]) <3: continue

        residue_chain_nums.append(residue_chain_num)
        residue_indices.append(residue_index)
        atomize_N = residue[0] + ("N",)
        atomize_C = residue[0] + ("C",)

        N_index = akeys_sm.index(atomize_N) + sum(Ls_prot)
        C_index = akeys_sm.index(atomize_C) + sum(Ls_prot)

        # if first residue in chain, no extra bond feats to previous residue
        if residue_index != 0 and residue_index not in Ls_prot:
            bond_feats[residue_index-1, N_index] = 6
            bond_feats[N_index, residue_index-1] = 6

        # if residue is last in chain, no extra bonds feats to following residue
        if residue_index not in [L-1 for L in Ls_prot]:
            bond_feats[residue_index+1, C_index] = 6
            bond_feats[C_index,residue_index+1] = 6

        lig_chain_num = np.digitize([N_index], chain_bins)[0] -1 # np.digitize is 1 indexed
        same_chain[chain_bins[lig_chain_num]:chain_bins[lig_chain_num+1], \
                   chain_bins[residue_chain_num]: chain_bins[residue_chain_num+1]] = 1
        same_chain[chain_bins[residue_chain_num]: chain_bins[residue_chain_num+1], \
                   chain_bins[lig_chain_num]:chain_bins[lig_chain_num+1]] = 1

    # remove atomized residues from feature tensors
    i_res = torch.tensor([i for i in range(sum(Ls)) if i not in residue_indices])
    msa = msa[:,i_res]
    ins = ins[:,i_res]
    xyz = xyz[:,i_res]
    mask = mask[:,i_res]
    bond_feats = bond_feats[i_res][:,i_res]
    idx = idx[i_res]
    xyz_t = xyz_t[:,i_res]
    f1d_t = f1d_t[:,i_res]
    mask_t = mask_t[:,i_res]
    same_chain = same_chain[i_res][:,i_res]
    ch_label = ch_label[i_res]

    for i_ch in residue_chain_nums:
        Ls_prot[i_ch] -= 1

    return msa, ins, xyz, mask, bond_feats, idx, xyz_t, f1d_t, mask_t, same_chain, ch_label, Ls_prot, Ls_sm




def atomize_discontiguous_residues(idxs, msa, xyz, mask, bond_feats, same_chain, dslfs=None):
    """
    this atomizes multiple discontiguous residues at the same time, this is the default interface into atomizing residues 
    (using the non assembly dataset)
    """
    protein_L = msa.shape[1]
    seq_atomize_all = []
    ins_atomize_all = []
    xyz_atomize_all = []
    mask_atomize_all = []
    frames_atomize_all = []
    chirals_atomize_all = []
    prev_C_index = None
    total_num_atoms = 0
    sgs = {}
    for idx in idxs:
        seq_atomize, ins_atomize, xyz_atomize, mask_atomize, frames_atomize, bond_feats_atomize, resatom2idx, chirals_atomize = \
            atomize_protein(idx, msa, xyz, mask, n_res_atomize=1)
        r,_ = resatom2idx.T
        #print ('atomize_discontiguous_residues', idx, resatom2idx)
        last_C = torch.all(resatom2idx==torch.tensor([r[-1],2]),dim=1).nonzero()
        sgs[idx.item()] = torch.all(resatom2idx==torch.tensor([r[-1],5]),dim=1).nonzero()

        natoms = seq_atomize.shape[0]
        L = bond_feats.shape[0]

        sgs[idx.item()] = L+sgs[idx.item()]

        # update the chirals to be after all the other atoms (still need to update to put it behind all the proteins)
        chirals_atomize[:, :-1] += total_num_atoms

        seq_atomize_all.append(seq_atomize)
        ins_atomize_all.append(ins_atomize)
        xyz_atomize_all.append(xyz_atomize)
        mask_atomize_all.append(mask_atomize)
        frames_atomize_all.append(frames_atomize)
        chirals_atomize_all.append(chirals_atomize)

        N_term = idx ==  0
        C_term = idx == protein_L-1

        # update bond_feats every iteration, update all other features at the end 
        bond_feats_new = torch.zeros((L+natoms, L+natoms))
        bond_feats_new[:L, :L] = bond_feats
        bond_feats_new[L:, L:] = bond_feats_atomize
        # add bond between protein and atomized N
        if not N_term and idx-1 not in idxs:
            bond_feats_new[idx-1, L] = 6 # protein (backbone)-atom bond 
            bond_feats_new[L, idx-1] = 6 # protein (backbone)-atom bond 
        # add bond between protein and C, assumes every residue is being atomized one at a time (eg n_res_atomize=1)
        if not C_term and idx+1 not in idxs:
            bond_feats_new[idx+1, L+int(last_C.numpy())] = 6 # protein (backbone)-atom bond 
            bond_feats_new[L+int(last_C.numpy()), idx+1] = 6 # protein (backbone)-atom bond 
        # handle drawing peptide bond between contiguous atomized residues
        if idx-1 in idxs:
            if prev_C_index is None:
                raise ValueError("prev_C_index is None even though the previous residue has been atomized")
            bond_feats_new[prev_C_index, L] = 1 # single bond
            bond_feats_new[L, prev_C_index] = 1 # single bond

        prev_C_index =  L+int(last_C.numpy()) #update prev_C to draw bond to upcoming residue
        # update same_chain every iteration
        same_chain_new = torch.zeros((L+natoms, L+natoms))
        same_chain_new[:L, :L] = same_chain
        residues_in_prot_chain = same_chain[idx].squeeze().nonzero()

        same_chain_new[L:, residues_in_prot_chain] = 1
        same_chain_new[residues_in_prot_chain, L:] = 1
        same_chain_new[L:, L:] = 1

        bond_feats = bond_feats_new
        same_chain = same_chain_new
        total_num_atoms += natoms

    # disulfides
    if dslfs is not None:
        for i,j in dslfs:
            start_idx = sgs[i].item()
            end_idx = sgs[j].item()
            bond_feats[start_idx, end_idx] = 1
            bond_feats[end_idx, start_idx] = 1

    seq_atomize_all = torch.cat(seq_atomize_all)
    ins_atomize_all = torch.cat(ins_atomize_all)
    xyz_atomize_all = cartprodcat(xyz_atomize_all)
    mask_atomize_all = cartprodcat(mask_atomize_all)
    
    # frames were calculated per residue -- we want them over all residues in case there are contiguous residues
    bond_feats_sm = bond_feats[protein_L:][:, protein_L:]
    G = nx.from_numpy_matrix(bond_feats_sm.detach().cpu().numpy())
    frames_atomize_all = get_atom_frames(seq_atomize_all, G)
    
    # frames_atomize_all = torch.cat(frames_atomize_all)
    chirals_atomize_all = torch.cat(chirals_atomize_all)

    return seq_atomize_all, ins_atomize_all, xyz_atomize_all, mask_atomize_all, frames_atomize_all, chirals_atomize_all, \
        bond_feats, same_chain

def get_atom_template_indices(msa,res_idxs_to_atomize):
    """
    chooses a random frame (with high probability of choosing later frames) from each atomized residue to provide as a 
    template to resemble diffusion training returns indices that represent that frame in the atomized region of the rosettafold input
    """
    aa2long_ = [[x.strip() if x is not None else None for x in y] for y in aa2long]
    residues_atomize = msa[0, res_idxs_to_atomize]
    residue_atomize_allatom_mask = allatom_mask[residues_atomize][:, :14] # the indices that have heavy atoms in that sidechain
    num_atoms_per_residue = torch.sum(residue_atomize_allatom_mask,dim=1)

    frames_per_residue = [frames[i] for i in residues_atomize]
    chosen_frame_per_residue = [random.choices(frames, weights=range(len(frames)), k=1)[0] for frames in frames_per_residue]

    atom_indices_per_residue = [[aa2long_[res].index(atom.strip()) for atom in frame] for res, frame in zip(residues_atomize, chosen_frame_per_residue)]
    atom_indices_per_residue = torch.tensor(atom_indices_per_residue)
    
    index_offset_per_residue = torch.roll(torch.cumsum(num_atoms_per_residue, dim=0), shifts=1)
    index_offset_per_residue[0] = 0
    
    atom_indices_per_residue += index_offset_per_residue[:, None].repeat(1,3)

    return atom_indices_per_residue.flatten()

def reindex_protein_feats_after_atomize(
    residues_to_atomize,
    prot_partners,
    msa, 
    ins,
    xyz,
    mask,
    bond_feats,
    idx,
    xyz_t,
    f1d_t,
    mask_t,
    same_chain,
    ch_label,
    Ls_prot,
    Ls_sm, 
    akeys_sm
):
    """
    Removes residues that have been atomized from protein features.
    """
    Ls = Ls_prot + Ls_sm
    chain_bins = [sum(Ls[:i]) for i in range(len(Ls)+1)]
    akeys_sm = list(itertools.chain.from_iterable(akeys_sm)) # list of list of tuples get flattened to a list of tuples

    # get tensor indices of atomized residues
    residue_chain_nums = []
    residue_indices = []
    for residue in residues_to_atomize:
        # residue object is a list of tuples:
        #   ((chain_letter, res_number, res_name), (chain_letter, xform_index))

        #### Need to identify what chain you're in to get correct res idx
        residue_chid_xf = residue[1]
        residue_chain_num = [p[:2] for p in prot_partners].index(residue_chid_xf)
        residue_index = (int(residue[0][1]) - 1) + sum(Ls_prot[:residue_chain_num])  # residues are 1 indexed in the cif files

        # skip residues with all backbone atoms masked
        if torch.sum(mask[0, residue_index, :3]) <3: continue

        residue_chain_nums.append(residue_chain_num)
        residue_indices.append(residue_index)
        atomize_N = residue[0] + ("N",)
        atomize_C = residue[0] + ("C",)

        N_index = akeys_sm.index(atomize_N) + sum(Ls_prot)
        C_index = akeys_sm.index(atomize_C) + sum(Ls_prot)

        # if first residue in chain, no extra bond feats to previous residue
        if residue_index != 0 and residue_index not in Ls_prot:
            bond_feats[residue_index-1, N_index] = 6
            bond_feats[N_index, residue_index-1] = 6

        # if residue is last in chain, no extra bonds feats to following residue
        if residue_index not in [L-1 for L in Ls_prot]:
            bond_feats[residue_index+1, C_index] = 6
            bond_feats[C_index,residue_index+1] = 6

        lig_chain_num = np.digitize([N_index], chain_bins)[0] -1 # np.digitize is 1 indexed
        same_chain[chain_bins[lig_chain_num]:chain_bins[lig_chain_num+1], \
                   chain_bins[residue_chain_num]: chain_bins[residue_chain_num+1]] = 1
        same_chain[chain_bins[residue_chain_num]: chain_bins[residue_chain_num+1], \
                   chain_bins[lig_chain_num]:chain_bins[lig_chain_num+1]] = 1

    # remove atomized residues from feature tensors
    msa, ins, xyz, mask, bond_feats, idx, xyz_t, f1d_t, mask_t, same_chain, ch_label = \
        pop_protein_feats(residue_indices, msa, ins, xyz, mask, bond_feats, idx, xyz_t, f1d_t, mask_t, same_chain, ch_label, Ls)

    for i_ch in residue_chain_nums:
        Ls_prot[i_ch] -= 1

    return msa, ins, xyz, mask, bond_feats, idx, xyz_t, f1d_t, mask_t, same_chain, ch_label, Ls_prot, Ls_sm


def pop_protein_feats(residue_indices, msa, ins, xyz, mask, bond_feats, idx, xyz_t, f1d_t, mask_t, same_chain, ch_label, Ls):
    """
    remove protein features for an arbitrary set of residue indices
    """
    pop = torch.ones((sum(Ls)))
    pop[residue_indices] = 0
    pop = pop.bool()

    msa = msa[:,pop]
    ins = ins[:,pop]
    xyz = xyz[:,pop]
    mask = mask[:,pop]
    bond_feats = bond_feats[pop][:,pop]
    idx = idx[pop]
    xyz_t = xyz_t[:,pop]
    f1d_t = f1d_t[:,pop]
    mask_t = mask_t[:,pop]
    same_chain = same_chain[pop][:,pop]
    ch_label = ch_label[pop]

    return msa, ins, xyz, mask, bond_feats, idx, xyz_t, f1d_t, mask_t, same_chain, ch_label

def cif_prot_to_xyz(ch, ch_xf, modres=dict()):
    """Given a protein chain and coordinate transform parsed from CIF file,
    return tensors with coordinates and masks

    Parameters
    ----------
    ch: namedtuple
        A protein chain as parsed by cifutils
    ch_xf : 2-tuple (chain_id, np.array:(4,4))
        The coordinate transform for this chain
    modres : dict
        Maps modified residue names to their canonical equivalents. Any
        modified residue will be converted to its standard equivalent and
        coordinates for atoms with matching names will be saved.

    Returns
    -------
    xyz_xf : torch.Tensor (L, NTOTAL, 3), float
        Transformed coordinates for all the atoms in each residue
    mask : torch.Tensor (L, NTOTAL,), bool
        Boolean mask with True if a certain atom is present
    seq : torch.Tensor (L,), long
        Integer encoded amino acid sequence
    chid : list of str (L,)
        Chain IDs for each residue
    resi : list of str (L,)
        Residue numbers for each residue
    """ 
    assert(ch.type == 'polypeptide(L)')
    assert(ch.id == ch_xf[0])

    # atom names from cif don't have whitespace
    aa2long_ = [[x.strip() if x is not None else None for x in y] for y in aa2long]

    idx = [int(k[1]) for k in ch.atoms]
    i_min, i_max = np.min(idx), np.max(idx)
    L = i_max - i_min + 1

    xyz = torch.zeros(L, NTOTAL, 3)
    mask = torch.zeros(L, NTOTAL).bool()
    seq = torch.full((L,), np.nan)
    chid = ['-']*L
    resi = ['-']*L

    unrec_elements = set()
    residues_to_atomize = set()
    for (ch_letter, res_num, res_name, atom_name), atom_val in ch.atoms.items():
        i_res = int(res_num)-i_min
        if res_name in aa2num: # standard AA
            aa = aa2num[res_name]
        elif res_name in modres and modres[res_name] in aa2num: # nonstandard AA, map to standard
            #print('nonstandard AA',k,modres[k[2]])
            aa = aa2num[modres[res_name]]
            residues_to_atomize.add((ch_letter, res_num, res_name))
        else: # unknown AA, still try to store BB atoms
            #print('unknown AA',k)
            aa = 20
        if atom_name in aa2long_[aa]: # atom name exists in RF nomenclature
            i_atom = aa2long_[aa].index(atom_name) # atom index
            xyz[i_res, i_atom, :] = torch.tensor(atom_val.xyz)
            mask[i_res, i_atom] = atom_val.occ
        seq[i_res] = aa
        chid[i_res] = ch_letter
        resi[i_res] = res_num

    xf = torch.tensor(ch_xf[1]).float()
    u,r = xf[:3,:3], xf[:3,3]
    xyz_xf = torch.einsum('ij,raj->rai', u, xyz) + r[None,None]

    return xyz_xf, mask, seq, chid, resi, residues_to_atomize


def get_ligand_atoms_bonds(ligand, chains, covale):
    """Gets the atoms and bonds belonging to a certain ligand, as identified by
    the ligand's chain ID(s) and residue number(s). Includes
    inter-chain bonds, for multi-residue ligands. `chains`, `covale`,
    `lig_atoms`, and `lig_bonds` are as parsed by cifutils.
    """
    lig_atoms = dict()
    lig_bonds = []
    for i_ch,ch in chains.items():
        for k,v in ch.atoms.items():
            if k[:3] in ligand:
                lig_atoms[k] = v
        for bond in ch.bonds:
            if bond.a[:3] in ligand or bond.b[:3] in ligand:
                lig_bonds.append(bond)
    for bond in covale:
        if bond.a[:3] in ligand and bond.b[:3] in ligand:
            lig_bonds.append(bond)
    return lig_atoms, lig_bonds


def cif_ligand_to_xyz(atoms, asmb_xfs, ch2xf, input_akeys=None):
    """Given ligand atoms from a parsed CIF file and coordinate transforms
    specific to a bio-assembly of interest, returns tensors with transformed
    coordinates and a mask for valid atoms.

    Parameters
    ----------
    atoms: dict
        Atom key-value pairs for the query ligand, as parsed by cifutils
    asmb_xfs : list of 2-tuples (chain_id, torch.Tensor(4,4))
        Coordinate transforms for the current assembly
    ch2xf : dict
        Maps chain letters to transform indices
    input_akeys : list of 4-tuples (chain_id, residue_num, residue_name, atom_name)
        Atom keys for the query ligand, used to enforce a specific atom order

    Returns
    -------
    xyz : torch.Tensor (N_atoms, 3), float
    occ : torch.Tensor (N_atoms,), float
        These values represent atom position occupancies and can be fractional.
    seq : torch.Tensor (N_atoms,), long
    chid : list (N_atoms,)
        Chain IDs for each ligand atom
    akeys : list of 4-tuples (chain_id, residue_num, residue_name, atom_name)
        Atom keys with information about each atom, in the same order as the returned coordinates
    """
    atoms_no_H = {k:v for k,v in atoms.items() if v.element != 1} # exclude hydrogens
    L = len(atoms_no_H)
    
    if input_akeys is None:
        input_akeys = atoms_no_H.keys()

    xyz = torch.zeros(L, 3)
    occ = torch.zeros(L,)
    seq = torch.full((L,), np.nan)
    chid = ['-']*L
    akeys = [None]*L

    # create coords, atom mask, and seq tokens
    for i,k in enumerate(input_akeys):
        v = atoms_no_H[k]
        xyz[i, :] = torch.tensor(v.xyz)
        occ[i] = v.occ # can contain fractionally occupied atom positions
        if v.element not in chemical.atomnum2atomtype:
            #print('Element not in alphabet:',v.element)
            seq[i] = chemical.aa2num['ATM']
        else:
            seq[i] = chemical.aa2num[chemical.atomnum2atomtype[v.element]]
        akeys[i] = k
        chid[i] = k[0]

    # apply transforms
    chid = np.array(chid)
    for i_ch in np.unique(chid):
        if i_ch not in ch2xf: continue # indicates ligand chains with zero occupied atoms
        idx = chid==i_ch
        xf = torch.tensor(asmb_xfs[ch2xf[i_ch]][1]).float()
        u,r = xf[:3,:3], xf[:3,3]
        xyz[idx] = torch.einsum('ij,aj->ai', u, xyz[idx]) + r[None,None]

    return xyz, occ, seq, chid, akeys

def remove_unresolved_substructures(akeys, lig_bonds, mask_sm):
    """
    returns a tensor mask of indices of atoms that are not resolved and do not have any resolved neighbors
    atoms with resolved neigbors =1, atoms without resolved neighbors =0
    """
    L = len(akeys)
    bond_feats = torch.zeros(L, L)
    for bond in lig_bonds:
        if bond.a not in akeys or bond.b not in akeys: continue # intended to skip bonds to H's
        i = akeys.index(bond.a)
        j = akeys.index(bond.b)
        bond_feats[i,j] = bond.order if not bond.aromatic else 4
        bond_feats[j,i] = bond_feats[i,j] 
    
    no_resolved_neighbors = torch.ones(L)
    for i in range(len(akeys)):
        neighbors = bond_feats[i].nonzero()
        resolved_neighbors = mask_sm[neighbors]
        if torch.sum(resolved_neighbors) == 0 and ~mask_sm[i]:
            no_resolved_neighbors[i] = 0
    return no_resolved_neighbors.bool()


def cif_ligand_to_obmol(xyz, akeys, atoms, bonds):
    """Given a ligand's coordinates and atom and bond information, return an
    openbabel molecule representing the ligand as well as its 2D bond features.

    Parameters
    ----------
    xyz : torch.Tensor (N_atoms, 3), float
    akeys : list of 4-tuples (chain_id, residue_num, residue_name, atom_name)
    atoms : dict
        Ligand atoms, as parsed by cifutils
    bonds : list of Bond
        Ligand bonds, as parsed by cifutils

    Returns
    -------
    mol : OBMol object
        Openbabel molecule containing coordinate, atom, and bond information for the ligand
    bond_feats : torch.Tensor (L,L)
        Bond features for the ligand
    """
    mol = openbabel.OBMol()    
    for i,k in enumerate(akeys):
        a = mol.NewAtom()
        a.SetAtomicNum(atoms[k].element)
        a.SetVector(float(xyz[i,0]), float(xyz[i,1]), float(xyz[i,2]))

    sm_L = len(akeys)
    bond_feats = torch.zeros((sm_L,sm_L))
    for bond in bonds:
        if bond.a not in akeys or bond.b not in akeys: continue # intended to skip bonds to H's
        i = akeys.index(bond.a)
        j = akeys.index(bond.b)
        bond_feats[i,j] = bond.order if not bond.aromatic else 4
        bond_feats[j,i] = bond_feats[i,j]

        obb = openbabel.OBBond()
        obb.SetBegin(mol.GetAtom(i+1))
        obb.SetEnd(mol.GetAtom(j+1))
        obb.SetBondOrder(bond.order)
        if bond.aromatic:
            obb.SetAromatic()

        mol.AddBond(obb)
    
    return mol, bond_feats

def get_alt_query_ligand(chains, ligand_name, partners, lig_akeys, asmb_xfs):
    """Given a query ligand name and its contacting chains & transforms, return coordinates
    of other ligands with the same name but different chain, transform, and/or residue number.

    Parameters
    ----------
    chains : dict
        All the chains for this PDB entry, as parsed by cifutils
    ligand_name : str
        Name of query ligand
    partners : list of 4-tuples (chain_id, transform_index, num_contacts, chain_type)
        Chains making contacts to the query ligand
    lig_akeys : list of atom keys (4-tuples) (chain_id, residue_num, residue_name, atom_name)
        Atom keys for the query ligand, used to ensure alternate ligands have same atom order
    asmb_xfs : list of 2-tuples (chain_id, torch.Tensor:(4,4))
        Coordinate transforms for the current assembly

    Returns
    -------
    xyz_alt_s : list of float tensors (N_symm, N_atoms, 3)
    mask_alt_s : list of bool tensors (N_symm, N_atoms)
    """
    lig_anames = [k[3] for k in lig_akeys]
    xyz_alt_s = []
    mask_alt_s = []

    for partner in partners:
        if partner[3] != 'nonpoly': continue # skip non-small-molecule chains

        # gather all atoms and bonds on partner chain with the same name as query ligand
        alt_lig_atoms = dict()
        alt_lig_bonds = []
        for k,v in chains[partner[0]].atoms.items():
            if ligand_name==k[2]:
                alt_lig_atoms[k] = v
        for bond in chains[partner[0]].bonds:
            if bond.a[2]==ligand_name and bond.b[2]==ligand_name:
                alt_lig_bonds.append(bond)

        lig_ch2xf = {partner[0]:partner[1]}

        # iterate through all unique residue numbers on partner chain
        # usually a ligand chain will only have one residue with a given ligand name, but this is to be safe
        alt_lig_res = set([k[:3] for k in alt_lig_atoms])
        for res in alt_lig_res:
            res_atoms = {k:v for k,v in alt_lig_atoms.items() if k[:3]==res}
            res_bonds = [bond for bond in alt_lig_bonds if bond.a[:3]==res and bond.b[:3]==res]

            res_anames = [k[3] for k in res_atoms.keys()]
            if not set(lig_anames).issubset(res_anames):
                print('Alternate ligand position/conformation does not match query ligand atom names. Skipping.', partner, res)
                continue

            res_akeys = [res+(aname,) for aname in lig_anames]
            xyz_alt, mask_alt, msa_alt, chid_alt, akeys_alt = cif_ligand_to_xyz(res_atoms, asmb_xfs, lig_ch2xf, input_akeys=res_akeys)
            mol_alt, bond_feats_alt = cif_ligand_to_obmol(xyz_alt, akeys_alt, res_atoms, res_bonds)
            xyz_alt, mask_alt = get_automorphs(mol_alt, xyz_alt, mask_alt)

            xyz_alt_s.append(xyz_alt)
            mask_alt_s.append(mask_alt)

    return xyz_alt_s, mask_alt_s

def get_automorphs(mol, xyz_sm, mask_sm):
    """Enumerate atom symmetry permutations."""
    try:
        automorphs = openbabel.vvpairUIntUInt()
        openbabel.FindAutomorphisms(mol, automorphs)

        automorphs = torch.tensor(automorphs)
        n_symmetry = automorphs.shape[0]

        xyz_sm = xyz_sm[None].repeat(n_symmetry,1,1)
        mask_sm = mask_sm[None].repeat(n_symmetry,1)

        xyz_sm = torch.scatter(xyz_sm, 1, automorphs[:,:,0:1].repeat(1,1,3),
                                    torch.gather(xyz_sm,1,automorphs[:,:,1:2].repeat(1,1,3)))
        mask_sm = torch.scatter(mask_sm, 1, automorphs[:,:,0],
                            torch.gather(mask_sm, 1, automorphs[:,:,1]))
    except Exception as e:
        xyz_sm = xyz_sm[None]
        mask_sm = mask_sm[None]

    return xyz_sm, mask_sm

def expand_xyz_sm_to_ntotal(xyz_sm, mask_sm, N_symmetry=None):
    """
    for small molecules, takes a 1d xyz tensor and converts to using N_total
    """
    N_symm_sm, L =  xyz_sm.shape[:2]
    if N_symmetry is None:
        N_symmetry = N_symm_sm
    xyz = torch.full((N_symmetry, L, NTOTAL, 3), np.nan).float()
    xyz[:N_symm_sm, :, 1, :] = xyz_sm

    mask = torch.full((N_symmetry, L, NTOTAL), False).bool()
    mask[:N_symm_sm, :, 1] = mask_sm
    return xyz, mask

def same_chain_2d_from_Ls(Ls):
    """Given list of chain lengths, returns binary matrix with 1 if two residues are on the same chain."""
    same_chain = torch.zeros((sum(Ls),sum(Ls))).long()
    i_curr = 0
    for L in Ls:
        same_chain[i_curr:i_curr+L, i_curr:i_curr+L] = 1
        i_curr += L
    return same_chain

def Ls_from_same_chain_2d(same_chain):
    """Given binary matrix indicating whether two residues are on same chain, returns list of chain lengths"""
    if len(same_chain.shape)==3: # remove batch dimension
        same_chain = same_chain.squeeze(0)
    Ls = []
    i_curr = 0
    while i_curr < len(same_chain):
        idx = torch.where(same_chain[i_curr])[0]
        Ls.append(int(idx[-1]-idx[0]+1))
        i_curr = idx[-1]+1
    return Ls

def get_prot_seqstring(ch, modres):
    """Return string representing amino acid sequence of a parsed CIF chain."""
    idx = [int(k[1]) for k in ch.atoms]
    i_min, i_max = np.min(idx), np.max(idx)
    L = i_max - i_min + 1
    seq = ["-"]*L

    for k,v in ch.atoms.items():
        i_res = int(k[1])-i_min
        if k[2] in to1letter: # standard AA
            aa = to1letter[k[2]]
        elif k[2] in modres and modres[k[2]] in to1letter: # nonstandard AA, map to standard
            aa = to1letter[modres[k[2]]]
        else: # unknown AA, still try to store BB atoms
            aa = 'X'
        seq[i_res] = aa
    return ''.join(seq)

def map_identical_prot_chains(partners, chains, modres):
    """Identifies which chain letters represent unique protein sequences,
    assigns a number to each unique sequence, and returns dicts mapping sequence
    numbers to chain letters and vice versa.
    
    Parameters
    ----------
    partners : list of tuples (partner, transform_index, num_contacts, partner_type)
        Information about neighboring chains to the query ligand in an
        assembly. This function will use the subset of these tuples that
        represent protein chains, where `partner_type = 'polypeptide(L)'`
        and `partner` contains the chain letter. `transform_index` is an
        integer index of the coordinate transform for each partner chain.
    chains : dict
        Dictionary mapping chain letters to cifutils.Chain objects representing
        the chains in a PDB entry.
    modres : dict
        Maps modified residue names to their canonical equivalents. Any
        modified residue will be converted to its standard equivalent and
        coordinates for atoms with matching names will be saved.

    Returns
    -------
    chnum2chlet : dict
        Dictionary mapping integers to lists of chain letters which represent
        identical chains
    """
    chlet2seq = OrderedDict()
    for p in partners:
        if p[-1] != 'polypeptide(L)': continue
        if p[0] not in chlet2seq:
            chlet2seq[p[0]] = get_prot_seqstring(chains[p[0]], modres)

    seq2chlet = OrderedDict()
    for chlet, seq in chlet2seq.items():
        if seq not in seq2chlet:
            seq2chlet[seq] = set()
        seq2chlet[seq].add(chlet)

    chnum2chlet = OrderedDict([(i,v) for i,(k,v) in enumerate(seq2chlet.items())])
    #chlet2chnum = OrderedDict([(chlet,chnum) for chnum,chlet_s in chnum2chlet.items() for chlet in chlet_s])

    return chnum2chlet 

def reassign_symmetry_after_cropping(sel, Ls_prot, ch_label, mask, item):
    """
    for cases with protein symmetry, after cropping the precomputed protein symmetries are no longer valid
    this updates the mask so that the invalid protein symmetries arent considered in resolve_equiv_natives
    """
    # if a homomer subunit is cropped you need to reorder the symmetry axis because the
    # small molecule will not longer be in the global frames in the projections of the single chain
    # across different symmetry axes
    # depending on how many subunits are cropped, you will have to mask out the invalid chain swaps

    protein_sel = sel[sel<sum(Ls_prot)]
    chain_break_idxs = [sum(Ls_prot[:i]) for i in range(len(Ls_prot))]
    chosen_prot_chains = torch.tensor(list(set(np.digitize(protein_sel, chain_break_idxs)))) -1 # 1 indexed

    # this code assumes that all permutations of each set of identical protein chains are computed and then they are
    # concatenated by catprodcat (See: featurize_asmb_prot)
    # this is not necessarily true for ligands because there could be multiple copies of the same ligand that are stored separately
    # so we will check that the protein chain chain labels are ascending
    ch_label_per_prot_chain = ch_label[chain_break_idxs] # makes ch_label chainwise instead of residue wise

    assert torch.all(torch.diff(ch_label_per_prot_chain) >=0), \
        f"all identical chains are not next to each other so symmetry resolution will not work. here is the order of chain labels\
                {ch_label_per_prot_chain}"
    
    all_perms = []
    ch_number = 0
    for num_repeats in Counter(ch_label_per_prot_chain.numpy()).values():
        perms_per_chain = torch.tensor(list(itertools.permutations(range(ch_number, ch_number + num_repeats)))) # torch doesnt have a permutations fx
        all_perms.append(perms_per_chain)
        ch_number += num_repeats
    perms = cartprodcat(all_perms)
    perms = perms[:, chosen_prot_chains.long()]
    is_valid_perm = torch.all( torch.isin(perms, chosen_prot_chains), dim=1)
    is_valid_perm = torch.nn.functional.pad(is_valid_perm, (0, mask.shape[0]-is_valid_perm.shape[0])) # pad in case there are more ligand symmetry dimensions
    if len(chosen_prot_chains) == 0:
        raise Exception(f"no protein was chosen in the crop for this item: {item}")
    else:
        # mask all invalid protein permutations
        mask[~is_valid_perm, :len(protein_sel)] = False 
    return mask

def cartprodcat(X_s):
    """Concatenate list of tensors on dimension 1 while taking their cartesian product
    over dimension 0."""
    X = X_s[0]
    for X_ in X_s[1:]:
        N, L = X.shape[:2]
        N_, L_ = X_.shape[:2]
        X_out = torch.full((N, N_, L+L_,)+X.shape[2:], np.nan)
        for i in range(N):
            for j in range(N_):
                X_out[i,j] = torch.concat([X[i], X_[j]], dim=0)
        dims = (N*N_,L+L_,)+X.shape[2:]
        X = X_out.view(*dims)
    return X

def idx_from_Ls(Ls):
    """Generate residue indexes from a list of chain lengths, 
    with a chain gap offset between indexes for each chain."""
    idx = []
    offset = 0
    for L in Ls:
        idx.append(torch.arange(L)+offset)
        offset = offset+L+CHAIN_GAP
    return torch.cat(idx, dim=0)


def bond_feats_from_Ls(Ls):
    """Generate protein (or DNA/RNA) bond features from a list of chain
    lengths"""
    bond_feats = torch.zeros((sum(Ls), sum(Ls))).long()
    offset = 0
    for L_ in Ls:
        bond_feats[offset:offset+L_, offset:offset+L_] = get_protein_bond_feats(L_)
        offset += L_
    return bond_feats

def same_chain_from_bond_feats(bond_feats):
    """Return binary matrix indicating if pairs of residues are on same chain,
    given their bond features.
    """
    assert(len(bond_feats.shape)==2) # assume no batch dimension
    L = bond_feats.shape[0]
    same_chain = torch.zeros((L,L))
    G = nx.from_numpy_matrix(bond_feats.detach().cpu().numpy())
    for idx in nx.connected_components(G):
        idx = list(idx)
        for i in idx:
            same_chain[i,idx] = 1
    return same_chain


def kabsch(xyz1, xyz2, eps=1e-6):
    """Superimposes `xyz2` coordinates onto `xyz1`, returns RMSD and rotation matrix."""
    # center to CA centroid
    xyz1 = xyz1 - xyz1.mean(0)
    xyz2 = xyz2 - xyz2.mean(0)

    # Computation of the covariance matrix
    C = xyz2.T @ xyz1

    # Compute optimal rotation matrix using SVD
    V, S, W = torch.linalg.svd(C)

    # get sign to ensure right-handedness
    d = torch.ones([3,3])
    d[:,-1] = torch.sign(torch.linalg.det(V)*torch.linalg.det(W))

    # Rotation matrix U
    U = (d*V) @ W

    # Rotate xyz2
    xyz2_ = xyz2 @ U

    L = xyz2_.shape[0]

    rmsd = torch.sqrt(torch.sum((xyz2_-xyz1)*(xyz2_-xyz1), axis=(0,1)) / L + eps)

    return rmsd, U


def get_contacts(xyz1, xyz2, max_dist=5):
    """
    gets a random contact between two xyz tensors within a certain distance cutoff
    """
    contacts = torch.cdist(xyz1, xyz2)
    is_close_euclidean = contacts < max_dist
    return is_close_euclidean


def get_triple_contact(is_close_euclidean_far_seq):
    """
    note: considered squaring the adjacency matrix for this computation but noticed it is slower than early stopping
    """
    contact_idxs = is_close_euclidean_far_seq.nonzero()
    contact_idxs = contact_idxs[torch.randperm(len(contact_idxs))]
    for i,j in contact_idxs:
        if j < i:
            continue
        K = (is_close_euclidean_far_seq[i,:] * is_close_euclidean_far_seq[j,:]).nonzero()
        if len(K):
            K = K[torch.randperm(len(K))]
            for k in K:
                return torch.tensor([i,j,k])
    return None


def get_residue_contacts(xyz, idx, max_dist=5, seq_dist_greater_than=10):
    """
    find two residues close in geometric space but far in sequence space 
    this mimics the theozyme task in diffusion so the structure prediction model can learn the task before 
    transfer learning
    """
    Cb = generate_Cbeta(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    is_close_euclidean = get_contacts(Cb, Cb, max_dist=max_dist)
    seqsep = torch.abs(idx[None, :] - idx[:, None])
    is_far_seqsep = seqsep > seq_dist_greater_than

    is_close_euclidean_far_seq = is_close_euclidean * is_far_seqsep

    triplet =  get_triple_contact(is_close_euclidean_far_seq) # returns absolute index in tensors but want residue indices for other functions

    if triplet is None:
        return None
    return idx[triplet]

def detect_sequence_type(sequence):
    if sum([1 for x in sequence if x in '-ACTGD']) == len(sequence):
        mode = 'dsdna' # auto assumes DNA should be duplex by default
    elif sum([1 for x in sequence if x in '-ACUGN']) == len(sequence):
        mode = 'rna'
    elif sum([1 for x in sequence if x in '-ACDEFGHIKLMNPQRSTVWXY']) == len(sequence):
        mode = 'protein'
    elif sum([1 for x in sequence if x in '-ACDEFGHIKLMNPQRSTVWXYacgtxbdhuy']) == len(sequence):
        mode = 'full'
    else:
        mode = 'smiles'
    return mode
    
