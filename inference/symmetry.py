"""Helper class for handle symmetric assemblies."""
from pyrsistent import v
from scipy.spatial.transform import Rotation
import functools as fn
import torch
import string
import logging
import numpy as np
import sys
import copy 
import pdb
from icecream import ic
import matplotlib.pyplot as plt 
from pdb import set_trace
format_rots = lambda r: torch.tensor(r).float()

T3_ROTATIONS = [
    torch.Tensor([
        [ 1.,  0.,  0.],
        [ 0.,  1.,  0.],
        [ 0.,  0.,  1.]]).float(),
    torch.Tensor([
        [-1., -0.,  0.],
        [-0.,  1.,  0.],
        [-0.,  0., -1.]]).float(),
    torch.Tensor([
        [-1.,  0.,  0.],
        [ 0., -1.,  0.],
        [ 0.,  0.,  1.]]).float(),
    torch.Tensor([
        [ 1.,  0.,  0.],
        [ 0., -1.,  0.],
        [ 0.,  0., -1.]]).float(),
]

saved_symmetries = ['tetrahedral', 'octahedral', 'icosahedral']

class SymGen:

    def __init__(self, global_sym, recenter, radius, model_only_neibhbors=False):
        self._log = logging.getLogger(__name__)
        self._recenter = recenter
        self._radius = radius

        if global_sym.lower().startswith('c'):
            # Cyclic symmetry
            if not global_sym[1:].isdigit():
                raise ValueError(f'Invalid cyclic symmetry {global_sym}')
            self._log.info(
                f'Initializing cyclic symmetry order {global_sym[1:]}.')
            self._init_cyclic(int(global_sym[1:]))
            self.apply_symmetry = self._apply_cyclic

        elif global_sym.lower().startswith('mc'):
            # 2-component cyclic symmetry
            order = global_sym[2:]
            if not order.isdigit():
                raise ValueError(f'Invalid cyclic symmetry {global_sym}')
            self._log.info(
                f'Initializing 2-component cyclic symmetry order {order}.')
            self._init_multi_cyclic(int(order), 2)
            self.apply_symmetry = self._apply_multi_cyclic

        elif global_sym.lower().startswith('d'):
            # Dihedral symmetry
            if not global_sym[1:].isdigit():
                raise ValueError(f'Invalid dihedral symmetry {global_sym}')
            self._log.info(
                f'Initializing dihedral symmetry order {global_sym[1:]}.')
            self._init_dihedral(int(global_sym[1:]))
            # Applied the same way as cyclic symmetry
            self.apply_symmetry = self._apply_cyclic

        elif global_sym.lower() == 't3':
            # Tetrahedral (T3) symmetry
            self._log.info('Initializing T3 symmetry order.')
            self.sym_rots = T3_ROTATIONS
            self.order = 4
            # Applied the same way as cyclic symmetry
            self.apply_symmetry = self._apply_cyclic

        elif global_sym == 'octahedral':
            # Octahedral symmetry
            self._log.info(
                'Initializing octahedral symmetry.')
            self._init_octahedral()
            self.apply_symmetry = self._apply_octahedral

        elif global_sym.lower() in saved_symmetries:
            # Using a saved symmetry 
            self._log.info('Initializing %s symmetry order.'%global_sym)
            self._init_from_symrots_file(global_sym)

            # Applied the same way as cyclic symmetry
            self.apply_symmetry = self._apply_cyclic
        else:
            raise ValueError(f'Unrecognized symmetry {global_sym}')

        # self.model_only_neibhbors = model_only_neibhbors
        # if self.model_only_neibhbors:
        #     self.close_rots = self.close_neighbors()
        # self.num_subunits = len(self.close_rots) if self.model_only_neibhbors else self.order
        self.res_idx_procesing = fn.partial(
            self._lin_chainbreaks, num_breaks=self.order)

     # DJ
    #####################
    ## pseudo symmetry ##
    #####################
    ### change to dimer A + B, works for dimmer only?
    def pseudo_chainbreak(self, pdb_idx, break_idx): 
        """
        Breaks the chain at desired index 
        """
        out_idx = torch.clone(pdb_idx)
        prev_break_idx = 0
        Ls = []
        for curr_break_idx in (str(break_idx).split("-")):
            curr_break_idx = int(curr_break_idx)
            #curr_break_idx = curr_break_idx - 1
            out_idx[:,curr_break_idx:] += 200 # 1-indexed
            Ls.append(curr_break_idx - prev_break_idx)
            prev_break_idx = curr_break_idx
        Ls.append(out_idx.shape[1] - prev_break_idx)
        # break_idx_1 = break_idx - 1
        # out_idx[:,break_idx_1:] += 200 # 1-indexed

        # La = break_idx_1
        # Lb = out_idx.shape[1] - La

        # out_chids = ['A']*La + ['B']*Lb
        out_chids = []
        for i, L in enumerate(Ls):
            out_chids += [string.ascii_uppercase[i]]*L
        return out_idx, out_chids


    #####################
    ## Cyclic symmetry ##
    #####################
    def _init_cyclic(self, order):
        sym_rots = []
        for i in range(order):
            deg = i * 360.0 / order
            r = Rotation.from_euler('z', deg, degrees=True)
            sym_rots.append(format_rots(r.as_matrix()))
        self.sym_rots = sym_rots
        self.order = order

    def _apply_cyclic(self, coords_in, seq_in):
        coords_out = torch.clone(coords_in)
        seq_out = torch.clone(seq_in)
        if seq_out.shape[0] % self.order != 0:
            raise ValueError(
                f'Sequence length must be divisble by {self.order}')
        subunit_len = seq_out.shape[0] // self.order
        for i in range(self.order):
            start_i = subunit_len * i
            end_i = subunit_len * (i+1)
            coords_out[start_i:end_i] = torch.einsum(
                'bnj,kj->bnk', coords_out[:subunit_len], self.sym_rots[i])
            seq_out[start_i:end_i]  = seq_out[:subunit_len]
        return coords_out, seq_out

    def _lin_chainbreaks(self, num_breaks, res_idx, offset=None):
        assert res_idx.ndim == 2
        res_idx = torch.clone(res_idx)
        subunit_len = res_idx.shape[-1] // num_breaks
        chain_delimiters = []
        if offset is None:
            offset = res_idx.shape[-1]
        for i in range(num_breaks):
            start_i = subunit_len * i
            end_i = subunit_len * (i+1)
            chain_labels = list(string.ascii_uppercase) + [str(i+j) for i in
                    string.ascii_uppercase for j in string.ascii_uppercase]
            chain_delimiters.extend(
                [chain_labels[i] for _ in range(subunit_len)]
            )
            res_idx[:, start_i:end_i] = res_idx[:, start_i:end_i] + offset * (i+1)
        return res_idx, chain_delimiters

    #####################################
    ## Multi-component cyclic symmetry ##
    #####################################
    def _init_multi_cyclic(self, order, num_subunits):
        subunit_rots = []
        subunit_order = order // num_subunits
        for i in range(subunit_order):
            deg = i * 360.0 / subunit_order
            r = Rotation.from_euler('z', deg, degrees=True)
            subunit_rots.append(format_rots(r.as_matrix()))
        self.subunit_rots = subunit_rots

        inter_rots = []
        for i in range(num_subunits):
            deg = i * 360.0 / order
            r = Rotation.from_euler('z', deg, degrees=True)
            inter_rots.append(format_rots(r.as_matrix()))
        self.inter_rots = inter_rots

        sym_rots = []
        for i in range(order):
            deg = i * 360.0 / order
            r = Rotation.from_euler('z', deg, degrees=True)
            sym_rots.append(format_rots(r.as_matrix()))
        self.sym_rots = sym_rots

        self.order = order
        self.num_subunits = num_subunits
    
    def _apply_multi_cyclic(self, coords_in, seq_in):
        coords_out = torch.clone(coords_in)
        seq_out = torch.clone(seq_in)
        if seq_out.shape[0] % self.order != 0:
            raise ValueError(
                f'Sequence length must be divisible by {self.order}')

        subunit_len = seq_out.shape[0] // self.order
        rot_indices = []
        for i in range(len(self.subunit_rots)):
            for _ in range(self.num_subunits):
                rot_indices.append(i)
        
        base_axis = torch.tensor([self._radius, 0., 0.])[None]
        for i, rot in enumerate(range(self.order)):
            subunit_idx = i % self.num_subunits
            inter_rot = self.inter_rots[subunit_idx]
            rot = self.subunit_rots[rot_indices[i]]

            subunit_chain = coords_in[
                (subunit_len * subunit_idx):(subunit_len * (subunit_idx+1))]
            subunit_chain = torch.einsum(
                'bnj,kj->bnk', subunit_chain, inter_rot)
            subunit_seq = seq_out[
                (subunit_len * subunit_idx):(subunit_len * (subunit_idx+1))]
            
            
            start_i = subunit_len * i
            end_i = subunit_len * (i+1)
            subunit_chain = torch.einsum(
                'bnj,kj->bnk', subunit_chain, rot)

            if self._recenter:
                center = torch.mean(subunit_chain[:, 1, :], axis=0)
                subunit_chain -= center[None, None, :]
                rotated_axis = torch.einsum(
                    'nj,kj->nk', base_axis, self.sym_rots[i]) 
                subunit_chain += rotated_axis[:, None, :]

            coords_out[start_i:end_i] = subunit_chain
            seq_out[start_i:end_i]  = subunit_seq

        return coords_out, seq_out

    #######################
    ## Dihedral symmetry ##
    #######################
    def _init_dihedral(self, order):
        sym_rots = []
        flip = Rotation.from_euler('x', 180, degrees=True).as_matrix()
        for i in range(order):
            deg = i * 360.0 / order
            rot = Rotation.from_euler('z', deg, degrees=True).as_matrix()
            sym_rots.append(format_rots(rot))
            rot2 = flip @ rot
            sym_rots.append(format_rots(rot2))
        self.sym_rots = sym_rots
        self.order = order * 2

    #########################
    ## Octahedral symmetry ##
    #########################
    def _init_octahedral(self):
        sym_rots = np.load("./inference/sym_rots.npz")
        self.sym_rots = [
            torch.tensor(v_i, dtype=torch.float32)
            for v_i in sym_rots['octahedral']
        ]
        self.order = len(self.sym_rots)

    def _apply_octahedral(self, coords_in, seq_in):
        coords_out = torch.clone(coords_in)
        seq_out = torch.clone(seq_in)
        if seq_out.shape[0] % self.order != 0:
            raise ValueError(
                f'Sequence length must be divisble by {self.order}')
        subunit_len = seq_out.shape[0] // self.order
        base_axis = torch.tensor([self._radius, 0., 0.])[None]
        for i in range(self.order):
            start_i = subunit_len * i
            end_i = subunit_len * (i+1)
            subunit_chain = torch.einsum(
                'bnj,kj->bnk', coords_in[:subunit_len], self.sym_rots[i])

            if self._recenter:
                center = torch.mean(subunit_chain[:, 1, :], axis=0)
                subunit_chain -= center[None, None, :]
                rotated_axis = torch.einsum(
                    'nj,kj->nk', base_axis, self.sym_rots[i]) 
                subunit_chain += rotated_axis[:, None, :]

            coords_out[start_i:end_i] = subunit_chain
            seq_out[start_i:end_i]  = seq_out[:subunit_len]
        return coords_out, seq_out

    #######################
    ## symmetry from file #
    #######################
    def _init_from_symrots_file(self, name):
        """ _init_from_symrots_file initializes using 
        ./inference/sym_rots.npz

        Args:
            name: name of symmetry (of tetrahedral, octahedral, icosahedral)

        sets self.sym_rots to be a list of torch.tensor of shape [3, 3]
        """
        assert name in saved_symmetries, name + " not in " + str(saved_symmetries)

        # Load in list of rotation matrices for `name`
        fn = "./inference/sym_rots.npz"
        obj = np.load(fn)
        symms = None
        for k, v in obj.items():
            if str(k) == name: symms = v
        assert symms is not None, "%s not found in %s"%(name, fn)

        
        self.sym_rots =  [torch.tensor(v_i, dtype=torch.float32) for v_i in symms]
        self.order = len(self.sym_rots)

        # Return if identity is the first rotation  
        if not np.isclose(((self.sym_rots[0]-np.eye(3))**2).sum(), 0):

            # Move identity to be the first rotation
            for i, rot in enumerate(self.sym_rots):
                if np.isclose(((rot-np.eye(3))**2).sum(), 0):
                    self.sym_rots = [self.sym_rots.pop(i)]  + self.sym_rots

            assert len(self.sym_rots) == self.order
            assert np.isclose(((self.sym_rots[0]-np.eye(3))**2).sum(), 0)

    def close_neighbors(self):
        """close_neighbors finds the rotations within self.sym_rots that
        correspond to close neighbors.

        Returns:
            list of rotation matrices corresponding to the identity and close neighbors
        """
        # set of small rotation angle rotations
        rel_rot = lambda M: np.linalg.norm(Rotation.from_matrix(M).as_rotvec())
        rel_rots = [(i+1, rel_rot(M)) for i, M in enumerate(self.sym_rots[1:])]
        min_rot = min(rel_rot_val[1] for rel_rot_val in rel_rots)
        close_rots = [np.eye(3)] + [
                self.sym_rots[i] for i, rel_rot_val in rel_rots if
                np.isclose(rel_rot_val, min_rot)
                ]
        return close_rots



##########################
# Franks Symmetry Stuff  # 
##########################

SYMA = 1.0

def generateC(angs, eps=1e-6):
    L = angs.shape[0]
    Rs = torch.eye(3,  device=angs.device).repeat(L,1,1)
    Rs[:,1,1] = torch.cos(angs)
    Rs[:,1,2] = -torch.sin(angs)
    Rs[:,2,1] = torch.sin(angs)
    Rs[:,2,2] = torch.cos(angs)
    return Rs

def generateCz(angs, eps=1e-6):
    L = angs.shape[0]
    Rs = torch.eye(3,  device=angs.device).repeat(L,1,1)

    Rs[:,0,0] = torch.cos(angs)
    Rs[:,0,1] = -torch.sin(angs)
    Rs[:,1,0] = torch.sin(angs)
    Rs[:,1,1] = torch.cos(angs)

    return Rs

def generateD(angs, eps=1e-6):
    L = angs.shape[0]
    Rs = torch.eye(3,  device=angs.device).repeat(2*L,1,1)
    Rs[:L,1,1] = torch.cos(angs)
    Rs[:L,1,2] = -torch.sin(angs)
    Rs[:L,2,1] = torch.sin(angs)
    Rs[:L,2,2] = torch.cos(angs)
    Rx = torch.tensor([[-1.,0,0],[0,-1,0],[0,0,1]],device=angs.device)
    Rs[L:] = torch.einsum('ij,bjk->bik',Rx,Rs[:L])
    return Rs

# used to be called find_symm_subs from FD 
def find_minimal_neighbors(indep, Rs, metasymm):
    """
    Parameters:
        indep: Indep object 

        Rs: (L,3,3) tensor of rotation matrices

        metasymm: (subsymms,nneighs) tuple of lists of symmetry subunits
            e.g., for icosahedral: metasymm = (
                                                [torch.arange(60)], # number of subunits
                                                [6] # number of subunits modelled 
                                                )
    """
    assert indep.is_sm.sum() == 0, "find_minimal_neighbors only works for non-sm"
    L_orig = indep.xyz.shape[0]
    xyz = indep.xyz[None]


    com = xyz[:,:,1].mean(dim=-2) # center of mass of single chain 
    rcoms = torch.einsum('sij,bj->si', Rs, com)

    # subsymms - list of length 1, torch.arange(nsubs)
    # nneighs - list of length 1, total number of subunits being modelled
    subsymms, nneighs = metasymm

    subs = []
    for i in range(len(subsymms)):
        # distances of com of subunits to the first one (index 0)
        drcoms = torch.linalg.norm(rcoms[0,:] - rcoms[subsymms[i],:], dim=-1)

         # indices of the topk closest subunits to the first one
        _,subs_i = torch.topk(drcoms,nneighs[i],largest=False)
        
        # indices of the canonical subunits that are closest 
        subs_i,_ = torch.sort( subsymms[i][subs_i] )
        subs.append(subs_i)

    subs=torch.cat(subs)
    # now that we know the canonical identity of closest, rotate the coordinates
    # according to the rotation matrix of the canonical subunit
    xyz_new = torch.einsum('sij,braj->bsrai', Rs[subs], xyz).reshape(
        xyz.shape[0],-1,xyz.shape[2],3)

    o = copy.deepcopy(indep)
    Ncopy = subs.shape[0]
    """
    Shapes that o needs to have: 

        L is total number of amino acids 

        xyz         : (L,14,3)
        atom_frames : (?,?,?)
        bond_feats  : (L, L)
        chirals     : (?)
        idx         : (L)
        is_sm       : (L)
        same_chain  : (L, L)
        seq         : (L)
    """

    # copy other tensors to match the new xyz shape
    # ic(o.xyz.shape)
    # ic(o.atom_frames.shape)
    # ic(o.bond_feats.shape)
    # ic(o.chirals.shape)
    # ic(o.idx.shape)
    # ic(o.is_sm.shape)
    # ic(o.same_chain.shape)
    # ic(o.seq.shape)
    # print('*'*40)

    # 1D information 
    o.xyz           = xyz_new.squeeze(0)
    o.seq           = torch.cat([o.seq[None]]*Ncopy,dim=1).squeeze(0) 
    o.atom_frames   = torch.cat([o.atom_frames[None]]*Ncopy,dim=1).squeeze(0)
    o.chirals       = torch.cat([o.chirals[None]]*Ncopy,dim=1).squeeze(0)

    o.idx           = torch.cat([o.idx[None]+(L_orig+200)*i for i in range(Ncopy)],dim=1).squeeze(0)
    o.is_sm         = torch.cat([o.is_sm[None]]*Ncopy,dim=1).squeeze(0)

    o.is_protein    = torch.cat([o.is_protein[None]]*Ncopy,dim=1).squeeze(0)
    o.is_dna        = torch.cat([o.is_dna[None]]*Ncopy,dim=1).squeeze(0)
    o.is_rna        = torch.cat([o.is_rna[None]]*Ncopy,dim=1).squeeze(0)

    # 

    o.terminus_type = torch.cat([o.terminus_type[None]]*Ncopy,dim=1).squeeze(0)

    # 2D information
    # bond_feats: 5 at (i,j) = (i,i-1) and (i,i+1) 
    new_bond_feats = torch.zeros((Ncopy*L_orig, Ncopy*L_orig),dtype=torch.long)
    new_same_chain = torch.zeros((Ncopy*L_orig, Ncopy*L_orig),dtype=torch.long)
    for i_copy in range(Ncopy):
        start = i_copy*L_orig
        end = (i_copy+1)*L_orig
        # slice in the original same chain tensors along main diag
        new_bond_feats[start:end,start:end] = o.bond_feats
        new_same_chain[start:end,start:end] = o.same_chain
    
    o.bond_feats = new_bond_feats
    o.same_chain = new_same_chain

    return o, subs

def get_symm_map(subs,O):
    symmmask = torch.zeros(O,dtype=torch.long)
    symmmask[subs] = torch.arange(1,subs.shape[0]+1)
    return symmmask


# def rotation_from_matrix(R, eps=1e-5):
#     w, W = torch.linalg.eig(R.T)

#     i = torch.where(abs(torch.real(w) - 1.0) < eps)[0]
    
#     if (len(i)==0):
#         print('This is R ', R)
#         i = torch.tensor(0)
#         print (torch.real(w))
#         print (torch.real(R.T))
        
#     axis = torch.real(W[:, i[-1]]).squeeze()

#     cosa = (torch.trace(R) - 1.0) / 2.0
#     if abs(axis[2]) > eps:
#         sina = (R[1, 0] + (cosa-1.0)*axis[0]*axis[1]) / axis[2]
#     elif abs(axis[1]) > eps:
#         sina = (R[0, 2] + (cosa-1.0)*axis[0]*axis[2]) / axis[1]
#     else:
#         sina = (R[2, 1] + (cosa-1.0)*axis[1]*axis[2]) / axis[0]
#     angle = torch.atan2(sina, cosa)

#     return angle, axis

# convert a (...,3,3) rot matrix stack to axis (...,3) and angle (...) of rotation
def rotation_from_matrix(R, eps=1e-4):
    Rorig = R.shape
    R = R.reshape(-1,3,3)
    w, W = torch.linalg.eig(R.transpose(-1,-2))
    i = torch.argmax(torch.real(w), dim=-1)
    axis = torch.gather(torch.real(W), -1, i[:,None,None].repeat(1,3,1)).squeeze(-1)
    
    tr = (torch.eye(3)*R).sum(dim=(-1,-2))
    cosa = (tr - 1.0) / 2.0
    sina = torch.zeros_like(cosa)
    mask1 = abs(axis[:,2]) > eps
    sina[mask1] = (R[mask1,1, 0] + (cosa[mask1]-1.0)*axis[mask1,0]*axis[mask1,1]) / axis[mask1,2]
    mask2 = ~mask1 * (abs(axis[:,1]) > eps)
    sina[mask2] = (R[mask2,0, 2] + (cosa[mask2]-1.0)*axis[mask2,0]*axis[mask2,2]) / axis[mask2,1]
    mask3 = ~(mask1+mask2)
    sina[mask3] = (R[mask3,2, 1] + (cosa[mask3]-1.0)*axis[mask3,1]*axis[mask3,2]) / axis[mask3,0]
    angle = torch.atan2(sina, cosa)
    angle = angle.reshape(Rorig[:-2])
    axis = axis.reshape((*Rorig[:-2],3))
    return angle, axis

# convert axis/angle to rotation matrix
def matrix_from_rotation(angle, axis):
    c,s = np.cos(angle), np.sin(angle)
    t = 1-c

    R = torch.zeros((3,3), device = axis.device)
    R[0,0] = c + axis[0]*axis[0]*t
    R[1,1] = c + axis[1]*axis[1]*t
    R[2,2] = c + axis[2]*axis[2]*t
    R[1,0] = axis[0]*axis[1]*t + axis[2]*s
    R[0,1] = axis[0]*axis[1]*t - axis[2]*s
    R[2,0] = axis[0]*axis[2]*t - axis[1]*s
    R[0,2] = axis[0]*axis[2]*t + axis[1]*s
    R[2,1] = axis[1]*axis[2]*t + axis[0]*s
    R[1,2] = axis[1]*axis[2]*t - axis[0]*s

    return R


def kabsch(pred, true):
    def rmsd(V, W, eps=1e-6):
        L = V.shape[0]
        return torch.sqrt(torch.sum((V-W)*(V-W)) / L + eps)
    def centroid(X):
        return X.mean(dim=-2, keepdim=True)

    cP = centroid(pred)
    cT = centroid(true)
    pred = pred - cP
    true = true - cT
    C = torch.matmul(pred.permute(1,0), true)
    V, S, W = torch.svd(C)
    d = torch.ones([3,3], device=pred.device)
    d[:,-1] = torch.sign(torch.det(V)*torch.det(W))
    U = torch.matmul(d*V, W.permute(1,0)) # (IB, 3, 3)
    rpred = torch.matmul(pred, U) # (IB, L*3, 3)
    rms = rmsd(rpred, true)
    return rms, U, cP, cT

# do lines X0->X and Y0->Y intersect?
def intersect(X0,X,Y0,Y,eps=0.1):
    mtx = torch.cat(
        (torch.stack((X0,X0+X,Y0,Y0+Y)), torch.ones((4,1))) , axis=1 
    )
    det = torch.linalg.det( mtx )
    return (torch.abs(det) <= eps)

def get_angle(X,Y):
    angle = torch.acos( torch.clamp( torch.sum(X*Y), -1., 1. ) )
    if (angle > np.pi/2):
        angle = np.pi - angle
    return angle

# given the coordinates of a subunit + 
# def get_symmetry(xyz, mask, rms_cut=2.5, nfold_cut=0.1, angle_cut=0.05, trans_cut=2.0):
#     nops = xyz.shape[0]
#     L = xyz.shape[1]//2

#     # PASS 1: find all symm axes
#     symmaxes = []
#     for i in range(nops):
#         # if there are multiple biomt records, this may occur.
#         # rather than try to rescue, we will take the 1st (typically author-assigned)
#         offset0 = torch.linalg.norm(xyz[i,:L,1]-xyz[0,:L,1], dim=-1)
#         if (torch.mean(offset0)>1e-4):
#             continue

#         # get alignment
#         mask_i = mask[i,:L,1]*mask[i,L:,1]
#         xyz_i = xyz[i,:L,1][mask_i,:]
#         xyz_j = xyz[i,L:,1][mask_i,:]

#         rms_ij, Uij, cI, cJ = kabsch(xyz_i, xyz_j)
#         if (rms_ij > rms_cut):
#             print (i,'rms',rms_ij)
#             continue

#         # get axis and point symmetry about axis
#         angle, axis = rotation_from_matrix(Uij)
#         nfold = 2*np.pi/torch.abs(angle)

#         # a) ensure integer # of subunits per rotation
#         if (torch.abs( nfold - torch.round(nfold) ) > nfold_cut ):
#             #print ('nfold fail',nfold)
#             continue

#         nfold = torch.round(nfold).long()
#         # b) ensure rotation only (no translation)
#         delCOM = torch.mean(xyz_i, dim=-2) - torch.mean(xyz_j, dim=-2)
#         trans_dot_symaxis = nfold * torch.abs(torch.dot(delCOM, axis))
#         if (trans_dot_symaxis > trans_cut ):
#             #print ('trans fail',trans_dot_symaxis)
#             continue


#         # 3) get a point on the symm axis from CoMs and angle 
#         cIJ = torch.sign(angle) * (cJ-cI).squeeze(0)
#         dIJ = torch.linalg.norm(cIJ)
#         p_mid = (cI+cJ).squeeze(0) / 2
#         u = cIJ / dIJ            # unit vector in plane of circle
#         v = torch.cross(axis, u) # unit vector from sym axis to p_mid
#         r = dIJ / (2*torch.sin(angle/2))
#         d = torch.sqrt( r*r - dIJ*dIJ/4 ) # distance from mid-chord to center
#         point = p_mid - (d)*v

#         # check if redundant
#         toadd = True
#         for j,(nf_j,ax_j,pt_j,err_j) in enumerate(symmaxes):
#             if (not intersect(pt_j,ax_j,point,axis)):
#                 continue
#             angle_j = get_angle(ax_j,axis)
#             if (angle_j < angle_cut):
#                 if (nf_j < nfold): # stored is a subsymmetry of complex, overwrite
#                     symmaxes[j] = (nfold, axis, point, i)
#                 toadd = False

#         if (toadd):
#             symmaxes.append( (nfold, axis, point, i) )

#     # PASS 2: combine
#     symmgroup = 'C1'
#     subsymm = []
#     if len(symmaxes)==1:
#         symmgroup = 'C%d'%(symmaxes[0][0])
#         subsymm = [symmaxes[0][3]]
#     elif len(symmaxes)>1:
#         symmaxes = sorted(symmaxes, key=lambda x: x[0], reverse=True)
#         angle = get_angle(symmaxes[0][1],symmaxes[1][1])
#         subsymm = [symmaxes[0][3],symmaxes[1][3]]

#         # 2-fold and n-fold intersect at 90 degress => Dn
#         if (symmaxes[1][0] == 2 and torch.abs(angle-np.pi/2) < angle_cut):
#             symmgroup = 'D%d'%(symmaxes[0][0])
#         else:
#             # polyhedral rules:
#             #   3-Fold + 2-fold intersecting at acos(-1/sqrt(3)) -> T
#             angle_tgt = np.arccos(-1/np.sqrt(3))
#             if (symmaxes[0][0] == 3 and symmaxes[1][0] == 2 and torch.abs(angle - angle_tgt) < angle_cut):
#                 symmgroup = 'T'

#             #   3-Fold + 2-fold intersecting at asin(1/sqrt(3)) -> O
#             angle_tgt = np.arcsin(1/np.sqrt(3))
#             if (symmaxes[0][0] == 3 and symmaxes[1][0] == 2 and torch.abs(angle - angle_tgt) < angle_cut):
#                 symmgroup = 'O'

#             #   4-Fold + 3-fold intersecting at acos(1/sqrt(3)) -> O
#             angle_tgt = np.arccos(1/np.sqrt(3))
#             if (symmaxes[0][0] == 4 and symmaxes[1][0] == 3 and torch.abs(angle - angle_tgt) < angle_cut):
#                 symmgroup = 'O'

#             #   3-Fold + 2-fold intersecting at 0.5*acos(sqrt(5)/3) -> I
#             angle_tgt = 0.5*np.arccos(np.sqrt(5)/3)
#             if (symmaxes[0][0] == 3 and symmaxes[1][0] == 2 and torch.abs(angle - angle_tgt) < angle_cut):
#                 symmgroup = 'I'

#             #   5-Fold + 2-fold intersecting at 0.5*acos(1/sqrt(5)) -> I
#             angle_tgt = 0.5*np.arccos(1/np.sqrt(5))
#             if (symmaxes[0][0] == 5 and symmaxes[1][0] == 2 and torch.abs(angle - angle_tgt) < angle_cut):
#                 symmgroup = 'I'

#             #   5-Fold + 3-fold intersecting at 0.5*acos((4*sqrt(5)-5)/15) -> I
#             angle_tgt = 0.5*np.arccos((4*np.sqrt(5)-5)/15)
#             if (symmaxes[0][0] == 5 and symmaxes[1][0] == 3 and torch.abs(angle - angle_tgt) < angle_cut):
#                 symmgroup = 'I'
#             else:
#                 pass
#                 #fd: we could use a single symmetry here instead.  
#                 #    But these cases mostly are bad BIOUNIT annotations...
#                 #print ('nomatch',angle, [(x,y) for x,_,_,y in symmaxes])

#     return symmgroup, subsymm

def get_symmetry(xyz, mask, rms_cut=2.5, nfold_cut=0.1, angle_cut=0.05, trans_cut=2.0):
    nops = xyz.shape[0]
    L = xyz.shape[1]//2

    # PASS 1: find all symm axes
    symmaxes = []
    for i in range(nops):
        # if there are multiple biomt records, this may occur.
        # rather than try to rescue, we will take the 1st (typically author-assigned)
        offset0 = torch.linalg.norm(xyz[i,:L,1]-xyz[0,:L,1], dim=-1)
        if (torch.mean(offset0)>1e-4):
            continue

        # get alignment
        mask_i = mask[i,:L,1]*mask[i,L:,1]
        xyz_i = xyz[i,:L,1][mask_i,:]
        xyz_j = xyz[i,L:,1][mask_i,:]
        rms_ij, Uij, cI, cJ = kabsch(xyz_i, xyz_j)
        if (rms_ij > rms_cut):
            #print (i,'rms',rms_ij)
            continue

        # get axis and point symmetry about axis
        angle, axis = rotation_from_matrix(Uij)
        nfold = 2*np.pi/torch.abs(angle)
        # a) ensure integer # of subunits per rotation
        if (torch.abs( nfold - torch.round(nfold) ) > nfold_cut ):
            #print ('nfold fail',nfold)
            continue
        nfold = torch.round(nfold).long()
        # b) ensure rotation only (no translation)
        delCOM = torch.mean(xyz_i, dim=-2) - torch.mean(xyz_j, dim=-2)
        trans_dot_symaxis = nfold * torch.abs(torch.dot(delCOM, axis))
        if (trans_dot_symaxis > trans_cut ):
            #print ('trans fail',trans_dot_symaxis)
            continue


        # 3) get a point on the symm axis from CoMs and angle 
        cIJ = torch.sign(angle) * (cJ-cI).squeeze(0)
        dIJ = torch.linalg.norm(cIJ)
        p_mid = (cI+cJ).squeeze(0) / 2
        u = cIJ / dIJ            # unit vector in plane of circle
        v = torch.cross(axis, u) # unit vector from sym axis to p_mid
        r = dIJ / (2*torch.sin(angle/2))
        d = torch.sqrt( r*r - dIJ*dIJ/4 ) # distance from mid-chord to center
        point = p_mid - (d)*v

        # check if redundant
        toadd = True
        for j,(nf_j,ax_j,pt_j,err_j) in enumerate(symmaxes):
            if (not intersect(pt_j,ax_j,point,axis)):
                continue
            angle_j = get_angle(ax_j,axis)
            if (angle_j < angle_cut):
                if (nf_j < nfold): # stored is a subsymmetry of complex, overwrite
                    symmaxes[j] = (nfold, axis, point, i)
                toadd = False

        if (toadd):
            symmaxes.append( (nfold, axis, point, i) )

    # PASS 2: combine
    symmgroup = 'C1'
    subsymm = []
    if len(symmaxes)==1:
        symmgroup = 'C%d'%(symmaxes[0][0])
        subsymm = [symmaxes[0][3]]
    elif len(symmaxes)>1:
        symmaxes = sorted(symmaxes, key=lambda x: x[0], reverse=True)
        angle = get_angle(symmaxes[0][1],symmaxes[1][1])
        subsymm = [symmaxes[0][3],symmaxes[1][3]]

        # 2-fold and n-fold intersect at 90 degress => Dn
        if (symmaxes[1][0] == 2 and torch.abs(angle-np.pi/2) < angle_cut):
            symmgroup = 'D%d'%(symmaxes[0][0])
        else:
            # polyhedral rules:
            #   3-Fold + 2-fold intersecting at acos(-1/sqrt(3)) -> T
            angle_tgt = np.arccos(-1/np.sqrt(3))
            if (symmaxes[0][0] == 3 and symmaxes[1][0] == 2 and torch.abs(angle - angle_tgt) < angle_cut):
                symmgroup = 'T'

            #   3-Fold + 2-fold intersecting at asin(1/sqrt(3)) -> O
            angle_tgt = np.arcsin(1/np.sqrt(3))
            if (symmaxes[0][0] == 3 and symmaxes[1][0] == 2 and torch.abs(angle - angle_tgt) < angle_cut):
                symmgroup = 'O'

            #   4-Fold + 3-fold intersecting at acos(1/sqrt(3)) -> O
            angle_tgt = np.arccos(1/np.sqrt(3))
            if (symmaxes[0][0] == 4 and symmaxes[1][0] == 3 and torch.abs(angle - angle_tgt) < angle_cut):
                symmgroup = 'O'

            #   3-Fold + 2-fold intersecting at 0.5*acos(sqrt(5)/3) -> I
            angle_tgt = 0.5*np.arccos(np.sqrt(5)/3)
            if (symmaxes[0][0] == 3 and symmaxes[1][0] == 2 and torch.abs(angle - angle_tgt) < angle_cut):
                symmgroup = 'I'

            #   5-Fold + 2-fold intersecting at 0.5*acos(1/sqrt(5)) -> I
            angle_tgt = 0.5*np.arccos(1/np.sqrt(5))
            if (symmaxes[0][0] == 5 and symmaxes[1][0] == 2 and torch.abs(angle - angle_tgt) < angle_cut):
                symmgroup = 'I'

            #   5-Fold + 3-fold intersecting at 0.5*acos((4*sqrt(5)-5)/15) -> I
            angle_tgt = 0.5*np.arccos((4*np.sqrt(5)-5)/15)
            if (symmaxes[0][0] == 5 and symmaxes[1][0] == 3 and torch.abs(angle - angle_tgt) < angle_cut):
                symmgroup = 'I'
            else:
                pass
                #fd: we could use a single symmetry here instead.  
                #    But these cases mostly are bad BIOUNIT annotations...
                #print ('nomatch',angle, [(x,y) for x,_,_,y in symmaxes])

    return symmgroup, subsymm, symmaxes
        
# used to be called symm_subunit_matrix
def get_pointsym_meta(symmid):
    """
    Finds the subunit matrix for a given symmetry, and the number of subunits to be modeled.

    Parameters: 
        symmid (str): symmetry identifier, e.g. 'C2', 'D3', 'T', 'O', 'I'

    Returns:
        symmatrix (torch.Tensor): subunit matrix denoting which blocks make equivalent interactions 

        Rs (torch.Tensor): rotation matrices for each subunit

        metasymm (tuple): metadata for the symmetry, including the subunit indices and the number of subunits to be modeled
    """

    if (symmid[0].upper()=='C'):
        print('Initializing circular RF model symmetry...')

        nsub = int(symmid[1:]) # number of subunits


        symmatrix = (
            torch.arange(nsub)[:,None]-torch.arange(nsub)[None,:]
        )%nsub

        angles = torch.linspace(0,2*np.pi,nsub+1)[:nsub]
        Rs = generateCz(angles)

        metasymm = (
            [torch.arange(nsub)],   # subunit indices
            [min(3,nsub)]           # number of subunits, capped at modelling 3 
        )

        if (nsub==1):
            D = 0.0
        else:
            est_radius = 2.0*SYMA
            theta = 2.0*np.pi/nsub
            D = est_radius/np.sin(theta/2)

        # offset = torch.tensor([ 0.0, 0.0, float(D) ]) # original for sym about X
        offset = torch.tensor([float(D), 0.0, 0.0])
    
    elif (symmid[0].upper()=='D'):
        nsub = int(symmid[1:])
        cblk=(torch.arange(nsub)[:,None]-torch.arange(nsub)[None,:])%nsub
        symmatrix=torch.zeros((2*nsub,2*nsub),dtype=torch.long)
        symmatrix[:nsub,:nsub] = cblk
        symmatrix[:nsub,nsub:] = cblk+nsub
        symmatrix[nsub:,:nsub] = cblk+nsub
        symmatrix[nsub:,nsub:] = cblk
        angles = torch.linspace(0,2*np.pi,nsub+1)[:nsub]
        Rs = generateD(angles)

        #metasymm = (
        #    [torch.arange(nsub), nsub+torch.arange(nsub)],
        #    [min(3,nsub),2]
        #)
        metasymm = (
            [torch.arange(2*nsub)],
            [min(2*nsub,5)]
        )

        est_radius = 2.0*SYMA
        theta1 = 2.0*np.pi/nsub
        theta2 = np.pi
        D1 = est_radius/np.sin(theta1/2)
        D2 = est_radius/np.sin(theta2/2)
        offset = torch.tensor([ float(D2),0.0,float(D1) ])
        # offset = torch.tensor([ 0.0,0.0,0.0 ])

    elif (symmid.upper()=='T'):
        symmatrix=torch.tensor(
            [[ 0,  1,  2,  3,  8, 11,  9, 10,  4,  6,  7,  5],
            [ 1,  0,  3,  2,  9, 10,  8, 11,  5,  7,  6,  4],
            [ 2,  3,  0,  1, 10,  9, 11,  8,  6,  4,  5,  7],
            [ 3,  2,  1,  0, 11,  8, 10,  9,  7,  5,  4,  6],
            [ 4,  6,  7,  5,  0,  1,  2,  3,  8, 11,  9, 10],
            [ 5,  7,  6,  4,  1,  0,  3,  2,  9, 10,  8, 11],
            [ 6,  4,  5,  7,  2,  3,  0,  1, 10,  9, 11,  8],
            [ 7,  5,  4,  6,  3,  2,  1,  0, 11,  8, 10,  9],
            [ 8, 11,  9, 10,  4,  6,  7,  5,  0,  1,  2,  3],
            [ 9, 10,  8, 11,  5,  7,  6,  4,  1,  0,  3,  2],
            [10,  9, 11,  8,  6,  4,  5,  7,  2,  3,  0,  1],
            [11,  8, 10,  9,  7,  5,  4,  6,  3,  2,  1,  0]])

        Rs = torch.zeros(12,3,3)
        Rs[ 0]=torch.tensor([[1.000000,0.000000,0.000000],[0.000000,1.000000,0.000000],[0.000000,0.000000,1.000000]])
        Rs[ 1]=torch.tensor([[-1.000000,0.000000,0.000000],[0.000000,-1.000000,0.000000],[0.000000,0.000000,1.000000]])
        Rs[ 2]=torch.tensor([[-1.000000,0.000000,0.000000],[0.000000,1.000000,0.000000],[0.000000,0.000000,-1.000000]])
        Rs[ 3]=torch.tensor([[1.000000,0.000000,0.000000],[0.000000,-1.000000,0.000000],[0.000000,0.000000,-1.000000]])
        Rs[ 4]=torch.tensor([[0.000000,0.000000,1.000000],[1.000000,0.000000,0.000000],[0.000000,1.000000,0.000000]])
        Rs[ 5]=torch.tensor([[0.000000,0.000000,1.000000],[-1.000000,0.000000,0.000000],[0.000000,-1.000000,0.000000]])
        Rs[ 6]=torch.tensor([[0.000000,0.000000,-1.000000],[-1.000000,0.000000,0.000000],[0.000000,1.000000,0.000000]])
        Rs[ 7]=torch.tensor([[0.000000,0.000000,-1.000000],[1.000000,0.000000,0.000000],[0.000000,-1.000000,0.000000]])
        Rs[ 8]=torch.tensor([[0.000000,1.000000,0.000000],[0.000000,0.000000,1.000000],[1.000000,0.000000,0.000000]])
        Rs[ 9]=torch.tensor([[0.000000,-1.000000,0.000000],[0.000000,0.000000,1.000000],[-1.000000,0.000000,0.000000]])
        Rs[10]=torch.tensor([[0.000000,1.000000,0.000000],[0.000000,0.000000,-1.000000],[-1.000000,0.000000,0.000000]])
        Rs[11]=torch.tensor([[0.000000,-1.000000,0.000000],[0.000000,0.000000,-1.000000],[1.000000,0.000000,0.000000]])
        nneigh = 5
        metasymm = (
            [torch.arange(12)],
            [6]
        )
        offset = torch.tensor([ 1.0, 1.0, 1.0 ]) # No idea what T offset should be - attempting this for now
    elif (symmid.upper()=='O'):
        symmatrix=torch.tensor(
            [[ 0,  1,  2,  3,  8, 11,  9, 10,  4,  6,  7,  5, 12, 13, 15, 14, 19, 17,
             18, 16, 22, 21, 20, 23],
            [ 1,  0,  3,  2,  9, 10,  8, 11,  5,  7,  6,  4, 13, 12, 14, 15, 18, 16,
             19, 17, 23, 20, 21, 22],
            [ 2,  3,  0,  1, 10,  9, 11,  8,  6,  4,  5,  7, 14, 15, 13, 12, 17, 19,
             16, 18, 20, 23, 22, 21],
            [ 3,  2,  1,  0, 11,  8, 10,  9,  7,  5,  4,  6, 15, 14, 12, 13, 16, 18,
             17, 19, 21, 22, 23, 20],
            [ 4,  6,  7,  5,  0,  1,  2,  3,  8, 11,  9, 10, 16, 18, 17, 19, 21, 22,
             23, 20, 15, 14, 12, 13],
            [ 5,  7,  6,  4,  1,  0,  3,  2,  9, 10,  8, 11, 17, 19, 16, 18, 20, 23,
             22, 21, 14, 15, 13, 12],
            [ 6,  4,  5,  7,  2,  3,  0,  1, 10,  9, 11,  8, 18, 16, 19, 17, 23, 20,
             21, 22, 13, 12, 14, 15],
            [ 7,  5,  4,  6,  3,  2,  1,  0, 11,  8, 10,  9, 19, 17, 18, 16, 22, 21,
             20, 23, 12, 13, 15, 14],
            [ 8, 11,  9, 10,  4,  6,  7,  5,  0,  1,  2,  3, 20, 23, 22, 21, 14, 15,
             13, 12, 17, 19, 16, 18],
            [ 9, 10,  8, 11,  5,  7,  6,  4,  1,  0,  3,  2, 21, 22, 23, 20, 15, 14,
             12, 13, 16, 18, 17, 19],
            [10,  9, 11,  8,  6,  4,  5,  7,  2,  3,  0,  1, 22, 21, 20, 23, 12, 13,
             15, 14, 19, 17, 18, 16],
            [11,  8, 10,  9,  7,  5,  4,  6,  3,  2,  1,  0, 23, 20, 21, 22, 13, 12,
             14, 15, 18, 16, 19, 17],
            [12, 13, 15, 14, 19, 17, 18, 16, 22, 21, 20, 23,  0,  1,  2,  3,  8, 11,
              9, 10,  4,  6,  7,  5],
            [13, 12, 14, 15, 18, 16, 19, 17, 23, 20, 21, 22,  1,  0,  3,  2,  9, 10,
              8, 11,  5,  7,  6,  4],
            [14, 15, 13, 12, 17, 19, 16, 18, 20, 23, 22, 21,  2,  3,  0,  1, 10,  9,
             11,  8,  6,  4,  5,  7],
            [15, 14, 12, 13, 16, 18, 17, 19, 21, 22, 23, 20,  3,  2,  1,  0, 11,  8,
             10,  9,  7,  5,  4,  6],
            [16, 18, 17, 19, 21, 22, 23, 20, 15, 14, 12, 13,  4,  6,  7,  5,  0,  1,
              2,  3,  8, 11,  9, 10],
            [17, 19, 16, 18, 20, 23, 22, 21, 14, 15, 13, 12,  5,  7,  6,  4,  1,  0,
              3,  2,  9, 10,  8, 11],
            [18, 16, 19, 17, 23, 20, 21, 22, 13, 12, 14, 15,  6,  4,  5,  7,  2,  3,
              0,  1, 10,  9, 11,  8],
            [19, 17, 18, 16, 22, 21, 20, 23, 12, 13, 15, 14,  7,  5,  4,  6,  3,  2,
              1,  0, 11,  8, 10,  9],
            [20, 23, 22, 21, 14, 15, 13, 12, 17, 19, 16, 18,  8, 11,  9, 10,  4,  6,
              7,  5,  0,  1,  2,  3],
            [21, 22, 23, 20, 15, 14, 12, 13, 16, 18, 17, 19,  9, 10,  8, 11,  5,  7,
              6,  4,  1,  0,  3,  2],
            [22, 21, 20, 23, 12, 13, 15, 14, 19, 17, 18, 16, 10,  9, 11,  8,  6,  4,
              5,  7,  2,  3,  0,  1],
            [23, 20, 21, 22, 13, 12, 14, 15, 18, 16, 19, 17, 11,  8, 10,  9,  7,  5,
              4,  6,  3,  2,  1,  0]])
        Rs = torch.zeros(24,3,3)
        Rs[0]=torch.tensor([[ 1.000000, 0.000000,0.000000],[ 0.000000, 1.000000,0.000000],[ 0.000000, 0.000000,1.000000]])
        Rs[1]=torch.tensor([[-1.000000, 0.000000,0.000000],[ 0.000000,-1.000000,0.000000],[ 0.000000, 0.000000,1.000000]])
        Rs[2]=torch.tensor([[-1.000000, 0.000000,0.000000],[ 0.000000, 1.000000,0.000000],[ 0.000000 ,0.000000,-1.000000]])
        Rs[3]=torch.tensor([[ 1.000000, 0.000000,0.000000],[ 0.000000,-1.000000,0.000000],[ 0.000000 ,0.000000,-1.000000]])
        Rs[4]=torch.tensor([[ 0.000000, 0.000000,1.000000],[ 1.000000, 0.000000,0.000000],[ 0.000000, 1.000000,0.000000]])
        Rs[5]=torch.tensor([[ 0.000000, 0.000000,1.000000],[-1.000000, 0.000000,0.000000],[ 0.000000,-1.000000,0.000000]])
        Rs[6]=torch.tensor([[ 0.000000 ,0.000000,-1.000000],[-1.000000, 0.000000,0.000000],[ 0.000000, 1.000000,0.000000]])
        Rs[7]=torch.tensor([[ 0.000000 ,0.000000,-1.000000],[ 1.000000, 0.000000,0.000000],[ 0.000000,-1.000000,0.000000]])
        Rs[8]=torch.tensor([[ 0.000000, 1.000000,0.000000],[ 0.000000, 0.000000,1.000000],[ 1.000000, 0.000000,0.000000]])
        Rs[9]=torch.tensor([[ 0.000000,-1.000000,0.000000],[ 0.000000, 0.000000,1.000000],[-1.000000, 0.000000,0.000000]])
        Rs[10]=torch.tensor([[ 0.000000, 1.000000,0.000000],[ 0.000000 ,0.000000,-1.000000],[-1.000000, 0.000000,0.000000]])
        Rs[11]=torch.tensor([[ 0.000000,-1.000000,0.000000],[ 0.000000 ,0.000000,-1.000000],[ 1.000000, 0.000000,0.000000]])
        Rs[12]=torch.tensor([[ 0.000000, 1.000000,0.000000],[ 1.000000, 0.000000,0.000000],[ 0.000000 ,0.000000,-1.000000]])
        Rs[13]=torch.tensor([[ 0.000000,-1.000000,0.000000],[-1.000000, 0.000000,0.000000],[ 0.000000 ,0.000000,-1.000000]])
        Rs[14]=torch.tensor([[ 0.000000, 1.000000,0.000000],[-1.000000, 0.000000,0.000000],[ 0.000000, 0.000000,1.000000]])
        Rs[15]=torch.tensor([[ 0.000000,-1.000000,0.000000],[ 1.000000, 0.000000,0.000000],[ 0.000000, 0.000000,1.000000]])
        Rs[16]=torch.tensor([[ 1.000000, 0.000000,0.000000],[ 0.000000, 0.000000,1.000000],[ 0.000000,-1.000000,0.000000]])
        Rs[17]=torch.tensor([[-1.000000, 0.000000,0.000000],[ 0.000000, 0.000000,1.000000],[ 0.000000, 1.000000,0.000000]])
        Rs[18]=torch.tensor([[-1.000000, 0.000000,0.000000],[ 0.000000 ,0.000000,-1.000000],[ 0.000000,-1.000000,0.000000]])
        Rs[19]=torch.tensor([[ 1.000000, 0.000000,0.000000],[ 0.000000 ,0.000000,-1.000000],[ 0.000000, 1.000000,0.000000]])
        Rs[20]=torch.tensor([[ 0.000000, 0.000000,1.000000],[ 0.000000, 1.000000,0.000000],[-1.000000, 0.000000,0.000000]])
        Rs[21]=torch.tensor([[ 0.000000, 0.000000,1.000000],[ 0.000000,-1.000000,0.000000],[ 1.000000, 0.000000,0.000000]])
        Rs[22]=torch.tensor([[ 0.000000 ,0.000000,-1.000000],[ 0.000000, 1.000000,0.000000],[ 1.000000, 0.000000,0.000000]])
        Rs[23]=torch.tensor([[ 0.000000 ,0.000000,-1.000000],[ 0.000000,-1.000000,0.000000],[-1.000000, 0.000000,0.000000]])
        nneigh = 5
        metasymm = (
            [torch.arange(24)],
            [6]
        )
        offset = torch.tensor([1.0, 1.0, 1.0]) # No idea what offset for O should be, using this for now 
    elif (symmid.upper()=='I'):
        symmatrix=torch.tensor(
            [[ 0,  4,  3,  2,  1,  5, 33, 49, 41, 22, 10, 27, 51, 59, 38, 15, 16, 17,
             18, 19, 40, 21,  9, 32, 48, 55, 39, 11, 28, 52, 45, 42, 23,  6, 34, 50,
             58, 37, 14, 26, 20,  8, 31, 47, 44, 30, 46, 43, 24,  7, 35, 12, 29, 53,
             56, 25, 54, 57, 36, 13],
            [ 1,  0,  4,  3,  2,  6, 34, 45, 42, 23, 11, 28, 52, 55, 39, 16, 17, 18,
             19, 15, 41, 22,  5, 33, 49, 56, 35, 12, 29, 53, 46, 43, 24,  7, 30, 51,
             59, 38, 10, 27, 21,  9, 32, 48, 40, 31, 47, 44, 20,  8, 36, 13, 25, 54,
             57, 26, 50, 58, 37, 14],
            [ 2,  1,  0,  4,  3,  7, 30, 46, 43, 24, 12, 29, 53, 56, 35, 17, 18, 19,
             15, 16, 42, 23,  6, 34, 45, 57, 36, 13, 25, 54, 47, 44, 20,  8, 31, 52,
             55, 39, 11, 28, 22,  5, 33, 49, 41, 32, 48, 40, 21,  9, 37, 14, 26, 50,
             58, 27, 51, 59, 38, 10],
            [ 3,  2,  1,  0,  4,  8, 31, 47, 44, 20, 13, 25, 54, 57, 36, 18, 19, 15,
             16, 17, 43, 24,  7, 30, 46, 58, 37, 14, 26, 50, 48, 40, 21,  9, 32, 53,
             56, 35, 12, 29, 23,  6, 34, 45, 42, 33, 49, 41, 22,  5, 38, 10, 27, 51,
             59, 28, 52, 55, 39, 11],
            [ 4,  3,  2,  1,  0,  9, 32, 48, 40, 21, 14, 26, 50, 58, 37, 19, 15, 16,
             17, 18, 44, 20,  8, 31, 47, 59, 38, 10, 27, 51, 49, 41, 22,  5, 33, 54,
             57, 36, 13, 25, 24,  7, 30, 46, 43, 34, 45, 42, 23,  6, 39, 11, 28, 52,
             55, 29, 53, 56, 35, 12],
            [ 5, 33, 49, 41, 22,  0,  4,  3,  2,  1, 15, 16, 17, 18, 19, 10, 27, 51,
             59, 38, 45, 42, 23,  6, 34, 50, 58, 37, 14, 26, 40, 21,  9, 32, 48, 55,
             39, 11, 28, 52, 25, 54, 57, 36, 13, 35, 12, 29, 53, 56, 30, 46, 43, 24,
              7, 20,  8, 31, 47, 44],
            [ 6, 34, 45, 42, 23,  1,  0,  4,  3,  2, 16, 17, 18, 19, 15, 11, 28, 52,
             55, 39, 46, 43, 24,  7, 30, 51, 59, 38, 10, 27, 41, 22,  5, 33, 49, 56,
             35, 12, 29, 53, 26, 50, 58, 37, 14, 36, 13, 25, 54, 57, 31, 47, 44, 20,
              8, 21,  9, 32, 48, 40],
            [ 7, 30, 46, 43, 24,  2,  1,  0,  4,  3, 17, 18, 19, 15, 16, 12, 29, 53,
             56, 35, 47, 44, 20,  8, 31, 52, 55, 39, 11, 28, 42, 23,  6, 34, 45, 57,
             36, 13, 25, 54, 27, 51, 59, 38, 10, 37, 14, 26, 50, 58, 32, 48, 40, 21,
              9, 22,  5, 33, 49, 41],
            [ 8, 31, 47, 44, 20,  3,  2,  1,  0,  4, 18, 19, 15, 16, 17, 13, 25, 54,
             57, 36, 48, 40, 21,  9, 32, 53, 56, 35, 12, 29, 43, 24,  7, 30, 46, 58,
             37, 14, 26, 50, 28, 52, 55, 39, 11, 38, 10, 27, 51, 59, 33, 49, 41, 22,
              5, 23,  6, 34, 45, 42],
            [ 9, 32, 48, 40, 21,  4,  3,  2,  1,  0, 19, 15, 16, 17, 18, 14, 26, 50,
             58, 37, 49, 41, 22,  5, 33, 54, 57, 36, 13, 25, 44, 20,  8, 31, 47, 59,
             38, 10, 27, 51, 29, 53, 56, 35, 12, 39, 11, 28, 52, 55, 34, 45, 42, 23,
              6, 24,  7, 30, 46, 43],
            [10, 27, 51, 59, 38, 15, 16, 17, 18, 19,  0,  4,  3,  2,  1,  5, 33, 49,
             41, 22, 50, 58, 37, 14, 26, 45, 42, 23,  6, 34, 55, 39, 11, 28, 52, 40,
             21,  9, 32, 48, 30, 46, 43, 24,  7, 20,  8, 31, 47, 44, 25, 54, 57, 36,
             13, 35, 12, 29, 53, 56],
            [11, 28, 52, 55, 39, 16, 17, 18, 19, 15,  1,  0,  4,  3,  2,  6, 34, 45,
             42, 23, 51, 59, 38, 10, 27, 46, 43, 24,  7, 30, 56, 35, 12, 29, 53, 41,
             22,  5, 33, 49, 31, 47, 44, 20,  8, 21,  9, 32, 48, 40, 26, 50, 58, 37,
             14, 36, 13, 25, 54, 57],
            [12, 29, 53, 56, 35, 17, 18, 19, 15, 16,  2,  1,  0,  4,  3,  7, 30, 46,
             43, 24, 52, 55, 39, 11, 28, 47, 44, 20,  8, 31, 57, 36, 13, 25, 54, 42,
             23,  6, 34, 45, 32, 48, 40, 21,  9, 22,  5, 33, 49, 41, 27, 51, 59, 38,
             10, 37, 14, 26, 50, 58],
            [13, 25, 54, 57, 36, 18, 19, 15, 16, 17,  3,  2,  1,  0,  4,  8, 31, 47,
             44, 20, 53, 56, 35, 12, 29, 48, 40, 21,  9, 32, 58, 37, 14, 26, 50, 43,
             24,  7, 30, 46, 33, 49, 41, 22,  5, 23,  6, 34, 45, 42, 28, 52, 55, 39,
             11, 38, 10, 27, 51, 59],
            [14, 26, 50, 58, 37, 19, 15, 16, 17, 18,  4,  3,  2,  1,  0,  9, 32, 48,
             40, 21, 54, 57, 36, 13, 25, 49, 41, 22,  5, 33, 59, 38, 10, 27, 51, 44,
             20,  8, 31, 47, 34, 45, 42, 23,  6, 24,  7, 30, 46, 43, 29, 53, 56, 35,
             12, 39, 11, 28, 52, 55],
            [15, 16, 17, 18, 19, 10, 27, 51, 59, 38,  5, 33, 49, 41, 22,  0,  4,  3,
              2,  1, 55, 39, 11, 28, 52, 40, 21,  9, 32, 48, 50, 58, 37, 14, 26, 45,
             42, 23,  6, 34, 35, 12, 29, 53, 56, 25, 54, 57, 36, 13, 20,  8, 31, 47,
             44, 30, 46, 43, 24,  7],
            [16, 17, 18, 19, 15, 11, 28, 52, 55, 39,  6, 34, 45, 42, 23,  1,  0,  4,
              3,  2, 56, 35, 12, 29, 53, 41, 22,  5, 33, 49, 51, 59, 38, 10, 27, 46,
             43, 24,  7, 30, 36, 13, 25, 54, 57, 26, 50, 58, 37, 14, 21,  9, 32, 48,
             40, 31, 47, 44, 20,  8],
            [17, 18, 19, 15, 16, 12, 29, 53, 56, 35,  7, 30, 46, 43, 24,  2,  1,  0,
              4,  3, 57, 36, 13, 25, 54, 42, 23,  6, 34, 45, 52, 55, 39, 11, 28, 47,
             44, 20,  8, 31, 37, 14, 26, 50, 58, 27, 51, 59, 38, 10, 22,  5, 33, 49,
             41, 32, 48, 40, 21,  9],
            [18, 19, 15, 16, 17, 13, 25, 54, 57, 36,  8, 31, 47, 44, 20,  3,  2,  1,
              0,  4, 58, 37, 14, 26, 50, 43, 24,  7, 30, 46, 53, 56, 35, 12, 29, 48,
             40, 21,  9, 32, 38, 10, 27, 51, 59, 28, 52, 55, 39, 11, 23,  6, 34, 45,
             42, 33, 49, 41, 22,  5],
            [19, 15, 16, 17, 18, 14, 26, 50, 58, 37,  9, 32, 48, 40, 21,  4,  3,  2,
              1,  0, 59, 38, 10, 27, 51, 44, 20,  8, 31, 47, 54, 57, 36, 13, 25, 49,
             41, 22,  5, 33, 39, 11, 28, 52, 55, 29, 53, 56, 35, 12, 24,  7, 30, 46,
             43, 34, 45, 42, 23,  6],
            [20,  8, 31, 47, 44, 30, 46, 43, 24,  7, 35, 12, 29, 53, 56, 25, 54, 57,
             36, 13,  0,  4,  3,  2,  1,  5, 33, 49, 41, 22, 10, 27, 51, 59, 38, 15,
             16, 17, 18, 19, 40, 21,  9, 32, 48, 55, 39, 11, 28, 52, 45, 42, 23,  6,
             34, 50, 58, 37, 14, 26],
            [21,  9, 32, 48, 40, 31, 47, 44, 20,  8, 36, 13, 25, 54, 57, 26, 50, 58,
             37, 14,  1,  0,  4,  3,  2,  6, 34, 45, 42, 23, 11, 28, 52, 55, 39, 16,
             17, 18, 19, 15, 41, 22,  5, 33, 49, 56, 35, 12, 29, 53, 46, 43, 24,  7,
             30, 51, 59, 38, 10, 27],
            [22,  5, 33, 49, 41, 32, 48, 40, 21,  9, 37, 14, 26, 50, 58, 27, 51, 59,
             38, 10,  2,  1,  0,  4,  3,  7, 30, 46, 43, 24, 12, 29, 53, 56, 35, 17,
             18, 19, 15, 16, 42, 23,  6, 34, 45, 57, 36, 13, 25, 54, 47, 44, 20,  8,
             31, 52, 55, 39, 11, 28],
            [23,  6, 34, 45, 42, 33, 49, 41, 22,  5, 38, 10, 27, 51, 59, 28, 52, 55,
             39, 11,  3,  2,  1,  0,  4,  8, 31, 47, 44, 20, 13, 25, 54, 57, 36, 18,
             19, 15, 16, 17, 43, 24,  7, 30, 46, 58, 37, 14, 26, 50, 48, 40, 21,  9,
             32, 53, 56, 35, 12, 29],
            [24,  7, 30, 46, 43, 34, 45, 42, 23,  6, 39, 11, 28, 52, 55, 29, 53, 56,
             35, 12,  4,  3,  2,  1,  0,  9, 32, 48, 40, 21, 14, 26, 50, 58, 37, 19,
             15, 16, 17, 18, 44, 20,  8, 31, 47, 59, 38, 10, 27, 51, 49, 41, 22,  5,
             33, 54, 57, 36, 13, 25],
            [25, 54, 57, 36, 13, 35, 12, 29, 53, 56, 30, 46, 43, 24,  7, 20,  8, 31,
             47, 44,  5, 33, 49, 41, 22,  0,  4,  3,  2,  1, 15, 16, 17, 18, 19, 10,
             27, 51, 59, 38, 45, 42, 23,  6, 34, 50, 58, 37, 14, 26, 40, 21,  9, 32,
             48, 55, 39, 11, 28, 52],
            [26, 50, 58, 37, 14, 36, 13, 25, 54, 57, 31, 47, 44, 20,  8, 21,  9, 32,
             48, 40,  6, 34, 45, 42, 23,  1,  0,  4,  3,  2, 16, 17, 18, 19, 15, 11,
             28, 52, 55, 39, 46, 43, 24,  7, 30, 51, 59, 38, 10, 27, 41, 22,  5, 33,
             49, 56, 35, 12, 29, 53],
            [27, 51, 59, 38, 10, 37, 14, 26, 50, 58, 32, 48, 40, 21,  9, 22,  5, 33,
             49, 41,  7, 30, 46, 43, 24,  2,  1,  0,  4,  3, 17, 18, 19, 15, 16, 12,
             29, 53, 56, 35, 47, 44, 20,  8, 31, 52, 55, 39, 11, 28, 42, 23,  6, 34,
             45, 57, 36, 13, 25, 54],
            [28, 52, 55, 39, 11, 38, 10, 27, 51, 59, 33, 49, 41, 22,  5, 23,  6, 34,
             45, 42,  8, 31, 47, 44, 20,  3,  2,  1,  0,  4, 18, 19, 15, 16, 17, 13,
             25, 54, 57, 36, 48, 40, 21,  9, 32, 53, 56, 35, 12, 29, 43, 24,  7, 30,
             46, 58, 37, 14, 26, 50],
            [29, 53, 56, 35, 12, 39, 11, 28, 52, 55, 34, 45, 42, 23,  6, 24,  7, 30,
             46, 43,  9, 32, 48, 40, 21,  4,  3,  2,  1,  0, 19, 15, 16, 17, 18, 14,
             26, 50, 58, 37, 49, 41, 22,  5, 33, 54, 57, 36, 13, 25, 44, 20,  8, 31,
             47, 59, 38, 10, 27, 51],
            [30, 46, 43, 24,  7, 20,  8, 31, 47, 44, 25, 54, 57, 36, 13, 35, 12, 29,
             53, 56, 10, 27, 51, 59, 38, 15, 16, 17, 18, 19,  0,  4,  3,  2,  1,  5,
             33, 49, 41, 22, 50, 58, 37, 14, 26, 45, 42, 23,  6, 34, 55, 39, 11, 28,
             52, 40, 21,  9, 32, 48],
            [31, 47, 44, 20,  8, 21,  9, 32, 48, 40, 26, 50, 58, 37, 14, 36, 13, 25,
             54, 57, 11, 28, 52, 55, 39, 16, 17, 18, 19, 15,  1,  0,  4,  3,  2,  6,
             34, 45, 42, 23, 51, 59, 38, 10, 27, 46, 43, 24,  7, 30, 56, 35, 12, 29,
             53, 41, 22,  5, 33, 49],
            [32, 48, 40, 21,  9, 22,  5, 33, 49, 41, 27, 51, 59, 38, 10, 37, 14, 26,
             50, 58, 12, 29, 53, 56, 35, 17, 18, 19, 15, 16,  2,  1,  0,  4,  3,  7,
             30, 46, 43, 24, 52, 55, 39, 11, 28, 47, 44, 20,  8, 31, 57, 36, 13, 25,
             54, 42, 23,  6, 34, 45],
            [33, 49, 41, 22,  5, 23,  6, 34, 45, 42, 28, 52, 55, 39, 11, 38, 10, 27,
             51, 59, 13, 25, 54, 57, 36, 18, 19, 15, 16, 17,  3,  2,  1,  0,  4,  8,
             31, 47, 44, 20, 53, 56, 35, 12, 29, 48, 40, 21,  9, 32, 58, 37, 14, 26,
             50, 43, 24,  7, 30, 46],
            [34, 45, 42, 23,  6, 24,  7, 30, 46, 43, 29, 53, 56, 35, 12, 39, 11, 28,
             52, 55, 14, 26, 50, 58, 37, 19, 15, 16, 17, 18,  4,  3,  2,  1,  0,  9,
             32, 48, 40, 21, 54, 57, 36, 13, 25, 49, 41, 22,  5, 33, 59, 38, 10, 27,
             51, 44, 20,  8, 31, 47],
            [35, 12, 29, 53, 56, 25, 54, 57, 36, 13, 20,  8, 31, 47, 44, 30, 46, 43,
             24,  7, 15, 16, 17, 18, 19, 10, 27, 51, 59, 38,  5, 33, 49, 41, 22,  0,
              4,  3,  2,  1, 55, 39, 11, 28, 52, 40, 21,  9, 32, 48, 50, 58, 37, 14,
             26, 45, 42, 23,  6, 34],
            [36, 13, 25, 54, 57, 26, 50, 58, 37, 14, 21,  9, 32, 48, 40, 31, 47, 44,
             20,  8, 16, 17, 18, 19, 15, 11, 28, 52, 55, 39,  6, 34, 45, 42, 23,  1,
              0,  4,  3,  2, 56, 35, 12, 29, 53, 41, 22,  5, 33, 49, 51, 59, 38, 10,
             27, 46, 43, 24,  7, 30],
            [37, 14, 26, 50, 58, 27, 51, 59, 38, 10, 22,  5, 33, 49, 41, 32, 48, 40,
             21,  9, 17, 18, 19, 15, 16, 12, 29, 53, 56, 35,  7, 30, 46, 43, 24,  2,
              1,  0,  4,  3, 57, 36, 13, 25, 54, 42, 23,  6, 34, 45, 52, 55, 39, 11,
             28, 47, 44, 20,  8, 31],
            [38, 10, 27, 51, 59, 28, 52, 55, 39, 11, 23,  6, 34, 45, 42, 33, 49, 41,
             22,  5, 18, 19, 15, 16, 17, 13, 25, 54, 57, 36,  8, 31, 47, 44, 20,  3,
              2,  1,  0,  4, 58, 37, 14, 26, 50, 43, 24,  7, 30, 46, 53, 56, 35, 12,
             29, 48, 40, 21,  9, 32],
            [39, 11, 28, 52, 55, 29, 53, 56, 35, 12, 24,  7, 30, 46, 43, 34, 45, 42,
             23,  6, 19, 15, 16, 17, 18, 14, 26, 50, 58, 37,  9, 32, 48, 40, 21,  4,
              3,  2,  1,  0, 59, 38, 10, 27, 51, 44, 20,  8, 31, 47, 54, 57, 36, 13,
             25, 49, 41, 22,  5, 33],
            [40, 21,  9, 32, 48, 55, 39, 11, 28, 52, 45, 42, 23,  6, 34, 50, 58, 37,
             14, 26, 20,  8, 31, 47, 44, 30, 46, 43, 24,  7, 35, 12, 29, 53, 56, 25,
             54, 57, 36, 13,  0,  4,  3,  2,  1,  5, 33, 49, 41, 22, 10, 27, 51, 59,
             38, 15, 16, 17, 18, 19],
            [41, 22,  5, 33, 49, 56, 35, 12, 29, 53, 46, 43, 24,  7, 30, 51, 59, 38,
             10, 27, 21,  9, 32, 48, 40, 31, 47, 44, 20,  8, 36, 13, 25, 54, 57, 26,
             50, 58, 37, 14,  1,  0,  4,  3,  2,  6, 34, 45, 42, 23, 11, 28, 52, 55,
             39, 16, 17, 18, 19, 15],
            [42, 23,  6, 34, 45, 57, 36, 13, 25, 54, 47, 44, 20,  8, 31, 52, 55, 39,
             11, 28, 22,  5, 33, 49, 41, 32, 48, 40, 21,  9, 37, 14, 26, 50, 58, 27,
             51, 59, 38, 10,  2,  1,  0,  4,  3,  7, 30, 46, 43, 24, 12, 29, 53, 56,
             35, 17, 18, 19, 15, 16],
            [43, 24,  7, 30, 46, 58, 37, 14, 26, 50, 48, 40, 21,  9, 32, 53, 56, 35,
             12, 29, 23,  6, 34, 45, 42, 33, 49, 41, 22,  5, 38, 10, 27, 51, 59, 28,
             52, 55, 39, 11,  3,  2,  1,  0,  4,  8, 31, 47, 44, 20, 13, 25, 54, 57,
             36, 18, 19, 15, 16, 17],
            [44, 20,  8, 31, 47, 59, 38, 10, 27, 51, 49, 41, 22,  5, 33, 54, 57, 36,
             13, 25, 24,  7, 30, 46, 43, 34, 45, 42, 23,  6, 39, 11, 28, 52, 55, 29,
             53, 56, 35, 12,  4,  3,  2,  1,  0,  9, 32, 48, 40, 21, 14, 26, 50, 58,
             37, 19, 15, 16, 17, 18],
            [45, 42, 23,  6, 34, 50, 58, 37, 14, 26, 40, 21,  9, 32, 48, 55, 39, 11,
             28, 52, 25, 54, 57, 36, 13, 35, 12, 29, 53, 56, 30, 46, 43, 24,  7, 20,
              8, 31, 47, 44,  5, 33, 49, 41, 22,  0,  4,  3,  2,  1, 15, 16, 17, 18,
             19, 10, 27, 51, 59, 38],
            [46, 43, 24,  7, 30, 51, 59, 38, 10, 27, 41, 22,  5, 33, 49, 56, 35, 12,
             29, 53, 26, 50, 58, 37, 14, 36, 13, 25, 54, 57, 31, 47, 44, 20,  8, 21,
              9, 32, 48, 40,  6, 34, 45, 42, 23,  1,  0,  4,  3,  2, 16, 17, 18, 19,
             15, 11, 28, 52, 55, 39],
            [47, 44, 20,  8, 31, 52, 55, 39, 11, 28, 42, 23,  6, 34, 45, 57, 36, 13,
             25, 54, 27, 51, 59, 38, 10, 37, 14, 26, 50, 58, 32, 48, 40, 21,  9, 22,
              5, 33, 49, 41,  7, 30, 46, 43, 24,  2,  1,  0,  4,  3, 17, 18, 19, 15,
             16, 12, 29, 53, 56, 35],
            [48, 40, 21,  9, 32, 53, 56, 35, 12, 29, 43, 24,  7, 30, 46, 58, 37, 14,
             26, 50, 28, 52, 55, 39, 11, 38, 10, 27, 51, 59, 33, 49, 41, 22,  5, 23,
              6, 34, 45, 42,  8, 31, 47, 44, 20,  3,  2,  1,  0,  4, 18, 19, 15, 16,
             17, 13, 25, 54, 57, 36],
            [49, 41, 22,  5, 33, 54, 57, 36, 13, 25, 44, 20,  8, 31, 47, 59, 38, 10,
             27, 51, 29, 53, 56, 35, 12, 39, 11, 28, 52, 55, 34, 45, 42, 23,  6, 24,
              7, 30, 46, 43,  9, 32, 48, 40, 21,  4,  3,  2,  1,  0, 19, 15, 16, 17,
             18, 14, 26, 50, 58, 37],
            [50, 58, 37, 14, 26, 45, 42, 23,  6, 34, 55, 39, 11, 28, 52, 40, 21,  9,
             32, 48, 30, 46, 43, 24,  7, 20,  8, 31, 47, 44, 25, 54, 57, 36, 13, 35,
             12, 29, 53, 56, 10, 27, 51, 59, 38, 15, 16, 17, 18, 19,  0,  4,  3,  2,
              1,  5, 33, 49, 41, 22],
            [51, 59, 38, 10, 27, 46, 43, 24,  7, 30, 56, 35, 12, 29, 53, 41, 22,  5,
             33, 49, 31, 47, 44, 20,  8, 21,  9, 32, 48, 40, 26, 50, 58, 37, 14, 36,
             13, 25, 54, 57, 11, 28, 52, 55, 39, 16, 17, 18, 19, 15,  1,  0,  4,  3,
              2,  6, 34, 45, 42, 23],
            [52, 55, 39, 11, 28, 47, 44, 20,  8, 31, 57, 36, 13, 25, 54, 42, 23,  6,
             34, 45, 32, 48, 40, 21,  9, 22,  5, 33, 49, 41, 27, 51, 59, 38, 10, 37,
             14, 26, 50, 58, 12, 29, 53, 56, 35, 17, 18, 19, 15, 16,  2,  1,  0,  4,
              3,  7, 30, 46, 43, 24],
            [53, 56, 35, 12, 29, 48, 40, 21,  9, 32, 58, 37, 14, 26, 50, 43, 24,  7,
             30, 46, 33, 49, 41, 22,  5, 23,  6, 34, 45, 42, 28, 52, 55, 39, 11, 38,
             10, 27, 51, 59, 13, 25, 54, 57, 36, 18, 19, 15, 16, 17,  3,  2,  1,  0,
              4,  8, 31, 47, 44, 20],
            [54, 57, 36, 13, 25, 49, 41, 22,  5, 33, 59, 38, 10, 27, 51, 44, 20,  8,
             31, 47, 34, 45, 42, 23,  6, 24,  7, 30, 46, 43, 29, 53, 56, 35, 12, 39,
             11, 28, 52, 55, 14, 26, 50, 58, 37, 19, 15, 16, 17, 18,  4,  3,  2,  1,
              0,  9, 32, 48, 40, 21],
            [55, 39, 11, 28, 52, 40, 21,  9, 32, 48, 50, 58, 37, 14, 26, 45, 42, 23,
              6, 34, 35, 12, 29, 53, 56, 25, 54, 57, 36, 13, 20,  8, 31, 47, 44, 30,
             46, 43, 24,  7, 15, 16, 17, 18, 19, 10, 27, 51, 59, 38,  5, 33, 49, 41,
             22,  0,  4,  3,  2,  1],
            [56, 35, 12, 29, 53, 41, 22,  5, 33, 49, 51, 59, 38, 10, 27, 46, 43, 24,
              7, 30, 36, 13, 25, 54, 57, 26, 50, 58, 37, 14, 21,  9, 32, 48, 40, 31,
             47, 44, 20,  8, 16, 17, 18, 19, 15, 11, 28, 52, 55, 39,  6, 34, 45, 42,
             23,  1,  0,  4,  3,  2],
            [57, 36, 13, 25, 54, 42, 23,  6, 34, 45, 52, 55, 39, 11, 28, 47, 44, 20,
              8, 31, 37, 14, 26, 50, 58, 27, 51, 59, 38, 10, 22,  5, 33, 49, 41, 32,
             48, 40, 21,  9, 17, 18, 19, 15, 16, 12, 29, 53, 56, 35,  7, 30, 46, 43,
             24,  2,  1,  0,  4,  3],
            [58, 37, 14, 26, 50, 43, 24,  7, 30, 46, 53, 56, 35, 12, 29, 48, 40, 21,
              9, 32, 38, 10, 27, 51, 59, 28, 52, 55, 39, 11, 23,  6, 34, 45, 42, 33,
             49, 41, 22,  5, 18, 19, 15, 16, 17, 13, 25, 54, 57, 36,  8, 31, 47, 44,
             20,  3,  2,  1,  0,  4],
            [59, 38, 10, 27, 51, 44, 20,  8, 31, 47, 54, 57, 36, 13, 25, 49, 41, 22,
              5, 33, 39, 11, 28, 52, 55, 29, 53, 56, 35, 12, 24,  7, 30, 46, 43, 34,
             45, 42, 23,  6, 19, 15, 16, 17, 18, 14, 26, 50, 58, 37,  9, 32, 48, 40,
             21,  4,  3,  2,  1,  0]])
        Rs = torch.zeros(60,3,3)
        Rs[0]=torch.tensor([[ 1.000000, 0.000000,0.000000],[ 0.000000, 1.000000,0.000000],[ 0.000000, 0.000000,1.000000]])
        Rs[1]=torch.tensor([[ 0.500000,-0.809017,0.309017],[ 0.809017 ,0.309017,-0.500000],[ 0.309017, 0.500000,0.809017]])
        Rs[2]=torch.tensor([[-0.309017,-0.500000,0.809017],[ 0.500000,-0.809017,-0.309017],[ 0.809017, 0.309017,0.500000]])
        Rs[3]=torch.tensor([[-0.309017, 0.500000,0.809017],[-0.500000,-0.809017,0.309017],[ 0.809017,-0.309017,0.500000]])
        Rs[4]=torch.tensor([[ 0.500000, 0.809017,0.309017],[-0.809017, 0.309017,0.500000],[ 0.309017,-0.500000,0.809017]])
        Rs[5]=torch.tensor([[-0.809017, 0.309017,0.500000],[ 0.309017,-0.500000,0.809017],[ 0.500000, 0.809017,0.309017]])
        Rs[6]=torch.tensor([[ 0.000000, 1.000000,0.000000],[ 0.000000, 0.000000,1.000000],[ 1.000000, 0.000000,0.000000]])
        Rs[7]=torch.tensor([[ 0.809017 ,0.309017,-0.500000],[ 0.309017, 0.500000,0.809017],[ 0.500000,-0.809017,0.309017]])
        Rs[8]=torch.tensor([[ 0.500000,-0.809017,-0.309017],[ 0.809017, 0.309017,0.500000],[-0.309017,-0.500000,0.809017]])
        Rs[9]=torch.tensor([[-0.500000,-0.809017,0.309017],[ 0.809017,-0.309017,0.500000],[-0.309017, 0.500000,0.809017]])
        Rs[10]=torch.tensor([[-0.500000,-0.809017,0.309017],[-0.809017 ,0.309017,-0.500000],[ 0.309017,-0.500000,-0.809017]])
        Rs[11]=torch.tensor([[-0.809017, 0.309017,0.500000],[-0.309017 ,0.500000,-0.809017],[-0.500000,-0.809017,-0.309017]])
        Rs[12]=torch.tensor([[ 0.000000, 1.000000,0.000000],[ 0.000000 ,0.000000,-1.000000],[-1.000000, 0.000000,0.000000]])
        Rs[13]=torch.tensor([[ 0.809017 ,0.309017,-0.500000],[-0.309017,-0.500000,-0.809017],[-0.500000 ,0.809017,-0.309017]])
        Rs[14]=torch.tensor([[ 0.500000,-0.809017,-0.309017],[-0.809017,-0.309017,-0.500000],[ 0.309017 ,0.500000,-0.809017]])
        Rs[15]=torch.tensor([[ 0.309017 ,0.500000,-0.809017],[ 0.500000,-0.809017,-0.309017],[-0.809017,-0.309017,-0.500000]])
        Rs[16]=torch.tensor([[ 0.309017,-0.500000,-0.809017],[-0.500000,-0.809017,0.309017],[-0.809017 ,0.309017,-0.500000]])
        Rs[17]=torch.tensor([[-0.500000,-0.809017,-0.309017],[-0.809017, 0.309017,0.500000],[-0.309017 ,0.500000,-0.809017]])
        Rs[18]=torch.tensor([[-1.000000, 0.000000,0.000000],[ 0.000000, 1.000000,0.000000],[ 0.000000 ,0.000000,-1.000000]])
        Rs[19]=torch.tensor([[-0.500000 ,0.809017,-0.309017],[ 0.809017 ,0.309017,-0.500000],[-0.309017,-0.500000,-0.809017]])
        Rs[20]=torch.tensor([[-0.500000,-0.809017,-0.309017],[ 0.809017,-0.309017,-0.500000],[ 0.309017,-0.500000,0.809017]])
        Rs[21]=torch.tensor([[-1.000000, 0.000000,0.000000],[ 0.000000,-1.000000,0.000000],[ 0.000000, 0.000000,1.000000]])
        Rs[22]=torch.tensor([[-0.500000 ,0.809017,-0.309017],[-0.809017,-0.309017,0.500000],[ 0.309017, 0.500000,0.809017]])
        Rs[23]=torch.tensor([[ 0.309017 ,0.500000,-0.809017],[-0.500000, 0.809017,0.309017],[ 0.809017, 0.309017,0.500000]])
        Rs[24]=torch.tensor([[ 0.309017,-0.500000,-0.809017],[ 0.500000 ,0.809017,-0.309017],[ 0.809017,-0.309017,0.500000]])
        Rs[25]=torch.tensor([[ 0.000000 ,0.000000,-1.000000],[-1.000000, 0.000000,0.000000],[ 0.000000, 1.000000,0.000000]])
        Rs[26]=torch.tensor([[-0.309017,-0.500000,-0.809017],[-0.500000 ,0.809017,-0.309017],[ 0.809017 ,0.309017,-0.500000]])
        Rs[27]=torch.tensor([[-0.809017,-0.309017,-0.500000],[ 0.309017 ,0.500000,-0.809017],[ 0.500000,-0.809017,-0.309017]])
        Rs[28]=torch.tensor([[-0.809017 ,0.309017,-0.500000],[ 0.309017,-0.500000,-0.809017],[-0.500000,-0.809017,0.309017]])
        Rs[29]=torch.tensor([[-0.309017 ,0.500000,-0.809017],[-0.500000,-0.809017,-0.309017],[-0.809017, 0.309017,0.500000]])
        Rs[30]=torch.tensor([[ 0.809017, 0.309017,0.500000],[-0.309017,-0.500000,0.809017],[ 0.500000,-0.809017,-0.309017]])
        Rs[31]=torch.tensor([[ 0.809017,-0.309017,0.500000],[-0.309017, 0.500000,0.809017],[-0.500000,-0.809017,0.309017]])
        Rs[32]=torch.tensor([[ 0.309017,-0.500000,0.809017],[ 0.500000, 0.809017,0.309017],[-0.809017, 0.309017,0.500000]])
        Rs[33]=torch.tensor([[ 0.000000, 0.000000,1.000000],[ 1.000000, 0.000000,0.000000],[ 0.000000, 1.000000,0.000000]])
        Rs[34]=torch.tensor([[ 0.309017, 0.500000,0.809017],[ 0.500000,-0.809017,0.309017],[ 0.809017 ,0.309017,-0.500000]])
        Rs[35]=torch.tensor([[-0.309017, 0.500000,0.809017],[ 0.500000 ,0.809017,-0.309017],[-0.809017 ,0.309017,-0.500000]])
        Rs[36]=torch.tensor([[ 0.500000, 0.809017,0.309017],[ 0.809017,-0.309017,-0.500000],[-0.309017 ,0.500000,-0.809017]])
        Rs[37]=torch.tensor([[ 1.000000, 0.000000,0.000000],[ 0.000000,-1.000000,0.000000],[ 0.000000 ,0.000000,-1.000000]])
        Rs[38]=torch.tensor([[ 0.500000,-0.809017,0.309017],[-0.809017,-0.309017,0.500000],[-0.309017,-0.500000,-0.809017]])
        Rs[39]=torch.tensor([[-0.309017,-0.500000,0.809017],[-0.500000, 0.809017,0.309017],[-0.809017,-0.309017,-0.500000]])
        Rs[40]=torch.tensor([[-0.500000, 0.809017,0.309017],[-0.809017,-0.309017,-0.500000],[-0.309017,-0.500000,0.809017]])
        Rs[41]=torch.tensor([[ 0.500000 ,0.809017,-0.309017],[-0.809017 ,0.309017,-0.500000],[-0.309017, 0.500000,0.809017]])
        Rs[42]=torch.tensor([[ 0.809017,-0.309017,-0.500000],[-0.309017 ,0.500000,-0.809017],[ 0.500000, 0.809017,0.309017]])
        Rs[43]=torch.tensor([[ 0.000000,-1.000000,0.000000],[ 0.000000 ,0.000000,-1.000000],[ 1.000000, 0.000000,0.000000]])
        Rs[44]=torch.tensor([[-0.809017,-0.309017,0.500000],[-0.309017,-0.500000,-0.809017],[ 0.500000,-0.809017,0.309017]])
        Rs[45]=torch.tensor([[ 0.809017,-0.309017,0.500000],[ 0.309017,-0.500000,-0.809017],[ 0.500000 ,0.809017,-0.309017]])
        Rs[46]=torch.tensor([[ 0.309017,-0.500000,0.809017],[-0.500000,-0.809017,-0.309017],[ 0.809017,-0.309017,-0.500000]])
        Rs[47]=torch.tensor([[ 0.000000, 0.000000,1.000000],[-1.000000, 0.000000,0.000000],[ 0.000000,-1.000000,0.000000]])
        Rs[48]=torch.tensor([[ 0.309017, 0.500000,0.809017],[-0.500000 ,0.809017,-0.309017],[-0.809017,-0.309017,0.500000]])
        Rs[49]=torch.tensor([[ 0.809017, 0.309017,0.500000],[ 0.309017 ,0.500000,-0.809017],[-0.500000, 0.809017,0.309017]])
        Rs[50]=torch.tensor([[-0.309017 ,0.500000,-0.809017],[ 0.500000, 0.809017,0.309017],[ 0.809017,-0.309017,-0.500000]])
        Rs[51]=torch.tensor([[ 0.000000 ,0.000000,-1.000000],[ 1.000000, 0.000000,0.000000],[ 0.000000,-1.000000,0.000000]])
        Rs[52]=torch.tensor([[-0.309017,-0.500000,-0.809017],[ 0.500000,-0.809017,0.309017],[-0.809017,-0.309017,0.500000]])
        Rs[53]=torch.tensor([[-0.809017,-0.309017,-0.500000],[-0.309017,-0.500000,0.809017],[-0.500000, 0.809017,0.309017]])
        Rs[54]=torch.tensor([[-0.809017 ,0.309017,-0.500000],[-0.309017, 0.500000,0.809017],[ 0.500000 ,0.809017,-0.309017]])
        Rs[55]=torch.tensor([[ 0.000000,-1.000000,0.000000],[ 0.000000, 0.000000,1.000000],[-1.000000, 0.000000,0.000000]])
        Rs[56]=torch.tensor([[-0.809017,-0.309017,0.500000],[ 0.309017, 0.500000,0.809017],[-0.500000 ,0.809017,-0.309017]])
        Rs[57]=torch.tensor([[-0.500000, 0.809017,0.309017],[ 0.809017, 0.309017,0.500000],[ 0.309017 ,0.500000,-0.809017]])
        Rs[58]=torch.tensor([[ 0.500000 ,0.809017,-0.309017],[ 0.809017,-0.309017,0.500000],[ 0.309017,-0.500000,-0.809017]])
        Rs[59]=torch.tensor([[ 0.809017,-0.309017,-0.500000],[ 0.309017,-0.500000,0.809017],[-0.500000,-0.809017,-0.309017]])

        nneigh = 5
        est_radius = 2.0*SYMA
        theta1 = 37/(2.0*np.pi)
        theta2 = 13.5/(2.0*np.pi)
        D1 = est_radius/np.sin(theta1/2)
        D2 = est_radius/np.sin(theta2/2)
        offset = torch.tensor([ float(D2),0.0,float(D1) ]) #+ torch.rand(3)*est_radius/2.0-est_radius/4.0
        #metasymm = (
        #    [torch.tensor([0,1,2,3,4]),torch.tensor([5,9,11,12,17,18,20,21,24,25])],
        #    [3,3]
        #)
        metasymm = (
            [torch.arange(60)],
            [6]
        )
    else:
        print ("Unknown symmetry",symmid)
        assert False

    return symmatrix, Rs,metasymm, offset

# find a particular subsymmetry within a larger symmetry
# def find_subsymmetry( xyz_t, symmgp, symmaxes, symmRs, eps=1e-4 ):
#     N,L = xyz_t.shape[:2]
#     S = symmRs.shape[0]

#     # only C subsymms for now...
#     if (len(symmaxes)!=1):
#         print ("Unsupported symmetry in find_subsymmetry:", symmgp)

#     nfold = symmaxes[0][0]

#     print('About to get rotation from matrix')
#     print(symmRs[1:].shape)
#     ang_i, ax_i = rotation_from_matrix(symmRs[1:]) # 1st xform is identity, skip

#     # a) find 'nfold' subunits with the same axis and rotations of [0...nfold-1]*2*pi/nfold
#     nfold_i  = (ang_i*nfold/(2*np.pi))
#     cand1    = (torch.abs(nfold_i-torch.round(nfold_i)) < eps).nonzero()[0,0]
#     similars = (torch.linalg.norm(ax_i - ax_i[cand1], dim=-1) < eps).nonzero()[:,0] + 1 # +1 since we skipped 1st

#     assert (similars.shape[0] == nfold-1)
#     ax_i = ax_i[cand1]

#     # b) rotate / translate symmaxes to ax_i
#     tgt_axis, tgt_origin = symmaxes[0][1], symmaxes[0][2]
#     ic(tgt_axis, tgt_origin)
#     # R_1: first rotate axes to be coincident
#     # R_2: random rotation about symm axis
#     #   inside/outside "flips" are handled in offset code
#     dotprod = torch.sum(tgt_axis*ax_i)
#     if ( torch.abs( dotprod ) > 1-eps):
#         R_1 = torch.eye(3, device=tgt_axis.device)
#     else:
#         axis_rot = torch.cross( tgt_axis, ax_i )
#         axis_rot = axis_rot / torch.linalg.norm(axis_rot)
#         angle_rot = torch.acos( dotprod )
#         R_1 = matrix_from_rotation( angle_rot, axis_rot )

#     angle_rot = np.random.rand()*2*np.pi
#     R_2 = matrix_from_rotation( angle_rot, tgt_axis )

#     R_12 = R_1 @ R_2
#     xyz = torch.einsum('ij,braj->brai', R_12, xyz_t-tgt_origin)

#     # c) make res-pair mask
#     mask_t = torch.zeros((1,S,S), dtype=torch.bool, device=tgt_axis.device)
#     mask_t[0,0,0] = True
#     mask_t[0,0,similars] = True
#     mask_t[0,similars,0] = True
#     mask_t[0,similars[:,None],similars[None,:]] = True
#     return xyz, mask_t, ax_i

def find_subsymmetry( xyz_t, symmgp, symmaxes, symmRs, eps=1e-6, all_subsymms=True ):
    N,L = xyz_t.shape[:2]
    S = symmRs.shape[0]

    # only C subsymms for now...
    if (len(symmaxes)!=1):
        print ("Unsupported symmetry in find_subsymmetry:", symmgp)

    nfold = symmaxes[0][0]
    ang_i, ax_i = rotation_from_matrix(symmRs[1:]) # 1st xform is identity, skip

    # a) find 'nfold' subunits with the same axis and rotations of [0...nfold-1]*2*pi/nfold
    nfold_i = (ang_i*nfold/(2*np.pi))
    cand1 = (torch.abs(nfold_i-torch.round(nfold_i)) < eps).nonzero()[0,0]
    similars = (torch.linalg.norm(ax_i - ax_i[cand1], dim=-1) < eps).nonzero()[:,0] + 1 # +1 since we skipped 1st
    assert (similars.shape[0] == nfold-1)
    ax_i = ax_i[cand1]

    # b) rotate / translate symmaxes to ax_i
    tgt_axis, tgt_origin = symmaxes[0][1], symmaxes[0][2]
    # R_1: first rotate axes to be coincident
    # R_2: random rotation about symm axis
    #   inside/outside "flips" are handled in offset code
    dotprod = torch.sum(tgt_axis*ax_i)
    if ( torch.abs( dotprod ) > 1-eps):
        R_1 = torch.eye(3, device=tgt_axis.device)
    else:
        axis_rot = torch.cross( tgt_axis, ax_i )
        axis_rot = axis_rot / torch.linalg.norm(axis_rot)
        angle_rot = torch.acos( dotprod )
        R_1 = matrix_from_rotation( angle_rot, axis_rot )

    angle_rot = np.random.rand()*2*np.pi
    R_2 = matrix_from_rotation( angle_rot, tgt_axis )

    R_12 = R_1 @ R_2
    xyz = torch.einsum('ij,braj->brai', R_12, xyz_t-tgt_origin)

    # c) make res-pair mask
    mask_t = torch.zeros((1,S,S), dtype=torch.bool, device=tgt_axis.device)
    mask_t[0,0,0] = True
    mask_t[0,0,similars] = True
    mask_t[0,similars,0] = True
    mask_t[0,similars[:,None],similars[None,:]] = True

    if (all_subsymms):
        symmRs_ij = torch.einsum('sji,tjk->stik', symmRs,symmRs)
        symmRs_ijk = torch.sum( torch.abs(symmRs_ij[:,:,None] - symmRs[similars][None,None]), dim=(-1,-2) )
        mask_t, _ = symmRs_ijk.min(dim=-1)
        mask_t = (mask_t<eps)[None]
        
    return xyz, mask_t, ax_i


def fill_square_diagonal(x, val, k=0):
    """
    Fills the k'th diagonal of x with value val. x must be square.
    """
    assert x.shape[0] == x.shape[1] and len(x.shape) == 2

    # compute how many entries we will fill 
    n = x.shape[0] - abs(k)

    # get the indices of the diagonal
    if k >= 0:
        row_idx = torch.arange(n)
        col_idx = row_idx + k
    else:
        col_idx = torch.arange(n)
        row_idx = col_idx - k

    print(row_idx, col_idx)

    # fill the diagonal
    x[row_idx, col_idx] = val

    return x


def propogate_repeat_features2(indep, Lasu, inf_conf):
    """
    Propogates tensor information for repeat proteins (V2)
    """
    Ncopy = inf_conf.n_repeats 
    L_orig = indep.xyz.shape[0]
    o = copy.deepcopy(indep)

    # 1D information 
    ################

    # copy xyz and add magic offset between initialized units 
    MAGIC_OFFSET = 0.5 # Angstroms
    orig_xyz = o.xyz.squeeze()
    newcrds = []
    for i in range(Ncopy):
        new = orig_xyz + MAGIC_OFFSET * (i) * torch.tensor([1,0,0]) # offset in x direction, arbitrarily
        newcrds.append(new)
    newcrds = torch.cat(newcrds, dim=0)
    o.xyz = newcrds

    # o.xyz2 = newcrds.clone() # for 3rd template
    orig_xyz2 = o.xyz2.squeeze().clone() 
    o.xyz2 = orig_xyz2.repeat(Ncopy,1,1)
    
    o.seq           = torch.cat([o.seq[None]]*Ncopy, dim=1).squeeze(0)
    o.atom_frames   = torch.cat([o.atom_frames[None]]*Ncopy,dim=1).squeeze(0)
    o.chirals       = torch.cat([o.chirals[None]]*Ncopy,dim=1).squeeze(0)
    o.is_sm         = torch.cat([o.is_sm[None]]*Ncopy,dim=1).squeeze(0)

    o.is_protein    = torch.cat([o.is_protein[None]]*Ncopy,dim=1).squeeze(0)
    o.is_dna        = torch.cat([o.is_dna[None]]*Ncopy,dim=1).squeeze(0)
    o.is_rna        = torch.cat([o.is_rna[None]]*Ncopy,dim=1).squeeze(0)

    ntotal = len(o.idx)*Ncopy
    o.idx = torch.arange(ntotal)
    

    o.terminus_type = torch.zeros(len(o.idx), dtype=o.terminus_type.dtype)
    o.terminus_type[0] = 1
    o.terminus_type[-1] = 2 
    
    # 2D information
    ################
    
    # bond features 
    # set k=1 and k=-1 diagonals to peptide bond value 
    # doesn't work for ligands yet 
    print('WARNING: Assuming NO LIGANDS/LIGAND BONDS')
    new_bond_feats = torch.zeros((Ncopy*L_orig, Ncopy*L_orig),dtype=torch.long)
    peptide_bond = 5

    new_bond_feats = fill_square_diagonal(new_bond_feats, peptide_bond, k=1)
    new_bond_feats = fill_square_diagonal(new_bond_feats, peptide_bond, k=-1)
    # set_trace()
    o.bond_feats = new_bond_feats

    # same chain all true because all in same chain
    new_same_chain = torch.ones((Ncopy*L_orig, Ncopy*L_orig),dtype=torch.bool)
    o.same_chain = new_same_chain


    return o


def symmetrize_repeat_features(indep, Lasu, main_block):
    """
    Takes in an already correct length indep and makes sure all relevant features are repeated
    """
    Ncopy = len(indep.xyz.squeeze()) // Lasu 

    o = copy.deepcopy(indep)

    # Helper functions
    ################ 
    ################
    def copy_1d_reverse(x,L,ncopy, main_block):

        if main_block == None:
            master = x[-L:]
        else: 
            # get master from main block 
            master = x[main_block*Lasu:(main_block+1)*Lasu]

        for i in range(ncopy):
            start = i*L
            end = (i+1)*L
            x[start:end] = master
        
        return x
    
    def copy_2d_diag_reverse(x, lasu, main_block):
        assert x.shape[0] % lasu == 0

        if main_block == None:
            master = x[-lasu:, -lasu:]
        else:
            master = x[main_block*lasu:(main_block+1)*lasu, main_block*lasu:(main_block+1)*lasu]

        # copy along the diagonal 
        new = torch.zeros_like(x)
        n = x.shape[0] // lasu 

        for i in range(n):
            start = i * lasu
            end = (i+1) * lasu
            new[start:end, start:end] = master
                

        return new
    
    def get_repeat_same_chain(is_metal):
        A = is_metal[:,None] == is_metal[None,:] # correct but has off diag metals as True (same chain)
        B = is_metal[:,None] &  is_metal[None,:] # places where i,j are metals 

        i,j = torch.where(B) # indices where metals 

        # set off diag metals to False
        use = (i != j)
        i = i[use]
        j = j[use]

        A[i,j] = False # set off diag metals to False (not same chain)

        return A 
    ################
    ################ 


    # 1D information 
    ################
    # copy first ASU of sequence over (reverse because SM is put last)
    o.seq   = copy_1d_reverse(o.seq,         Lasu, Ncopy, main_block)
    o.is_sm = copy_1d_reverse(o.is_sm,       Lasu, Ncopy, main_block)


    # duplicate atom frames for the metal atoms
    print('WARNING: Duplicating atom frames in a non-general way (metals only, single atom)')
    o.atom_frames = o.atom_frames.repeat(Ncopy,1,1)

    # increment each sm index by (Lasu + 200)
    (i_sm,) = torch.where(o.is_sm) # only single dim 

    for i in i_sm:
        o.idx[i]    += (Lasu*i + 200) # make this its own chain
        
        if i < len(o.idx) - 2:
            o.idx[1+i:] -= 1              # de-increment the next atoms so that the chain remains continuous

    # 2D information
    # Bond features
    o.bond_feats = copy_2d_diag_reverse(o.bond_feats, Lasu, main_block=main_block)
    
    o.same_chain = get_repeat_same_chain(o.is_sm)

    return o

