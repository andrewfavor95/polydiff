import sys

import numpy as np
import torch

import scipy.sparse

from rf2aa.chemical import *
from rf2aa.scoring import *

import rf2aa.kinematics
import rf2aa.util
from pdb import set_trace
import matplotlib.pyplot as plt

def generate_Cbeta(N,Ca,C):
    # recreate Cb given N,Ca,C
    b = Ca - N 
    c = C - Ca
    a = torch.cross(b, c, dim=-1)
    #Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca
    # fd: below matches sidechain generator (=Rosetta params)
    Cb = -0.57910144*a + 0.5689693*b - 0.5441217*c + Ca

    return Cb


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

def find_contiguous_true_indices(arr):
    """
    Find the (start, stop) indices of each contiguous set of True values in a boolean array.

    Args:
    arr (numpy.ndarray): Input boolean array.

    Returns:
    list of tuples: List of (start, stop) index pairs for each contiguous True segment.
    """
    if torch.is_tensor(arr):
        arr = arr.numpy()
    assert arr.dtype == bool
    
    # Find the indices where True values start and stop
    stops = np.where(arr[:-1] & ~arr[1:])[0] + 1
    starts = np.where(~arr[:-1] & arr[1:])[0] + 1

    # Handle the case where the array starts with True
    if arr[0]:
        starts = np.insert(starts, 0, 0)

    # Handle the case where the array ends with True
    if arr[-1]:
        stops = np.append(stops, len(arr))

    # Pair the start and stop indices
    indices = list(zip(starts, stops))

    return indices
    

def mask_sequence_chunks(is_masked, p=0.5):
    """
    Take some regions that are unmasked and mask them with probability p, in a contiguous/chunked fashion.
    """
    is_masked = is_masked.copy()
    is_revealed = ~is_masked

    revealed_indices = find_contiguous_true_indices(is_revealed)

    # print('revealed indices are: ', revealed_indices)
    for start,stop in revealed_indices:

        if np.random.rand() < p:
            n_mask      = np.random.randint(1, stop - start+1)             # how many within here to mask? 
            mask_start  = np.random.randint(start, stop - n_mask+1)
            mask_stop   = mask_start + n_mask
            # print('masking from ', mask_start, ' to ', mask_stop, ' out of ', start, ' to ', stop)
            is_masked[mask_start:mask_stop+1] = True # mask it 
    
    return is_masked

    
# # More complicated version splits error in CA-N and CA-C (giving more accurate CB position)
# # It returns the rigid transformation from local frame to global frame
# def rigid_from_3_points(N, Ca, C, non_ideal=False, eps=1e-8):
#     #N, Ca, C - [B,L, 3]
#     #R - [B,L, 3, 3], det(R)=1, inv(R) = R.T, R is a rotation matrix
#     B,L = N.shape[:2]
    
#     v1 = C-Ca
#     v2 = N-Ca
#     e1 = v1/(torch.norm(v1, dim=-1, keepdim=True)+eps)
#     u2 = v2-(torch.einsum('bli, bli -> bl', e1, v2)[...,None]*e1)
#     e2 = u2/(torch.norm(u2, dim=-1, keepdim=True)+eps)
#     e3 = torch.cross(e1, e2, dim=-1)
#     R = torch.cat([e1[...,None], e2[...,None], e3[...,None]], axis=-1) #[B,L,3,3] - rotation matrix
    
#     if non_ideal:
#         v2 = v2/(torch.norm(v2, dim=-1, keepdim=True)+eps)
#         cosref = torch.sum(e1*v2, dim=-1) # cosine of current N-CA-C bond angle
#         costgt = cos_ideal_NCAC.item()
#         cos2del = torch.clamp( cosref*costgt + torch.sqrt((1-cosref*cosref)*(1-costgt*costgt)+eps), min=-1.0, max=1.0 )
#         cosdel = torch.sqrt(0.5*(1+cos2del)+eps)
#         sindel = torch.sign(costgt-cosref) * torch.sqrt(1-0.5*(1+cos2del)+eps)
#         Rp = torch.eye(3, device=N.device).repeat(B,L,1,1)
#         Rp[:,:,0,0] = cosdel
#         Rp[:,:,0,1] = -sindel
#         Rp[:,:,1,0] = sindel
#         Rp[:,:,1,1] = cosdel
    
#         R = torch.einsum('blij,bljk->blik', R,Rp)

#     return R, Ca

# build a frame from 3 points
#fd  -  more complicated version splits angle deviations between CA-N and CA-C (giving more accurate CB position)
#fd  -  makes no assumptions about input dims (other than last 1 is xyz)

def rigid_from_3_points(N, Ca, C, is_na=None, eps=1e-4):

    assert 0, "nothing wrong here, just trying to redirect all calls to the function versions in rf2aa.util.py"
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
       costgt[is_na] = -0.2744

    cos2del = torch.clamp( cosref*costgt + torch.sqrt((1-cosref*cosref)*(1-costgt*costgt)+eps), min=-1.0, max=1.0 )

    cosdel = torch.sqrt(0.5*(1+cos2del)+eps)

    sindel = torch.sign(costgt-cosref) * torch.sqrt(1-0.5*(1+cos2del)+eps)

    Rp = torch.eye(3, device=N.device).repeat(*dims,1,1)
    Rp[...,0,0] = cosdel
    Rp[...,0,1] = -sindel
    Rp[...,1,0] = sindel
    Rp[...,1,1] = cosdel
    R = torch.einsum('...ij,...jk->...ik', R,Rp)

    return R, Ca

def get_tor_mask(seq, torsion_indices, mask_in=None):
    assert 0, "nothing wrong here, just trying to redirect all calls to the function versions in rf2aa.util.py"

    B,L = seq.shape[:2]
    tors_mask = torch.ones((B,L,10), dtype=torch.bool, device=seq.device)
    tors_mask[...,3:7] = torsion_indices[seq,:,-1] > 0
    tors_mask[:,0,1] = False
    tors_mask[:,-1,0] = False

    # mask for additional angles
    tors_mask[:,:,7] = seq!=aa2num['GLY']
    tors_mask[:,:,8] = seq!=aa2num['GLY']
    tors_mask[:,:,9] = torch.logical_and( seq!=aa2num['GLY'], seq!=aa2num['ALA'] )
    tors_mask[:,:,9] = torch.logical_and( tors_mask[:,:,9], seq!=aa2num['UNK'] )
    tors_mask[:,:,9] = torch.logical_and( tors_mask[:,:,9], seq!=aa2num['MAS'] )

    if mask_in != None:
        # mask for missing atoms
        # chis
        ti0 = torch.gather(mask_in,2,torsion_indices[seq,:,0])
        ti1 = torch.gather(mask_in,2,torsion_indices[seq,:,1])
        ti2 = torch.gather(mask_in,2,torsion_indices[seq,:,2])
        ti3 = torch.gather(mask_in,2,torsion_indices[seq,:,3])
        is_valid = torch.stack((ti0, ti1, ti2, ti3), dim=-2).all(dim=-1)
        tors_mask[...,3:7] = torch.logical_and(tors_mask[...,3:7], is_valid)
        tors_mask[:,:,7] = torch.logical_and(tors_mask[:,:,7], mask_in[:,:,4]) # CB exist?
        tors_mask[:,:,8] = torch.logical_and(tors_mask[:,:,8], mask_in[:,:,4]) # CB exist?
        tors_mask[:,:,9] = torch.logical_and(tors_mask[:,:,9], mask_in[:,:,5]) # XG exist?

    return tors_mask

def get_torsions(xyz_in, seq, torsion_indices, torsion_can_flip, ref_angles, mask_in=None):
    # set_trace()
    assert 0, "nothing wrong here, just trying to redirect all calls to the function versions in rf2aa.util.py"
    B,L = xyz_in.shape[:2]
    
    tors_mask = get_tor_mask(seq, torsion_indices, mask_in)
    
    # torsions to restrain to 0 or 180degree
    tors_planar = torch.zeros((B, L, 10), dtype=torch.bool, device=xyz_in.device)
    tors_planar[:,:,5] = seq == aa2num['TYR'] # TYR chi 3 should be planar

    # idealize given xyz coordinates before computing torsion angles
    xyz = xyz_in.clone()
    Rs, Ts = rigid_from_3_points(xyz[...,0,:],xyz[...,1,:],xyz[...,2,:])
    Nideal = torch.tensor([-0.5272, 1.3593, 0.000], device=xyz_in.device)
    Cideal = torch.tensor([1.5233, 0.000, 0.000], device=xyz_in.device)
    xyz[...,0,:] = torch.einsum('brij,j->bri', Rs, Nideal) + Ts
    xyz[...,2,:] = torch.einsum('brij,j->bri', Rs, Cideal) + Ts

    torsions = torch.zeros( (B,L,10,2), device=xyz.device )
    # avoid undefined angles for H generation
    torsions[:,0,1,0] = 1.0
    torsions[:,-1,0,0] = 1.0

    # omega
    torsions[:,:-1,0,:] = th_dih(xyz[:,:-1,1,:],xyz[:,:-1,2,:],xyz[:,1:,0,:],xyz[:,1:,1,:])
    # phi
    torsions[:,1:,1,:] = th_dih(xyz[:,:-1,2,:],xyz[:,1:,0,:],xyz[:,1:,1,:],xyz[:,1:,2,:])
    # psi
    torsions[:,:,2,:] = -1 * th_dih(xyz[:,:,0,:],xyz[:,:,1,:],xyz[:,:,2,:],xyz[:,:,3,:])

    # chis
    ti0 = torch.gather(xyz,2,torsion_indices[seq,:,0,None].repeat(1,1,1,3))
    ti1 = torch.gather(xyz,2,torsion_indices[seq,:,1,None].repeat(1,1,1,3))
    ti2 = torch.gather(xyz,2,torsion_indices[seq,:,2,None].repeat(1,1,1,3))
    ti3 = torch.gather(xyz,2,torsion_indices[seq,:,3,None].repeat(1,1,1,3))
    torsions[:,:,3:7,:] = th_dih(ti0,ti1,ti2,ti3)
    
    # CB bend
    NC = 0.5*( xyz[:,:,0,:3] + xyz[:,:,2,:3] )
    CA = xyz[:,:,1,:3]
    CB = xyz[:,:,4,:3]
    t = th_ang_v(CB-CA,NC-CA)
    t0 = ref_angles[seq][...,0,:]
    torsions[:,:,7,:] = torch.stack( 
        (torch.sum(t*t0,dim=-1),t[...,0]*t0[...,1]-t[...,1]*t0[...,0]),
        dim=-1 )
    
    # CB twist
    NCCA = NC-CA
    NCp = xyz[:,:,2,:3] - xyz[:,:,0,:3]
    NCpp = NCp - torch.sum(NCp*NCCA, dim=-1, keepdim=True)/ torch.sum(NCCA*NCCA, dim=-1, keepdim=True) * NCCA
    t = th_ang_v(CB-CA,NCpp)
    t0 = ref_angles[seq][...,1,:]
    torsions[:,:,8,:] = torch.stack( 
        (torch.sum(t*t0,dim=-1),t[...,0]*t0[...,1]-t[...,1]*t0[...,0]),
        dim=-1 )

    # CG bend
    CG = xyz[:,:,5,:3]
    t = th_ang_v(CG-CB,CA-CB)
    t0 = ref_angles[seq][...,2,:]
    torsions[:,:,9,:] = torch.stack( 
        (torch.sum(t*t0,dim=-1),t[...,0]*t0[...,1]-t[...,1]*t0[...,0]),
        dim=-1 )
    
    mask0 = torch.isnan(torsions[...,0]).nonzero()
    mask1 = torch.isnan(torsions[...,1]).nonzero()
    torsions[mask0[:,0],mask0[:,1],mask0[:,2],0] = 1.0
    torsions[mask1[:,0],mask1[:,1],mask1[:,2],1] = 0.0

    # alt chis
    torsions_alt = torsions.clone()
    torsions_alt[torsion_can_flip[seq,:]] *= -1

    return torsions, torsions_alt, tors_mask, tors_planar

def get_tips(xyz, seq):
    B,L = xyz.shape[:2]

    xyz_tips = torch.gather(xyz, 2, tip_indices.to(xyz.device)[seq][:,:,None,None].expand(-1,-1,-1,3)).reshape(B, L, 3)
    mask = ~(torch.isnan(xyz_tips[:,:,0]))
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
    return xyz_tips, mask

# process ideal frames
def make_frame(X, Y):
    Xn = X / torch.linalg.norm(X)
    Y = Y - torch.dot(Y, Xn) * Xn
    Yn = Y / torch.linalg.norm(Y)
    Z = torch.cross(Xn,Yn)
    Zn =  Z / torch.linalg.norm(Z)

    return torch.stack((Xn,Yn,Zn), dim=-1)

def cross_product_matrix(u):
    B, L = u.shape[:2]
    matrix = torch.zeros((B, L, 3, 3), device=u.device)
    matrix[:,:,0,1] = -u[...,2]
    matrix[:,:,0,2] = u[...,1]
    matrix[:,:,1,0] = u[...,2]
    matrix[:,:,1,2] = -u[...,0]
    matrix[:,:,2,0] = -u[...,1]
    matrix[:,:,2,1] = u[...,0]
    return matrix

# writepdb
def writepdb(filename, atoms, seq, binderlen=None, idx_pdb=None, bfacts=None, chain_idx=None):
    f = open(filename,"w")
    ctr = 1
    scpu = seq.cpu().squeeze()
    atomscpu = atoms.cpu().squeeze()
    if bfacts is None:
        bfacts = torch.zeros(atomscpu.shape[0])
    if idx_pdb is None:
        idx_pdb = 1 + torch.arange(atomscpu.shape[0])

    Bfacts = torch.clamp( bfacts.cpu(), 0, 1)
    for i,s in enumerate(scpu):
        if chain_idx is None:
            if binderlen is not None:
                if i < binderlen:
                    chain = 'A'
                else:
                    chain = 'B'
            else:
                chain = 'A'
        else:
            chain = chain_idx[i]
        if (len(atomscpu.shape)==2):
            f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                    "ATOM", ctr, " CA ", num2aa[s],
                    chain, idx_pdb[i], atomscpu[i,0], atomscpu[i,1], atomscpu[i,2],
                    1.0, Bfacts[i] ) )
            ctr += 1
        elif atomscpu.shape[1]==3:
            for j,atm_j in enumerate([" N  "," CA "," C  "]):
                f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                        "ATOM", ctr, atm_j, num2aa[s],
                        chain, idx_pdb[i], atomscpu[i,j,0], atomscpu[i,j,1], atomscpu[i,j,2],
                        1.0, Bfacts[i] ) )
                ctr += 1
        else:
            natoms = atomscpu.shape[1]
            if (natoms!=14 and natoms!=27):
                assert False, f'bad size: {atoms.shape}, must be [L, 14|27,...]'
            atms = aa2long[s]
            # his prot hack
            if (s==8 and torch.linalg.norm( atomscpu[i,9,:]-atomscpu[i,5,:] ) < 1.7):
                atms = (
                    " N  "," CA "," C  "," O  "," CB "," CG "," NE2"," CD2"," CE1"," ND1",
                      None,  None,  None,  None," H  "," HA ","1HB ","2HB "," HD2"," HE1",
                    " HD1",  None,  None,  None,  None,  None,  None) # his_d

            for j,atm_j in enumerate(atms):
                if (j<natoms and atm_j is not None): # and not torch.isnan(atomscpu[i,j,:]).any()):
                    f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                        "ATOM", ctr, atm_j, num2aa[s],
                        chain, idx_pdb[i], atomscpu[i,j,0], atomscpu[i,j,1], atomscpu[i,j,2],
                        1.0, Bfacts[i] ) )
                    ctr += 1


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



# hbtypes = torch.full((NNAPROTAAS,NTOTAL,3),-1, dtype=torch.long) # (donortype, acceptortype, acchybtype)
# hbbaseatoms = torch.full((NNAPROTAAS,NTOTAL,2),-1, dtype=torch.long) # (B,B0) for acc; (D,-1) for don
# hbpolys = torch.zeros((HbDonType.NTYPES,HbAccType.NTYPES,3,15)) # weight,xmin,xmax,ymin,ymax,c9,...,c0
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
    

# af: commenting these out: should use values from rf2aa.chemical.py
# N_BACKBONE_ATOMS = 3
# N_HEAVY = 14 

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


def writepdb_multi(filename, atoms_stack, bfacts, seq_stack, backbone_only=False, chain_ids=None, use_hydrogens=True):
    """
    Function for writing multiple structural states of the same sequence into a single 
    pdb file. 
    """

    f = open(filename,"w")

    if seq_stack.ndim != 2:
        T = atoms_stack.shape[0]
        seq_stack = torch.tile(seq_stack, (T,1))
    seq_stack = seq_stack.cpu()
    for atoms, scpu in zip(atoms_stack, seq_stack):
        ctr = 1
        atomscpu = atoms.cpu()
        Bfacts = torch.clamp( bfacts.cpu(), 0, 1)
        for i,s in enumerate(scpu):
            atms = aa2long[s]
            for j,atm_j in enumerate(atms):

                if backbone_only and j >= N_BACKBONE_ATOMS:
                    break
                if not use_hydrogens and j >= N_HEAVY:
                    break 
                if (atm_j is None) or (torch.all(torch.isnan(atomscpu[i,j]))):
                    continue
                chain_id = 'A'
                if chain_ids is not None:
                    chain_id = chain_ids[i]
                f.write ("%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"%(
                    "ATOM", ctr, atm_j, num2aa[s],
                    chain_id, i+1, atomscpu[i,j,0], atomscpu[i,j,1], atomscpu[i,j,2],
                    1.0, Bfacts[i] ) )
                ctr += 1

        f.write('ENDMDL\n')

def get_mu_xt_x0(xt, px0, t, schedule, alphabar_schedule, eps=1e-6):
    """
    Given xt, predicted x0 and the timestep t, give mu of x(t-1)
    Assumes t is 0 indexed
    """
    #sigma is predefined from beta. Often referred to as beta tilde t
    t_idx = t-1

    sigma = ((1-alphabar_schedule[t_idx-1])/(1-alphabar_schedule[t_idx]))*schedule[t_idx]

    xt_ca = xt[:,1,:]
    px0_ca = px0[:,1,:]

    a = ((torch.sqrt(alphabar_schedule[t_idx-1] + eps)*schedule[t_idx])/(1-alphabar_schedule[t_idx]))*px0_ca
    b = ((torch.sqrt(1-schedule[t_idx] + eps)*(1-alphabar_schedule[t_idx-1]))/(1-alphabar_schedule[t_idx]))*xt_ca
    mu = a + b

    return mu, sigma

def get_t2d(xyz_t, is_sm, atom_frames, mask_t_2d=None):
    '''
    Returns t2d for a template.

    Parameters:
        xyz_t: [T, L, 36, 3]
        is_sm: [L]
        atom_frames: [F, 3, 2]
    '''

    # assert seq_cat.ndim == 1
    # L = seq_cat.shape[0]
    # seq_cat = seq_cat[None]
    L = xyz_t.shape[1]

    # # This will be all True because of the above
    # mask_t_2d = rf2aa.util.get_prot_sm_mask(mask_t, seq_cat).to(same_chain.device) # (T, L)
    # mask_t_2d = mask_t_2d[:,None]*mask_t_2d[:,:,None] # (T, L, L)

    # mask_t_2d = mask_t_2d.float() * same_chain.float()[None] # (ignore inter-chain region)
    # TODO(Look into atom_frames)
    xyz_t_frames = rf2aa.util.xyz_t_to_frame_xyz_sm_mask(xyz_t[None], is_sm, atom_frames[None])

    if mask_t_2d is None:
        mask_t_2d = torch.ones(1,L,L).bool().to(xyz_t_frames.device)
    else:
        pass

    t2d = rf2aa.kinematics.xyz_to_t2d(xyz_t_frames, mask_t_2d[None])
    # Strip batch dimension
    t2d = t2d[0]
    
    return t2d, mask_t_2d

def polymer_content_check(seq, polymer_focus='protein', polymer_frac_cutoff=0.2):

    
    if polymer_focus is not None:
        if polymer_focus in ['protein','prot','aa','AA','a','A']:
            lower_lim = 0
            upper_lim = 21

        elif polymer_focus in ['dna','DNA','d','D']:
            lower_lim = 22
            upper_lim = 26

        elif polymer_focus in ['rna','RNA','r','R']:
            lower_lim = 27
            upper_lim = 31

        resi_in_focus = torch.ge(seq, lower_lim) & torch.le(seq, upper_lim)

        frac_in_focus = resi_in_focus.sum()/len(seq)

        if frac_in_focus >= polymer_frac_cutoff:
            passes_check = True
        else:
            passes_check = False
        

    else:
        passes_check = True

    return passes_check


def get_chain_lengths(pdb_idx, return_dict=False):

    prev_chain = '_'
    chain_letter_list = []
    chain_length_list = []
    chain_length_dict = {}
    for chain_letter_i, resdex_i in pdb_idx:
        if not chain_letter_i==prev_chain:
            chain_letter_list.append(chain_letter_i)
            chain_length_dict[chain_letter_i] = 1
        else:
            chain_length_dict[chain_letter_i] += 1

        prev_chain = chain_letter_i

    chain_length_list = [chain_length_dict[chain_letter] for chain_letter in chain_letter_list]

    if return_dict:
        return chain_length_list, chain_length_dict
    else:
        return chain_length_list



def sstr_to_matrix(ss_string, only_basepairs=True):

    def find_any_bracket_pairs(s, open_symbol, close_symbol):
        stack = []
        pairs = []

        for i, char in enumerate(s):
            if char == open_symbol:
                stack.append(i)
            elif char == close_symbol:
                if stack:
                    pairs.append((stack.pop(), i))
                else:
                    raise ValueError(f"No matching open parenthesis for closing parenthesis at index {i}")

        if stack:
            raise ValueError(f"No matching closing parenthesis for open parenthesis at index {stack[0]}")

        return pairs

    def find_loop_bases(s):
        loop_bases = []
        for i, char in enumerate(s):
            if char == '.':
                loop_bases.append(i)
                
        return loop_bases
    open_symbols  = ['(','[','{','<','5','i','f','b']
    close_symbols = [')',']','}','>','3','j','t','e']

    all_pair_dict = {}
    for pair_ind, (open_symbol, close_symbol) in enumerate(zip(open_symbols, close_symbols)):

        num_opens = len([ _ for _ in ss_string if _ == open_symbol])
        num_close = len([ _ for _ in ss_string if _ == close_symbol])
        assert num_opens==num_close , "number of base pairs must be an even number... must be an error somewhere..."
        
        all_pair_dict[pair_ind] = find_any_bracket_pairs(ss_string, open_symbol, close_symbol)
        
        

    

    # pairs
    L = len(ss_string)
    ss_adj_mat = np.zeros((L,L))

    # if not only_basepairs:
    #     loop_inds = find_loop_bases(ss_string)
    #     for i in loop_inds:
    #         ss_adj_mat[i,i] = 1
        
    for pair_ind in all_pair_dict.keys():
        paired_base_list = all_pair_dict[pair_ind]
        for i,j in paired_base_list:
            ss_adj_mat[i,j] = 1
            ss_adj_mat[j,i] = 1
            # ss_adj_mat[i,j] = pair_ind + 1
            # ss_adj_mat[j,i] = pair_ind + 1
            # # ss_adj_mat[i,j] = pair_ind + 2
            # # ss_adj_mat[j,i] = pair_ind + 2
            
    return ss_adj_mat
        

def ss_matrix_to_t2d_feats(ss_matrix):

    ss_matrix = torch.from_numpy(ss_matrix).long()
    ss_templ_onehot = torch.nn.functional.one_hot(ss_matrix, num_classes=3)
    ss_templ_onehot = ss_templ_onehot.reshape(1, 1, *ss_templ_onehot.shape).repeat(1,3,1,1,1)

    return ss_templ_onehot
    


#########################. ANDREW DONOR ACCEPTOR NOTES: ################################################


    # Donors:
    # D A : N6.    (  20  )
    # D C : N4     (  16  )
    # D G : N1, N2 (  15*, 20  )
    # D T : N3     (  14*  )
    # R A : N6.    (  18 )
    # R C : N4     (  17  )
    # R G : N1, N2 (  12*, 14  )
    # R U : N3     (  15*  )

    # Acceptors:
    # D A : N1     ( 15* ) 
    # D C : O2, N3 ( 13, 14*  )
    # D G : O6     ( 21   )
    # D T : O4     ( 16   )
    # R A : N1     ( 12* )
    # R C : O2, N3 ( 14, 15*  )
    # R G : O6     ( 19  )
    # R U : O4     ( 17   )

    # FOR HOODSTEIN:
    # donors at:
    # DC: N3 : 14
    # DT:

    # acceptors at:
    # DA: N7 : 18
    # DG: N7 : 18
    # DT: 
    # RA: 


    # # donor_atoms    = torch.zeros((sum(len_s),2), dtype=torch.long, device=xyz.device)
    # donor_atoms    = torch.zeros((len_s,2), dtype=torch.long, device=xyz.device)
    # donor_atoms[seq==22] = torch.tensor([20,20])
    # donor_atoms[seq==23] = torch.tensor([16,16])
    # donor_atoms[seq==24] = torch.tensor([15,20])
    # donor_atoms[seq==25] = torch.tensor([14,14])
    # donor_atoms[seq==27] = torch.tensor([18,18])
    # donor_atoms[seq==28] = torch.tensor([17,17])
    # donor_atoms[seq==29] = torch.tensor([12,14])
    # donor_atoms[seq==30] = torch.tensor([15,15])

    # # acceptor_atoms = torch.zeros((sum(len_s),2), dtype=torch.long, device=xyz.device)
    # acceptor_atoms = torch.zeros((len_s,2), dtype=torch.long, device=xyz.device)
    # acceptor_atoms[seq==22] = torch.tensor([15,15])
    # acceptor_atoms[seq==23] = torch.tensor([13,14])
    # acceptor_atoms[seq==24] = torch.tensor([21,21])
    # acceptor_atoms[seq==25] = torch.tensor([16,16])
    # acceptor_atoms[seq==27] = torch.tensor([12,12])
    # acceptor_atoms[seq==28] = torch.tensor([14,15])
    # acceptor_atoms[seq==29] = torch.tensor([19,19])
    # acceptor_atoms[seq==30] = torch.tensor([17,17])

    # donor_xyz = torch.gather(xyz, 1, donor_atoms[:,:,None].repeat(1,1,3)).squeeze(1)
    # acceptor_xyz = torch.gather(xyz, 1, acceptor_atoms[:,:,None].repeat(1,1,3)).squeeze(1)

    # all_hbond_dists, _ = torch.min(torch.cat((torch.cdist(donor_xyz[:,0,:], acceptor_xyz[:,0,:]).unsqueeze(-1),torch.cdist(donor_xyz[:,1,:], acceptor_xyz[:,1,:]).unsqueeze(-1)),dim=-1),dim=-1)
    # cond_new = all_hbond_dists < bp_cutoff

    # # cond_new = torch.logical_or(cond_new, torch.transpose(cond_new))
    # cond_new = torch.logical_or(cond_new, cond_new.t())
    
    #0  ala.  # ALA: CB:  4
    #1  arg.  # ARG: CZ:  8
    #2  asn.  # 6 (accept)  7 (donor)
    #3  asp.  # ASP: OD1: 6: 
    #4  cys.  # CYS: S:   5 (donor and accept)
    #5  gln.  # Gln: NE2: 8 (donor) # Gln: OE2: 7 (accept)
    #6  glu.  # GLU: OE2: 8 (accept)
    #7  gly.  # GLY: CA:  1 
    #8  his.  # HIS: CE1: 8 
    #9  ile.  # ILE: C  7 (donor and accept)
    #10 leu.  # LEU: C  7 (donor and accept)
    #11 lys.  # Lys: NZ: 8 (donor)
    #12 met.  # MET: CE:  7 (donot and accept)
    #13 phe.  # PHE: CZ: 10 (donor and accept)
    #14 pro.  # PRO: CD: 6
    #15 ser.  # SER: O : 5 (accept)
    #16 thr.  # THR: O : 5 (donot)
    #17 trp.  # TRPL C  11 (donor and accept)
    #18 tyr.  # TYR: OH: 11 (donor)
    #19 val.  # VAL: CB: 4


# def get_pair_ss_partners(seq, xyz, mask, sel, len_s, vert_diff_cutoff=3.0,
#                         negative=False, incl_protein=True, cutoff=12.0, bp_cutoff=4.0, eps=1e-6, seq_cutoff=2,
#                          use_base_angles=False, base_angle_cutoff=0.5234, compute_aa_contacts=False, use_repatom=True, canonical_partner_filter=False):




#     seq_neighbors = torch.le(torch.abs(sel[:,None]-sel[None,:]), seq_cutoff)
#     is_protein = torch.logical_and((0 <= seq),(seq <= 21))
#     is_dna = torch.logical_and((22 <= seq),(seq <= 26))
#     is_rna = torch.logical_and((27 <= seq),(seq <= 31))

#     # We return an integer tensor (to later be converted to onehot), where: 
#     # 0 means unpaired
#     # 1 means paired
#     # 2 means masked/unspecified

#     # Initialize as all unspecified, and fill in. Everything should technically get defined if we even call this function, 
#     # but safer to do it this way, so we fall back to the default way this template gets provided.
#     cond_num = 2*torch.ones((len_s,len_s), dtype=torch.long, device=xyz.device)
#     # # get base pairing NA bases
#     if use_repatom: # Using Frank's method with distance between representative atoms:
#         repatom = torch.zeros(len_s, dtype=torch.long, device=xyz.device)
#         repatom[seq==22] = 15 # DA - N1
#         repatom[seq==23] = 14 # DC - N3
#         repatom[seq==24] = 15 # DG - N1
#         repatom[seq==25] = 14 # DT - N3
#         repatom[seq==27] = 12 # A - N1
#         repatom[seq==28] = 15 # C - N3
#         repatom[seq==29] = 12 # G - N1
#         repatom[seq==30] = 15 # U - N3

#         if compute_aa_contacts:
#             repatom[seq==0 ] = 4
#             repatom[seq==1 ] = 8
#             repatom[seq==2 ] = 7
#             repatom[seq==3 ] = 6
#             repatom[seq==4 ] = 5
#             repatom[seq==5 ] = 6
#             repatom[seq==6 ] = 6
#             repatom[seq==7 ] = 1
#             repatom[seq==8 ] = 8
#             repatom[seq==9 ] = 7
#             repatom[seq==10] = 7
#             repatom[seq==11] = 8
#             repatom[seq==12] = 7
#             repatom[seq==13] = 10
#             repatom[seq==14] = 6
#             repatom[seq==15] = 5
#             repatom[seq==16] = 5
#             repatom[seq==17] = 11
#             repatom[seq==18] = 11
#             repatom[seq==19] = 4


#         xyz_na_rep = torch.gather(xyz, 1, repatom[:,None,None].repeat(1,1,3)).squeeze(1)
#         contact_dist = torch.cdist(xyz_na_rep, xyz_na_rep) < bp_cutoff

#     else:

#         donor_atoms    = torch.ones((len_s,1), dtype=torch.long, device=xyz.device)
#         donor_atoms[seq==22] = torch.tensor([20])
#         donor_atoms[seq==23] = torch.tensor([16])
#         donor_atoms[seq==24] = torch.tensor([15])
#         donor_atoms[seq==25] = torch.tensor([14])
#         donor_atoms[seq==27] = torch.tensor([18])
#         donor_atoms[seq==28] = torch.tensor([17])
#         donor_atoms[seq==29] = torch.tensor([12])
#         donor_atoms[seq==30] = torch.tensor([15])

#         acceptor_atoms = torch.ones((len_s,1), dtype=torch.long, device=xyz.device)
#         acceptor_atoms[seq==22] = torch.tensor([15])
#         acceptor_atoms[seq==23] = torch.tensor([14])
#         acceptor_atoms[seq==24] = torch.tensor([21])
#         acceptor_atoms[seq==25] = torch.tensor([16])
#         acceptor_atoms[seq==27] = torch.tensor([12])
#         acceptor_atoms[seq==28] = torch.tensor([15])
#         acceptor_atoms[seq==29] = torch.tensor([19])
#         acceptor_atoms[seq==30] = torch.tensor([17])

#         donor_xyz = torch.gather(xyz, 1, donor_atoms[:,:,None].repeat(1,1,3)).squeeze(1)
#         acceptor_xyz = torch.gather(xyz, 1, acceptor_atoms[:,:,None].repeat(1,1,3)).squeeze(1)

#         contact_dist = torch.cdist(donor_xyz, acceptor_xyz) < bp_cutoff


#     cond = torch.logical_and(contact_dist, ~seq_neighbors)

#     if use_base_angles: # Additional Andrew Filter: check for angle between normal vector of any two bases

#         len_s_na = (~is_protein).sum()

#         base_atom_xyz = torch.zeros((len_s_na,11,3), dtype=torch.float, device=xyz.device)
#         mask_na = mask[~is_protein].unsqueeze(-1).repeat(1,1,3)

#         base_xyz_masked = torch.where(mask_na, xyz[~is_protein], torch.nan)

#         base_atom_xyz[is_dna[~is_protein],:,:] = base_xyz_masked[is_dna[~is_protein],11:22,:]
#         base_atom_xyz[is_rna[~is_protein],:,:] = base_xyz_masked[is_rna[~is_protein],12:23,:]


#         # Compute the centroid of the points
#         centroid = torch.nanmean(base_atom_xyz, dim=1, keepdim=True)

#         # Center the points
#         centered_points = base_atom_xyz - centroid
#         centered_nan_mask = ~torch.isnan(centered_points)
#         centered_zero_nan = torch.where(centered_nan_mask, centered_points, 0.0)

#         ###    COMPUTING THE BASE ANGLES    ###
#         # Compute the covariance matrix
#         covariance_matrix_unscaled = torch.matmul(centered_zero_nan.transpose(-1, -2), centered_zero_nan)
#         denom = ( centered_nan_mask.sum(-2) - 1 ).unsqueeze(-1).repeat((1,1,3))
#         covariance_matrix = covariance_matrix_unscaled / (denom + eps)

#         # Compute the eigenvectors and eigenvalues
#         eigenvalues, eigenvectors = torch.linalg.eig(covariance_matrix)

#         # The normal to the plane is the eigenvector associated with the smallest eigenvalue
#         base_normals = torch.real(eigenvectors)[torch.arange(eigenvectors.shape[0]), torch.argmin(torch.real(eigenvalues),dim=-1)]
#         cosines = torch.clamp(torch.einsum('ni,mi->nm', base_normals, base_normals), -1, 1)
#         angle_differences = torch.acos(cosines)
#         bases_in_plane = (angle_differences <= base_angle_cutoff)

#         cond[~is_protein,:][:,~is_protein] = torch.logical_and(cond[~is_protein,:][:,~is_protein], bases_in_plane)
    


#         ###    COMPUTING THE PLANE DISTANCE   ###
#         r_ij_mat = centroid - centroid.transpose(0, 1)
#         d_ij_on_norm_i = torch.norm(torch.sum(r_ij_mat * base_normals.unsqueeze(1), dim=-1).unsqueeze(-1) * base_normals.unsqueeze(1) , dim=-1)

#         cond_01 = (d_ij_on_norm_i <= vert_diff_cutoff)
#         # cond_01 = (d_ij_on_norm_i < 2.0)
#         # cond_01 = (d_ij_on_norm_i < 1.)
#         cond_01_sym = torch.logical_or(cond_01, cond_01.transpose(0,1))
#         # cond_01_sym = torch.logical_and(cond_01, cond_01.transpose(0,1))
#         # cond = torch.logical_and(~seq_neighbors,cond_01_sym)
#         # cond_02 = ( torch.norm(r_ij_mat , dim=-1) < 10.0 )
#         # cond_02_sym = torch.logical_or(cond_02, cond_02.transpose(0,1))
#         # cond = torch.logical_and(cond,cond_02_sym)

#         cond = torch.logical_and(cond,cond_01_sym)

#         # if just_use_proj:

            



#             # cond = torch.logical_and(cond,bases_in_plane)
#             # bases_in_plane = (angle_differences <= 2.0)
#             # cond = torch.logical_and(cond[~is_protein,:][:,~is_protein],bases_in_plane)
#             # cond = torch.logical_and(cond[~is_protein,:][:,~is_protein],bases_in_plane)
            

#             # cond[~is_protein,:][:,~is_protein] = torch.logical_and(cond[~is_protein,:][:,~is_protein], bases_in_plane)

#             # cond



#             # cond_vdist = torch.logical_and(cond_01_sym, cond_02_sym)
#             # ~seq_neighbors

#             # cond = torch.logical_and(cond,cond_vdist)
#             # cond = torch.logical_and(cond,cond_01_sym)
#             # cond = torch.logical_and(cond,cond_01_sym)
#             # cond = torch.logical_or(cond, cond.t())
#             # cond_num[:,:] = cond[:,:].long()
#             # print('JUST USING PROJECTION METHOD')
#             # return cond_num





#     cond = torch.logical_or(cond, cond.t())
    

#     # Andrew filter: check for canonical base pairing.
#     # probably bad for RNA, but helps to add in traning occasionally.
#     if canonical_partner_filter:
#         bp_partners_canon = torch.zeros((len_s, len_s), dtype=torch.bool, device=xyz.device)

#         # Define the conditions as boolean masks
#         cond_AA = (seq[:, None] == 22) | (seq[:, None] == 27)
#         cond_TU = (seq[:, None] == 25) | (seq[:, None] == 30)
#         cond_CC = (seq[:, None] == 23) | (seq[:, None] == 28)
#         cond_GG = (seq[:, None] == 24) | (seq[:, None] == 29)

#         # Update the matrix based on the conditions
#         bp_partners_canon[cond_AA & cond_TU.T] = True
#         bp_partners_canon[cond_TU & cond_AA.T] = True
#         bp_partners_canon[cond_CC & cond_GG.T] = True
#         bp_partners_canon[cond_GG & cond_CC.T] = True

#         cond = torch.logical_and(cond, bp_partners_canon)

#     # If we want to return this matrix as integer, rather than boolean.
#     cond_num[:,:] = cond[:,:].long()

#     return cond_num



def get_pair_ss_partners(seq, xyz, mask, sel, len_s, vert_diff_cutoff=6.69730945, incl_protein=True, centroid_cutoff=6.20250967, bp_cutoff=3.20732419, eps=1e-6, seq_cutoff=2, base_angle_cutoff=0.06184608, canonical_partner_filter=False):

    seq_neighbors = torch.le(torch.abs(sel[:,None]-sel[None,:]), seq_cutoff)
    is_protein = torch.logical_and((0 <= seq),(seq <= 21))
    is_dna = torch.logical_and((22 <= seq),(seq <= 26))
    is_rna = torch.logical_and((27 <= seq),(seq <= 31))

    len_s_na = (~is_protein).sum()

    # Using Frank's method with distance between representative atoms:
    repatom = torch.zeros(len_s, dtype=torch.long, device=xyz.device)

    repatom[seq==22] = rf2aa.chemical.aa2long[22].index(' N1 ') # DA - N1
    repatom[seq==23] = rf2aa.chemical.aa2long[23].index(' N3 ') # DC - N3
    repatom[seq==24] = rf2aa.chemical.aa2long[24].index(' N1 ') # DG - N1
    repatom[seq==25] = rf2aa.chemical.aa2long[25].index(' N3 ') # DT - N3
    repatom[seq==27] = rf2aa.chemical.aa2long[27].index(' N1 ') # A - N1
    repatom[seq==28] = rf2aa.chemical.aa2long[28].index(' N3 ') # C - N3
    repatom[seq==29] = rf2aa.chemical.aa2long[29].index(' N1 ') # G - N1
    repatom[seq==30] = rf2aa.chemical.aa2long[30].index(' N3 ') # U - N3


    xyz_na_rep = torch.gather(xyz, 1, repatom[:,None,None].repeat(1,1,3)).squeeze(1)
    contact_dist = torch.cdist(xyz_na_rep, xyz_na_rep) < bp_cutoff
    cond = torch.logical_and(contact_dist, ~seq_neighbors)

    

    base_atom_xyz = torch.zeros((len_s_na,11,3), dtype=torch.float, device=xyz.device)
    mask_na = mask[~is_protein].unsqueeze(-1).repeat(1,1,3)

    base_xyz_masked = torch.where(mask_na, xyz[~is_protein], torch.nan)

    # Select full range for guanines, since those have the most atoms
    dna_base_atoms_start = rf2aa.chemical.aa2long[24].index(' N9 ')
    dna_base_atoms_stop = rf2aa.chemical.aa2long[24].index(' O6 ')
    rna_base_atoms_start = rf2aa.chemical.aa2long[29].index(' N1 ')
    rna_base_atoms_stop = rf2aa.chemical.aa2long[29].index(' N9 ')


    base_atom_xyz[is_dna[~is_protein],:,:] = base_xyz_masked[is_dna[~is_protein],dna_base_atoms_start:dna_base_atoms_stop+1,:]
    base_atom_xyz[is_rna[~is_protein],:,:] = base_xyz_masked[is_rna[~is_protein],rna_base_atoms_start:rna_base_atoms_stop+1,:]


    # Compute the centroid of the points
    centroid = torch.nanmean(base_atom_xyz, dim=1, keepdim=True)
    
    centroid_contact_dist = torch.cdist(centroid[:,0,:],centroid[:,0,:])

    centroid_in_contact = (torch.cdist(centroid[:,0,:],centroid[:,0,:]) < centroid_cutoff)

    cond[~is_protein,:][:,~is_protein] = torch.logical_and(cond[~is_protein,:][:,~is_protein], centroid_in_contact)


    # Center the points
    centered_points = base_atom_xyz - centroid
    centered_nan_mask = ~torch.isnan(centered_points)
    centered_zero_nan = torch.where(centered_nan_mask, centered_points, 0.0)

    ###    COMPUTING THE BASE ANGLES    ###
    # Compute the covariance matrix
    covariance_matrix_unscaled = torch.matmul(centered_zero_nan.transpose(-1, -2), centered_zero_nan)
    denom = ( centered_nan_mask.sum(-2) - 1 ).unsqueeze(-1).repeat((1,1,3))
    covariance_matrix = covariance_matrix_unscaled / (denom + eps)

    # Compute the eigenvectors and eigenvalues
    eigenvalues, eigenvectors = torch.linalg.eig(covariance_matrix)

    # The normal to the plane is the eigenvector associated with the smallest eigenvalue
    base_normals = torch.real(eigenvectors)[torch.arange(eigenvectors.shape[0]), torch.argmin(torch.real(eigenvalues),dim=-1)]
    cosines = torch.clamp(torch.einsum('ni,mi->nm', base_normals, base_normals), -1, 1)
    angle_differences = torch.acos(torch.abs(cosines))
    bases_in_plane = (angle_differences <= base_angle_cutoff)

    # Don't use base-angle logic for protein contacts:
    cond[~is_protein,:][:,~is_protein] = torch.logical_and(cond[~is_protein,:][:,~is_protein], bases_in_plane)

    ###    COMPUTING THE PLANE DISTANCE   ###
    r_ij_mat = centroid - centroid.transpose(0, 1)
    d_ij_on_norm_i = torch.norm(torch.sum(r_ij_mat * base_normals.unsqueeze(1), dim=-1).unsqueeze(-1) * base_normals.unsqueeze(1) , dim=-1)

    base_close_vert_dist = (d_ij_on_norm_i <= vert_diff_cutoff)
    cond[~is_protein,:][:,~is_protein] = torch.logical_and(cond[~is_protein,:][:,~is_protein], base_close_vert_dist)
    
    cond = torch.logical_or(cond, cond.t())
    
    return cond.long()



def get_start_stop_inds(mask_vec):
    shifted_tensor = torch.roll(mask_vec, shifts=1, dims=0)
    diff_tensor = mask_vec != shifted_tensor
    change_indices = torch.where(diff_tensor)[0]
    start_stop_indices = [(start.item(), stop.item()) for start, stop in zip(change_indices[:-1], change_indices[1:])]
    if mask_vec[0]:
        start_stop_indices = [(0, change_indices[0].item())] + start_stop_indices
    if mask_vec[-1]:
        start_stop_indices.append((change_indices[-1].item(), len(mask_vec)))

    values_list = []
    for start, stop in start_stop_indices:
        value = mask_vec[start].item()
        values_list.append(value)

    return start_stop_indices, values_list







# # fit_params_pSSA = {
# # 'fc1_w': torch.tensor([[-1.0338e-01,  3.5331e-02,  4.9069e-02,  2.2999e-02, -1.2684e-02,
# #           6.4225e-04, -8.4047e-02, -4.2551e-02, -2.9523e-02, -6.8141e-02,
# #          -8.3562e-02,  3.3802e-02,  3.4457e-02, -9.8110e-02,  1.8489e-03,
# #           4.1041e-03,  4.6931e-02,  1.6970e-02,  5.4350e-02, -5.0783e-02,
# #           3.2512e-02,  4.2157e-02, -1.0197e-01, -8.8423e-02, -4.8389e-03,
# #           2.3583e-02, -6.3560e-02, -1.2945e-02, -2.1890e-02, -7.3247e-04,
# #          -6.8314e-02,  1.7561e-02,  2.3275e-02, -8.8858e-02, -2.5093e-02,
# #          -2.6956e-02,  1.6792e-03,  5.3077e-02,  1.0609e-04, -4.1331e-02,
# #           5.3074e-02, -5.3591e-02, -4.6379e-02, -1.6524e-02, -8.5086e-02,
# #          -1.3087e-02, -8.1809e-02, -2.6400e-02, -2.4189e-02, -1.5339e-02,
# #          -9.2064e-02, -1.0170e-01,  1.2357e-02,  4.4801e-02, -9.8422e-02,
# #           9.2845e-03, -8.2776e-02, -1.6720e-02,  5.5308e-02,  4.1154e-02,
# #          -8.2719e-02,  2.8745e-02,  1.9451e-02, -2.9291e-02,  4.8628e-02,
# #          -5.5326e-02, -4.1230e-02, -6.4582e-02, -1.3779e-02, -4.2797e-03,
# #          -1.7984e-02, -6.8588e-02, -5.9289e-02,  6.2086e-03, -9.5116e-02,
# #          -3.4776e-02, -2.3497e-02, -5.5893e-02,  6.9436e-03, -2.4496e-02,
# #           2.6514e-02, -3.8333e-02, -7.5282e-02,  3.5022e-02, -1.3097e-02,
# #          -4.9274e-02, -8.0056e-02, -3.3419e-03, -1.0136e-01, -1.7403e-04,
# #           1.9120e-02, -9.4831e-02, -7.5795e-02, -8.2323e-02,  4.2873e-02,
# #           2.8375e-02, -2.6185e-02,  4.9994e-02, -5.1200e-02, -1.0317e-01,
# #           5.0472e-03,  3.1869e-03,  5.5251e-02, -2.3547e-02,  4.2159e-02,
# #           2.9035e-02,  3.8108e-02, -1.0461e-01,  4.7078e-02, -9.3357e-02,
# #           5.1856e-02, -2.8687e-02, -1.0577e-01, -3.6952e-03, -3.8753e-02,
# #          -9.6765e-02, -5.3993e-02, -1.7855e-02,  3.8144e-02, -4.9053e-02,
# #           2.7447e-02, -1.1727e-02, -1.9127e-02, -5.5620e-03, -7.6151e-02,
# #          -9.1368e-02, -4.8582e-02, -1.0213e-01, -7.0996e-02, -1.8699e-02,
# #          -7.6743e-02, -5.6831e-02, -3.9887e-02, -3.6057e-02, -7.1751e-02,
# #          -2.0521e-02, -3.4119e-02,  1.6365e-03, -1.8133e-02, -6.3003e-02,
# #          -5.0019e-02, -2.2778e-02, -7.4803e-02, -2.3145e-02],
# #         [-2.7816e-02,  9.8378e-03, -2.4415e-03,  4.0384e-02, -2.4526e-02,
# #          -8.3001e-02,  1.5791e-02, -7.7633e-02, -3.2582e-02, -4.7877e-02,
# #          -5.2763e-02, -5.2232e-02,  5.5173e-02,  2.6401e-02,  1.4319e-03,
# #           5.6200e-02, -9.7404e-02, -8.4639e-03,  3.1323e-02,  3.3278e-02,
# #          -8.8002e-02,  8.2725e-03, -6.8853e-02, -2.4605e-02, -3.8600e-02,
# #          -9.0634e-02, -1.0060e-01, -7.6126e-02,  1.6435e-02,  1.6808e-02,
# #          -9.3844e-02, -5.3117e-02, -2.0522e-02, -9.3911e-02,  2.4911e-02,
# #           5.3032e-02, -7.0845e-03, -8.3100e-02, -1.8995e-03,  8.8910e-03,
# #           4.4605e-02, -6.5614e-02, -9.5747e-02,  1.5696e-02, -1.6014e-02,
# #          -4.8527e-02, -5.6377e-02, -6.4779e-02,  4.5572e-02, -3.6254e-02,
# #           3.7948e-02,  5.2257e-02, -1.1307e-02, -3.7387e-03, -2.7036e-02,
# #           3.5880e-02, -9.6278e-02,  1.2329e-02, -3.4551e-02,  1.6744e-02,
# #          -2.1773e-02, -5.2321e-02, -3.0931e-02,  2.2934e-02, -7.9807e-02,
# #          -7.4859e-02, -4.0657e-02, -5.2229e-02,  1.0835e-02,  3.4644e-02,
# #          -3.0194e-02,  5.7428e-02, -5.6172e-02, -6.7842e-02, -5.8126e-02,
# #          -7.8540e-02, -7.9786e-02, -9.5628e-02, -9.5433e-02, -2.3245e-04,
# #           2.8894e-02,  5.6278e-02,  4.9087e-02, -2.6840e-02, -5.8833e-03,
# #          -7.1291e-02,  1.0901e-02, -1.0119e-01,  1.7887e-02, -6.1946e-03,
# #           4.5630e-03,  9.7278e-03, -5.9352e-02,  5.0153e-02,  1.8175e-02,
# #           3.2781e-02, -1.0881e-02,  3.0464e-02,  3.4085e-02,  1.5365e-02,
# #           4.1440e-02, -3.6809e-02,  3.3333e-02, -2.1845e-03,  2.8844e-02,
# #           6.0068e-02, -2.0465e-03, -5.1941e-02, -2.9716e-02,  1.5718e-03,
# #          -8.0324e-03, -4.4665e-02,  4.3367e-02,  2.8109e-02, -7.3612e-02,
# #          -2.8843e-02, -2.9974e-02, -7.3860e-02, -9.2198e-03, -4.6952e-02,
# #          -6.4388e-02, -4.8134e-02, -5.0730e-02,  5.8827e-02,  1.0772e-02,
# #           6.0444e-02,  9.5161e-04, -3.1406e-03,  1.2329e-02, -8.0648e-03,
# #          -8.7292e-02, -7.1164e-02, -4.2299e-02, -2.0566e-02,  1.0186e-02,
# #           1.0739e-02,  4.4716e-02, -4.5126e-03, -4.9820e-02, -8.3458e-02,
# #          -8.1905e-02, -7.1129e-02,  3.3288e-03,  2.2996e-02],
# #         [ 2.7126e-01,  1.4502e-01,  1.5561e-01,  1.3703e-01,  1.1476e-01,
# #           7.1077e-02,  5.9538e-02,  1.0015e-01,  4.5924e-02, -2.5426e-02,
# #           3.5914e-03,  7.5152e-02,  1.9214e-01,  1.4955e-01,  1.3165e-01,
# #           2.5983e-01,  1.9904e-01,  1.6965e-01,  1.1819e-01,  1.0790e-01,
# #           1.5805e-01,  3.3138e-01,  1.5718e-01,  1.0049e-02, -9.0633e-02,
# #           6.3039e-04,  1.5805e-01,  2.4640e-01,  1.2110e-01,  2.0778e-01,
# #           1.8756e-01,  1.6659e-01,  2.2584e-01,  1.3597e-01,  1.2017e-01,
# #           7.3271e-02,  2.5205e-01,  4.3657e-01,  2.1529e-01,  7.1325e-03,
# #          -1.1572e-01, -1.0897e-01, -1.2057e-01,  1.6960e-01,  1.3377e-01,
# #           1.8403e-01,  1.8440e-01,  2.7987e-01,  2.0318e-01,  2.1030e-01,
# #           1.0755e-01,  6.1883e-02,  3.0245e-01,  2.6660e-01,  1.7915e-01,
# #          -7.6741e-02, -5.1639e-02,  3.9901e-02,  1.3324e-01,  1.6962e-01,
# #           2.0445e-01,  1.9128e-01,  2.3609e-01,  2.0584e-01,  2.3769e-01,
# #           8.5186e-02,  1.8973e-01,  1.5971e-01,  1.4497e-01,  1.1044e-01,
# #          -1.7699e-01, -1.1641e-01,  9.0692e-02,  2.1250e-01,  1.9478e-01,
# #           1.8022e-01,  2.1109e-01,  1.7864e-01,  1.7002e-01,  2.4764e-01,
# #           1.4556e-01,  1.9088e-01,  1.3911e-01,  7.7808e-02,  1.5617e-02,
# #          -1.3425e-01, -6.3802e-02, -9.7848e-02, -5.3549e-03,  3.4925e-02,
# #           1.7280e-01,  1.2308e-01,  2.1563e-01,  9.0998e-02,  2.1960e-01,
# #           1.6982e-01,  2.4067e-01,  1.4771e-01,  1.4479e-01,  9.2695e-02,
# #           3.5370e-01,  2.8883e-01, -6.1576e-03, -6.4103e-02, -1.4769e-01,
# #           6.6170e-02,  2.2672e-01,  1.8338e-01,  1.2975e-01,  1.8980e-01,
# #           1.2068e-01,  2.7014e-01,  2.4337e-01,  2.3082e-01,  1.7984e-01,
# #           4.0959e-02, -1.5578e-01, -2.8747e-01, -1.7127e-01, -9.1027e-02,
# #           1.4858e-01,  4.2064e-01,  2.4329e-01,  2.2118e-01,  8.6948e-02,
# #           1.5887e-01,  9.4973e-02,  2.6160e-01,  1.3898e-01,  1.3179e-01,
# #           2.1873e-01,  2.7912e-01,  3.8992e-01,  4.3909e-01,  9.4120e-02,
# #          -4.4665e-02, -1.3440e-01,  3.5483e-02, -8.1354e-02,  1.7336e-01,
# #           2.4281e-01,  1.1911e-01,  2.1074e-01,  2.7794e-01],
# #         [ 2.5804e-01,  1.3387e-01,  1.3179e-01,  1.9153e-01,  5.5519e-02,
# #          -9.0090e-02, -3.6132e-02,  3.3289e-01,  1.0520e-01, -7.9361e-02,
# #          -3.9609e-01, -2.0253e-01,  1.5844e-01,  9.2043e-02,  1.6224e-01,
# #           3.5108e-01,  1.1318e-01,  1.0658e-01,  8.8094e-02,  1.2429e-01,
# #           1.7422e-01, -2.1384e-02,  2.3167e-01, -7.4548e-02, -2.3115e-01,
# #          -2.5651e-01, -1.6363e-01,  9.7431e-02,  1.4542e-01,  1.4314e-01,
# #           1.7804e-01,  2.7403e-01,  1.4289e-01,  2.3776e-01,  1.2284e-01,
# #           5.8025e-02,  4.7836e-03, -2.4114e-01, -2.4553e-01, -2.3957e-01,
# #          -6.0037e-02,  9.5083e-02,  2.4294e-01,  4.9732e-01,  2.3886e-01,
# #           2.1141e-01,  1.3931e-01,  3.9377e-01,  2.2025e-01,  1.5545e-01,
# #           1.9411e-01,  1.2224e-01,  4.9635e-01,  1.9691e-01,  2.6158e-01,
# #          -6.8881e-02, -3.0038e-01, -4.2805e-01, -4.3456e-01,  4.9864e-02,
# #           1.5588e-01,  1.8386e-01,  2.0292e-01,  3.4799e-01,  2.2522e-01,
# #           8.1561e-02,  6.0740e-02,  1.0061e-01, -1.6242e-01, -1.7182e-01,
# #          -7.3501e-02, -2.2005e-01, -1.3347e-01, -3.0173e-02,  2.4687e-01,
# #           1.6678e-01,  5.2869e-02,  1.3374e-01,  1.6562e-01,  2.8572e-01,
# #           2.1893e-01,  2.2138e-01,  5.4650e-02, -3.9893e-02, -4.5479e-01,
# #          -8.3597e-01, -5.1918e-01, -1.3543e-01,  1.8753e-01,  4.2470e-01,
# #           4.8829e-01,  1.6683e-01,  1.8864e-01,  1.1017e-01,  1.2805e-01,
# #           2.6466e-01,  2.3952e-01,  1.6207e-01,  1.4063e-01,  3.3912e-01,
# #           5.9711e-01,  3.5110e-01,  4.2385e-01, -1.7135e-01, -8.4309e-02,
# #          -4.2335e-01, -6.5586e-01, -1.5610e-03,  1.1905e-01,  2.0605e-01,
# #           1.5631e-01,  2.7859e-01,  1.3922e-01,  2.3945e-01,  5.4050e-02,
# #           5.7067e-02, -3.1652e-01, -3.3191e-01, -1.0737e-01, -3.6651e-02,
# #           3.8004e-02, -3.6062e-01, -1.4360e-01,  1.8070e-01,  1.4570e-01,
# #           1.1716e-01,  1.5365e-01,  2.8061e-01,  2.4132e-01,  1.3188e-01,
# #           1.2689e-01,  1.6298e-01,  8.5855e-02, -3.3433e-01, -1.1558e-01,
# #          -7.8668e-02, -5.8021e-02,  6.0356e-02,  7.5198e-02,  1.8321e-01,
# #           1.6180e-01,  1.5633e-01,  2.3702e-01,  2.7541e-01]]),
# # 'fc1_b': torch.tensor([-0.0073, -0.0736,  0.7235,  1.2319]),
# # 'fc2_w': torch.tensor([[ 0.1245,  0.3830, -0.2434, -0.3621],
# #         [-0.5115,  0.2924,  0.7716,  0.7454],
# #         [ 0.2076,  0.4347, -0.6673,  0.1229],
# #         [ 0.0469,  0.0250,  1.1549,  1.7516]]),
# # 'fc2_b': torch.tensor([0.6202, 0.4566, 0.3264, 0.0704]),
# # 'fc3_w': torch.tensor([[-0.9361,  0.4330, -0.6616,  0.8476],
# #         [-0.1534,  0.3012, -0.4654,  0.2928],
# #         [ 0.6743, -0.0571,  0.4280, -0.7742],
# #         [-1.1054,  0.2829, -0.4307,  0.5904]]),
# # 'fc3_b': torch.tensor([ 0.5191,  0.5326, -0.6463,  0.2321]),
# # 'fc_out_w': torch.tensor([[-0.4532, -0.5225,  0.5296, -0.6015]]),
# # 'fc_out_b': torch.tensor([0.2702]),
# # }

        
# def run_inference(x, 
#                   fit_params = fit_params_pSSA
#                  ):
    
        
#         B, N, M, D = x.shape # batch_size, n_rows, m_cols, feature_dim
#         x = x.reshape(B*N*M, D)
        

#         # Pass through the network
#         x = torch.matmul(x,fit_params['fc1_w'].T) + fit_params['fc1_b']
#         x = torch.matmul(x,fit_params['fc2_w'].T) + fit_params['fc2_b']
#         x = torch.matmul(x,fit_params['fc3_w'].T) + fit_params['fc3_b']
#         x = torch.matmul(x,fit_params['fc_out_w'].T) + fit_params['fc_out_b']
        
#         x = x.reshape(B, N, M, 1)
        
#         x = x + x.permute(0,2,1,3)
        
#         x = torch.sigmoid(x)  # Using sigmoid to get values between 0 and 1

        
#         return x[0,:,:,0] # batch size should always be 1. 
    
    
    

# def make_rbf_feats(X_i, 
#                    d_min=2., 
#                    d_max=22., 
#                    n_bases=16, 
#                    noise_val=None,
#                     kernel=None):

#     def _gaussian(D_i, d_min, d_max, n_bases):
#         # device = D_i.device
#         D_mu = torch.linspace(d_min, d_max, n_bases)
#         D_sigma = (d_max - d_min) / n_bases
        
#         RBF = torch.exp(-((D_i[...,None] - D_mu[None,...]) / D_sigma)**2)

#         return RBF
    
#     def _null_base(D_i, d_min, d_max, n_bases):
#         # device = D_i.device
#         D_mu = torch.linspace(d_min, d_max, n_bases)
#         D_sigma = (d_max - d_min) / n_bases
#         RBF = ((D_i[...,None] - D_mu[None,...]) / D_sigma)

#         return RBF
    
#     def _quadratic(D_i, d_min, d_max, n_bases):
#         # device = D_i.device
#         D_mu = torch.linspace(d_min, d_max, n_bases)
#         D_sigma = (d_max - d_min) / n_bases
#         RBF = ((D_i[...,None] - D_mu[None,...]) / D_sigma)**2

#         return RBF

#     nresi, natoms, ndims = X_i.shape

#     if noise_val is not None:
#         noise_i = torch.rand_like(X_i) * (2 * noise_val) - noise_val
#         X_i += noise_i


#     D_sigma = (d_max - d_min) / n_bases
#     nresi, natoms, ndims = X_i.shape
#     frame_atom_combo_inds = [(i,j) for i in range(natoms) for j in range(natoms)]
#     # D_i = torch.zeros((1, nresi, nresi, len(frame_atom_combo_inds)), device=device)
#     D_i = torch.zeros((1, nresi, nresi, len(frame_atom_combo_inds)))

#     for pair_ind, (i,j) in enumerate(frame_atom_combo_inds):

#         D_i[:,:,:,pair_ind] = torch.cdist(X_i[:,i,:],X_i[:,j,:])
        
#     if kernel is not None:
#         if kernel=='gaussian':
#             RBF_i = _gaussian(D_i, d_min, d_max, n_bases)
#         elif kernel=='quadratic':
#             RBF_i = _quadratic(D_i, d_min, d_max, n_bases)
            
#         else:
#             RBF_i = _null_base(D_i, d_min, d_max, n_bases)
            
#     else:
#         RBF_i = _null_base(D_i, d_min, d_max, n_bases)
        
        

#     t2d_i = RBF_i.reshape(1, nresi, nresi, len(frame_atom_combo_inds)*n_bases)
    
    
    
#     return t2d_i
        

# # def compute_pSSA(ss_string, xyz, seq, 
# #                 network_params_in=None,
# #                 representative_res_ind=29, # RG has most atoms to reference
# #                 frame_atom_0=" O4'", 
# #                 frame_atom_1=" C1'", 
# #                 frame_atom_2=" C2'",
# #                 only_basepairs=False):




# #     def find_any_bracket_pairs(s, open_symbol, close_symbol):
# #         stack = []
# #         pairs = []

# #         for i, char in enumerate(s):
# #             if char == open_symbol:
# #                 stack.append(i)
# #             elif char == close_symbol:
# #                 if stack:
# #                     pairs.append((stack.pop(), i))
# #                 else:
# #                     raise ValueError(f"No matching open parenthesis for closing parenthesis at index {i}")

# #         if stack:
# #             raise ValueError(f"No matching closing parenthesis for open parenthesis at index {stack[0]}")

# #         return pairs

# #     def find_loop_bases(s):
# #         loop_bases = []
# #         for i, char in enumerate(s):
# #             if char == '.':
# #                 loop_bases.append(i)

# #         return loop_bases


# #     open_symbols  = ['(','[','{','<','5','i','f','b']
# #     close_symbols = [')',']','}','>','3','j','t','e']
# #     loop_symbols  = ['.','&',','] # For now we just treating breaks as loops, cuz dssr is fallible.


# #     all_pair_dict = {}
# #     for pair_ind, (open_symbol, close_symbol) in enumerate(zip(open_symbols, close_symbols)):

# #         num_opens = len([ _ for _ in ss_string if _ == open_symbol])
# #         num_close = len([ _ for _ in ss_string if _ == close_symbol])
# #         assert num_opens==num_close , "number of base pairs must be an even number... must be an error somewhere..."

# #         all_pair_dict[pair_ind] = find_any_bracket_pairs(ss_string, open_symbol, close_symbol)


# #     ss_element_list = []

# #     for pair_ind in all_pair_dict.keys():
# #         paired_base_list = all_pair_dict[pair_ind]
# #         for i,j in paired_base_list:
# #             ss_element_list.append(('P',i,j))


# #     if not only_basepairs:
# #         for i, char in enumerate(ss_string):
# #             if char in loop_symbols: 
# #                 ss_element_list.append(('L',i,i))



# #     # Get frame atoms:
# #     frame_atom_inds = torch.tensor([rf2aa.chemical.aa2long[representative_res_ind].index(frame_atom_0),
# #                                     rf2aa.chemical.aa2long[representative_res_ind].index(frame_atom_1),
# #                                     rf2aa.chemical.aa2long[representative_res_ind].index(frame_atom_2)])

# #     frame_xyz = xyz[0,:,frame_atom_inds,:]

# #     L = len(ss_string)

# #     frame_t2d = make_rbf_feats(frame_xyz, kernel='gaussian').to('cpu')

# #     P_mat = run_inference(frame_t2d, fit_params=fit_params_pSSA)

# #     pSSA_vec = torch.zeros(L)
# #     num_positions_counted = 0
# #     for i, ss_element_i in enumerate(ss_element_list):

# #         if ss_element_i[0]=='L':
# #             P_not_i = torch.cat((P_mat[ss_element_i[1],:ss_element_i[2]], P_mat[ss_element_i[1],ss_element_i[2]+1:]),dim=-1)
# #             pSSA_vec[ss_element_i[1]] = (1-torch.max(P_not_i))+1e-8

# #         elif ss_element_i[0]=='P':
# #             pSSA_vec[1] = P_mat[ss_element_i[1],ss_element_i[2]]


# #     return pSSA_vec





# def compute_pSSA(ss_string, xyz, seq, 
#                 network_params_in=None,
#                 representative_res_ind=29, # RG has most atoms to reference
#                 frame_atom_0=" O4'", 
#                 frame_atom_1=" C1'", 
#                 frame_atom_2=" C2'",
#                 only_basepairs=False):




#     def find_any_bracket_pairs(s, open_symbol, close_symbol):
#         stack = []
#         pairs = []

#         for i, char in enumerate(s):
#             if char == open_symbol:
#                 stack.append(i)
#             elif char == close_symbol:
#                 if stack:
#                     pairs.append((stack.pop(), i))
#                 else:
#                     raise ValueError(f"No matching open parenthesis for closing parenthesis at index {i}")

#         if stack:
#             raise ValueError(f"No matching closing parenthesis for open parenthesis at index {stack[0]}")

#         return pairs

#     def find_loop_bases(s):
#         loop_bases = []
#         for i, char in enumerate(s):
#             if char == '.':
#                 loop_bases.append(i)

#         return loop_bases


#     open_symbols  = ['(','[','{','<','5','i','f','b']
#     close_symbols = [')',']','}','>','3','j','t','e']
#     loop_symbols  = ['.','&',','] # For now we just treating breaks as loops, cuz dssr is fallible.


#     all_pair_dict = {}
#     for pair_ind, (open_symbol, close_symbol) in enumerate(zip(open_symbols, close_symbols)):

#         num_opens = len([ _ for _ in ss_string if _ == open_symbol])
#         num_close = len([ _ for _ in ss_string if _ == close_symbol])
#         assert num_opens==num_close , "number of base pairs must be an even number... must be an error somewhere..."

#         all_pair_dict[pair_ind] = find_any_bracket_pairs(ss_string, open_symbol, close_symbol)


#     ss_element_list = []

#     for pair_ind in all_pair_dict.keys():
#         paired_base_list = all_pair_dict[pair_ind]
#         for i,j in paired_base_list:
#             ss_element_list.append(('P',i,j))


#     if not only_basepairs:
#         for i, char in enumerate(ss_string):
#             if char in loop_symbols: 
#                 ss_element_list.append(('L',i,i))


#     if network_params_in:
#         network_params = network_params_in
#     else:

#         network_params ={
#             'fc1_w': torch.tensor([[-0.2386, -0.3275, -0.3491,  0.4477,  0.8526, -3.0067, -1.5363],
#                                     [-0.1860,  0.3721, -0.1883,  0.1263,  0.5563, -2.2224, -1.4701],
#                                     [-0.1579,  0.2265, -1.0069,  0.3789,  0.4825, -2.0054, -1.1969]]),
#             'fc1_b': torch.tensor([-1.6262, -1.7848, -1.9741]),
#             'fc2_w': torch.tensor([[-1.5385, -0.2681, -0.1609],
#                                     [ 1.3806,  0.4991,  0.1363],
#                                     [ 0.0748,  0.0310,  0.4322]]),
#             'fc2_b': torch.tensor([ 1.2666, -1.4560,  0.0421]),
#             'fc_out_w': torch.tensor([[-1.3313,  1.1436,  0.1296]]),
#             'fc_out_b': torch.tensor([-1.1276]),
#         }


#     # Get frame atoms:
#     frame_atom_inds = torch.tensor([rf2aa.chemical.aa2long[representative_res_ind].index(frame_atom_0),
#                                     rf2aa.chemical.aa2long[representative_res_ind].index(frame_atom_1),
#                                     rf2aa.chemical.aa2long[representative_res_ind].index(frame_atom_2)])
#     frame_xyz = xyz[None,None,0,:,frame_atom_inds,:]


#     B, T, L = frame_xyz.shape[:3]

#     c6d = rf2aa.kinematics.xyz_to_c6d(frame_xyz[:,:,:,:3].view(B*T,L,3,3))
#     c6d = c6d.view(B, T, L, L, 4)

#     orien = torch.cat((torch.sin(c6d[...,1:]), torch.cos(c6d[...,1:])), dim=-1) # (B, T, L, L, 6)
#     t2d = torch.cat((c6d[...,0:1], orien), dim=-1)[0,...]
#     D_in = t2d.shape[-1] # input feature dimension
#     D_out = 1 # output feature dimension

#     # Prepare t2d feats for forward pass
#     x = t2d.reshape(B*L*L, D_in)

#     # Pass through the network
#     x = torch.matmul(x,network_params['fc1_w'].T) + network_params['fc1_b']
#     x = torch.matmul(x,network_params['fc2_w'].T) + network_params['fc2_b']
#     x = torch.matmul(x,network_params['fc_out_w'].T) + network_params['fc_out_b']

#     x = x.reshape(B, L, L, D_out)

#     P_mat = torch.sigmoid(x + x.permute(0,2,1,3))[0,:,:,0]  # Using sigmoid to get values between 0 and 1 (matrix of bp probabilities)

#     pSSA_vec = torch.zeros(L)
#     num_positions_counted = 0
#     for i, ss_element_i in enumerate(ss_element_list):

#         if ss_element_i[0]=='L':
#             P_not_i = torch.cat((P_mat[ss_element_i[1],:ss_element_i[2]], P_mat[ss_element_i[1],ss_element_i[2]+1:]),dim=-1)
#             pSSA_vec[ss_element_i[1]] = (1-torch.max(P_not_i))+1e-8

#         elif ss_element_i[0]=='P':
#             pSSA_vec[1] = P_mat[ss_element_i[1],ss_element_i[2]]


#     return pSSA_vec





def compute_pSSA(ss_string, xyz, seq, 
                network_params_in=None,
                representative_res_ind=29, # RG has most atoms to reference
                frame_atom_0=" O4'", 
                frame_atom_1=" C1'", 
                frame_atom_2=" C2'",
                only_basepairs=False):




    def find_any_bracket_pairs(s, open_symbol, close_symbol):
        stack = []
        pairs = []

        for i, char in enumerate(s):
            if char == open_symbol:
                stack.append(i)
            elif char == close_symbol:
                if stack:
                    pairs.append((stack.pop(), i))
                else:
                    raise ValueError(f"No matching open parenthesis for closing parenthesis at index {i}")

        if stack:
            raise ValueError(f"No matching closing parenthesis for open parenthesis at index {stack[0]}")

        return pairs

    def find_loop_bases(s):
        loop_bases = []
        for i, char in enumerate(s):
            if char == '.':
                loop_bases.append(i)
                
        return loop_bases


    open_symbols  = ['(','[','{','<','5','i','f','b']
    close_symbols = [')',']','}','>','3','j','t','e']
    loop_symbols  = ['.','&',','] # For now we just treating breaks as loops, cuz dssr is fallible.


    all_pair_dict = {}
    for pair_ind, (open_symbol, close_symbol) in enumerate(zip(open_symbols, close_symbols)):

        num_opens = len([ _ for _ in ss_string if _ == open_symbol])
        num_close = len([ _ for _ in ss_string if _ == close_symbol])
        assert num_opens==num_close , "number of base pairs must be an even number... must be an error somewhere..."
        
        all_pair_dict[pair_ind] = find_any_bracket_pairs(ss_string, open_symbol, close_symbol)
        

    ss_element_list = []
        
    for pair_ind in all_pair_dict.keys():
        paired_base_list = all_pair_dict[pair_ind]
        for i,j in paired_base_list:
            ss_element_list.append(('P',i,j))


    if not only_basepairs:
        for i, char in enumerate(ss_string):
            if char in loop_symbols: 
                ss_element_list.append(('L',i,i))


    if network_params_in:
        network_params = network_params_in
    else:

        network_params ={
            'fc1_w': torch.tensor([[-0.2386, -0.3275, -0.3491,  0.4477,  0.8526, -3.0067, -1.5363],
                                    [-0.1860,  0.3721, -0.1883,  0.1263,  0.5563, -2.2224, -1.4701],
                                    [-0.1579,  0.2265, -1.0069,  0.3789,  0.4825, -2.0054, -1.1969]]),
            'fc1_b': torch.tensor([-1.6262, -1.7848, -1.9741]),
            'fc2_w': torch.tensor([[-1.5385, -0.2681, -0.1609],
                                    [ 1.3806,  0.4991,  0.1363],
                                    [ 0.0748,  0.0310,  0.4322]]),
            'fc2_b': torch.tensor([ 1.2666, -1.4560,  0.0421]),
            'fc_out_w': torch.tensor([[-1.3313,  1.1436,  0.1296]]),
            'fc_out_b': torch.tensor([-1.1276]),
        }


    # Get frame atoms:
    frame_atom_inds = torch.tensor([rf2aa.chemical.aa2long[representative_res_ind].index(frame_atom_0),
                                    rf2aa.chemical.aa2long[representative_res_ind].index(frame_atom_1),
                                    rf2aa.chemical.aa2long[representative_res_ind].index(frame_atom_2)])
    frame_xyz = xyz[None,None,0,:,frame_atom_inds,:]


    B, T, L = frame_xyz.shape[:3]

    c6d = rf2aa.kinematics.xyz_to_c6d(frame_xyz[:,:,:,:3].view(B*T,L,3,3))
    c6d = c6d.view(B, T, L, L, 4)

    orien = torch.cat((torch.sin(c6d[...,1:]), torch.cos(c6d[...,1:])), dim=-1) # (B, T, L, L, 6)
    t2d = torch.cat((c6d[...,0:1], orien), dim=-1)[0,...]
    D_in = t2d.shape[-1] # input feature dimension
    D_out = 1 # output feature dimension

    # Prepare t2d feats for forward pass
    x = t2d.reshape(B*L*L, D_in)
    
    # Pass through the network
    x = torch.matmul(x,network_params['fc1_w'].T) + network_params['fc1_b']
    x = torch.matmul(x,network_params['fc2_w'].T) + network_params['fc2_b']
    x = torch.matmul(x,network_params['fc_out_w'].T) + network_params['fc_out_b']

    x = x.reshape(B, L, L, D_out)
    
    P_mat = torch.sigmoid(x + x.permute(0,2,1,3))[0,:,:,0]  # Using sigmoid to get values between 0 and 1 (matrix of bp probabilities)

    pSSA_vec = torch.zeros(L)
    num_positions_counted = 0
    for i, ss_element_i in enumerate(ss_element_list):

        if ss_element_i[0]=='L':
            P_not_i = torch.cat((P_mat[ss_element_i[1],:ss_element_i[2]], P_mat[ss_element_i[1],ss_element_i[2]+1:]),dim=-1)
            pSSA_vec[ss_element_i[1]] = (1-torch.max(P_not_i))+1e-8

        elif ss_element_i[0]=='P':
            pSSA_vec[1] = P_mat[ss_element_i[1],ss_element_i[2]]


    print('WARNING! THE FUNCTIONALITY OF OUR pSSA SYSTEM IS STILL SUB-OPTIMAL!!!')
    print('WARNING! THE FUNCTIONALITY OF OUR pSSA SYSTEM IS STILL SUB-OPTIMAL!!!')
    print('WARNING! THE FUNCTIONALITY OF OUR pSSA SYSTEM IS STILL SUB-OPTIMAL!!!')


    return pSSA_vec
    




