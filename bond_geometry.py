import itertools
from collections import OrderedDict, defaultdict
from itertools import combinations

import networkx as nx

import torch
from icecream import ic
import numpy as np
import scipy
import rf2aa.chemical

def calc_atom_bond_loss(indep, pred_xyz, is_diffused):
    """
    Loss on distances between bonded atoms
    """
    # Uncomment in future to distinguish between ligand / atomized_residue
    # is_residue = ~indep.is_sm
    # is_atomized = indep.is_sm & (indep.seq < rf2aa.chemical.NPROTAAS)
    # is_ligand = indep.is_sm & ~(indep.seq < rf2aa.chemical.NPROTAAS)
    mask_by_name = {}
    for k, v in {
        'residue': ~indep.is_sm,
        'atom': indep.is_sm,
    }.items():
        for prefix, mask in {
            'diffused': is_diffused,
            'motif': ~is_diffused
        }.items():
            mask_by_name[f'{prefix}_{k}'] = v*mask

    bond_losses = {}
    true_xyz = indep.xyz
    is_bonded = torch.triu(indep.bond_feats > 0)
    for (a, a_mask), (b, b_mask) in itertools.combinations_with_replacement(mask_by_name.items(), 2):
        is_pair = a_mask[..., None] * b_mask[None, ...]
        is_pair = torch.triu(is_pair)
        is_bonded_pair = is_bonded * is_pair
        i, j = torch.where(is_bonded_pair)
        
        true_dist = torch.norm(true_xyz[i,1]-true_xyz[j,1],dim=-1)
        pred_dist = torch.norm(pred_xyz[i,1]-pred_xyz[j,1],dim=-1)
        bond_losses[f'{a}:{b}'] = torch.mean(torch.abs(true_dist - pred_dist))
    return bond_losses


def find_all_rigid_groups(bond_feats):
    """
    Params:
        bond_feats: torch.tensor([N, N])
    
    Returns:
        list of tensors, where each tensor contains the indices of atoms within the same rigid group.
    """
    rigid_atom_bonds = (bond_feats>1)*(bond_feats<5)
    any_atom_bonds = bond_feats != 0
    rigid_edges = (rigid_atom_bonds.int() @ any_atom_bonds.int()).bool() + rigid_atom_bonds
    rigid_atom_bonds_np = rigid_edges.cpu().numpy()
    G = nx.from_numpy_array(rigid_atom_bonds_np)
    connected_components = nx.connected_components(G)
    connected_components = [cc for cc in connected_components if len(cc)>1]
    connected_components = [torch.tensor(list(cc)) for cc in connected_components]
    return connected_components

def align(xyz1, xyz2, eps=1e-6):

    # center to CA centroid
    xyz1_mean = xyz1.mean(0)
    xyz1 = xyz1 - xyz1_mean
    xyz2 = xyz2 - xyz2.mean(0)

    # Computation of the covariance matrix
    C = xyz2.T @ xyz1

    # Compute optimal rotation matrix using SVD
    V, S, W = np.linalg.svd(C)

    # get sign to ensure right-handedness
    d = np.ones([3,3])
    d[:,-1] = np.sign(np.linalg.det(V)*np.linalg.det(W))

    # Rotation matrix U
    U = (d*V) @ W

    # Rotate xyz2
    xyz2_ = xyz2 @ U

    return xyz2_ + xyz1_mean

def calc_rigid_loss(indep, pred_xyz, is_diffused, atomizer=None):
    '''
    Params:
        indep: atomized aa_mode.Indep corresponding to the true structure
        pred_xyz: atomized xyz coordinates [L, A, 3]
    Returns:
        Dictionary mapping the composition of a rigid group to the maximum aligned RMSD of the group to the true 
        coordinates in indep.
        i.e. {'diffused_atom_motif_atom': 5.5} implies that the worst predicted rigid group which has
        at least 1 diffused atom and at least 1 motif atom has an RMSD to the corresponding set of atoms
        in the true structure of 5.5.

        The suffix '_determined' is added to groups which contain >=3 motif atoms, as all DoFs of these
        groups are determined by the motif atoms.
    '''
    rigid_groups = find_all_rigid_groups(indep.bond_feats)
    if atomizer:
        named_rigid_groups = [atomize.res_atom_name(atomizer, g) for g in rigid_groups]

    mask_by_name = OrderedDict()
    for k, v in {
        'residue': ~indep.is_sm,
        'atom': indep.is_sm,
    }.items():
        for prefix, mask in {
            'diffused': is_diffused,
            'motif': ~is_diffused
        }.items():
            mask_by_name[f'{prefix}_{k}'] = v*mask


    atom_types = torch.full((indep.length(),), -1)
    for i, (mask_name, mask) in enumerate(mask_by_name.items()):
        atom_types[mask] = i
    mask_names = list(mask_by_name)

    is_motif_key = np.char.startswith(np.array(mask_names), 'motif')
    motif_keys = np.nonzero(is_motif_key)[0]

    dist_by_composition =  defaultdict(list)
    for rigid_idx in rigid_groups:
        composition = atom_types[rigid_idx].tolist()
        n_motif = np.in1d(np.array(composition), motif_keys).sum()
        composition = tuple(sorted(list(set(composition))))
        composition_string = ':'.join(mask_names[e] for e in composition)
        if n_motif >= 3:
            composition_string += '_determined'
        true_ca = indep.xyz[rigid_idx, 1]
        pred_ca = pred_xyz[rigid_idx, 1]
        # motif_rmsd = benchmark.util.af2_metrics.calc_rmsd(true_ca, pred_ca)
        # pred_ca_aligned = rf2aa.util.superimpose(pred_ca[None], true_ca[None])[0]
        pred_ca_aligned = align(true_ca, pred_ca)
        dists = torch.norm(pred_ca_aligned - true_ca, dim=-1)
        dist_by_composition[composition_string].append(dists.max())

    out = {k:max(v) for k,v in dist_by_composition.items()}
    return out
