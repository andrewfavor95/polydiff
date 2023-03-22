import itertools

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
