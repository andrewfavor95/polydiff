import copy
from collections import defaultdict

import torch
import assertpy

import aa_model
import rf2aa.util
import rf2aa.chemical

def set_nonexistant_atoms_to_nan(xyz, seq, H_exists=False):
    atom_mask = rf2aa.util.allatom_mask[seq]
    if not H_exists:
        atom_mask[:, rf2aa.chemical.NHEAVYPROT:] = False # no Hs
    xyz = xyz.clone()
    xyz[~atom_mask] = torch.nan
    return xyz

def atomize(indep, is_residue_atomized, atomize_H=False):
    assert not atomize_H, 'not supported'
    indep = copy.deepcopy(indep)

    assert not (is_residue_atomized * indep.is_sm).any(), 'cannot atomize small molecule atoms'
    atomizer = aa_model.AtomizeResidues(indep, is_residue_atomized)
    atom_mask = rf2aa.util.allatom_mask[indep.seq]
    atom_mask[:, rf2aa.chemical.NHEAVYPROT:] = False # no Hs
    atomizer.featurize_atomized_residues(atom_mask)
    indep_atomized, input_str_mask, input_seq_mask = atomizer.return_input_tensors()
    # is_sm = atomized_indices_res(atomizer, is_residue_atomized)
    indep_atomized.is_sm = rf2aa.util.is_atom(indep_atomized.seq)
    atomizer.atomized_length = indep_atomized.length()
    return indep_atomized, atomizer

def deatomize(atomizer, indep_atomized):
    indep = copy.deepcopy(atomizer.indep_initial_copy)
    seq_one_hot = torch.nn.functional.one_hot(indep_atomized.seq, rf2aa.chemical.NAATOKENS)
    deatomized_seq_one_hot, indep.xyz, indep.idx, indep.bond_feats, _ = atomizer.get_deatomized_features(
        seq_one_hot, indep_atomized.xyz)
    
    indep.seq = torch.argmax(deatomized_seq_one_hot, dim=-1)

    return indep

def atomized_indices_atoms(atomizer, atom_names_by_res):
    atom_idx_by_res = atomizer.get_atom_idx_by_res()
    named_i = []
    for res_i, atom_names in atom_names_by_res.items():
        assert isinstance(res_i, int), res_i
        atomized_residue_idxs = atom_idx_by_res[res_i]
        original_aa = atomizer.indep_initial_copy.seq[res_i]
        within_res_atom_idxs = {atom_name:i for i,atom_name in enumerate(e for e in rf2aa.chemical.aa2long[original_aa] if e is not None)}

        # Strip whitespace
        within_res_atom_idxs = {atom_name.strip():i for atom_name,i in within_res_atom_idxs.items()}
        atom_names = [a.strip() for a in atom_names]

        atom_i = atomized_residue_idxs[[within_res_atom_idxs[atom_name] for atom_name in atom_names]]
        named_i.extend(atom_i.tolist())
    return named_i

def atomized_indices_res(atomizer, mask):
    atomized_res_idx_from_res = atomizer.get_atomized_res_idx_from_res()
    atomized_res_idx = []
    mask_idx = torch.nonzero(mask)[:,0]
    for i in mask_idx:
        atomized_res_idx.append(atomized_res_idx_from_res[i.item()])
    return atomized_res_idx

def get_res_atom_name_by_atomized_idx(atomizer):
    '''
    Returns a dictionary mapping the index of an atom in the atomized protein
    to the original (0-index residue, atom_name) from pre-atomization.
    '''
    atomized_res_idx_from_res = atomizer.get_atom_idx_by_res()
    res_idx_atom_name_by_atomized_idx = {}
    for res_idx, atomized_res_idx in atomized_res_idx_from_res.items():
        original_aa = atomizer.indep_initial_copy.seq[res_idx]
        atom_name_by_within_res_idx = {i:atom_name for i,atom_name in enumerate(e for e in rf2aa.chemical.aa2long[original_aa] if e is not None)}
        for within_res_atom_idx, atom_idx in enumerate(atomized_res_idx):
            res_idx_atom_name_by_atomized_idx[atom_idx.item()] = (
                # f'{rf2aa.chemical.num2aa[original_aa]}{atomizer.indep_initial_copy.idx[res_idx]}'
                res_idx, atom_name_by_within_res_idx[within_res_atom_idx].strip()
            )
    return res_idx_atom_name_by_atomized_idx

def res_atom_name(atomizer, atomized_idx):
    '''
    Params:
        Indices of atoms in the atomized protein
    Returns:
        List o (0-index residue, atom_name) from pre-atomization.
    '''
    res_idx_atom_name_by_atomized_idx = get_res_atom_name_by_atomized_idx(atomizer)
    return [res_idx_atom_name_by_atomized_idx[i.item()] for i in atomized_idx]

def convert_atomized_mask(atomizer, mask):
    '''
    Params:
        atomizer: aa_model.AtomizeResidues
        mask: binary mask, the length of an atomized protein
    Returns:
        Dictionary mapping deatomized 0-indexed residues to the atom names corresponding to True in the mask, i.e.
            {0: ['CB', 'CG], 1: ['ALL'], ...}
    '''
    atomized_idx = mask.nonzero()[:,0]
    atomized_res_idx_from_res = atomizer.get_atomized_res_idx_from_res()
    res_idx_from_atomized_res_idx = {v:k for k,v in atomized_res_idx_from_res.items()}

    res_idx_atom_name_by_atomized_idx = get_res_atom_name_by_atomized_idx(atomizer)
    o = defaultdict(list)
    for atomized_i in atomized_idx.tolist():
        if atomized_i in res_idx_atom_name_by_atomized_idx:
            deatomized_i, atom_name = res_idx_atom_name_by_atomized_idx[atomized_i]
            o[deatomized_i].append(atom_name)

        elif atomized_i in res_idx_from_atomized_res_idx:
            deatomized_i = res_idx_from_atomized_res_idx[atomized_i]
            o[deatomized_i].append('ALL')
        else:
            raise Exception(f'{atomized_i} not found')
    return o


def atom_indices(atomizer, res_mask, atom_names_by_res):
    res_i = atomized_indices_res(atomizer, res_mask)
    atom_i = atomized_indices_atoms(atomizer, atom_names_by_res)
    assert set(res_i).isdisjoint(set(atom_i))
    return res_i + atom_i

def create_masks(atomizer, is_res_str_shown, is_atom_str_shown):

    is_atom_seq_shown = {res_i: [e for e in rf2aa.chemical.aa2long[atomizer.indep_initial_copy.seq[res_i]][:rf2aa.chemical.NHEAVYPROT] if e is not None]
                            for res_i in is_atom_str_shown.keys()}
    is_res_seq_shown = is_res_str_shown
    return create_masks_str_seq(atomizer, is_res_str_shown, is_res_seq_shown, is_atom_str_shown, is_atom_seq_shown)

def create_masks_str_seq(atomizer, is_res_str_shown, is_res_seq_shown, is_atom_str_shown, is_atom_seq_shown):
    L = atomizer.atomized_length
    str_shown_indices = atom_indices(atomizer, is_res_str_shown, is_atom_str_shown)
    seq_shown_indices = atom_indices(atomizer, is_res_seq_shown, is_atom_seq_shown)
    is_diffused = torch.ones(L).bool()
    is_masked_seq = torch.ones(L).bool()
    is_diffused[str_shown_indices] = False
    is_masked_seq[seq_shown_indices] = False

    return is_diffused, is_masked_seq

def atomize_and_mask(indep, is_res_str_shown, is_atom_str_shown):
    '''
    Params:
        is_atom_str_shown: map from 0-indexed residues to motif atom names:
            Example: {0: ['OD1', 'CB'], 4: ['CA']}
        indep: aa_model.Indep
    Returns:
        atomized_indep
        is_diffused
        is_masked_seq
    '''
    assertpy.assert_that(len(is_res_str_shown)).is_equal_to(indep.length())
    is_atomized = torch.zeros(indep.length()).bool()
    for k in is_atom_str_shown.keys():
        is_atomized[k] = True

    indep, atomizer = atomize(indep, is_atomized)
    is_diffused, is_masked_seq = create_masks(atomizer, is_res_str_shown, is_atom_str_shown)
    return indep, is_diffused, is_masked_seq, atomizer
