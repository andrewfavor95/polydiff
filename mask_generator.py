import random
import sys

import torch
import scipy.stats
import kinematics
import numpy as np
from icecream import ic
import rf2aa.util
import networkx as nx
from functools import wraps
import itertools
import assertpy
import ipdb

#####################################
# Misc functions for mask generation
#####################################

def sample_gaps(n, M):
    """
    Samples n chunks that sum to M. 

    Adapts below solution for getting close to uniform distibution of numbers: 
    https://stackoverflow.com/questions/2640053/getting-n-random-numbers-whose-sum-is-m/2640079#2640079
    """

    nums = np.random.dirichlet(np.ones(n))*M

    # now round to nearest integer, conserving the total sum
    rounded = []
    round_up = True
    for i in range(len(nums)):
        if round_up:
            rounded.append(np.ceil(nums[i]))
            round_up = False
        else:
            rounded.append(np.floor(nums[i]))
            round_up = True

    # ensure all > 0
    for i in range(len(rounded)):
        if rounded[i] < 1:
            rounded[i] = 1


    while sum(rounded) > M:
        ix = np.random.randint(0, len(rounded))
        if rounded[ix] < 2: # must be at least 1
            continue
        rounded[ix] -= 1

    while sum(rounded) != M:
        ix = np.random.randint(0, len(rounded))
        rounded[ix] += 1

    assert all([x >= 1 for x in rounded])
    return [int(x) for x in rounded]


def get_chunked_mask(xyz, low_prop, high_prop, max_motif_chunks=8):
    """
    Produces a mask of discontiguous protein chunks that are revealed. 
    Also produces a tensor indicating which chunks are given relative geom. info

    Parameters:
    -----------

    xyz (torch.tensor): (L, 3) tensor of protein coordinates

    low_prop (float): lower bound on proportion of protein that is masked

    high_prop (float): upper bound on proportion of protein that is masked

    """
    L = xyz.shape[0]
    # L = 100

    # decide number of chunks 
    n_motif_chunks = random.randint(2, max_motif_chunks)
    # decide what proportion of the protein is masked 

    # prop cannot result in n_unmasked < n_motif_chunks --> clamp high prop
    max_prop = 1-(n_motif_chunks+1)/L       # add +1 to be safe 
    high_prop = min(high_prop, max_prop)

    prop = random.uniform(low_prop, high_prop)
    n_masked   = int(L * prop)
    n_unmasked = L - n_masked

    # decide the length of each chunk by randomly sampling 
    # positions to cut a line with n_chunks - 1 cuts
    cuts    = sorted(random.sample(range(1, n_unmasked), n_motif_chunks - 1))
    lengths = [cuts[0]] + [cuts[i] - cuts[i-1] for i in range(1, len(cuts))] + [n_unmasked - cuts[-1]]
    # decide which chunks are given relative geom. info
    # walk over all unique pairs 
    motif_pairs = list(itertools.combinations(range(n_motif_chunks), 2))
    # 33% chance that a pair can see each other
    pairs_can_see = [random.choice([True, False, False]) for _ in range(len(motif_pairs))]

    # decide location of chunks within the protein
    # (1) decide order 
    random.shuffle(lengths)

    # (2) split available space remaining into other chunks between 
    #     the chunks that have been assigned a length
    ngap_low  = len(lengths) - 1
    ngap_high = ngap_low + 1
    ngap = random.randint(ngap_low, ngap_high)
    
    if ngap == (ngap_low): # there's no cterm/nterm gaps 
        Nterm_gap = False
        Cterm_gap = False
    elif ngap == (ngap_low + 1): # there's either a cterm or nterm gap
        Nterm_gap = random.choice([True, False])
        Cterm_gap = not Nterm_gap
    else: # there's both a cterm and nterm gap
        Nterm_gap = True
        Cterm_gap = True

    gaps = sample_gaps(ngap, n_masked) # gaps between unmasked chunks 
    random.shuffle(gaps)
    # print('This is gaps ', gaps)
    # print()
    # print('*********')


    chunks = []
    is_motif = []
    motif_ids = []
    cur_motif_id = 0
    
    if Nterm_gap:
        chunks.append(gaps.pop())
        is_motif.append(False)
        motif_ids.append(-1)

    
    for i in range(len(lengths)):
        chunks.append(lengths[i])
        is_motif.append(True)
        motif_ids.append(cur_motif_id)
        cur_motif_id += 1

        if len(gaps) > 1:                       # more to spare 
            chunks.append(gaps.pop())
            is_motif.append(False)
            motif_ids.append(-1)
        elif len(gaps) == 1 and not Cterm_gap:  # only one left, but no Cterm gap
            chunks.append(gaps.pop())
            is_motif.append(False)
            motif_ids.append(-1)
        else:
            pass # no more to spare

    if Cterm_gap:
        assert len(gaps) == 1
        chunks.append(gaps.pop())
        is_motif.append(False)
        motif_ids.append(-1)

    
    assert sum(chunks) == L, f'chunks sum to {sum(chunks)} but should sum to {L}'

    return chunks, is_motif, motif_ids, motif_pairs, pairs_can_see


def _get_diffusion_mask_chunked(xyz, prop_low, prop_high, max_motif_chunks=6):
    """
    More complicated masking strategy to create discontiguous chunks
    """

    chunks, chunk_is_motif, motif_ids, ij, ij_can_see = get_chunked_mask(xyz, prop_low, prop_high, max_motif_chunks)
    chunk_starts = np.cumsum([0] + chunks[:-1])
    chunk_ends   = np.cumsum(chunks)

    # make 1D array designating which chunks are motif
    L = xyz.shape[0]
    mask = torch.zeros(L, L)
    is_motif = torch.zeros(L)
    for i in range(len(chunks)):
        is_motif[chunk_starts[i]:chunk_ends[i]] = chunk_is_motif[i]


    # 2D array designating which chunks can see each other
    for i in range(len(chunks)):
        for j in range(len(chunks)):

            i_is_motif = chunk_is_motif[i]
            j_is_motif = chunk_is_motif[j]
            if (i_is_motif and j_is_motif): # both are motif, so possibly reveal info 
                
                
                ID_i = motif_ids[i]
                ID_j = motif_ids[j]
                assert ID_i != -1 and ID_j != -1, 'both motif but one has no ID'
                
                # always reveal self vs self 
                if i == j:
                    mask[chunk_starts[i]:chunk_ends[i], chunk_starts[j]:chunk_ends[j]] = 1
                
                else:
                    # find out of this (i,j) are allowed to see each other
                    ix = tuple(sorted([ID_i, ID_j]))
                    can_see = ij_can_see[ij.index(ix)]
                    if can_see:
                        mask[chunk_starts[i]:chunk_ends[i], chunk_starts[j]:chunk_ends[j]] = 1

    return mask.bool(), is_motif.bool()

def _get_diffusion_mask_chunked_na(xyz, is_na, prop_low, prop_high, 
                                    max_motif_chunks=6,
                                    na_fixed_intra=False,
                                    na_fixed_inter=False,
                                ):
    """
    More complicated masking strategy to create discontiguous chunks
    """

    def get_na_start_stop_inds(is_na):
        shifted_tensor = torch.roll(is_na, shifts=1, dims=0)
        diff_tensor = is_na != shifted_tensor
        change_indices = torch.where(diff_tensor)[0]
        start_stop_indices = [(start.item(), stop.item()) for start, stop in zip(change_indices[:-1], change_indices[1:])]
        if is_na[0]:
            start_stop_indices = [(0, change_indices[0].item())] + start_stop_indices
        if is_na[-1]:
            start_stop_indices.append((change_indices[-1].item(), len(is_na)))

        values_list = []
        for start, stop in start_stop_indices:
            value = is_na[start].item()
            values_list.append(value)

        return start_stop_indices, values_list

    # Subselect region of xyz to chunk:
    if na_fixed_intra:
        na_chunk_inds, chunk_is_na = get_na_start_stop_inds(is_na)
        chunks, chunk_is_motif, motif_ids, ij, ij_can_see = get_chunked_mask(xyz[~is_na], 
                                                                            prop_low, 
                                                                            prop_high, 
                                                                            max_motif_chunks)

        L = xyz[~is_na].shape[0]

    else:
        # xyz_to_chunk = xyz.clone()
        chunks, chunk_is_motif, motif_ids, ij, ij_can_see = get_chunked_mask(xyz,
                                                                            prop_low, 
                                                                            prop_high, 
                                                                            max_motif_chunks)
        L = xyz.shape[0]


    chunk_starts = np.cumsum([0] + chunks[:-1])
    chunk_ends   = np.cumsum(chunks)



    # make 1D array designating which chunks are motif
    # L = xyz.shape[0]
    mask = torch.zeros(L, L)
    is_motif = torch.zeros(L)
    for i in range(len(chunks)):
        is_motif[chunk_starts[i]:chunk_ends[i]] = chunk_is_motif[i]


    # 2D array designating which chunks can see each other
    for i in range(len(chunks)):
        for j in range(len(chunks)):

            i_is_motif = chunk_is_motif[i]
            j_is_motif = chunk_is_motif[j]
            if (i_is_motif and j_is_motif): # both are motif, so possibly reveal info 
                
                ID_i = motif_ids[i]
                ID_j = motif_ids[j]
                assert ID_i != -1 and ID_j != -1, 'both motif but one has no ID'
                
                # always reveal self vs self 
                if i == j:
                    mask[chunk_starts[i]:chunk_ends[i], chunk_starts[j]:chunk_ends[j]] = 1
                
                else:
                    # find out of this (i,j) are allowed to see each other
                    ix = tuple(sorted([ID_i, ID_j]))
                    can_see = ij_can_see[ij.index(ix)]
                    if can_see:
                        mask[chunk_starts[i]:chunk_ends[i], chunk_starts[j]:chunk_ends[j]] = 1

    if na_fixed_intra:
        full_L = xyz.shape[0]
        full_mask = torch.zeros(full_L, full_L)
        full_is_motif = torch.zeros(full_L)

        non_aa_indices = torch.nonzero(~is_na).squeeze()
        full_mask[non_aa_indices[:, None], non_aa_indices] = mask
        full_is_motif[non_aa_indices] = is_motif

        for i in range(len(chunk_is_na)):
            for j in range(len(chunk_is_na)):
                i_is_na = chunk_is_na[i]
                j_is_na = chunk_is_na[j]

                if i_is_na or j_is_na:
                    if i == j : # just fill the internal chunks with NA motif mask
                        full_mask[na_chunk_inds[i][0]:na_chunk_inds[i][1],na_chunk_inds[j][0]:na_chunk_inds[j][1]] = 1

                    elif na_fixed_inter: # fill off-diagonals with NA motif mask
                        full_mask[na_chunk_inds[i][0]:na_chunk_inds[i][1],na_chunk_inds[j][0]:na_chunk_inds[j][1]] = 1
                    
        return full_mask.bool(), full_is_motif.bool()

    else:
        return mask.bool(), is_motif.bool()


def get_diffusion_mask_chunked(indep, atom_mask, low_prop, high_prop, broken_prop, max_motif_chunks=6):
    # wrapper to accomodate indep/atom mask input 
    assert indep.is_sm.sum() == 0, 'small molecules not supported currently for this masking'

    if indep.is_na.any(): # IF we have nucleic acids, we can choose to process differently
        mask2d, is_motif = _get_diffusion_mask_chunked_na(indep.xyz, indep.is_na, low_prop, high_prop, 
                                                            max_motif_chunks=max_motif_chunks,
                                                            na_fixed_intra=indep.na_fixed_intra,
                                                            na_fixed_inter=indep.na_fixed_inter)

    else:
        mask2d, is_motif = _get_diffusion_mask_chunked(indep.xyz, low_prop, high_prop, max_motif_chunks)
    
    # spoofing a return of two items: "diffusion_mask, is_atom_motif"
    return (mask2d, is_motif), None


def get_masks(L, min_length, max_length, min_flank, max_flank):
    """
    Makes a random contiguous mask, with (or without) flanking residues masked.
    """
    flank_width = random.randint(min_flank, max_flank)
    max_length = min([max_length, L-2*flank_width - 20]) #require at least 20 visible residues in any masking regime.
    central_width = random.randint(min_length,max_length)
    assert central_width > min_length - 1
    assert max_length > min_length
    
    start = random.randint(flank_width,L-flank_width-central_width-1)
    return (start,start+central_width),flank_width


def get_diffusion_pos(L,min_length, max_length=None):
    """
    Random contiguous mask generation to denote which residues are being diffused 
    and which are not. 

    TODO: This does not support multi-chain diffusion training at the moment 

    Returns:

        start,end : indices between which residues are allowed to be diffused. 
                    Otherwise, residues are held fixed and revealed 
    """
    if (max_length is None) or (max_length > L):
        max_length = L 

    assert min_length <= max_length 

    # choose a length to crop 
    chosen_length = np.random.randint(min_length, max_length)

    # choose a start position - between 0 (inclusive) and L-chosen_length (exclusive)
    start_idx = random.randint(0, L-chosen_length)
    end_idx   = start_idx + chosen_length

    return start_idx, end_idx 

def get_cb_distogram(xyz):
    Cb = kinematics.get_Cb(xyz)
    dist = kinematics.get_pair_dist(Cb, Cb)
    return dist

def get_contacts(xyz, xyz_less_than=5, seq_dist_greater_than=10):
    L = xyz.shape[0]
    dist = get_cb_distogram(xyz)

    is_close_xyz = dist < xyz_less_than

    idx = torch.ones_like(dist).nonzero()
    seq_dist = torch.abs(torch.arange(L)[None] - torch.arange(L)[:,None])
    is_far_seq = torch.abs(seq_dist) > seq_dist_greater_than

    contacts = is_far_seq * is_close_xyz
    return contacts

def sample_around_contact(L, indices, len_low, len_high):
    diffusion_mask = torch.zeros(L).bool()
    for anchor in indices:
        mask_length = int(np.floor(random.uniform(len_low, len_high)))
        l = anchor - mask_length // 2
        r = anchor + (mask_length - mask_length//2)
        l = max(0, l)
        r = min(r, L)
        diffusion_mask[l:r] = True
    return diffusion_mask


def _get_double_contact(xyz, low_prop, high_prop, broken_prop, xyz_less_than=5, seq_dist_greater_than=25, len_low=5, len_high=10):
    contacts = get_contacts(xyz, xyz_less_than, seq_dist_greater_than)
    if not contacts.any():
        return _get_diffusion_mask_simple(xyz, low_prop, high_prop, broken_prop)
    contact_idxs = contacts.nonzero()
    contact_idx = np.random.choice(np.arange(len(contact_idxs)))
    indices = contact_idxs[contact_idx]
    L = xyz.shape[0]
    return sample_around_contact(L, indices, len_low, len_high)

def find_third_contact(contacts):
    contact_idxs = contacts.nonzero()
    contact_idxs = contact_idxs[torch.randperm(len(contact_idxs))]
    for i,j in contact_idxs:
        if j < i:
            continue
        K = (contacts[i,:] * contacts[j,:]).nonzero()
        if len(K):
            K = K[torch.randperm(len(K))]
            for k in K:
                return torch.tensor([i,j,k])
    return None

def get_sm_contacts(
        indep, atom_mask,
    d_beyond_closest = 1.5,
    n_beyond_closest = 2,
    n_sample_low = 1,
    n_sample_high = 8, **kwargs):

    xyz, is_sm = indep.xyz, indep.is_sm

    assert len(xyz.shape) == 3
    assert is_sm.any()

    L = xyz.shape[0]
    L_prot = (~is_sm).sum()
    n_sample = np.random.randint(n_sample_low, n_sample_high)

    crds = torch.clone(xyz)
    crds[~atom_mask] = torch.nan
    prot_crds = crds[~is_sm]
    sm_crds = crds[is_sm]
    dist = (prot_crds[:, None] - sm_crds[ None]).pow(2).sum(dim=-1).sqrt()
    dist = dist.nan_to_num(99999)
    dist = dist.min(dim=-1)[0].min(dim=-1)[0]
    dist_cutoff = dist.min() + d_beyond_closest

    is_sampled = torch.zeros(L_prot).bool()
    _, closest_idx = torch.topk(dist, n_sample + n_beyond_closest, largest=False)
    is_sampled[closest_idx] = True
    is_sampled[dist < dist_cutoff] = True

    is_sampled_het = torch.zeros(L).bool()
    is_sampled_het[~is_sm] = is_sampled

    candidate_indices = is_sampled_het.nonzero().flatten()
    indices = np.random.choice(candidate_indices, n_sample, replace=False)
    is_motif = torch.zeros(L).bool()
    is_motif[is_sm] = True
    is_motif[indices] = True

    # Verification
    picked = crds[is_motif]
    dist_conf = (picked[:, None] - sm_crds[ None]).pow(2).sum(dim=-1).sqrt()
    dist_conf = dist_conf.nan_to_num(9999)
    picked_distances = dist_conf.min(-1)[0].min(-1)[0]
    #ic(is_motif, n_sample, picked_distances, dist_cutoff, indices)

    return is_motif, None

def get_triple_contact_atomize(*args, **kwargs):
    raise Exception('not implemented')

def get_closest_tip_atoms(indep, atom_mask,
    d_beyond_closest = 1.0,
    n_beyond_closest = 1,
    n_sample_low = 1,
    n_sample_high = 5, **kwargs):

    assert len(indep.xyz.shape) == 3
    assert indep.is_sm.any()

    L = indep.length()
    L_prot = (~indep.is_sm).sum()
    n_sample = np.random.randint(n_sample_low, n_sample_high)

    crds = torch.clone(indep.xyz)
    crds[~atom_mask] = torch.nan
    prot_crds = crds[~indep.is_sm]
    sm_crds = crds[indep.is_sm][:, 1]
    # ic(prot_crds.shape)
    # ic(sm_crds.shape)
    dist_res_sidechain_ligand = (prot_crds[:,:, None,...] - sm_crds[ None,None,...]).pow(2).sum(dim=-1).sqrt()
    # ic(dist_res_sidechain_ligand.shape)
    dist_res_sidechain_ligand = dist_res_sidechain_ligand.nan_to_num(9999)
    dist_res_sidechain = dist_res_sidechain_ligand.min(dim=-1)[0]
    dist = dist_res_sidechain.min(dim=-1)[0]
    is_valid_for_atomization = indep.is_valid_for_atomization(atom_mask)[~indep.is_sm]
    # ic(is_valid_for_atomization.sum())
    if not is_valid_for_atomization.any():
        ic('No valid residues for atomization, falling back to unconditional generation')
        return torch.zeros(L).bool(), None
    dist[~is_valid_for_atomization] = 9999

    # Calculate distance cutoff
    dist_cutoff = dist.min() + d_beyond_closest
    is_sampled = torch.zeros(L_prot).bool()
    _, closest_idx = torch.topk(dist, n_sample + n_beyond_closest, largest=False)
    is_sampled[closest_idx] = True
    is_sampled[dist < dist_cutoff] = True
    n_contacts_before = is_sampled.sum()
    is_sampled[~is_valid_for_atomization] = False
    n_contacts_after = is_sampled.sum()
    #ic(f'After removing residue contacts with unresolved heavy atoms: {n_contacts_before} --> {n_contacts_after}')

    is_sampled_het = torch.zeros(L).bool()
    is_sampled_het[~indep.is_sm] = is_sampled
    candidate_indices = is_sampled_het.nonzero().flatten()

    n_sample = min(n_sample, len(candidate_indices))
    print(f'choosing {n_sample} out of {len(candidate_indices)}')
    indices = np.random.choice(candidate_indices, n_sample, replace=False)

    # Verification for debugging
    if False:
        picked = crds[indices]
        dist_conf = (picked[:, None] - sm_crds[ None]).pow(2).sum(dim=-1).sqrt()
        dist_conf = dist_conf.nan_to_num(9999)
        picked_distances = dist_conf.min(-1)[0].min(-1)[0]
        ic(picked_distances, dist_cutoff, indices)

    is_atom_diffused = {}
    sm_prot_transition_types = (indep.is_sm[1:].int() - indep.is_sm[:-1].int()).unique().tolist()
    # If the ligands do not come in a single block after the protein, using dist_sc_to_sm will provide incorrect indices,
    assertpy.assert_that(sm_prot_transition_types).is_equal_to([0,1])
    # prot_by_het = torch.full((indep.length(),), torch.nan)
    # prot_by_het[~indep.is_sm] = torch.arange((~indep.is_sm).sum())
    # torch.nonzero(~indep.is_sm).flatten()
    for het_i in indices:
        # prot_i = prot_by_het[het_i]
        prot_i = het_i
        closest_atom = torch.argmin(dist_res_sidechain[prot_i]).item()
        n_bonds = np.random.randint(1, 3)
        is_atom_diffused[het_i] = get_atom_names_within_n_bonds(indep.seq[het_i], closest_atom, n_bonds)
    is_motif = torch.zeros(L).bool()
    is_motif[indep.is_sm] = True

    return is_motif, is_atom_diffused


def get_atom_names_within_n_bonds(res, source_node, n_bonds):
    bond_feats = get_residue_bond_feats(res)
    bond_graph = nx.from_numpy_matrix(bond_feats.numpy())
    paths = nx.single_source_shortest_path_length(bond_graph, source=source_node,cutoff=n_bonds)
    atoms_within_n_bonds = paths.keys()
    atom_names = [rf2aa.chemical.aa2long[res][i] for i in atoms_within_n_bonds]
    ic(atom_names)
    return atom_names

def tip_crd(indep, i):
    '''Returns the coordinates of the tip atom of residue index i'''
    aa = indep.seq[i]
    tip_atom_name = rf2aa.chemical.aa2tip[aa].strip()
    tip_idx_within_res = next(i for i, atom_name in enumerate(rf2aa.chemical.aa2long[aa]) if atom_name.strip() == tip_atom_name)
    return indep.xyz[i, tip_idx_within_res]

def get_tip_gaussian_mask(indep, atom_mask, *args, std_dev=8, **kwargs):
    '''
    Params:
        indep: aa_model.Indep, a description of a protein complex
        atom_mask: [L, 36] mask of whether an atom is resolved in indep
        std_dev: standard deviation of the multivariate gaussian (see below)
        *args: ignored, necessary to match masker function signature
        **kwargs: ignored, necessary to match masker function signature
    Returns:
        is_motif: binary mask that is True where a non-atomized residue is motif
        is_atom_motif: dictionary mapping residue indices to the atom names which are motif
    
    This masking function provides a few partial sidechains as motif.

    The protocol for selecting those sidechains is as follows:
        1. Find all atomizable residue
        2. Select one at random, call it origin
        3. Sample 1-6 atomizable residues with probabilities given by evaluation of a
            multivariate gaussian centered at origin at the tips of the atomizable residues
        4. Select a random atom in each residue, weighted towards selecting the tip
        5. Expand the mask starting from each selected atom by traversing 1-3 bonds within the residue.
    '''
    assert not indep.is_sm.any()
    is_valid_for_atomization = indep.has_heavy_atoms_and_seq(atom_mask)
    if not is_valid_for_atomization.any():
        ic('No valid residues for atomization, falling back to unconditional generation')
        is_motif = torch.zeros(indep.length()).bool()
        is_motif[indep.is_sm] = True
        return is_motif, None
    valid_idx = is_valid_for_atomization.nonzero()[:,0]
    
    origin_i = np.random.choice(valid_idx, 1)[0]
    origin_tip_crd = tip_crd(indep, origin_i)
    tip_crds = [tip_crd(indep, i) for i in valid_idx]
    tip_crds = np.stack(tip_crds, axis=0)
    gaussian = scipy.stats.multivariate_normal(origin_tip_crd, std_dev)
    probs = gaussian.pdf(tip_crds)
    probs /= probs.sum()
    n_atomize = random.randint(1, 6)
    n_atomize = min(n_atomize, len(valid_idx))
    atomize_i = np.random.choice(valid_idx, n_atomize, p=probs, replace=False)

    is_atom_motif = {}
    for i in atomize_i:
        atom_crds = indep.xyz[i][atom_mask[i]]
        closest_atom_i = torch.argmin(torch.norm(atom_crds - origin_tip_crd), axis=-1)
        n_atoms = len(atom_crds)
        prob_non_closest = 0.5 / (n_atoms-1)
        probs = np.full((n_atoms,), prob_non_closest)
        probs[closest_atom_i] = 0.5
        p_tip_only = 0.5
        if np.random.rand() < p_tip_only:
            probs[:4] = 1e-6
        probs = probs.astype('float64')
        probs /= probs.sum()
        seed_atom = np.random.choice(np.arange(n_atoms), 1, p=probs)[0]
        n_bonds = np.random.randint(1, 3)
        atom_names = get_atom_names_within_n_bonds(indep.seq[i], seed_atom, n_bonds)
        assertpy.assert_that(atom_names).does_not_contain(None)
        is_atom_motif[i] = atom_names

    is_motif = torch.zeros(indep.length()).bool()
    return is_motif, is_atom_motif


def _get_triple_contact(xyz, low_prop, high_prop, broken_prop, xyz_less_than=6, seq_dist_greater_than=10, len_low=1, len_high=3):
    contacts = get_contacts(xyz, xyz_less_than, seq_dist_greater_than)
    if not contacts.any():
        return _get_diffusion_mask_simple(xyz, low_prop, high_prop, broken_prop)
    indices = find_third_contact(contacts)
    if indices is None:
        return _get_diffusion_mask_simple(xyz, low_prop, high_prop, broken_prop)
    L = xyz.shape[0]
    return sample_around_contact(L, indices, len_low, len_high)


def _get_triple_contact_3template(xyz, 
                                  low_prop, 
                                  high_prop, 
                                  xyz_less_than=6, 
                                  seq_dist_greater_than=10, 
                                  len_low=1, 
                                  len_high=7):

    contacts = get_contacts(xyz, xyz_less_than, seq_dist_greater_than)
    if not contacts.any():
        return _get_diffusion_mask_chunked(xyz, low_prop, high_prop, max_motif_chunks=6)

    # find the third contact
    indices = find_third_contact(contacts)
    if indices is None:
        return _get_diffusion_mask_chunked(xyz, low_prop, high_prop, max_motif_chunks=6)
    
    L = xyz.shape[0]
    # 1d tensor describing which residues are motif
    is_motif = sample_around_contact(L, indices, len_low, len_high)

    # now get the 2d tensor describing which residues can see each other
    # For these, all motif chunks can see each other
    mask_2d = is_motif[:, None] * is_motif[None, :]

    return mask_2d, is_motif

def _get_triple_contact_3template_na(xyz, 
                                    is_na,
                                    low_prop, 
                                    high_prop, 
                                    xyz_less_than=6, 
                                    seq_dist_greater_than=10, 
                                    len_low=1, 
                                    len_high=7,
                                    na_fixed_intra=False,
                                    na_fixed_inter=False,
                                  ):

    contacts = get_contacts(xyz, xyz_less_than, seq_dist_greater_than)
    if not contacts.any():
        return _get_diffusion_mask_chunked_na(xyz, is_na, low_prop, high_prop, 
                                            max_motif_chunks=6,
                                            na_fixed_intra=False,
                                            na_fixed_inter=False,
                                        )

    # find the third contact
    indices = find_third_contact(contacts)
    if indices is None:
        return _get_diffusion_mask_chunked_na(xyz, is_na, low_prop, high_prop, 
                                                max_motif_chunks=6,
                                                na_fixed_intra=False,
                                                na_fixed_inter=False,
                                        )
    
    L = xyz.shape[0]
    # 1d tensor describing which residues are motif
    is_motif = sample_around_contact(L, indices, len_low, len_high)
    # now get the 2d tensor describing which residues can see each other
    # For these, all motif chunks can see each other
    mask_2d = is_motif[:, None] * is_motif[None, :]


    # Now, modify to force nucleic info to be passed through, 
    # depending on input params:
    na_mask_inter = torch.logical_xor(is_na[:,None], is_na[None,:])
    na_mask_intra = torch.logical_and(is_na[:,None], is_na[None,:])
    if na_fixed_intra:
        mask_2d = torch.logical_or(mask_2d, na_mask_intra)
        is_motif = torch.logical_or(is_motif, is_na)

    if na_fixed_inter:
        mask_2d = torch.logical_or(mask_2d, na_mask_inter)

    return mask_2d, is_motif


def _get_multi_triple_contact_3template(xyz,
                                         low_prop,
                                         high_prop,
                                         max_triples=2,
                                         xyz_less_than=6,
                                         seq_dist_greater_than=10,
                                         len_low=1,
                                         len_high=7):
    """
    Gets 2d mask + 1d is motif for multiple triple contacts. 
    """
    
    contacts = get_contacts(xyz, xyz_less_than, seq_dist_greater_than)

    if not contacts.any():
        print('***returning simple diffusion mask')
        return _get_diffusion_mask_chunked(xyz, low_prop, high_prop, max_motif_chunks=6)

    is_motif_stack = []
    mask_2d_stack = []
    n_triples = random.randint(1, max_triples)
    for i in range(n_triples): 
        indices = find_third_contact(contacts)
        if indices is None:
            print('***returning simple diffusion mask')
            return _get_diffusion_mask_chunked(xyz, low_prop, high_prop, max_motif_chunks=6)
        L = xyz.shape[0]
        # 1d tensor describing which residues are motif
        tmp_is_motif = sample_around_contact(L, indices, len_low, len_high)
        # now get the 2d tensor describing which residues can see each other
        # For these, all motif chunks can see each other
        tmp_mask_2d = tmp_is_motif[:, None] * tmp_is_motif[None, :]

        is_motif_stack.append(tmp_is_motif)
        mask_2d_stack.append(tmp_mask_2d)

    is_motif = torch.stack(is_motif_stack, dim=0).bool()
    mask_2d = torch.stack(mask_2d_stack, dim=0).bool()

    is_motif = torch.any(is_motif, dim=0)
    mask_2d = torch.any(mask_2d, dim=0)

    return mask_2d, is_motif




def get_triple_contact_3template(indep, 
                                 atom_mask, 
                                 low_prop, 
                                 high_prop, 
                                 broken_prop):
    """
    Triple contact fxn that is compatable w/ 3template mode
    """
    assert indep.is_sm.sum() == 0, 'small molecules not yet supported'

    if indep.is_na.any(): # IF we have nucleic acids, we can choose to process differently
        mask2d, is_motif = _get_triple_contact_3template_na(indep.xyz, indep.is_na, low_prop, high_prop, 
                                                    na_fixed_intra=indep.na_fixed_intra,
                                                    na_fixed_inter=indep.na_fixed_inter)

    else:
        mask2d, is_motif = _get_triple_contact_3template(indep.xyz, low_prop, high_prop)

    # spoofing a return of two items: "diffusion_mask, is_atom_motif"
    return (mask2d, is_motif), None


def get_multi_triple_contact_3template(indep,
                                        atom_mask,
                                        low_prop,
                                        high_prop,
                                        broken_prop,
                                        max_triples=2):
    """
    Get multiple triple contacts
    """
    assert indep.is_sm.sum() == 0, 'small molecules not yet supported'
    mask2d, is_motif = _get_multi_triple_contact_3template(indep.xyz, low_prop, high_prop, max_triples=max_triples)

    # spoofing a return of two items: "diffusion_mask, is_atom_motif"
    return (mask2d, is_motif), None


 
def _get_diffusion_mask_simple(xyz, low_prop, high_prop, broken_prop):
    """
    Function to make a diffusion mask.
    Options:
        low_prop - lower bound on the proportion of the protein masked
        high_prop - upper bound on the proportion of the protein masked
        broken_prop - proportion of the time the mask is in the middle (broken motif), vs at the ends
    Output:
        1D diffusion mask. True is unmasked, False is masked/diffused
    """
    L = xyz.shape[0]
    diffusion_mask = torch.ones(L).bool()
    if L <= 3:
        # Too small to mask
        return torch.zeros(L).bool()
    mask_length = int(np.floor(random.uniform(low_prop, high_prop) * L))
    # decide if mask goes in the middle or the ends
    if random.uniform(0,1) < broken_prop or mask_length < 3:
        high_start = L-mask_length-1
        start = random.randint(0, high_start)
        diffusion_mask[start:start+mask_length] = False
    else:
        # split mask in two
        split = random.randint(1, mask_length-2)
        diffusion_mask[:split] = False
        diffusion_mask[-(mask_length-split):] = False
    return diffusion_mask

def _get_unconditional_diffusion_mask(xyz, *args, **kwargs):
    """
    unconditional generation of proteins, if a small molecule is present it will be given as context
    """
    L = xyz.shape[0]
    is_motif = torch.zeros(L).bool()
    return is_motif


def make_sm_compatible(get_mask):
    @wraps(get_mask)
    def out_get_mask(indep, atom_mask, *args, **kwargs):
        diffusion_mask = get_mask(indep.xyz[~indep.is_sm], *args, **kwargs)
        L = indep.length()
        diffusion_mask = torch.ones(L).bool()
        diffusion_mask_prot = get_mask(indep.xyz[~indep.is_sm], *args, **kwargs)
        diffusion_mask[~indep.is_sm] = diffusion_mask_prot
        return diffusion_mask, None
    return out_get_mask


def make_atomized(get_mask, min_atomized_residues=1, max_atomized_residues=5):
    @wraps(get_mask)
    def out_get_mask(indep, atom_mask, *args, **kwargs):
        is_motif, is_atom_motif = get_mask(indep, atom_mask, *args, **kwargs)
        assert is_atom_motif is None, 'attempting to atomize a masking function that is already returning atomization masks'
        can_be_atomized = is_motif * indep.is_valid_for_atomization(atom_mask)
        if not can_be_atomized.any():
            return is_motif, None
        atomize_indices = torch.nonzero(can_be_atomized).flatten()
        n_sample = random.randint(min_atomized_residues, max_atomized_residues)
        n_sample = min(len(atomize_indices), n_sample)
        atomize_indices = np.random.choice(atomize_indices, n_sample, replace=False)
        is_atom_motif = {i:choose_contiguous_atom_motif(indep.seq[i]) for i in atomize_indices}
        is_motif[atomize_indices] = False
        return is_motif, is_atom_motif
    return out_get_mask

get_diffusion_mask_simple = make_sm_compatible(_get_diffusion_mask_simple)
get_triple_contact = make_sm_compatible(_get_triple_contact)
get_double_contact = make_sm_compatible(_get_double_contact)
atomize_get_triple_contact = make_atomized(get_triple_contact)
atomize_get_double_contact = make_atomized(get_double_contact)
get_unconditional_diffusion_mask = make_sm_compatible(_get_unconditional_diffusion_mask)

sm_mask_fallback = {
    get_closest_tip_atoms: get_tip_gaussian_mask
}


def _get_sm_contact_3template(xyz, 
                              is_sm, 
                              low_prop, 
                              high_prop, 
                              contact_cut=8,
                              chunk_size_min=1,
                              chunk_size_max=7,
                              min_seq_dist=9): 
    """
    Produces mask2d and is_motif for small molecule, possibly with contacting protein chunks
    """
    print('Entered _get_sm_contact_3template')
    assert len(xyz.shape) == 3
    ca = xyz[~is_sm, 1,:]
    
    if ca.shape[0] == 0:
        sm_only = True 
    else:
        sm_only = False

    sm_xyz  = xyz[is_sm, 1,:]

    dmap = torch.cdist(ca, sm_xyz)
    dmap = dmap < contact_cut   
    protein_is_contacting = dmap.any(dim=-1) # which CA's are contacting sm 
    where_is_contacting = protein_is_contacting.nonzero().squeeze()

    
    n_chunk_revealed = random.randint(0,4)

    if (n_chunk_revealed == 0) or (sm_only):
        is_motif = is_sm.clone()
        is_motif_2d = is_motif[:, None] * is_motif[None, :]
        return is_motif_2d, is_motif
    
    else: 
        is_motif = is_sm.clone()
        cur_min_seq_dist = min_seq_dist # could possibly increment this if needed 

        for i in range(n_chunk_revealed):
            print('on chunk ',i) 
            chunk_size = torch.randint(chunk_size_min, chunk_size_max, size=(1,)).item()
            
            if len(where_is_contacting.shape) == 0:
                # ensures where_is_contacting is a 1d tensor
                where_is_contacting = where_is_contacting.unsqueeze(0)

            p = torch.ones_like(where_is_contacting)/len(where_is_contacting)
            chosen_idx = p.multinomial(num_samples=1, replacement=False)
            chosen_idx = chosen_idx.item()
            chosen_idx = where_is_contacting[chosen_idx]

            # find min and max indices for revealed chunk
            min_index = max(0, chosen_idx - chunk_size//2)
            max_index = min(protein_is_contacting.numel(), 1+chosen_idx + chunk_size//2)
            # reveal chunk 
            is_motif[min_index:max_index] = True

            # update where_is_contacting
            start = max(0,min_index-cur_min_seq_dist)
            end = min(protein_is_contacting.numel(), max_index+cur_min_seq_dist)
            protein_is_contacting[start:end] = False # remove this option from where_is_contacting 
            
            where_is_contacting = protein_is_contacting.nonzero().squeeze()
            
            if protein_is_contacting.sum() == 0:
                break # can't make any more chunks

        
        is_motif_2d = is_motif[:, None] * is_motif[None, :]

        return is_motif_2d, is_motif
    

def _get_na_contact_3template(xyz, 
                              is_na, 
                              low_prop, 
                              high_prop, 
                              contact_cut=8, # Should I increase size?
                              chunk_size_min=1,
                              chunk_size_max=7,
                              max_num_chunks=4,
                              na_fixed_intra=False, # is full na struct given with fixed internal structure?
                              na_fixed_inter=False, # is full na struct given with fixed relative orientation?
                              min_seq_dist=9,
                              ): 
    """
    Produces mask2d and is_motif for nucleic acids, possibly with contacting protein chunks
    """
    print('Entered _get_na_contact_3template')
    assert len(xyz.shape) == 3
    
    ca = xyz[~is_na, 1,:]
    
    if ca.shape[0] == 0:
        na_only = True 
    else:
        na_only = False

    na_xyz  = xyz[is_na, 1,:] # Select the P atoms (TODO: CHECK IF THIS IS THE BEST ATOM TO USE HERE)

    dmap = torch.cdist(ca, na_xyz)
    dmap = dmap < contact_cut   
    protein_is_contacting = dmap.any(dim=1) # which protein CA's are contacting na P's
    nucleic_is_contacting = dmap.any(dim=0) # which protein CA's are contacting na P's

    if na_fixed_intra: # IF give fixed nucleic structure, then only look at protein contact regions
        chains_are_contacting = protein_is_contacting.clone()
    else: #otherside, we can chunk up at any part of the complex
        chains_are_contacting = torch.concatenate((protein_is_contacting,nucleic_is_contacting))
    
    where_is_contacting = chains_are_contacting.nonzero().squeeze()

    
    n_chunk_revealed = random.randint(0,max_num_chunks)

    

    if (n_chunk_revealed == 0) or (na_only):
        is_motif = is_na.clone()
        is_motif_2d = is_motif[:, None] * is_motif[None, :]
        return is_motif_2d, is_motif
    
    else: 
        # Automatically set all nucleic resis to true if na_fixed_intra is given
        # Otherwise initially have all regions hidden, then add chunks
        is_motif = na_fixed_intra*is_na.clone()
            
        cur_min_seq_dist = min_seq_dist # could possibly increment this if needed 

        for i in range(n_chunk_revealed):
            print('on chunk ',i) 
            chunk_size = torch.randint(chunk_size_min, chunk_size_max, size=(1,)).item()
            
            if len(where_is_contacting.shape) == 0:
                # ensures where_is_contacting is a 1d tensor
                where_is_contacting = where_is_contacting.unsqueeze(0)

            p = torch.ones_like(where_is_contacting)/len(where_is_contacting)
            chosen_idx = p.multinomial(num_samples=1, replacement=False)
            chosen_idx = chosen_idx.item()
            chosen_idx = where_is_contacting[chosen_idx]

            # find min and max indices for revealed chunk
            min_index = max(0, chosen_idx - chunk_size//2)
            max_index = min(chains_are_contacting.numel(), 1+chosen_idx + chunk_size//2)
            # reveal chunk 
            is_motif[min_index:max_index] = True

            # update where_is_contacting
            start = max(0,min_index-cur_min_seq_dist)
            end = min(chains_are_contacting.numel(), max_index+cur_min_seq_dist)
            chains_are_contacting[start:end] = False # remove this option from where_is_contacting 
            
            where_is_contacting = chains_are_contacting.nonzero().squeeze()
            
            if chains_are_contacting.sum() == 0:
                break # can't make any more chunks
        
        is_motif_2d = is_motif[:, None] * is_motif[None, :]

        return is_motif_2d, is_motif
    

def get_unconditional_3template(indep, atom_mask, low_prop, high_prop, broken_prop):
    """
    Unconditional protein generation task. Nothing is motif. 
    """
    L = indep.length()
    is_motif = torch.zeros(L).bool()
    is_motif_2d = is_motif[:, None] * is_motif[None, :]

    return (is_motif_2d, is_motif), None


def get_sm_contact_mask(indep, 
                        atom_mask, 
                        low_prop, 
                        high_prop, 
                        broken_prop):
    """
    Gets a small molecule contact mask. Either SM alone, SM+1protein chunk, or SM+2protein chunks
    """
    # indep.write_pdb('check_indep_sm.pdb')
    mask2d, is_motif = _get_sm_contact_3template(indep.xyz, indep.is_sm, low_prop, high_prop)

    return (mask2d, is_motif), None

def get_na_contact_mask(indep, 
                        atom_mask, 
                        low_prop, 
                        high_prop, 
                        broken_prop,
                        contact_cut=8,
                        chunk_size_min=1,
                        chunk_size_max=7,
                        max_num_chunks=4,
                        na_fixed_intra=False,
                        na_fixed_inter=False,
                        ):
    """
    AF: create masks for protein na complexes
    """
    # indep.write_pdb('check_indep_na.pdb')

    is_na = get_nucleic_acid_residues(indep.seq)
    mask2d, is_motif = _get_na_contact_3template(indep.xyz, is_na, low_prop, high_prop,
        contact_cut=contact_cut,
        chunk_size_min=chunk_size_min,
        chunk_size_max=chunk_size_max,
        max_num_chunks=max_num_chunks,
        na_fixed_intra=na_fixed_intra,
        na_fixed_inter=na_fixed_inter
        )
    return (mask2d, is_motif), None


def get_diffusion_mask(
        indep, atom_mask, low_prop, high_prop, broken_prop,
        diff_mask_probs):
    
    mask_probs = list(diff_mask_probs.items())
    masks = [m for m, _ in mask_probs]
    props = [p for _, p in mask_probs]
    get_mask = np.random.choice(masks, p=props)

    # Use fallback mask if no small molecule present.
    # if not indep.is_sm.any():
    #     get_mask = sm_mask_fallback.get(get_mask, get_mask)


    # sm_compatable masks only
    if indep.is_sm.any(): 
        get_mask = get_sm_contact_mask



    ic(get_mask)
    
    return get_mask(indep, atom_mask, low_prop=low_prop, high_prop=high_prop, broken_prop=broken_prop)


def get_diffusion_mask_na(
        indep, atom_mask, low_prop, high_prop, broken_prop, diff_mask_probs,
        contact_cut=8,
        chunk_size_min=1,
        chunk_size_max=7,
        max_num_chunks=4,
        na_fixed_intra=False,
        na_fixed_inter=False,
        ):


    mask_probs = list(diff_mask_probs.items())
    masks = [m for m, _ in mask_probs]
    props = [p for _, p in mask_probs]
    get_mask = np.random.choice(masks, p=props)


    # Use fallback mask if no small molecule present.
    # if not indep.is_sm.any():
    #     get_mask = sm_mask_fallback.get(get_mask, get_mask)

    if indep.is_sm.any() and get_nucleic_acid_residues(indep.seq).any():
        sys.exit(f'small molecule and nucleic training not currently supported for {task} task!')

    # nucleic masks only
    if get_nucleic_acid_residues(indep.seq).any():
        get_mask = get_na_contact_mask

    return get_mask(indep, atom_mask, 
                low_prop=low_prop, 
                high_prop=high_prop, 
                broken_prop=broken_prop,
                contact_cut=contact_cut,
                chunk_size_min=chunk_size_min,
                chunk_size_max=chunk_size_max,
                max_num_chunks=max_num_chunks,
                na_fixed_intra=na_fixed_intra,
                na_fixed_inter=na_fixed_inter,
                )




def generate_sm_mask(prot_masks, is_sm):
    # Not currently used, but may become part of a better way to do this
    L = is_sm.shape[0]
    input_seq_mask = torch.ones(L).bool()
    input_str_mask = torch.ones(L).bool()
    input_floating_mask = -1
    input_t1d_str_conf_mask = torch.ones(L)
    input_t1d_seq_conf_mask = torch.ones(L)
    loss_seq_mask = torch.ones(L).bool()
    loss_str_mask = torch.ones(L).bool()
    loss_str_mask_2d = torch.ones(L,L).bool()

    mask_dict = {'input_seq_mask':input_seq_mask,
                'input_str_mask':input_str_mask,
                'input_floating_mask':input_floating_mask,
                'input_t1d_str_conf_mask':input_t1d_str_conf_mask,
                'input_t1d_seq_conf_mask':input_t1d_seq_conf_mask,
                'loss_seq_mask':loss_seq_mask,
                'loss_str_mask':loss_str_mask,
                'loss_str_mask_2d':loss_str_mask_2d}
    #is_motif = torch.ones(L).bool()
    #is_motif_prot = mask_dict['input_str_mask']
    for k, v in mask_dict.items():
        if type(v) is not torch.Tensor:
            continue
        if k == 'loss_str_mask_2d':
            continue
        #ic(k, v.shape, prot_masks[k].shape, is_sm.shape)
        v[~is_sm] = prot_masks[k]
        mask_dict[k] = v
    mask_dict['input_seq_mask']
    
    return mask_dict

###################
# Functions for making a mask for nearby contigs - DT
###################
def closest_distance(group1: torch.Tensor, group2: torch.Tensor, 
                     include_point1: torch.Tensor, include_point2: torch.Tensor) -> torch.Tensor:
    '''
    Given two groups of points, how close are the closest pair of points?
    
    Args
        group1 (batch1, n_points1, 3)
        group2 (batch2, n_points2, 3)
        include_point1 (batch1, n_points1): True = the coordinates should be considered in the distance calculation.
    
    Returns
        closest_dist: (batch1, batch2)
    '''
    assert group1.shape[:-1] == include_point1.shape
    assert group2.shape[:-1] == include_point2.shape
    
    # Expand shapes so we can broadcast
    group1 = group1[:,:,None,None,:]
    group2 = group2[None,None,:,:,:]
    include_point1 = include_point1[:,:,None,None]
    include_point2 = include_point2[None,None,:,:]

    # Distance calc
    dists = torch.linalg.norm(group1 - group2, ord=2, dim=-1)
    
    # Both points must be "included" to consider the dist between them
    include_dist = torch.logical_and(include_point1, include_point2)
    dists[~include_dist] = torch.inf

    # find min over all pairs of atom in each group. Would be clearner to do with a "topk_any_dims" like function.
    closest_dist = dists.min(dim=1)[0]
    closest_dist = closest_dist.min(dim=2)[0]
    
    return closest_dist

def get_neighboring_residues(xyz: torch.Tensor, atom_mask: torch.Tensor, 
                             i: int, r: float) -> torch.Tensor:
    '''
    Args
        xyz (L, 14, 3): Atom coordinates in the protien
        atom_mask (L, 14): True = atom is "really" there.
        i: Index of the central residue
        r: Contact radius.
            
    Returns
        neighboring_residues (L,): Boolean mask. True if any atom in the central residue is 
            closer than r to any atom in another residue, they are considered neighbors.
            DOES NOT INCLUDE THE CENTRAL RESIDUE! This is a mask of the *neighbors*.
    '''
    res_xyz = xyz[[i]]
    res_atom_mask = atom_mask[[i]]
    closest_dist = closest_distance(
        group1=res_xyz, 
        group2=xyz,
        include_point1=res_atom_mask,
        include_point2=atom_mask
    )[0]
    neighboring_residues = closest_dist < r
    return neighboring_residues

def dilate_1d(mask: torch.Tensor) -> torch.Tensor:
    '''
    Args
        mask: A 1D boolean mask
        
    Returns
        dilated: A boolean mask where True values have "spread" one space
            to the left and right.
    '''
    
    mask = mask[None,None].float()
    kernel = torch.ones(1,1,3).float()
    dilated = torch.nn.functional.conv1d(mask, kernel, padding=1)
    dilated = torch.clamp(dilated, 0, 1)
    return dilated[0,0].bool()

def erode_1d(mask: torch.Tensor) -> torch.Tensor:
    '''
    Args
        mask: A 1D boolean mask
        
    Returns
        eroded: A boolean mask where True values have "contracted" one space
            from the left and right. Isolated islands of True are removed.
    '''
    return ~dilate_1d(~mask)

def merge_islands(mask: torch.Tensor, n: int=1) -> torch.Tensor:
    '''
    If two Trues are separated by 2*n or fewer spaces,
    the interviening spaces are set to True.
    
    Ex for n=2.
        in:  [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
        out: [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
    
    Args
        mask: A 1D boolean mask        
    '''
    
    for _ in range(n):
        mask = dilate_1d(mask)
    for _ in range(n):
        mask = erode_1d(mask)
        
    return mask

def remove_small_islands(mask: torch.Tensor, n: int = 1) -> torch.Tensor:
    '''
    If a contiguous chunk has less than or equal to 2*n Trues, it is removed.
    
    Ex for n=2.
        in:  [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
        out: [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
    
    Args
        mask: A 1D boolean mask        
    '''
    for _ in range(n):
        mask = erode_1d(mask)
    for _ in range(n):
        mask = dilate_1d(mask)
        
    return mask

def get_contigs_around_residue(xyz: torch.Tensor, atom_mask: torch.Tensor,
                                i: int, r: float) -> torch.Tensor:
    '''
    Given a residue in a protein, find contigs that have residues with at least
    one atom within r Angstroms of any atom in the central residue. Essentially
    it selects residues in a sphere around a central residue, then joins isolated
    residues into contigs. Small contigs are then removed.
    
    Args
        xyz (L, 14, 3): Atom coordinates in the protien
        atom_mask = True = Atom is "really" there.
        i: Index of the central residue
        r: Contact radius.
       
    Returns
        mask (L,): True = residue is in the motif.
    '''
    mask = get_neighboring_residues(xyz, atom_mask, i, r)
    mask[i] = True  # include the central resiude in the motif
    mask = merge_islands(mask, n=1)
    mask = remove_small_islands(mask, n=2)
    
    return mask

def get_nearby_contigs(indep, atom_mask, low_prop, high_prop, broken_prop):
    '''
    Randomly samples a central residue and radius, and returns a contig mask
    of residues in that radius. 
    
    Args: NOTE: These must match the call signature of "get_mask", hence the unused args.
    
    Return
        mask: True = residue is in the contig(s)
        is_atom_motif: Currently this contig selector only works for proteins.
            This is spoofed to match the "get_mask" output signature.
    '''
    max_tries = 100
    xyz = indep.xyz
    L_ptn = xyz.shape[0]
    
    for _ in range(max_tries):
        # Get nearby contig mask
        i = int(torch.randint(high=L_ptn, size=(1,)))
        r = float(torch.rand(size=(1,))) * 15. + 5.
        mask = get_contigs_around_residue(xyz, atom_mask, i, r)
        
        # Do the contigs cover enough/too much of the protein?
        prop = mask.sum() / L_ptn
        if low_prop <= prop <= high_prop:
            break

    # Spoof is_atom_motif output
    is_atom_motif = None

    return mask, is_atom_motif

def get_nucleic_acid_residues(seq):
    # return torch.logical_and((seq >= 22),(seq <= 26))
    return torch.logical_and((seq >= 22),(seq <= 31))


def get_diffusion_mask_simple(L, loader_params, indep=None):
    """
    Function to make a diffusion mask.
    Options:
        full_prop - proportion of time whole protein is masked.
        low_prop - lower bound on the proportion of the protein masked
        high_prop - upper bound on the proportion of the protein masked
        broken_prop - proportion of the time the mask is in the middle (broken motif), vs at the ends
    Output:
        1D diffusion mask. True is unmasked, False is masked/diffused
    """
    diffusion_mask = torch.ones(L).bool()

    mask_length = int(
        np.floor(
        random.uniform(
        loader_params['MASK_MIN_PROPORTION'], loader_params['MASK_MAX_PROPORTION']) * L))
    # decide if mask goes in the middle or the ends
    try:
        if random.uniform(0,1) < loader_params['MASK_BROKEN_PROPORTION']:
            high_start = L-mask_length-1
            start = random.randint(0, high_start)
            diffusion_mask[start:start+mask_length] = False
        else:
            # split mask in two
            split = random.randint(1, mask_length-2)
            diffusion_mask[:split] = False
            diffusion_mask[-(mask_length-split):] = False
    except:
        return diffusion_mask
    return diffusion_mask

def get_atom_coordinates(NA_seq, NA_coords, residue_token_index, atom_token_indices):
    indices = NA_seq == residue_token_index
    coordinates = NA_coords[indices]
    coordinates = coordinates[:, torch.tensor(atom_token_indices), :].reshape(-1, 3)
    return coordinates

# def get_na_contacts(seq, xyz, loader_params, is_nucleic_acid):
#     """
#     want to find all contacts to 
#     C - C6, C5, N4 -> seqtoken 23, atom indices 18, 17, 16
#     A - N7, N6 -> seqtoken 22, atom indices 18, 20
#     G - N7, O6 -> seqtoken 24, atom_indices 18, 21
#     T - C6, C7 -> seqtoken 25, atom_indices 19, 18
    
#     so we first isolate these atoms from xyz
#     then we compute pairwise distances
#     then we find indices where these pairwise distances is smaller then X
#     and we return those indices
#     """
#     na_seq = seq[is_nucleic_acid]
#     na_coords = xyz[is_nucleic_acid]
#     DNAC_coords = get_atom_coordinates(na_seq, na_coords, 23, [16, 17, 18])
#     DNAA_coords = get_atom_coordinates(na_seq, na_coords, 22, [18, 20])
#     DNAG_coords = get_atom_coordinates(na_seq, na_coords, 24, [18, 21])
#     DNAT_coords = get_atom_coordinates(na_seq, na_coords, 25, [18, 19])

#     DNA_coords = torch.cat((DNAC_coords, DNAA_coords, DNAG_coords, DNAT_coords), dim=0)

#     # prot_seq = indep.seq[~indep.is_nucleic_acid]
#     prot_coords = xyz[~is_nucleic_acid]

#     distances = torch.cdist(prot_coords, DNA_coords)
#     contact_residues = torch.logical_and((distances < loader_params['BINDING_DISTANCE_CUTOFF']), (distances != 0)).any(dim=1).nonzero()
#     prot_contact_residues = contact_residues[:, 0].unique()
#     return prot_contact_residues

def get_na_contacts(seq, xyz, loader_params, is_nucleic_acid):
    """
    want to find all contacts to 
    C - C6, C5, N4 -> seqtoken 23, atom indices 18, 17, 16
    A - N7, N6 -> seqtoken 22, atom indices 18, 20
    G - N7, O6 -> seqtoken 24, atom_indices 18, 21
    T - C6, C7 -> seqtoken 25, atom_indices 19, 18
    
    so we first isolate these atoms from xyz
    then we compute pairwise distances
    then we find indices where these pairwise distances is smaller then X
    and we return those indices
    """
    na_seq = seq[is_nucleic_acid]
    na_coords = xyz[is_nucleic_acid]
    # DNAC_coords = get_atom_coordinates(na_seq, na_coords, 23, [16, 17, 18])
    # DNAA_coords = get_atom_coordinates(na_seq, na_coords, 22, [18, 20])
    # DNAG_coords = get_atom_coordinates(na_seq, na_coords, 24, [18, 21])
    # DNAT_coords = get_atom_coordinates(na_seq, na_coords, 25, [18, 19])

    # RNAC_coords = get_atom_coordinates(na_seq, na_coords, 28, [16, 17, 18])
    # RNAA_coords = get_atom_coordinates(na_seq, na_coords, 27, [18, 20])
    # RNAG_coords = get_atom_coordinates(na_seq, na_coords, 29, [18, 21])
    # RNAU_coords = get_atom_coordinates(na_seq, na_coords, 30, [18, 19])



    # DNA_coords = torch.cat(
    #     (DNAC_coords, DNAA_coords, DNAG_coords, DNAT_coords, 
    #         RNAC_coords, RNAA_coords, RNAG_coords, RNAU_coords,
    #         ), dim=0)

    # prot_seq = indep.seq[~indep.is_nucleic_acid]
    prot_coords = xyz[~is_nucleic_acid]

    # distances = torch.cdist(prot_coords, DNA_coords)
    distances = torch.cdist(prot_coords, na_coords)
    contact_residues = torch.logical_and((distances < loader_params['BINDING_DISTANCE_CUTOFF']), (distances != 0)).any(dim=1).nonzero()
    prot_contact_residues = contact_residues[:, 0].unique()
    return prot_contact_residues



def get_motif_block(contact_residues, sequence_length, loader_params):
    """
    Get all the indices between contact_residues, + some random additions to the ends. 
    """
    blocks = []
    last_block_start = contact_residues[0]
    last_block_end = contact_residues[0]
    for contact in contact_residues[1:]:
        if contact - last_block_end < 5:
            last_block_end = contact
        else:
            blocks.append([last_block_start, last_block_end])
            last_block_start = contact
            last_block_end = contact
    blocks.append([last_block_start, last_block_end])


    # randomly padding to the ends of the blocks
    for i in range(len(blocks)):
        additional_leftend = np.random.randint(0, 6)
        additional_rightend = np.random.randint(0, 6)
        blocks[i][0] = max(0, blocks[i][0] - additional_leftend)
        blocks[i][1] = min(sequence_length - 1, blocks[i][1] + additional_rightend)

    block_indices = torch.cat([torch.arange(block[0], block[1]+1) for block in blocks])

    return block_indices

def get_diff_mask_fn(diff_mask_probs):
    mask_probs = list(diff_mask_probs.items())
    masks = [m for m, _ in mask_probs]
    props = [p for _, p in mask_probs]
    mask_fn = np.random.choice(masks, p=props)
    return mask_fn

#####################################
# Main mask generator function
#####################################

def generate_masks(indep, task, loader_params, chosen_dataset, full_chain=None, xyz=None, atom_mask=None, is_sm=None, seq=None): #full_chain is for complexes, to signify which chain is complete
    '''
    Slimmed down function that outputs 1D masks for inputs and loss calculations.
    Input masks are defined as True=(unmasked)/False=masked (except for input_t1dconf, which is a scalar value, and seq2str_mask which is the msa mask for the seq2str task)
    Loss masks are defined as True=(loss applied)/False=(no loss applied)
    
    Input masks:
        -input_seq
        -input_str
        -input_floating = points to be represented as floating points (structure present but side chains masked out)
        -input_t1d_str_conf = scalar to multiply input str t1d confidences by
        -input_t1d_seq_conf = scalar to multiply input seq t1d confidences by

    Output masks:
        -loss_seq
        -loss_str
        -loss_str_2d = additional coordinate pair masking to be applied on top of loss_str 1d masking.
    '''

    L = indep.length()

    input_seq_mask = torch.ones(L).bool()
    input_str_mask = torch.ones(L).bool()
    input_floating_mask = -1
    input_t1d_str_conf_mask = torch.ones(L).bool() * 0.9
    input_t1d_seq_conf_mask = torch.ones(L).bool() * 0.9
    loss_seq_mask = torch.ones(L).bool()
    loss_str_mask = torch.ones(L).bool()
    loss_str_mask_2d = torch.ones(L,L).bool()
    is_atom_motif = None
    chunk_sees_relative = None
    t2d_is_revealed = None 

    if task == 'seq2str':
        '''
        Classic structure prediction task.
        '''
        #input masks
        # Currently msa loss masking is performed in train_multi_EMA
        #input_seq_mask = torch.clone(seq2str_mask) #this is not 1D
        input_str_mask = torch.ones(L).bool()
        input_floating_mask = torch.ones(L).bool()
        input_t1d_str_conf_mask = torch.ones(L)*0.9 #scale seq2str t1d confidences by 0.9
        input_t1d_seq_conf_mask = torch.ones(L) # Very confident about the true sequence

        #loss masks
        # Currently msa loss masking is performed in train_multi_EMA
        # loss_seq_mask = torch.clone(seq2str_mask) #this is not 1D
        #loss_str_mask = seq2str_str_mask
        #loss_str_mask_2d = seq2_str_mask[None, :] * seq2str_str_mask[:, None]

    # dj - only perform diffusion hal on pdb and fb for now 
    elif task == 'diff' and chosen_dataset not in ['complex','negative','na_compl','tf_distil']:
    # elif task == 'diff' and chosen_dataset not in ['complex','negative']:
        """
        Hal task but created for the diffusion-based training. 
        """ 
        mask_fns = list( loader_params['DIFF_MASK_PROBS'].keys() )
        indep.is_na = get_nucleic_acid_residues(indep.seq) # Check for nucleic acid resis
        # TEMPORARY LOCATION: ADD THIS AS OBJECT ATTRIBUTE IN AA MODEL LATER
        diffusion_mask, is_atom_motif = get_diffusion_mask(
            indep,
            atom_mask,
            low_prop=loader_params['MASK_MIN_PROPORTION'],
            high_prop=loader_params['MASK_MAX_PROPORTION'],
            broken_prop=loader_params['MASK_BROKEN_PROPORTION'],
            diff_mask_probs=loader_params['DIFF_MASK_PROBS'],
            ) 

        # 3 template stuff
        cond_A = ((get_diffusion_mask_chunked in mask_fns) or (get_triple_contact in mask_fns))
        cond_B = (type(diffusion_mask) == tuple)
        if  (cond_A and cond_B):
            # this means a mask generator which is aware of 3template diffusion was used 
            assert (len(diffusion_mask) == 2) and (type(diffusion_mask) == tuple)
            (t2d_is_revealed, diffusion_mask) = diffusion_mask 

        # ic(diffusion_mask)
        # ic(is_atom_motif)
        # sys.exit('Exiting early for debugging')

        # ic(is_atom_motif, torch.nonzero(diffusion_mask), diffusion_mask.sum())
        input_str_mask = diffusion_mask.clone()
        input_seq_mask = diffusion_mask.clone()
        # t1dconf scaling will be taken care of by diffuser, so just leave those at 1 here 
        input_t1d_str_conf_mask = torch.ones(L)
        input_t1d_seq_conf_mask = torch.ones(L)

        ## loss masks 
        loss_seq_mask[diffusion_mask] = False  # Dont score where diffusion mask is True (i.e., where things are not diffused)

    # AF: making proper masking for nucleic datasets
    elif task == 'diff' and chosen_dataset in ['na_compl','tf_distil']:

        mask_fns = list( loader_params['DIFF_MASK_PROBS'].keys() )

        indep.na_fixed_inter=loader_params["NA_FIXED_INTER"]
        indep.na_fixed_intra=loader_params["NA_FIXED_INTRA"]
        indep.is_na = get_nucleic_acid_residues(indep.seq)
        diffusion_mask, is_atom_motif = get_diffusion_mask(
                indep,
                atom_mask,
                low_prop=loader_params['MASK_MIN_PROPORTION'],
                high_prop=loader_params['MASK_MAX_PROPORTION'],
                broken_prop=loader_params['MASK_BROKEN_PROPORTION'],
                diff_mask_probs=loader_params['DIFF_MASK_PROBS'],
            )

        # 3 template stuff
        cond_A = ((get_diffusion_mask_chunked in mask_fns) or (get_triple_contact in mask_fns))
        cond_B = (type(diffusion_mask) == tuple)
        if  (cond_A and cond_B):
            # this means a mask generator which is aware of 3template diffusion was used 
            assert (len(diffusion_mask) == 2) and (type(diffusion_mask) == tuple)
            (t2d_is_revealed, diffusion_mask) = diffusion_mask 

        # ic(diffusion_mask)
        # ic(is_atom_motif)
        # sys.exit('Exiting early for debugging')

        # ic(is_atom_motif, torch.nonzero(diffusion_mask), diffusion_mask.sum())
        input_str_mask = diffusion_mask.clone()
        input_seq_mask = diffusion_mask.clone()
        # t1dconf scaling will be taken care of by diffuser, so just leave those at 1 here 
        input_t1d_str_conf_mask = torch.ones(L)
        input_t1d_seq_conf_mask = torch.ones(L)

        ## loss masks 
        loss_seq_mask[diffusion_mask] = False  # Dont score where diffusion mask is True (i.e., where things are not diffused)


    elif task == 'diff' and chosen_dataset == 'complex':
        '''
        Diffusion task for complexes. Default is to diffuse the whole of the complete chain.
        Takes full_chain as input, which is [full_chain_start_idx, full_chain_end_idx]
        '''
        assert full_chain[1] is not None
        
        input_str_mask = torch.clone(full_chain)
        input_seq_mask = torch.clone(input_str_mask)

    elif task == 'hal' and chosen_dataset != 'complex':
        '''
        This is Joe's hallucination task, where a contiguous region is masked, along with flanks, and two residues either end are given (but not their angles).
        Scored on everything but the flank regions (may want to change this to only score central inpainted region
        '''
        splice, flank_width = get_masks(L, loader_params['HAL_MASK_LOW'], loader_params['HAL_MASK_HIGH'], loader_params['FLANK_LOW'], loader_params['FLANK_HIGH'])
        hal_mask_len = splice[1]-splice[0]

        #input masks
        input_seq_mask = torch.ones(L).bool()
        input_seq_mask[splice[0]-flank_width:splice[1]+flank_width] = False #mask out flanks and central region
        input_str_mask = torch.clone(input_seq_mask)
        input_str_mask[splice[0]-1] = True #give structure of two flanking residues
        input_str_mask[splice[1]] = True

        input_floating_mask = torch.ones(L).bool()
        input_floating_mask[splice[0]-1] = False #immediate two flanking residues are set to false/floating
        input_floating_mask[splice[1]] = False
        input_t1d_str_conf_mask = torch.ones(L) #t1d confidences are unscaled in hal task
        input_t1d_seq_conf_mask = torch.ones(L) #t1d confidences are unscaled in hal task

        #loss masks
        loss_seq_mask = torch.zeros(L).bool()
        loss_seq_mask[splice[0]:splice[1]] = True #only apply a loss on the central region
        loss_str_mask = torch.ones(L).bool()
        loss_str_mask[splice[0]-flank_width:splice[0]] = False #don't apply a loss in the flanking regions
        loss_str_mask[splice[1]:splice[1] + flank_width] = False
        loss_str_mask_2d[~loss_str_mask,:] = False
        loss_str_mask_2d[:,~loss_str_mask] = False

    elif task == 'hal_ar' and chosen_dataset != 'complex':
        '''
        This is Joe's hallucination task, where a contiguous region is masked, along with flanks, and two residues either end are given (but not their angles).
        This is autoregressive sequence unmasking.
        Scored on everything but the flank regions (may want to change this to only score central inpainted region
        '''
        splice, flank_width = get_masks(L, loader_params['HAL_MASK_LOW'], loader_params['HAL_MASK_HIGH'], loader_params['FLANK_LOW'], loader_params['FLANK_HIGH'])
        hal_mask_len = splice[1]-splice[0]

        to_unmask = random.uniform(0,0.5) #up to 50% of sequence unmasked
        #input masks
        input_seq_mask = torch.where(torch.rand(L) < to_unmask, True, False)
        input_seq_mask[:splice[0]-flank_width] = True
        input_seq_mask[splice[1]+flank_width:] = True
        input_seq_mask[splice[0]-flank_width:splice[0]] = False #mask out flanks
        input_seq_mask[splice[1]:splice[1]+flank_width] = False
        
        input_str_mask[splice[0]-flank_width:splice[1]+flank_width] = False
        input_str_mask[splice[0]-1] = True #give structure of two flanking residues
        input_str_mask[splice[1]] = True

        input_floating_mask = torch.ones(L).bool()
        input_floating_mask[splice[0]-1] = False #immediate two flanking residues are set to false/floating
        input_floating_mask[splice[1]] = False
        input_t1d_str_conf_mask = torch.ones(L) #t1d confidences are unscaled in hal task
        input_t1d_seq_conf_mask = torch.ones(L) #t1d confidences are unscaled in hal task

        #loss masks
        loss_seq_mask = torch.zeros(L).bool()
        loss_seq_mask[splice[0]:splice[1]] = True #only apply a loss on the central region
        loss_str_mask = torch.ones(L).bool()
        loss_str_mask[splice[0]-flank_width:splice[0]] = False #don't apply a loss in the flanking regions
        loss_str_mask[splice[1]:splice[1] + flank_width] = False
        loss_str_mask_2d[~loss_str_mask,:] = False
        loss_str_mask_2d[:,~loss_str_mask] = False

    elif task == 'hal' and chosen_dataset == 'complex':
        ''' 
        This is Joe's complex hal task, where a contiguous region is masked.
        Everything is scored.
        This is for complexes
        '''
        len_full_chain = full_chain[1]-full_chain[0]+1 #full_chain has start and end index
        high_limit = min([loader_params['COMPLEX_HAL_MASK_HIGH'],len_full_chain-20])
        low_limit = min([loader_params['COMPLEX_HAL_MASK_LOW'],high_limit])
        len_to_mask = random.randint(low_limit, high_limit)
        start = random.randint(full_chain[0], len_full_chain-len_to_mask + full_chain[0])
        #input masks
        input_seq_mask = torch.ones(L).bool()
        input_seq_mask[start:start+len_to_mask] = False
        input_str_mask = torch.ones(L).bool()
        input_str_mask[start:start+len_to_mask] = False #not doing flanking masking now
        input_t1d_str_conf_mask = torch.ones(L) #t1d confidences are unscaled in str2seq task
        input_t1d_seq_conf_mask = torch.ones(L) #t1d confidences are unscaled in str2seq task

        #loss masks
        loss_seq_mask = torch.zeros(L).bool()
        loss_seq_mask[start:start+len_to_mask] = True
        loss_str_mask = torch.ones(L).bool()
        loss_str_mask_2d[~loss_str_mask,:] = False
        loss_str_mask_2d[:,~loss_str_mask] = False
   
    elif task == 'hal_ar' and chosen_dataset == 'complex':
        ''' 
        This is Joe's complex hal task, where a contiguous region is masked, but with some sequence tokens visible (to mimic autoregressive unmasking).
        Everything is scored.
        This is for complexes
        '''
        len_full_chain = full_chain[1]-full_chain[0]+1 #full_chain has start and end index
        high_limit = np.min([loader_params['COMPLEX_HAL_MASK_HIGH_AR'],len_full_chain-20])
        low_limit=np.min([loader_params['COMPLEX_HAL_MASK_LOW_AR'],high_limit])
        len_to_mask = random.randint(low_limit, high_limit)
        start = random.randint(full_chain[0], len_full_chain-len_to_mask + full_chain[0])
 
        to_unmask = random.uniform(0,0.5) #up to 50% of sequence unmasked
        #input masks
        input_seq_mask = torch.where(torch.rand(L) < to_unmask, True, False)
        input_seq_mask[:start] = True
        input_seq_mask[start+len_to_mask:] = True
        input_str_mask = torch.ones(L).bool()
        input_str_mask[start:start+len_to_mask] = False #not doing flanking masking now. No AR unmasking of structure
        input_t1d_str_conf_mask = torch.ones(L) #t1d confidences are unscaled in str2seq task
        input_t1d_seq_conf_mask = torch.ones(L) #t1d confidences are unscaled in str2seq task

        #loss masks
        loss_seq_mask = torch.zeros(L).bool()
        loss_seq_mask[start:start+len_to_mask] = True
        loss_str_mask = torch.ones(L).bool()
        loss_str_mask_2d[~loss_str_mask,:] = False
        loss_str_mask_2d[:,~loss_str_mask] = False

    elif task == 'str2seq' and chosen_dataset != 'complex':
        '''
        This is Joe's str2seq task, where a contiguous region is masked, along with flanks.
        Everything, but the flanked regions, is scored
        This is only if the protein is monomeric
        '''
        splice, flank_width = get_masks(L, loader_params['HAL_MASK_LOW'], loader_params['HAL_MASK_HIGH'], loader_params['FLANK_LOW'], loader_params['FLANK_HIGH'])
        hal_mask_len = splice[1]-splice[0]
        #input masks
        input_seq_mask = torch.ones(L).bool()
        input_seq_mask[splice[0]-flank_width:splice[1]+flank_width] = False #mask out flanks and central region
        input_str_mask = torch.ones(L).bool()
        input_str_mask[splice[0] - flank_width:splice[0]] = False #mask out flanks only (i.e. provide structure of the central region)
        input_str_mask[splice[1]:splice[1] + flank_width] = False
        input_floating_mask = torch.ones(L).bool()
        input_t1d_str_conf_mask = torch.ones(L) #t1d confidences are unscaled in str2seq task
        input_t1d_seq_conf_mask = torch.ones(L) #t1d confidences are unscaled in str2seq task

        #loss masks
        loss_seq_mask = torch.zeros(L).bool()
        loss_seq_mask[splice[0]:splice[1]] = True #only apply a loss on sequence recovery in the central region
        loss_str_mask = torch.clone(input_str_mask) #don't apply a structure loss on the flanking regions
        loss_str_mask_2d[~loss_str_mask,:] = False
        loss_str_mask_2d[:,~loss_str_mask] = False 
    
    elif task == 'str2seq' and chosen_dataset == 'complex':
        ''' 
        This is Joe's str2seq task, where a contiguous region is masked.
        Everything is scored
        This is for complexes
        '''
        len_full_chain = full_chain[1]-full_chain[0]+1 #full_chain has start and end index
        high_limit = np.min([loader_params['COMPLEX_HAL_MASK_HIGH'],len_full_chain-20])
        low_limit=np.min([loader_params['COMPLEX_HAL_MASK_LOW'],high_limit])
        len_to_mask = random.randint(low_limit, high_limit)
        start = random.randint(full_chain[0], len_full_chain-len_to_mask + full_chain[0])

        #input masks
        input_seq_mask = torch.ones(L).bool()
        input_seq_mask[start:start+len_to_mask] = False
        input_str_mask = torch.ones(L).bool()
        input_t1d_str_conf_mask = torch.ones(L) #t1d confidences are unscaled in str2seq task
        input_t1d_seq_conf_mask = torch.ones(L) #t1d confidences are unscaled in str2seq task
 
        #loss masks
        loss_seq_mask = torch.zeros(L).bool()
        loss_seq_mask[start:start+len_to_mask] = True
        loss_str_mask = torch.clone(input_str_mask)
        loss_str_mask_2d[~loss_str_mask,:] = False
        loss_str_mask_2d[:,~loss_str_mask] = False

    elif task == 'str2seq_full':
        '''
        This is David's str2seq task, where most (default 90-100%) of sequence is masked.
        Score on everything.
        '''
        #NOTE this has not been implemented yet for e.g. complexes
        rand_prop = random.uniform(loader_params['STR2SEQ_FULL_LOW'], loader_params['STR2SEQ_FULL_HIGH']) #random proportion masked, between two extremes
        
        #input masks
        input_seq_mask = torch.rand(L).bool() < rand_prop
        input_str_mask = torch.ones(L).bool()
        input_floating_mask = torch.ones(L).bool()

        input_t1d_str_conf_mask = torch.ones(L) #t1d confidences are not scaled in str2seq_full task
        input_t1d_seq_conf_mask = torch.ones(L) #t1d confidences are not scaled in str2seq_full task

        #loss masks
        loss_seq_mask = torch.clone(input_seq_mask)
        loss_str_mask = torch.ones(L).bool() #apply a loss on the whole structure        
        loss_str_mask_2d[~loss_str_mask,:] = False
        loss_str_mask_2d[:,~loss_str_mask] = False
        

    # # elif task == 'diff' and chosen_dataset in ['complex','negative','na_compl','tf_distil']:
    # elif task == 'diff' and chosen_dataset == 'na_compl':
    #     """
    #     The plan:
    #         provide sequence and structure of binding residues
    #         provide structure and masked sequence motif residues
    #     Make an initial diffusion mask over the protein (True where given, false where diffused)

    #     legal diffusion positions (True where diffusable, False where not diffusable)

    #     diffusion mask is ~(~simple diff mask AND legal diffusion position)

    #     Make a sequence_mask that converts sequence to mask token at specific residues. 

    #     """
    #     seq = seq.squeeze()
    #     xyz = xyz.squeeze()
    #     ic(seq.shape)
    #     ic(xyz.shape)
    #     is_nucleic_acid = get_nucleic_acid_residues(seq)
    #     L = len(is_nucleic_acid) - is_nucleic_acid.sum() # length of protein sequence is L

    #     if random.uniform(0,1) < loader_params['P_UNCOND']:
    #         diffusion_mask = torch.zeros(L).bool()
        
    #     else:
    #         mask_fn = get_diff_mask_fn(loader_params['DIFF_MASK_PROBS'])
    #         diffusion_mask = mask_fn(L, loader_params)

    #         ## loss masks 
    #         # loss_seq_mask[diffusion_mask] = False
        
    #     # find binding contacts
    #     if random.uniform(0,1) > loader_params['P_FREE']:

    #         contact_residue_indices = get_na_contacts(seq, xyz, loader_params, is_nucleic_acid)

    #         if len(contact_residue_indices) == 0:
    #             legal_diffusion_positions = torch.ones(L).bool()
    #             provide_sequence = diffusion_mask
    #         else:
    #         # find binding motifs
    #             motif_blocks_residue_indices = get_motif_block(contact_residue_indices, L, loader_params)

    #             legal_diffusion_positions = torch.ones(L).bool()
    #             legal_diffusion_positions[motif_blocks_residue_indices] = False # can't diffuse the motif blocks

    #             # making the sequence mask. We want to put a mask token at all the residues in the binding motif but not the contact itself. 
    #             provide_sequence = diffusion_mask
    #             provide_sequence[motif_blocks_residue_indices] = False # don't provide sequence in the motif
    #             provide_sequence[contact_residue_indices] = True # but provide the sequence in the actual binding locations


    #         diffusion_mask = ~torch.logical_and(~diffusion_mask,legal_diffusion_positions)

    #     else:
    #         provide_sequence = diffusion_mask

    #     # need to now extend both to cover the NAs as well
    #     len_NA = sum(is_nucleic_acid)
    #     NA_diffusion_mask = torch.ones(len_NA).bool() # don't diffuse any NA positions
    #     NA_provide_sequence = torch.ones(len_NA).bool() # provide all the sequence

    #     diffusion_mask = torch.cat((diffusion_mask, NA_diffusion_mask))
    #     provide_sequence = torch.cat((provide_sequence, NA_provide_sequence))
        
    #     is_atomize_example = False
    #     input_seq_mask = provide_sequence
    #     input_str_mask = diffusion_mask
    #     loss_seq_mask[diffusion_mask] = False
    #     input_t1d_str_conf_mask = torch.ones(len(input_seq_mask))
    #     input_t1d_seq_conf_mask = torch.ones(len(input_seq_mask))



    else:
        sys.exit(f'Masks cannot be generated for the {task} task!')
    
    if (task != 'seq2str') and (not loader_params['SM_ONLY']):
       assert torch.sum(~input_seq_mask) > 0, f'Task = {task}, dataset = {chosen_dataset}, full chain = {full_chain}'

    mask_dict = {'input_seq_mask':input_seq_mask,
                'input_str_mask':input_str_mask,
                'input_floating_mask':input_floating_mask,
                'input_t1d_str_conf_mask':input_t1d_str_conf_mask,
                'input_t1d_seq_conf_mask':input_t1d_seq_conf_mask,
                'loss_seq_mask':loss_seq_mask,
                'loss_str_mask':loss_str_mask,
                'loss_str_mask_2d':loss_str_mask_2d,
                'is_atom_motif': is_atom_motif,
                't2d_is_revealed': t2d_is_revealed,
                }
    
    return mask_dict


def choose_contiguous_atom_motif(res):
    """
    chooses a contiguous 3 or 4 atom motif
    """
    bond_feats = get_residue_bond_feats(res)
    natoms = bond_feats.shape[0]
    # choose atoms to be given as the motif 
    is_atom_motif = torch.zeros((natoms),dtype=bool)
    bond_graph = nx.from_numpy_matrix(bond_feats.numpy())
    paths = rf2aa.util.find_all_paths_of_length_n(bond_graph, 2)
    paths.extend(rf2aa.util.find_all_paths_of_length_n(bond_graph, 3))
    chosen_path = random.choice(paths)
    atom_names = [rf2aa.chemical.aa2long[res][i] for i in chosen_path]
    return atom_names


def get_residue_bond_feats(res, include_H=False):
    bond_feats = torch.zeros((rf2aa.chemical.NTOTAL, rf2aa.chemical.NTOTAL))
    for j, bond in enumerate(rf2aa.chemical.aabonds[res]):
        start_idx = rf2aa.chemical.aa2long[res].index(bond[0])
        end_idx = rf2aa.chemical.aa2long[res].index(bond[1])

        # maps the 2d index of the start and end indices to btype
        bond_feats[start_idx, end_idx] = rf2aa.chemical.aabtypes[res][j]
        bond_feats[end_idx, start_idx] = rf2aa.chemical.aabtypes[res][j]
    
    if not include_H:
        bond_feats = bond_feats[:rf2aa.chemical.NHEAVYPROT, :rf2aa.chemical.NHEAVYPROT]
    return bond_feats
