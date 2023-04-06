from __future__ import annotations  # allows circular references in type hinting
'''
    Defs:
        motif residue: A residue who's cartesian position is fixed and is part 
            of the same chain that is being diffused. Its location in 
            the primary sequence is pre-determined by the user.
        guide post residue: Like a motif resiude, its cartesian position remains fixed.
            However, it is not part of the chain being diffused. Its 
            location is a hint to the network that one of diffused frames
            should superimpose on the guide post residue at the end of the trajectory.
            The network has to figure out which position in the diffused
            chain should superimpose on the guide post residue.

    This module supports a different way to diffuse a protein such that it has a 
    desired backbone motif in the final stucture. In the "standard" motif scaffolding
    approach, the location the motif will appear in the primary sequence is fixed
    from the beginning, often times from brute force position sampling. With the guide
    post approach, the user doesn't have to speicfy the part of the diffused backbone
    that will adopt the desired geometry. It is determined on the fly during diffusion.
    The user could also specify general areas the motif geometry could appear, for example,
    just on chain A for multichain hallucination, or an exact location if desired.
    (Currently, inference does not support this general area approach. However, it just a matter
    of feature engineering. No additional training is required.)
'''

import torch
import numpy as np
import rf2aa.chemical
from typing import Tuple, List

def make_guideposts(indep: Indep, is_gp: torch.Tensor, placement: str=None) -> Tuple[Indep, torch.Tensor]:
    '''
    Takes the geometry of residues masked by `is_gp` and adds them as guide post features
    to indep.

    Args
    ------------
    indep: Indep instance (of length L).
    is_gp (L,): Boolean tensor. True = Residue should be added as a guide post feature.
    placement: See `convert_motif_to_guide_posts` for a description.

    Returns
    ------------
    indep: Original indep instance modified to have guide post features with a new length L+number_guidepost_residues.
    is_diffused (L+number_guidepost_residues,): True = residue should be diffused. (ie - not stay stationary.)
    gp_to_ptn_idx0: A map from the idx0 location of each guide post (keys; gp_idx0)
        to the idx0 location of the "true" corresponding frame (values; ptn_idx0).     '''
    assert not indep.is_sm.any(), 'Guide post feature does not currently support atomized residues or small molecules.'

    # Add *new* frames that will act as the guide posts
    indep, gp_to_ptn_idx0 = copy_and_append_indep_features(
        mask=is_gp,
        indep=indep,
        shuffle=True,  # Unsure if necessary
    )

    # Make the newly appended frames behave as guide posts instead of a regular (fixed) motif.
    indep, is_diffused = convert_motif_to_guide_posts(
        gp_to_ptn_idx0=gp_to_ptn_idx0,
        indep=indep,
        placement=placement,
    )

    return indep, is_diffused, gp_to_ptn_idx0

def copy_and_append_indep_features(mask: torch.tensor,
                                   indep: Indep,
                                   shuffle: bool=False) -> Tuple[Indep, dict]:
    '''
    Args
        mask (L): True = copy and append these residue's features to the end of their respective tensor.
        indep: Indep,
        shuffle: Should the features be randomly shuffled before being appended?

    Returns
        indep: Indep. Note - This does not change Indep.chirals or Indep.atom_frames
        new_to_ptn_idx0 - A map from the appended features idx0 to their original idx0 (in the protein).
    '''

    # What is the mapping from the feature's new idx0 to their original idx0 (ptn_idx0)?
    ptn_idx0 = torch.where(mask)[0]
    n_features = len(ptn_idx0)

    if shuffle:
        # Shuffle the order that the new features are appended.
        shuffle = torch.randperm(n_features)
        ptn_idx0 = ptn_idx0[shuffle]

    L_exist = mask.shape[0]
    new_idx0 = torch.arange(L_exist, L_exist + n_features)
    new_to_ptn_idx0 = dict(zip(new_idx0.tolist(), ptn_idx0.tolist()))

    # Append indep features
    indep.seq = append_indices(indep.seq, ptn_idx0, to_fill_value=rf2aa.chemical.aa2num['UNK'])
    indep.xyz = append_indices(indep.xyz, ptn_idx0)
    indep.idx = append_indices(indep.idx, ptn_idx0)

    indep.bond_feats = append_indices(indep.bond_feats, ptn_idx0, dim=0)
    indep.bond_feats = append_indices(indep.bond_feats, ptn_idx0, dim=1)
    indep.same_chain = append_indices(indep.same_chain, ptn_idx0, dim=0)
    indep.same_chain = append_indices(indep.same_chain, ptn_idx0, dim=1)
    indep.is_sm = append_indices(indep.is_sm, ptn_idx0, to_fill_value=False)
    indep.terminus_type = append_indices(indep.terminus_type, ptn_idx0, to_fill_value=0)

    return indep, new_to_ptn_idx0

def convert_motif_to_guide_posts(gp_to_ptn_idx0: dict, 
                                 indep: Indep, 
                                 placement: str=None) -> Indep:
    '''
    In indep, convert a motif residue to a guide post residue. 
    See definitions at the top. Essentially, this function makes
    each guide post residue its own chain, aa type "UNK" and adds
    guide bonds to a variable number of protien residue(s).

    Args
        gp_to_ptn_idx0: A map from the idx0 location of each guide post (keys; gp_idx0)
            to the idx0 location of the "true" corresponding frame (values; ptn_idx0). 
            The exact mapping is usually only known during training.
            The value of the ptn_idx0 only really matters when `placement` 
            is "exact" or "nearby".
        indep: Independent RF features.
        placement: choices=("exact", "local", "anywhere")
            See `add_guide_post_bond_feats` for a desription of each.

    Returns
        indep: Modified to have guide post features.
        is_diffused: True = the residues should move during diffusion.
    '''
    gp_idx0, ptn_idx0 = zip(*gp_to_ptn_idx0.items())
    gp_idx0, ptn_idx0 = torch.tensor(gp_idx0), torch.tensor(ptn_idx0)
    L_tot = indep.xyz.shape[0]
    n_gp = len(gp_idx0)
    is_protein = ~indep.is_sm[:-n_gp]
    assert guide_post_residues_at_end(gp_idx0.tolist(), L=L_tot), 'The guide posts must be clustered at the "right side" of the RF features.'

    # Set guide post tokens as UNK
    indep.seq[gp_idx0] = rf2aa.chemical.aa2num['UNK']

    # Add large idx jumps between guide post frames. Unsure if necesary.
    idx_last_non_gp = indep.idx[-n_gp - 1]
    idx_gp = rf2aa.chemical.CHAIN_GAP * torch.arange(1, n_gp + 1) + idx_last_non_gp
    indep.idx[-n_gp:] = idx_gp

    # Add in the guide bonds
    indep.bond_feats = add_guide_post_bond_feats(
        indep.bond_feats[:-n_gp, :-n_gp], 
        ptn_idxs=ptn_idx0, 
        window_size_low=5,
        window_size_high=40, 
        is_ptn=is_protein,
        placement=placement
    )
   
    # Make each guide post their own chain
    indep.same_chain = add_guide_post_same_chain(indep.same_chain[:-n_gp, :-n_gp], n_gp)

    # Note which residues should diffuse
    is_diffused = torch.ones_like(indep.is_sm, dtype=bool)
    is_diffused[gp_idx0] = False
    is_diffused[indep.is_sm] = False

    return indep, is_diffused

def guide_post_residues_at_end(gp_idx0: List[int], L: int) -> bool:
    '''Checks if all of the guide post residues are in a contigous block
    at the "C-terminus" (right side) of RF features.'''
    n_gp = len(gp_idx0)
    return set(gp_idx0) == set(range(L - n_gp, L))

def append_indices(t: torch.Tensor, idxs: torch.Tensor, dim: int=0, 
                   from_fill_value=None, to_fill_value=None) -> torch.Tensor:
    '''
    Append values at "idxs" to the end of the tensor "t" along the given dim

    Args
        from_fill_value: Replace the original idxs with this value.
        to_fill_value: Force this value in the appended indices. If None, copy the original values.
    '''
    t = t.clone()
    assert idxs.dim() == 1
    
    to_append = torch.index_select(t, dim, idxs)
    if from_fill_value is not None:
        t[idxs] = torch.full_like(t[idxs], from_fill_value, dtype=t.dtype)  # just ensures correct dtype is maintained
    if to_fill_value is not None:
        to_append = torch.full_like(to_append, to_fill_value, dtype=t.dtype)

    return torch.concat([t, to_append], dim)

def calc_window_boundaries(idx: int, window_size: int, offset: int, max_window_end: int=None) -> Tuple[int]:
    '''
    Return the first and last index (exclusive) of a defined window that overlaps the idx with the desired offset.
    '''
    window_start = idx - offset
    window_start = max(window_start, 0)
    window_end = window_start + window_size
    
    if max_window_end is not None:
        window_end = min(max_window_end, window_end)
    
    return window_start, window_end

def sample_window_boundaries(idx: int, window_size_low: int, window_size_high: int,
                             max_window_end: int=None) -> Tuple[int]:
    '''
    Return the first and last index (exclusive) of a variably sized
    window that overlaps the idx.
    '''
    sampled_window_size = int(torch.randint(window_size_low, window_size_high, size=(1,)))
    window_anchor = int(torch.randint(sampled_window_size, size=(1,)))
    window_start, window_end = calc_window_boundaries(idx, sampled_window_size, window_anchor, max_window_end)
    return window_start, window_end

def add_guide_post_bond_feats(bond_feats: torch.Tensor, ptn_idxs: torch.Tensor, 
                              window_size_low: int, window_size_high: int, is_ptn: torch.Tensor=None,
                              placement: str=None) -> torch.Tensor:
    '''Add bond features between the guide_posts and the indicated ptn_idxs
    bond_feats: (L_prev, L_prev). Existing bond features in the protein

    Args
        ptn_idxs: (None,). The true idx the guide posts should end up at.
        window_size_low: If `placement` is "local", this is the smallest window that will be drawn around the true protein index.
        window_size_high: If `placement` is "local", this is the largest window that will be drawn around the true protein index.
        is_ptn: (L_prev). True = this RFrame represents a protein.
        placement:
            "exact": Only one "guide bond" is added between the guide post and the protein location it should end up at.
            "local": Add "guide bonds" between the guide post and protein residues in a window around the true protein location.
            "anywhere": Add "guide bonds" between the guide post and all protein residues.
            None: Sample one of the above three modes for each contig.

    Returns
        Expanded and updated bond feature tensor.
    '''
    bond_feats = bond_feats.clone()
    L_prev = bond_feats.shape[0]
    n_gp = len(ptn_idxs)
    L_tot = L_prev + n_gp
    btype = 7  # the only unused bond type
    ptn_to_gp_idx = {j: i for i,j in enumerate(ptn_idxs.int().tolist())}
    
    # add existing bond feats
    bond_feats_new = torch.zeros(L_tot,L_tot, dtype=bond_feats.dtype)
    bond_feats_new[:L_prev, :L_prev] = bond_feats

    # reconstruct motif_mask
    motif_mask = torch.zeros(L_prev, dtype=bool)
    motif_mask[ptn_idxs] = True
    
    # Add guide bond features indepedently for each contig
    corner = torch.zeros(L_prev, n_gp, dtype=bond_feats.dtype)
    for contig_mask in split_islands(motif_mask):
        contig_ptn_idxs = torch.where(contig_mask)[0]
        contig_gp_idxs = torch.tensor([ptn_to_gp_idx[int(x)] for x in contig_ptn_idxs])
        
        if placement is None:
            placement = np.random.choice(['exact', 'local', 'anywhere'], p=[0.1, 0.3, 0.6])
            
        if placement == 'exact':
            corner[contig_ptn_idxs, contig_gp_idxs] = btype
        
        elif placement == 'local':
            sampled_window_size = int(torch.randint(window_size_low, window_size_high, size=(1,)))
            window_anchor = int(torch.randint(sampled_window_size, size=(1,)))
            
            for gp_idx, ptn_idx in zip(contig_gp_idxs, contig_ptn_idxs):
                window_start, window_end = calc_window_boundaries(ptn_idx, sampled_window_size, window_anchor, L_prev)
                corner[window_start:window_end, gp_idx] = btype
                
        elif placement == 'anywhere':
            corner[:, contig_gp_idxs] = btype
            
    # make sure spread out guide bonds don't spill over to sm frames
    if is_ptn is not None:
        corner[~is_ptn] = 0
    
    # add to bond feats new
    bond_feats_new[:L_prev, -n_gp:] = corner
    bond_feats_new[-n_gp:, :L_prev] = corner.T
    
    return bond_feats_new

def add_guide_post_same_chain(same_chain: torch.Tensor, n_gp: int) -> torch.Tensor:
    '''
    Modify the same_chain feature to include the guide posts.
    Add each guide post as a new chain. They have no correspondence to the chain they came from.
    same_chain: 2D tensor. True if two frames are from the same chain, else False.
    n_gp: Number of guide post frames that are being added.
    '''
    same_chain = same_chain.clone()
    L_prev = same_chain.shape[0]
    L_tot = L_prev + n_gp

    same_chain_new = torch.zeros(L_tot, L_tot, dtype=bool)
    same_chain_new[:L_prev, :L_prev] = same_chain
    same_chain_new[L_prev:L_tot, L_prev:L_tot] = torch.eye(n_gp, dtype=bool)

    return same_chain_new

def number_islands(x: torch.Tensor) -> torch.Tensor:
    '''
    Sequentially numbers "islands" of True.
    
    Ex: 
    In:  [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]  (but a bool dtype)
    Out: [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0]
    
    '''
    assert x.dtype == torch.bool
    assert x.ndim == 1
    
    # add padding (to prevent edge effects)
    n_pad = 5
    x = torch.cat([torch.zeros(n_pad, dtype=bool), x, torch.zeros(n_pad, dtype=bool)])

    x = torch.diff(x).int()
    x = torch.cat([torch.tensor([0]), torch.cumsum(x, 0)])
    x = (x + 1) / 2  # all islands are integers now
    x[x%1 != 0] = 0

    # remove padding
    x = x[n_pad:-n_pad]
    
    return x.int()

def split_islands(x: torch.Tensor) -> torch.Tensor:
    '''
    Split each island into its own row/category.
    
    Ex:
    In:   [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    Out: [[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]]
    ''' 
    assert x.dtype == torch.bool
    assert x.ndim == 1
    
    numbered_islands = number_islands(x).long()
    n_islands = numbered_islands.max()
    categorical_islands = torch.eye(n_islands+1)[numbered_islands]
    categorical_islands = categorical_islands.T[1:]  # remove the "0" category. It's the "sea".
    return categorical_islands.bool()

#######################
# Functions to adapt existing contig inputs/outputs if using the guide post features
#######################

def greedy_guide_post_correspondence(ptn_xyz: torch.Tensor, gp_xyz: torch.Tensor) -> dict:
    '''
    Which guide post frame is closest to which protein frame? Distance metric: RMSD over N, CA, C atoms.
    Assignment done in a greedy manner.
    
    Args
        ptn_xyz (L_ptn, 14, 3). Protein xyz coordinates, with the first three atoms being the N, CA and C coordinates.
        gp_xyz (n_gp, 3, 3). Guide post N, CA, C coordinates. 
            (Really, (n_gp, None, 3) will be chopped down to (n_gp, 3, 3).)
    
    Returns
        gp_to_ptn_idx: dict[guide_post_index: protein_index]. These are 0-indexed. They can be used to index ptn_xyz.
        sampled_mask: Contig string of infered guide post locations
    '''
    bb_xyz = ptn_xyz[:, :3, :]
    gp_xyz = gp_xyz[:, :3, :]
    n_gp = gp_xyz.shape[0]
    
    bb_xyz = bb_xyz[:, None, :, :]
    gp_xyz = gp_xyz[None, :, :, :]    
    diff = bb_xyz - gp_xyz
    rmsd = diff.pow(2).sum(-1).sum(-1).sqrt()
    
    # greedy assignment
    ptn_idxs = []
    gp_idxs = []
    for _ in range(n_gp):
        vals, idxs = rmsd.min(-1)
        ptn_idx = vals.argmin()
        gp_idx = idxs[ptn_idx]
        ptn_idxs.append(int(ptn_idx))
        gp_idxs.append(int(gp_idx))
        
        rmsd[ptn_idx, :] = torch.inf
        rmsd[:, gp_idx] = torch.inf
        
    assert len(ptn_idxs) == len(set(ptn_idxs)), 'Two guide posts mapped to the same protein index!'
    assert len(gp_idxs) == len(set(gp_idxs)), 'Two protein indices mapped to the same guide post!'
    
    gp_to_ptn_idx = dict(zip(gp_idxs, ptn_idxs))
    return gp_to_ptn_idx

def contig_to_guide_post_string(contig_str: str, length_range: str=None) -> str:
    '''
    Converts a "contig string" where you specify the contig locations in the sequence,
    to a "guide post string" where you don't.
    
    Ex:
      contig_str: '0-40,A1051-1051,10-40,A1083-1083,10-40,A1110-1110,10-40,A1180-1180,0-40'
      guide_post_str: '30-200,A1051-1051,A1083-1083,A1110-1110,A1180-1180'
      
    Args:
        contig_str: Specifies contig location in the sequence with variably sized gaps.
        length_range: Overrides the calculated L_low, L_high values from contig_str
    '''
    
    L_low = 0
    L_high = 0
    contigs = []

    for chunk in contig_str.split(','):
        if chunk[0].isalpha():
            contigs.append(chunk)
        else:
            s, e = parse_range(chunk)
            L_low += s
            L_high += e
            
    if length_range is None:
        length_range = f'{L_low}-{L_high}'

    guide_post_str = f'{length_range},{",".join(contigs)}'
    return guide_post_str
        
def parse_range(x: str) -> Tuple[int, int]:
    '''
    Ex:
    '10-40' -> (10, 40)
    '10' -> (10, 10)
    '''
    if '-' in x:
        s, e = x.split('-')
        return int(s), int(e)
    else:
        return int(x), int(x)

def parse_contig(contig):
    '''
    Return the chain, start and end residue in a contig or gap str.

    Ex:
    'A4-8' --> 'A', 4, 8
    'A5'   --> 'A', 5, 5
    '4-8'  --> None, 4, 8
    'A'    --> 'A', None, None
    '''

    # is contig
    if contig[0].isalpha():
      ch = contig[0]
      if len(contig) > 1:
        s, e = parse_range(contig[1:])
      else:
        s, e = None, None
    # is gap
    else:
      ch = None
      s, e = parse_range(contig)

    return ch, s, e

def expand(mask_str):
    '''
    Ex: '2,A3-5,3' --> [None, None, (A,3), (A,4), (A,5), None, None, None]
    '''
    expanded = []
    for l in mask_str.split(','):
      ch, s, e = parse_contig(l)
      
      # contig
      if ch:
        expanded += [(ch, res) for res in range(s, e+1)]
      # gap
      else:
        expanded += [None for _ in range(s)]
    
    return expanded
  
def contract(pdb_idx):
    '''
    Inverse of expand
    Ex: [None, None, (A,3), (A,4), (A,5), None, None, None] --> '2,A3-5,3'
    '''
    
    contracted = []
    l_prev = (None, -200)
    first_el_written = False
    
    for l_curr in pdb_idx:
      if l_curr is None:
        l_curr = (None, -100)
        
      # extend gap
      if l_curr == l_prev:
        L_gap += 1
        
      # extend con
      elif l_curr == (l_prev[0], l_prev[1]+1):
        con_e = l_curr[1]
        
      # new gap
      elif (l_curr != l_prev) and (l_curr[0] is None):
        # write prev con
        if 'con_ch' in locals():
          contracted.append(f'{con_ch}{con_s}-{con_e}')
        
        L_gap = 1
        
      # new con
      elif (l_curr != l_prev) and isinstance(l_curr[0], str):
        # write prev con
        if isinstance(l_prev[0], str) and ('con_ch' in locals()):
          contracted.append(f'{con_ch}{con_s}-{con_e}')
        # write prev gap
        elif 'L_gap' in locals():
          contracted.append(f'{L_gap}-{L_gap}')

        con_ch = l_curr[0]
        con_s = l_curr[1]
        con_e = l_curr[1]
        
      # update l_prev
      l_prev = l_curr
      
    # write last element
    if isinstance(l_prev[0], str) and ('con_ch' in locals()):
      contracted.append(f'{con_ch}{con_s}-{con_e}')
    elif 'L_gap' in locals():
      contracted.append(f'{L_gap}-{L_gap}')
    
    return ','.join(contracted)

def get_infered_mappings(gp_to_sampled_mask_idx0: dict, gp_alone_to_diffused_idx0: dict, original_mappings: dict) -> dict:
    '''
    The original contig mappings are useless. Because guide posts are free to end up anywhere in the protein,
    we need to make new mappings.

    Args
    -------------
    gp_to_sampled_mask_idx0: This mapping was set in the `insert_contig` function.
        (key) gp_idx0: idx0 of the new frame that was appended as a guide post. Could use this to index into indep.xyz.
        (value) sampled_mask_idx0: If the original sampled mask were expanded, the gp would have come from this index.
    gp_alone_to_diffused_idx0: This mapping was inferred from the final diffused structure.
        (key) gp_alone_idx0: gp_alone_idx0 = gp_idx0 - L_protein. If the guide posts were the only features, this is the gp idx0.
        (value) diffused_idx0: Idx0 of the inferred placement of the guidepost from the final diffusion timestep. 
            Could use this to index into indep.xyz.
    original_contig_map: The original contig_map made when the contig string was sampled at the BEGINNING of a trajectory.

    Returns
    -------------
    new_mappings: Dictionary of the (infered) correspondence between the motif in the diffused protein and the reference protein.
    '''
    original_sampled_mask_expanded = expand(original_mappings['sampled_mask'][0])
    L_ptn = len(original_sampled_mask_expanded)

    # If you removed the diffused protein features, map from gp idx0 to idx0 of the (expanded) original sampled mask.
    gp_alone_to_sampled_mask_idx0 = {k-L_ptn: v for k, v in gp_to_sampled_mask_idx0.items()}

    # What diffused idx0 should have the reference pdb geometry?
    ref_pdb_idx_to_diffused_idx0 = {original_sampled_mask_expanded[gp_alone_to_sampled_mask_idx0[gp_idx0]]: diff_idx0
                                    for gp_idx0, diff_idx0 in gp_alone_to_diffused_idx0.items()}
    con_ref_pdb_idx = original_mappings['con_ref_pdb_idx']
    con_hal_idx0 = [ref_pdb_idx_to_diffused_idx0[pdb_idx] for pdb_idx in con_ref_pdb_idx]
    con_hal_pdb_idx = [('A', i+1) for i in con_hal_idx0]  # Assumes we diffused one chain.

    # Make a new sampled mask based on the infered motif placement.
    diffused_idx0_to_ref_pdb_idx = {v: k for k,v in ref_pdb_idx_to_diffused_idx0.items()}
    new_sampled_mask_expanded = [diffused_idx0_to_ref_pdb_idx.get(i) for i in range(L_ptn)]

    new_mappings = {
        'con_ref_pdb_idx': original_mappings['con_ref_pdb_idx'],
        'con_ref_idx0': original_mappings['con_ref_idx0'],
        'con_hal_pdb_idx': con_hal_pdb_idx,
        'con_hal_idx0': con_hal_idx0,
        'sampled_mask': contract(new_sampled_mask_expanded)
    }

    return new_mappings
   
