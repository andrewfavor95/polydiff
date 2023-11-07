import functools
import torch
import assertpy
from collections import defaultdict
import torch.nn.functional as F
import dataclasses
from icecream import ic
from assertpy import assert_that
from rf2aa.chemical import NAATOKENS, MASKINDEX, NTOTAL, NHEAVYPROT, NHEAVY, NHEAVYNUC, UNKINDEX
import rf2aa.util
from rf2aa import parsers
from dataclasses import dataclass
from rf2aa.data_loader import MSAFeaturize, MSABlockDeletion, merge_a3m_homo, merge_a3m_hetero
from rf2aa.kinematics import xyz_to_c6d, c6d_to_bins, xyz_to_t2d, get_chirals
from rf2aa.util_module import XYZConverter
import rf2aa.tensor_util
import torch
import copy
import numpy as np
from kinematics import get_init_xyz
import chemical
from rf2aa.chemical import MASKINDEX, DNAMASKINDEX, RNAMASKINDEX
import rf2aa.chemical
import util
import inference.utils
import networkx as nx
import itertools
import random
from typing import Optional 
import guide_posts as gp
import rotation_conversions
import atomize
import sys 
import mask_generator
from rf2aa.util_module import XYZConverter
import ipdb

NINDEL=1
NTERMINUS=2
NMSAFULL=NAATOKENS+NINDEL+NTERMINUS
NMSAMASKED=NAATOKENS+NAATOKENS+NINDEL+NINDEL+NTERMINUS

MSAFULL_N_TERM = NAATOKENS+NINDEL
MSAFULL_C_TERM = MSAFULL_N_TERM+1

MSAMASKED_N_TERM = 2*NAATOKENS + 2*NINDEL
MSAMASKED_C_TERM = 2*NAATOKENS + 2*NINDEL + 1

N_TERMINUS = 1
C_TERMINUS = 2

@dataclass
class Indep:
    seq:  torch.Tensor # [L]
    xyz:  torch.Tensor # [L, 36?, 3]
    xyz2: torch.Tensor # DJ - the original xyz
    natstack: torch.Tensor 
    maskstack: torch.Tensor
    Ls: list
    idx:  torch.Tensor 

    # SM specific
    bond_feats: torch.Tensor
    chirals: torch.Tensor
    atom_frames: torch.Tensor
    same_chain: torch.Tensor
    is_sm: torch.Tensor
    terminus_type: torch.Tensor

    # Nucleic specific
    is_protein: torch.Tensor
    is_dna: torch.Tensor
    is_rna: torch.Tensor
    
    # # DJ new - introduced for refinement model
    # metadata: dict

    # # dj - new subsymmetric template information 
    # subsymm_seq: Optional[torch.Tensor] = None
    # subsymm_xyz: Optional[torch.Tensor] = None
    # mask_t_2d_subsymm: Optional[torch.Tensor] = None




    def write_pdb(self, path, **kwargs):
        # assert 0, "PROBABLY SHOULDNT USE THIS WITH NA"
        seq = self.seq
        seq = torch.where(seq == 20, 0, seq)
        seq = torch.where(seq == 21, 0, seq)
        seq = torch.where(seq == 26, 22, seq)
        seq = torch.where(seq == 31, 27, seq)
        # This is a hacky way of specifying same_chain.
        # TODO: convert same_chain from a 2D input to a 1D input
        chain_Ls = []
        G = nx.from_numpy_array(self.same_chain.numpy())
        connected_components = list(nx.connected_components(G))
        longest_connected_component = list(connected_components.pop(connected_components.index(max(connected_components, key=len))))
        # other_component = list(functools.reduce(lambda a,b: a.update(b), connected_components))
        other_component = []
        for cc in connected_components:
            other_component.extend(list(cc))
        # chain_letters = np.zeros((self.length(),), dtype='string_')
        chain_letters = np.chararray((self.length(),), unicode=True)
        chain_letters[longest_connected_component] = 'A'
        chain_letters[other_component] = 'B'
        # rf2aa.util.writepdb(path,
        #     torch.nan_to_num(self.xyz[:,:14]), seq, idx_pdb=self.idx, chain_letters=chain_letters, **kwargs)
        rf2aa.util.writepdb(path, torch.nan_to_num(self.xyz), seq, self.bond_feats, 
                            idx_pdb=self.idx, 
                            chain_letters=chain_letters, 
                            **kwargs)

    def ca_dists(self):
        xyz_ca = self.xyz[:,1]
        ca_sm = xyz_ca[self.is_sm]
        ca_prot = xyz_ca[~self.is_sm]
        dist = torch.norm(ca_prot[:,None] - ca_sm[None], dim=-1)
        dist = dist.min(dim=-1)[0]
        return dist
    
    def center_of_mass(self, mask=None):
        xyz_ca = self.xyz[:,1]
        if mask is not None:
            xyz_ca = xyz_ca[mask]
        return torch.mean(xyz_ca, dim=0)
    
    def length(self):
        return self.seq.shape[0]
    

    def has_c_terminal_residue(self):
        return is_monotonic(self.idx)

    def has_n_terminal_residue(self):
        return torch.flip(is_monotonic(-torch.flip(self.idx,(0,))),(0,))
    
    def human_readable_seq(self):
        return human_readable_seq(self.seq)
        # return np.chararray([rf2aa.chemical.num2aa[s] for s in self.seq], unicode=False)
    
    def has_heavy_atoms_and_seq(self, atom_mask):
        ipdb.set_trace()
        want_atom_mask = rf2aa.util.allatom_mask[self.seq]
        has_all_heavy_atoms = (want_atom_mask[:,:rf2aa.chemical.NHEAVYPROT] == atom_mask[:,:rf2aa.chemical.NHEAVYPROT]).all(dim=-1)
        has_sequence = self.seq < 20
        print('aa_model.py, line 140, only selecting AA tokens...')
        return has_all_heavy_atoms * ~self.is_sm * has_sequence
    
    def is_valid_for_atomization(self, atom_mask):
        return self.has_c_terminal_residue() * self.has_n_terminal_residue() * self.has_heavy_atoms_and_seq(atom_mask)

def human_readable_seq(seq):
    return [rf2aa.chemical.num2aa[s] for s in seq]

def is_monotonic(idx):
    idx_pad = torch.concat([idx, torch.tensor([9999])])
    return (idx_pad[1:] - idx_pad[:-1] == 1)


@dataclass
class RFI:
    msa_latent: torch.Tensor
    msa_full: torch.Tensor
    seq: torch.Tensor
    seq_unmasked: torch.Tensor
    xyz: torch.Tensor
    sctors: torch.Tensor
    idx: torch.Tensor
    bond_feats: torch.Tensor
    dist_matrix: torch.Tensor
    chirals: torch.Tensor
    atom_frames: torch.Tensor
    t1d: torch.Tensor
    t2d: torch.Tensor
    xyz_t: torch.Tensor
    alpha_t: torch.Tensor
    mask_t: torch.Tensor
    same_chain: torch.Tensor
    is_motif: torch.Tensor
    msa_prev: torch.Tensor
    pair_prev: torch.Tensor
    state_prev: torch.Tensor

@dataclass
class RFO:
    logits: torch.Tensor      # ([1, 61, L, L], [1, 61, L, L], [1, 37, L, L], [1, 19, L, L])
    logits_aa: torch.Tensor   # [1, 80, 115]
    logits_pae: torch.Tensor  # [1, 64, L, L]
    logits_pde: torch.Tensor  # [1, 64, L, L]
    p_bind: torch.Tensor      # [1,1]
    xyz: torch.Tensor         # [40, 1, L, 3, 3]
    alpha_s: torch.Tensor     # [40, 1, L, 20, 2]
    xyz_allatom: torch.Tensor # [1, L, 36, 3]
    lddt: torch.Tensor        # [1, 50, L]
    msa: torch.Tensor
    pair: torch.Tensor
    state: torch.Tensor
    symmsub: torch.Tensor

    # dataclass.astuple returns a deepcopy of the dataclass in which
    # gradients of member tensors are detached, so we define a 
    # custom unpacker here.
    def unsafe_astuple(self):
        return tuple([self.__dict__[field.name] for field in dataclasses.fields(self)])

    def get_seq_logits(self):
        return self.logits_aa.permute(0,2,1)
    
    def get_xyz(self):
        return self.xyz_allatom[0]

def filter_het(pdb_lines, ligand):
    lines = []
    hetatm_ids = []
    for l in pdb_lines:
        if 'HETATM' not in l:
            continue
        if l[17:17+4].strip() != ligand:
            continue
        lines.append(l)
        hetatm_ids.append(int(l[7:7+5].strip()))

    violations = []
    for l in pdb_lines:
        if 'CONECT' not in l:
            continue
        ids = [int(e.strip()) for e in l[6:].split()]
        if all(i in hetatm_ids for i in ids):
            lines.append(l)
            continue
        if any(i in hetatm_ids for i in ids):
            ligand_atms_bonded_to_protein = [i for i in ids if i in hetatm_ids]
            violations.append(f'line {l} references atom ids in the target ligand {ligand}: {ligand_atms_bonded_to_protein} and another atom')
    if violations:
        raise Exception('\n'.join(violations))
    return lines



def make_indep(pdb, ligand=None):
    # self.target_feats = iu.process_target(self.inf_conf.input_pdb, parse_hetatom=True, center=False)
    # init_protein_tmpl=False, init_ligand_tmpl=False, init_protein_xyz=False, init_ligand_xyz=False,
    #     parse_hetatm=False, n_cycle=10, random_noise=5.0)
    chirals = torch.Tensor()
    atom_frames = torch.zeros((0,3,2))

    # xyz_prot, mask_prot, idx_prot, seq_prot = parsers.parse_pdb(pdb, xyz27=True, seq=True)
    xyz_prot, mask_prot, idx_prot, seq_prot = parsers.parse_pdb(pdb, seq=True)

    target_feats = inference.utils.parse_pdb(pdb)
    xyz_prot, mask_prot, idx_prot, seq_prot = target_feats['xyz'], target_feats['mask'], target_feats['idx'], target_feats['seq']
    
    is_protein = np.logical_and((0  <= seq_prot),(seq_prot <= 21)).squeeze()
    is_nucleic = np.logical_and((22 <= seq_prot),(seq_prot <= 31)).squeeze()

    chain_length_list = util.get_chain_lengths(target_feats['pdb_idx'])

    # remove hydrogens

    xyz_prot[is_protein,NHEAVYPROT:] = 0
    xyz_prot[is_nucleic,NHEAVYNUC:] = 0
    mask_prot[is_protein,NHEAVYPROT:] = False
    mask_prot[is_nucleic,NHEAVYNUC:] = False
    # xyz_prot[:,14:] = 0 # remove hydrogens
    # mask_prot[:,14:] = False

    xyz_prot = torch.tensor(xyz_prot)
    mask_prot = torch.tensor(mask_prot)
    protein_L, nprotatoms, _ = xyz_prot.shape
    msa_prot = torch.tensor(seq_prot)[None].long()
    ins_prot = torch.zeros(msa_prot.shape).long()
    a3m_prot = {"msa": msa_prot, "ins": ins_prot}
    if ligand:
        with open(pdb, 'r') as fh:
            stream = [l for l in fh if "HETATM" in l or "CONECT" in l]
        stream = filter_het(stream, ligand)
        if not len(stream):
            raise Exception(f'ligand {ligand} not found in pdb: {pdb}')

        mol, msa_sm, ins_sm, xyz_sm, _ = parsers.parse_mol("".join(stream), filetype="pdb", string=True)
        a3m_sm = {"msa": msa_sm.unsqueeze(0), "ins": ins_sm.unsqueeze(0)}
        G = rf2aa.util.get_nxgraph(mol)
        atom_frames = rf2aa.util.get_atom_frames(msa_sm, G)
        N_symmetry, sm_L, _ = xyz_sm.shape
        Ls = [protein_L, sm_L]
        a3m = merge_a3m_hetero(a3m_prot, a3m_sm, Ls)
        msa = a3m['msa'].long()
        chirals = get_chirals(mol, xyz_sm[0])
        if chirals.numel() !=0:
            chirals[:,:-1] += protein_L
    else:
        Ls = [msa_prot.shape[-1], 0]
        N_symmetry = 1
        msa = msa_prot

    xyz = torch.full((N_symmetry, sum(Ls), NTOTAL, 3), np.nan).float()
    mask = torch.full(xyz.shape[:-1], False).bool()
    xyz[:, :Ls[0], :nprotatoms, :] = xyz_prot.expand(N_symmetry, Ls[0], nprotatoms, 3)
    if ligand:
        xyz[:, Ls[0]:, 1, :] = xyz_sm
    xyz = xyz[0]
    mask[:, :protein_L, :nprotatoms] = mask_prot.expand(N_symmetry, Ls[0], nprotatoms)
    idx_sm = torch.arange(max(idx_prot),max(idx_prot)+Ls[1])+200
    idx_pdb = torch.concat([torch.tensor(idx_prot), idx_sm])
    
    seq = msa[0]
    # seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, 
    #     p_mask=0.0, params={'MAXLAT': 128, 'MAXSEQ': 1024, 'MAXCYCLE': n_cycle}, tocpu=True)
    bond_feats = torch.zeros((sum(Ls), sum(Ls))).long()

    bond_feats[:Ls[0], :Ls[0]] = rf2aa.util.get_protein_bond_feats(Ls[0])
    # bond_feats[:Ls[0], :Ls[0]] = rf2aa.util.bond_feats_from_Ls(chain_length_list)
    # bond_feats2 = rf2aa.util.bond_feats_from_Ls(chain_length_list)

    # ipdb.set_trace()
    if ligand:
        bond_feats[Ls[0]:, Ls[0]:] = rf2aa.util.get_bond_feats(mol)


    same_chain = torch.zeros((sum(Ls), sum(Ls))).long()
    same_chain[:Ls[0], :Ls[0]] = 1
    same_chain[Ls[0]:, Ls[0]:] = 1
    is_sm = torch.zeros(sum(Ls)).bool()
    is_sm[Ls[0]:] = True
    assert len(Ls) <= 2, 'multi chain inference not implemented yet'
    terminus_type = torch.zeros(sum(Ls))
    terminus_type[0] = N_TERMINUS
    terminus_type[Ls[0]-1] = C_TERMINUS

    xyz = get_init_xyz(xyz[None, None], is_sm).squeeze()
    ###TODO: currently network needs values at 0,2 indices of tensor, need to remove this reliance
    xyz[is_sm, 0] = 0
    xyz[is_sm, 2] = 0

    # Now check where diff polymer types are:
    is_protein, is_dna, is_rna  = mask_generator.get_polymer_type_masks(seq)

    indep = Indep(
        seq,
        xyz,
        xyz.clone(), # DJ -- three template, dummy xyz for now 
        target_feats,
        mask,
        Ls,
        idx_pdb,
        # SM specific
        bond_feats,
        chirals,
        atom_frames,
        same_chain,
        is_sm,
        terminus_type,
        # NA specific
        is_protein,
        is_dna,
        is_rna,
        )
    return indep



def add_fake_frame_legs(xyz, is_atom):
    # HACK.  ComputeAllAtom in the network requires N and C coords even for atomized residues,
    # However, these have no semantic value.  TODO: Remove the network's reliance on these coordinates.
    xyz = xyz.clone()
    atom_xyz = xyz[is_atom, 1]
    xyz[is_atom,:3] = atom_xyz[...,None,:]
    xyz[is_atom, 0] += torch.normal(torch.zeros_like(xyz[is_atom, 0]), std=1.0)
    xyz[is_atom, 2] += torch.normal(torch.zeros_like(xyz[is_atom, 2]), std=1.0)
    return xyz

class Model:

    def __init__(self, conf):
        self.conf = conf
        self.NTOKENS = rf2aa.chemical.NAATOKENS
        self.atomizer = None
        self.converter = XYZConverter()
        self.make_indep = make_indep

    # def forward(self, rfi, **kwargs):
    #     # ipdb.set_trace()
    #     rfi_dict = dataclasses.asdict(rfi)
    #     # assert set(rfi_dict.keys()) - set()
    #     return RFO(*self.model(**{**rfi_dict, **kwargs}))

    def forward(self, rfi, N_cycle=1, **kwargs):
        model = self.model 
        model.eval()
        assert N_cycle > 0

        if N_cycle == 1:
            rfi_dict = dataclasses.asdict(rfi)
            # ipdb.set_trace()
            return RFO(*model(**{**rfi_dict, **kwargs}))

        else:

            # Do the first N-1 recycles 
            with torch.no_grad():
                for i in range(N_cycle-1):

                    if i == 0:
                        rfi_dict = dataclasses.asdict(rfi)
                        input = {**rfi_dict, **kwargs}
                        input['msa_prev'] = None
                        input['pair_prev'] = None
                        input['state_prev'] = None
                        input['return_raw'] = True
                        out = model(**input)

                    else:
                        msa_prev, pair_prev, xyz_prev, state, alpha_prev, _ = out
                        rfi_dict = dataclasses.asdict(rfi)
                        rfi_dict['msa_prev']    = msa_prev
                        rfi_dict['pair_prev']   = pair_prev
                        rfi_dict['xyz']         = xyz_prev
                        rfi_dict['state_prev']  = state

                        input = {**rfi_dict, **kwargs}
                        input['return_raw'] = True
                        out = model(**input)

                # Do the last recycle, and return RFO 
                msa_prev, pair_prev, xyz_prev, state, alpha_prev, _ = out
                rfi_dict = dataclasses.asdict(rfi)
                rfi_dict['msa_prev']    = msa_prev
                rfi_dict['pair_prev']   = pair_prev
                rfi_dict['xyz']         = xyz_prev
                rfi_dict['state_prev']  = state

                input = {**rfi_dict, **kwargs}
                input['return_raw'] = False
                # ipdb.set_trace()
                # with grad 
                return RFO(*model(**input))


    def insert_contig(self, indep, contig_map, partial_T=False, refine=False):
        """
        Assembl
        """
        o = copy.deepcopy(indep)


       # Some Andrew shit up in here:
        # First, assign polymer ids to diff chains:
        full_complex_idx0 = torch.tensor(contig_map.rf, dtype=torch.int64)
        chain_breaks = inference.utils.get_breaks(full_complex_idx0)
        chain_range_inds = inference.utils.find_template_ranges(full_complex_idx0, return_inds=True)

        o.is_protein = torch.full((len(contig_map.hal),), 0).bool()
        o.is_dna = torch.full((len(contig_map.hal),), 0).bool()
        o.is_rna = torch.full((len(contig_map.hal),), 0).bool()

        o.Ls = []
        contig_map_hal = []
        for chain_num, ((start_ind,stop_ind), polymer_type) in enumerate(zip(chain_range_inds, contig_map.polymer_chains)):

            # Assign polymer types
            if polymer_type in ['protein','prot','aa','AA','a','A']:
                o.is_protein[start_ind:stop_ind+1] = True

            elif polymer_type in ['dna','DNA','d','D']:
                o.is_dna[start_ind:stop_ind+1] = True

            elif polymer_type in ['rna','RNA','r','R']:
                o.is_rna[start_ind:stop_ind+1] = True

            o.Ls.append(1+stop_ind-start_ind)

            # Now modify contig_map.hal to make things into all the real chain breaks
            for ind in range(start_ind, stop_ind+1):
                contig_map_hal.append((contig_map.chain_order[chain_num], contig_map.hal[ind][1]))

        print('TO DO: add input bond features')
        # # get ref inds:
        # ref_ranges = []
        # for contig in contig_map.contigs[0].split(' '):
        #     for subcontig in contig.split(','):
        #         if subcontig[0].isalpha():
        #             start, stop = 

        #     all_chunk_ranges = []
        # is_motif_chunk = []
        # last_idx = 0
        # for contig_i in contig_map.sampled_mask:
        #     for subcontig_ij in contig_i.split(','):
        #         if subcontig_ij[0].isalpha(): # if it is a template region:
        #             chain_ij = subcontig_ij[0]
        #             start_resi_ij, stop_resi_ij = [int(idx) for idx in subcontig_ij[1:].split('-')]
                    
        #             len_ij = stop_resi_ij - start_resi_ij
        #             new_idx = last_idx + len_ij

        #             # contig_chunk_ranges.append((last_idx , new_idx))
        #             is_motif_chunk.append(1)
        #             all_chunk_ranges.append((last_idx , new_idx))

        contig_map.hal = contig_map_hal

        print('WARNING: in insert_contig(),ASSUMING NO SMALL MOLECULE, BUT CHANGE LATER!')
        print('Just modify o.Ls[-1] after computing sm stuff')
        o.Ls.append(0)


        # Insert small mol into contig_map
        all_chains = set(ch for ch,_ in contig_map.hal)

        # Not yet implemented due to index shifting
        # assert_that(len(all_chains)).is_equal_to(1)
        print(f'WARNING: only 1 chain supported for now. Found {len(all_chains)} chains: {all_chains}')

        # string
        next_unused_chain = next(e for e in contig_map.chain_order if e not in all_chains)

        # number of small molecule atoms
        n_sm = indep.is_sm.sum()

        # list of indices of small molecule atoms - 0 indexed
        is_sm_idx0 = torch.nonzero(indep.is_sm, as_tuple=True)[0].tolist()
        contig_map.ref_idx0.extend(is_sm_idx0)

        n_protein_hal       = len(contig_map.hal)
        contig_map.hal_idx0 = np.concatenate((contig_map.hal_idx0, np.arange(n_protein_hal, n_protein_hal+n_sm)))

        max_hal_idx = max(i for _, i  in contig_map.hal)

        # NOTE - this makes all small molecules in the same chain
        print(f'WARNING: all small molecules in the same chain. Chain: {next_unused_chain}')
        contig_map.hal.extend(zip([next_unused_chain]*n_sm, range(max_hal_idx+200,max_hal_idx+200+n_sm)))

        chain_id = np.array([c for c, _ in contig_map.hal])

        L_mapped = len(contig_map.hal)
        n_prot   = L_mapped - n_sm
        L_in, NATOMS, _ = indep.xyz.shape


 
        # o.Ls.append(len(is_sm_idx0))
        assert L_mapped==sum(o.Ls)
        

        # initialize xyz for trajectory - slice in protein atoms from original indep
        if not partial_T and not refine:
            o.xyz = torch.full((L_mapped, NATOMS, 3), np.nan)
            o.xyz[contig_map.hal_idx0] = indep.xyz[contig_map.ref_idx0]  
        else:
            o.xyz = indep.xyz 
            # The initial coordinates should be exactly the coordinates that 
            # were parsed out of the protein 

        # initialize seq - slice in sequence from original indep
        o.seq = torch.full((L_mapped,), MASKINDEX)
        o.seq[o.is_protein] = MASKINDEX
        o.seq[o.is_dna] = DNAMASKINDEX
        o.seq[o.is_rna] = RNAMASKINDEX
        o.seq[contig_map.hal_idx0] = indep.seq[contig_map.ref_idx0]

        # slice over the "is_sm" mask from the original indep
        o.is_sm = torch.full((L_mapped,), 0).bool()
        o.is_sm[contig_map.hal_idx0] = indep.is_sm[contig_map.ref_idx0]
        o.same_chain = torch.tensor(chain_id[None, :] == chain_id[:, None])
        
        o.xyz = get_init_xyz(o.xyz[None, None], o.is_sm).squeeze()

        # HACK.  ComputeAllAtom in the network requires N and C coords even for atomized residues,
        # However, these have no semantic value.  TODO: Remove the network's reliance on these coordinates.
        sm_ca = o.xyz[o.is_sm, 1]
        o.xyz[o.is_sm,:3] = sm_ca[...,None,:]
        o.xyz[o.is_sm] += chemical.INIT_CRDS

        

        o.bond_feats = torch.full((L_mapped, L_mapped), 0).long()
        # o.bond_feats[:n_prot, :n_prot] = rf2aa.util.get_protein_bond_feats(n_prot)
        poly_bond_feats = rf2aa.util.bond_feats_from_Ls(o.Ls[:-1])
        for bond_feats_i, (start_i, end_i) in zip(poly_bond_feats, chain_range_inds):
            o.bond_feats[start_i:end_i, start_i:end_i] = poly_bond_feats[start_i:end_i, start_i:end_i]

        # n_prot_ref = L_in-n_sm
        # o.bond_feats[n_prot:, n_prot:] = indep.bond_feats[n_prot_ref:, n_prot_ref:]

        hal_by_ref_d = dict(zip(contig_map.ref_idx0, contig_map.hal_idx0))
        def hal_by_ref(ref):
            return hal_by_ref_d[ref]
        hal_by_ref = np.vectorize(hal_by_ref, otypes=[float])
        o.chirals[...,:-1] = torch.tensor(hal_by_ref(o.chirals[...,:-1]))

        
        o.terminus_type = torch.zeros(L_mapped)
        o.idx = torch.tensor([i for _, i in contig_map.hal])
        last_start = 0
        for start_ind, stop_ind in chain_range_inds:
            o.terminus_type[start_ind] = N_TERMINUS
            o.terminus_type[stop_ind] = C_TERMINUS

            o.idx[start_ind:stop_ind+1] += last_start
            last_start += 200

        # ipdb.set_trace()

        # is_diffused = torch.
        is_diffused_prot = ~torch.from_numpy(contig_map.inpaint_str)
        is_diffused_sm = torch.zeros(n_sm).bool()
        is_diffused = torch.cat((is_diffused_prot, is_diffused_sm))

        # To see the shapes of the indep struct with contig inserted
        # print(rf2aa.tensor_util.info(rf2aa.tensor_util.to_ordered_dict(o)))
        if refine: 
            pass # want to keep original xyz2 because it contains perfect motif  
        else:
            o.xyz2 = o.xyz.clone() # DJ - three template, dummy xyz for now
        return o, is_diffused

    # def insert_contig(self, indep, contig_map, partial_T=False):
    #     o = copy.deepcopy(indep)

    #     # Insert small mol into contig_map
    #     all_chains = set(ch for ch,_ in contig_map.hal)
    #     # Not yet implemented due to index shifting
    #     assert_that(len(all_chains)).is_equal_to(1)
    #     next_unused_chain = next(e for e in contig_map.chain_order if e not in all_chains)
    #     n_sm = indep.is_sm.sum()
    #     is_sm_idx0 = torch.nonzero(indep.is_sm, as_tuple=True)[0].tolist()
    #     contig_map.ref_idx0.extend(is_sm_idx0)
    #     n_protein_hal = len(contig_map.hal)
    #     contig_map.hal_idx0 = np.concatenate((contig_map.hal_idx0, np.arange(n_protein_hal, n_protein_hal+n_sm)))
    #     max_hal_idx = max(i for _, i  in contig_map.hal)
    #     contig_map.hal.extend(zip([next_unused_chain]*n_sm, range(max_hal_idx+200,max_hal_idx+200+n_sm)))
    #     chain_id = np.array([c for c, _ in contig_map.hal])
    #     L_mapped = len(contig_map.hal)
    #     n_prot = L_mapped - n_sm
    #     L_in, NATOMS, _ = indep.xyz.shape
    #     o.xyz = torch.full((L_mapped, NATOMS, 3), np.nan)

    #     o.xyz[contig_map.hal_idx0] = indep.xyz[contig_map.ref_idx0]
    #     o.seq = torch.full((L_mapped,), MASKINDEX)
    #     o.seq[contig_map.hal_idx0] = indep.seq[contig_map.ref_idx0]
    #     o.is_sm = torch.full((L_mapped,), 0).bool()
    #     o.is_sm[contig_map.hal_idx0] = indep.is_sm[contig_map.ref_idx0]
    #     o.same_chain = torch.tensor(chain_id[None, :] == chain_id[:, None])
    #     o.xyz = get_init_xyz(o.xyz[None, None], o.is_sm).squeeze()

    #     o.bond_feats = torch.full((L_mapped, L_mapped), 0).long()
    #     o.bond_feats[:n_prot, :n_prot] = rf2aa.util.get_protein_bond_feats(n_prot)
    #     n_prot_ref = L_in-n_sm
    #     o.bond_feats[n_prot:, n_prot:] = indep.bond_feats[n_prot_ref:, n_prot_ref:]

    #     hal_by_ref_d = dict(zip(contig_map.ref_idx0, contig_map.hal_idx0))
    #     def hal_by_ref(ref):
    #         return hal_by_ref_d[ref]
    #     hal_by_ref = np.vectorize(hal_by_ref, otypes=[float])
    #     o.chirals[...,:-1] = torch.tensor(hal_by_ref(o.chirals[...,:-1]))

    #     o.idx = torch.tensor([i for _, i in contig_map.hal])

    #     o.terminus_type = torch.zeros(L_mapped)
    #     o.terminus_type[0] = N_TERMINUS
    #     o.terminus_type[n_prot-1] = C_TERMINUS

    #     is_diffused_prot = ~torch.from_numpy(contig_map.inpaint_str)
    #     is_diffused_sm = torch.zeros(n_sm).bool()
    #     is_diffused = torch.cat((is_diffused_prot, is_diffused_sm))
    #     is_atom_str_shown = contig_map.atomize_indices2atomname
    #     # The motifs for atomization are double-counted.
    #     if is_atom_str_shown:
    #         is_diffused[list(is_atom_str_shown.keys())] = True
    #     is_res_str_shown = ~is_diffused
    #     # ipdb.set_trace()
    #     use_guideposts = 'inference' in self.conf and self.conf.inference.contig_as_guidepost
    #     o, is_diffused, is_seq_masked, self.atomizer, contig_map.gp_to_ptn_idx0 = transform_indep(o, is_res_str_shown, is_atom_str_shown, use_guideposts, 'anywhere')

    #     # HACK.  ComputeAllAtom in the network requires N and C coords even for atomized residues,
	#     # However, these have no semantic value.  TODO: Remove the network's reliance on these coordinates.
    #     sm_ca = o.xyz[o.is_sm, 1]
    #     o.xyz[o.is_sm,:3] = sm_ca[...,None,:]
    #     o.xyz[o.is_sm] += chemical.INIT_CRDS

    #     # To see the shapes of the indep struct with contig inserted
    #     # print(rf2aa.tensor_util.info(rf2aa.tensor_util.to_ordered_dict(o)))
    #     return o, is_diffused, is_seq_masked


    def prepro(self, 
                indep, 
                t, 
                is_diffused, 
                twotemplate=True, 
                threetemplate=True, 
                t2d_is_revealed=None, 
                is_motif=None, 
                polymer_type_masks=None,
                rfmotif=None):
        """
        Function to prepare inputs to diffusion model
        
            seq (L,22) one-hot sequence 

            msa_masked (1,1,L,48)

            msa_full (1,1,L,25)
        
            xyz_t (L,14,3) template crds (diffused) 

            t1d (1,L,28) this is the t1d before tacking on the chi angles:
                - seq + unknown/mask (21)
                - global timestep (1-t/T if not motif else 1) (1)
                - contacting residues: for ppi. Target residues in contact with biner (1)
                - chi_angle timestep (1)
                - ss (H, E, L, MASK) (4)
            
            t2d (1, L, L, 45)
                - last plane is block adjacency
        """

        # Process motifs by polymer type:
        if polymer_type_masks:
            is_na = torch.logical_or(indep.is_dna, indep.is_rna)
            is_diffused_na = torch.logical_and(is_diffused, is_na)
            is_diffused_protein = torch.logical_and(is_diffused, ~is_na)

            is_protein_motif = is_motif * indep.is_protein
            is_dna_motif = is_motif * indep.is_dna
            is_rna_motif = is_motif * indep.is_rna
            is_na_motif = torch.logical_or(is_dna_motif, is_rna_motif)

        else:
            is_protein_motif = is_motif






        # if indep.metadata.get('refinement', None) is not None:
        #     assert twotemplate and threetemplate 
        #     ref_dict = indep.metadata['refinement']
        #     refine=True
        #     src_con_hal_idx0 = torch.from_numpy( ref_dict['con_hal_idx0'] )
        #     src_con_ref_idx0 = torch.from_numpy( ref_dict['con_ref_idx0'] )

        #     # replace the sequence w/ sequence from original motif
        #     src_motif_seq = indep.seq2[src_con_ref_idx0]
        # else:
        refine=False

        xyz_t = indep.xyz
        seq_one_hot = torch.nn.functional.one_hot(
                indep.seq, num_classes=self.NTOKENS).float()
        L = seq_one_hot.shape[0]


        '''
        msa_full:   NSEQ,NINDEL,NTERMINUS,
        msa_masked: NSEQ,NSEQ,NINDEL,NINDEL,NTERMINUS
        '''
        NTERMINUS = 2
        NINDEL = 1
        ### msa_masked ###
        ##################
        msa_masked = torch.zeros((1,1,L,2*NAATOKENS+NINDEL*2+NTERMINUS))

        

        msa_masked[:,:,:,:NAATOKENS] = seq_one_hot[None, None]
        msa_masked[:,:,:,NAATOKENS:2*NAATOKENS] = seq_one_hot[None, None]
        msa_masked[:,:,:,MSAMASKED_N_TERM] = (indep.terminus_type == N_TERMINUS).float()
        msa_masked[:,:,:,MSAMASKED_C_TERM] = (indep.terminus_type == C_TERMINUS).float()

        ### msa_full ###
        ################
        msa_full = torch.zeros((1,1,L,NAATOKENS+NINDEL+NTERMINUS))
        msa_full[:,:,:,:NAATOKENS] = seq_one_hot[None, None]
        msa_full[:,:,:,MSAFULL_N_TERM] = (indep.terminus_type == N_TERMINUS).float()
        msa_full[:,:,:,MSAFULL_C_TERM] = (indep.terminus_type == C_TERMINUS).float()

        ### t1d ###
        ########### 
        # Here we need to go from one hot with 22 classes to one hot with 21 classes
        # If sequence is masked, it becomes unknown
        # t1d = torch.zeros((1,1,L,NAATOKENS-1))

        #seqt1d = torch.clone(seq)
        # We only want to replace mask token with unknown aa token
        # DNA and RNA were already assigned unknown tokens, so don't shift them.
        seq_cat_shifted = seq_one_hot.argmax(dim=-1)
        # seq_cat_shifted[seq_cat_shifted>=MASKINDEX] -= 1
        seq_cat_shifted[seq_cat_shifted==MASKINDEX] -= 1
        t1d = torch.nn.functional.one_hot(seq_cat_shifted, num_classes=NAATOKENS-1)
        t1d = t1d[None, None] # [L, NAATOKENS-1] --> [1,1,L, NAATOKENS-1]
        # for idx in range(L):
            
        #     if seqt1d[idx,MASKINDEX] == 1:
        #         seqt1d[idx, MASKINDEX-1] = 1
        #         seqt1d[idx,MASKINDEX] = 0
        # t1d[:,:,:,:NPROTAAS+1] = seqt1d[None,None,:,:NPROTAAS+1]
        
        ## Str Confidence
        # Set confidence to 1 where diffusion mask is True, else 1-t/T
        strconf = torch.zeros((L,)).float()
        strconf[~is_diffused] = 1.
        strconf[is_diffused] = 1. - t/self.conf.diffuser.T
        strconf = strconf[None,None,...,None]

        t1d = torch.cat((t1d, strconf), dim=-1)
        t1d = t1d.float()

        num_backbone_atoms_protein = 3 # we can make this variable later if we want
        num_backbone_atoms_nucleic = self.conf['preprocess']['num_atoms_na']
        atom_trim_range = self.conf['inference']['num_atoms_modeled']
        if self.conf['inference']['num_atoms_modeled']>=num_backbone_atoms_nucleic :
            atom_trim_range = self.conf['inference']['num_atoms_modeled']
        else:
            atom_trim_range = NHEAVY
        # num_backbone_atoms_nucleic = 3
        ### xyz_t ###
        #############
        if self.conf.preprocess.sidechain_input:
            raise Exception('not implemented')
            xyz_t[torch.where(seq_one_hot == 21, True, False),3:,:] = float('nan')
        else:

            # Different number of atoms if protein, DNA, RNA, etc
            
            if polymer_type_masks:

                # How many backbone atoms do we need per polymer type?
                xyz_t[is_diffused_na,num_backbone_atoms_nucleic:,:] = float('nan')
                xyz_t[is_diffused_protein,num_backbone_atoms_protein:,:] = float('nan')


            else:
                xyz_t[is_diffused,3:,:] = float('nan')
        #xyz_t[:,3:,:] = float('nan')
        # ipdb.set_trace()

        # DJ - check if using inference.start_from_input
        # if so, chop off the atoms after 14: 
        # AF - only do this when doing inference, not during training.
        # if ('inference' in self.conf) and (self.conf.preprocess.sequence_decode):
        if ('inference' in self.conf) and (self.conf.inference.start_from_input):
            print('Warning: start_from_input is True, so chopping off atoms after 14')
            # assert 0, 'FOR ANDREW (note to self): check this line to see if we actually need to change, or how to best handle with both prot and NA: xyz_t = xyz_t.squeeze()[:,:14,:]'
            
            # atom_trim_range = self.conf['preprocess']['num_atoms_input'] # AF: default to 14, but can control how many

            xyz_t = xyz_t.squeeze()[:,:atom_trim_range,:]
            # xyz_t = xyz_t.squeeze()[:,:14,:]
            # xyz_t = xyz_t.squeeze()[:,:NHEAVY,:]
            # ipdb.set_trace()
            # xyz_t = xyz_t.squeeze()[:,:NHEAVY,:]
            if polymer_type_masks:
                # How many backbone atoms do we need per polymer type?
                print('AF: MAYBE THIS PART IS MESSING UP THE MODEL! LINE 857 aa_model.py')
                xyz_t[is_diffused_na, num_backbone_atoms_nucleic:, :] = float('nan')
                xyz_t[is_diffused_protein, num_backbone_atoms_protein:, :] = float('nan')
                # xyz_t[:,3:,:] = float('nan')

            else:
                xyz_t[:,3:,:] = float('nan')


        #     # atom_trim_range = self.conf['preprocess']['num_atoms_input'] # AF: default to 14, but can control how many
        #     print('AF: Check if this is the best way to handle this... (line 787, aa_model.py)')
        #     atom_trim_range = xyz_t.shape[1]

        # AF: since we diffuse NA now, switch to NHEAVY
        # if not atom_trim_range==NHEAVYPROT:
            # print(f'WARNING: preprocessed inputs are being trimmed to {atom_trim_range} instead of {NHEAVYPROT} (the NHEAVYPROT default). Hopefully this is for a good reason!')

        assert_that(xyz_t.shape).is_equal_to((L,NHEAVY,3))
        xyz_t=xyz_t[None, None]
        xyz_t = torch.cat((xyz_t, torch.full((1,1,L,NTOTAL-NHEAVY,3), float('nan'))), dim=3)
        # ipdb.set_trace()
        # assert_that(xyz_t.shape).is_equal_to((L,atom_trim_range,3))
        # xyz_t=xyz_t[None, None]
        # xyz_t = torch.cat((xyz_t, torch.full((1,1,L,NTOTAL-atom_trim_range,3), float('nan'))), dim=3)
        # assert_that(xyz_t.shape).is_equal_to((L,NHEAVYPROT,3))
        # xyz_t=xyz_t[None, None]
        # xyz_t = torch.cat((xyz_t, torch.full((1,1,L,NTOTAL-NHEAVYPROT,3), float('nan'))), dim=3)

        ### t2d ###
        ###########
        t2d = None
        # t2d = xyz_to_t2d(xyz_t)
        # B = 1
        # zeros = torch.zeros(B,1,L,36-3,3).float().to(px0_xyz.device)
        # xyz_t = torch.cat((px0_xyz.unsqueeze(1),zeros), dim=-2) # [B,T,L,27,3]
        # t2d, mask_t_2d_remade = get_t2d(
        #     xyz_t[0], mask_t[0], seq_scalar[0], same_chain[0], atom_frames[0])
        # t2d = t2d[None] # Add batch dimension # [B,T,L,L,44]
        
        ### idx ###
        ###########
        """
        idx = torch.arange(L)[None]
        if ppi_design:
            idx[:,binderlen:] += 200
        """
        # JW Just get this from the contig_mapper now. This handles chain breaks
        #idx = torch.tensor(self.contig_map.rf)[None]

        # ### alpha_t ###
        # ###############
        # seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,L)
        # alpha, _, alpha_mask, _ = util.get_torsions(xyz_t.reshape(-1,L,27,3), seq_tmp, TOR_INDICES, TOR_CAN_FLIP, REF_ANGLES)
        # alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
        # alpha[torch.isnan(alpha)] = 0.0
        # alpha = alpha.reshape(1,-1,L,10,2)
        # alpha_mask = alpha_mask.reshape(1,-1,L,10,1)
        # alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 30)


        # get torsion angles from templates
        seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,L)

        alpha, _, alpha_mask, _ = self.converter.get_torsions(xyz_t.reshape(-1,L,rf2aa.chemical.NTOTAL,3), seq_tmp)
            #rf2aa.util.torsion_indices, rf2aa.util.torsion_can_flip, rf2aa.util.reference_angles)
        alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
        alpha[torch.isnan(alpha)] = 0.0
        alpha = alpha.reshape(-1,L,rf2aa.chemical.NTOTALDOFS,2)
        alpha_mask = alpha_mask.reshape(-1,L,rf2aa.chemical.NTOTALDOFS,1)
        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(-1, L, 3*rf2aa.chemical.NTOTALDOFS) # [n,L,30]

        alpha_t = alpha_t.unsqueeze(1) # [n,I,L,30]

        if twotemplate and (not threetemplate):
            alpha_t = alpha_t.tile((1,2,1,1)) # add dim for second template 
        elif twotemplate and threetemplate:
            alpha_t = alpha_t.tile((1,3,1,1))
        else:
            pass 


        # if not (self.conf.preprocess.motif_only_2d):
        #     # only 2 templates (default)
        #     alpha_t = alpha_t.tile((1,2,1,1))
        # else:
        #     # 3 templates now 
        #     alpha_t = alpha_t.tile((1,3,1,1))


        # #put tensors on device
        # msa_masked = msa_masked.to(self.device)
        # msa_full = msa_full.to(self.device)
        # seq = seq.to(self.device)
        # xyz_t = xyz_t.to(self.device)
        # #idx = idx.to(self.device)
        # t1d = t1d.to(self.device)
        # # t2d = t2d.to(self.device)
        # alpha_t = alpha_t.to(self.device)
        
        ### added_features ###
        ######################
        # NB the hotspot input has been removed in this branch. 
        # JW added it back in, using pdb indexing

        if self.conf.preprocess.d_t1d == 24: # add hotpot residues
            raise Exception('not implemented')
            if self.ppi_conf.hotspot_res is None:
                print("WARNING: you're using a model trained on complexes and hotspot residues, without specifying hotspots. If you're doing monomer diffusion this is fine")
                hotspot_idx=[]
            else:
                hotspots = [(i[0],int(i[1:])) for i in self.ppi_conf.hotspot_res]
                hotspot_idx=[]
                for i,res in enumerate(self.contig_map.con_ref_pdb_idx):
                    if res in hotspots:
                        hotspot_idx.append(self.contig_map.hal_idx0[i])
            hotspot_tens = torch.zeros(L).float()
            hotspot_tens[hotspot_idx] = 1.0
            t1d=torch.cat((t1d, hotspot_tens[None,None,...,None].to(self.device)), dim=-1)
        
        # return msa_masked, msa_full, seq[None], torch.squeeze(xyz_t, dim=0), idx, t1d, t2d, xyz_t, alpha_t
        if twotemplate and not threetemplate:
            mask_t = torch.ones(1,2,L,L).bool() # 2 for second template
        elif twotemplate and threetemplate:
            mask_t = torch.ones(1,3,L,L).bool()
        else:
            mask_t = torch.ones(1,1,L,L).bool()
        # mask_t = torch.ones(1,2,L,L).bool()
        sctors = torch.zeros((1,L,rf2aa.chemical.NTOTALDOFS,2))

        xyz = torch.squeeze(xyz_t, dim=0)

        # NO SELF COND
        if twotemplate and (not threetemplate):
            xyz_t = torch.zeros(1,2,L,3)
            t2d   = torch.zeros(1,2,L,L,68)

            t2d_xt, mask_t_2d_remade = util.get_t2d(
                xyz, indep.is_sm, indep.atom_frames)
            t2d[0,0]   = t2d_xt[0]
            xyz_t[0,0] = xyz[0,:,1]
        
        elif twotemplate and threetemplate:
            assert (is_protein_motif is not None) 
            assert (t2d_is_revealed is not None)

            # only inputting motif in 2d template --> need 3rd template 
            xyz_t  = torch.zeros(1,3,L,3)
            t2d    = torch.zeros(1,3,L,L,68)

            ## BUGFIX - the third mask needs to be altered 
            mask_t = torch.ones(1,3,L,L).bool() 
            mask_t[0,2,:,:] = t2d_is_revealed.bool()

            ##### 2nd template #####
            t2d_xt, mask_t_2d_remade = util.get_t2d(xyz, indep.is_sm, indep.atom_frames)
            t2d[0,0]   = t2d_xt[0]
            xyz_t[0,0] = xyz[0,:,1]

            ##### 3rd template #####
            # set of diffused crds w/ motif sliced in
            xyz_xt_w_motif = xyz.clone()

            # DJ - if refine, selection in xyz2 needs to reference the original src pdb for motif
            sel2_prot = src_con_ref_idx0 if refine else is_protein_motif
            sel2_nuc = src_con_ref_idx0 if refine else is_na_motif

            # AF change: make NHEAVY (see if we need this change)
            # xyz_xt_w_motif[0,is_protein_motif,:NHEAVY] = indep.xyz2[sel2,:NHEAVY]
            if sum(is_protein_motif) > 0:
                xyz_xt_w_motif[0,is_protein_motif,:NHEAVYPROT] = indep.xyz2[sel2_prot,:NHEAVYPROT]
            if sum(is_na_motif) > 0:
                xyz_xt_w_motif[0,is_na_motif,:NHEAVYNUC] = indep.xyz2[sel2_nuc,:NHEAVYNUC]

            # t2d containing desired motif 
            # this construction uses bool mask to allow certain motifs to see others
            # all motifs can see themselves
            t2d_motif, _ = util.get_t2d(xyz_xt_w_motif, indep.is_sm, indep.atom_frames, t2d_is_revealed[None])


            # put it in as third template
            t2d[0,2]   = t2d_motif[0]
            xyz_t[0,2] = xyz_xt_w_motif[0,:,1]

            # fig, ax = plt.subplots(2,2,figsize=(5,5), dpi=300)

            # ax[0][0].imshow(t2d[0,0].argmax(-1))
            # ax[0][0].set_title('First template - Xt in')

            # ax[0][1].imshow(t2d[0,1].argmax(-1))
            # ax[0][1].set_title('Second template - self cond')

            # ax[1][0].imshow(t2d[0,2].argmax(-1))
            # ax[1][0].set_title('Third template - motif ')

            # ax[1][1].imshow(t2d_is_revealed)
            # ax[1][1].set_title('Third template motif mask')
            
            # plt.tight_layout()
            # plt.show()
            # plt.savefig('repeat_DBP_t2d.png', bbox_inches='tight', dpi='figure')
            # sys.exit('Exiting early')


            # stack on final feature 
            blank = torch.ones_like(t2d_is_revealed)*-1 # first two templates will have -1 in this channel
            cattable_t2d_is_revealed = torch.stack((blank, blank, t2d_is_revealed.int()), dim=-1)
            cattable_t2d_is_revealed = cattable_t2d_is_revealed.permute(2,0,1) # (L,L,3) -> (3,L,L)
            cattable_t2d_is_revealed = cattable_t2d_is_revealed[None,...,None] # (3,L,L) -> (1,3,L,L,1)

            t2d = torch.cat((t2d, cattable_t2d_is_revealed), dim=-1)    # (1,3,L,L,69)



        else:
            xyz_t = torch.zeros(1,1,L,3)
            t2d   = torch.zeros(1,1,L,L,68)




        # if is_protein_motif is None:
        if is_motif is None:
            # DJ - adding this bc if motif_only_t2, is_diffused will be 
            # entire protein --> can't trust it to calculate is_protein_motif
            # is_protein_motif = ~is_diffused * ~indep.is_sm
            is_motif = ~is_diffused * ~indep.is_sm

        # xyz = torch.nan_to_num(xyz)

        # We probably want to include more atoms if diffusing a nucleic acid chain
        if polymer_type_masks:
            xyz[0, is_diffused_protein*~indep.is_sm, num_backbone_atoms_protein:] = torch.nan
            xyz[0, is_diffused_na*~indep.is_sm, num_backbone_atoms_nucleic:] = torch.nan
        else:
            xyz[0, is_diffused*~indep.is_sm,3:] = torch.nan

        # ipdb.set_trace()
        # xyz[0, indep.is_sm,14:] = 0
        # xyz[0, is_motif, 14:] = 0
        xyz[0, indep.is_sm, NHEAVYPROT:] = 0
        xyz[0, is_protein_motif, NHEAVYPROT:] = 0
        xyz[0, is_na_motif, NHEAVYNUC:] = 0
        dist_matrix = rf2aa.data_loader.get_bond_distances(indep.bond_feats)

        # minor tweaks to rfi to match gp training
        # if ('inference' in self.conf) and (self.conf.inference.get('contig_as_guidepost', False)):
        #     '''Manually inspecting the pickled features passed to RF during training, 
        #     I did not see markers for the N and C termini. This is to more accurately 
        #     replicate the features seen during training at inference.'''
        #     # Erase N/C termini markers
        #     msa_masked[...,-2:] = 0
        #     msa_full[...,-2:] = 0
        if ('inference' in self.conf) and (self.conf.inference.get('contig_as_guidepost', False)):
            print('>'*80)
            print('Erase N/C termini markers')
            '''Manually inspecting the pickled features passed to RF during training, 
            I did not see markers for the N and C termini. This is to more accurately 
            replicate the features seen during training at inference.'''
            # Erase N/C termini markers
            msa_masked[...,-2:] = 0
            msa_full[...,-2:] = 0
        

        # # T1D 
        # if not (self.conf.preprocess.motif_only_2d): # classic 2template t1d 
        #     t1d = torch.tile(t1d, (1,2,1,1))
        #     t1d[0,1,:,-1] = -1

        # else: # new "3template" t1d
        #     t1d = torch.tile(t1d, (1,3,1,1))
        #     t1d[0,1,:,-1] = -1 # second template (Xt) gets -1 for timestep feature 
        #     t1d[0,2,:,-1] = -1 # third template (motif) gets -1 for timestep feature
            
        #     # feature for 3rd template - is it motif or not?
        #     # cattable_is_protein_motif = torch.tile(is_protein_motif[None,None,:,None], (1,3,1,1)).to(device=t1d.device, dtype=t1d.dtype)
        #     # cattable_is_protein_motif[:,:-1,...] = -1  # first two templates get -1 for the motif feature 
        #     # t1d = torch.cat((t1d, cattable_is_protein_motif), dim=-1) 
        #     cattable_is_motif = torch.tile(is_motif[None,None,:,None], (1,3,1,1)).to(device=t1d.device, dtype=t1d.dtype)
        #     cattable_is_motif[:,:-1,...] = -1  # first two templates get -1 for the motif feature 
        #     t1d = torch.cat((t1d, cattable_is_motif), dim=-1) 

        
        ### T1D ### 
        if twotemplate and (not threetemplate):
            t1d = torch.tile(t1d, (1,2,1,1)) # add dim for second template
            t1d[0,1,:,-1] = -1
        
        elif twotemplate and threetemplate:  # new "3 template" t1d - has extra feature 
            t1d = torch.tile(t1d, (1,3,1,1))
            t1d[0,1,:,-1] = -1 # second template (Xt) gets -1 for timestep feature 
            t1d[0,2,:,-1] = -1 # third template (motif) gets -1 for timestep feature

            # # feature for 3rd template - is it motif or not?
            # cattable_is_protein_motif = torch.tile(is_protein_motif[None,None,:,None], (1,3,1,1)).to(device=t1d.device, dtype=t1d.dtype)
            # cattable_is_protein_motif[:,:-1,...] = -1  # first two templates get -1 for the motif feature 
            # t1d = torch.cat((t1d, cattable_is_protein_motif), dim=-1) 
            # feature for 3rd template - is it motif or not?
            cattable_is_motif = torch.tile(is_motif[None,None,:,None], (1,3,1,1)).to(device=t1d.device, dtype=t1d.dtype)
            cattable_is_motif[:,:-1,...] = -1  # first two templates get -1 for the motif feature 
            t1d = torch.cat((t1d, cattable_is_motif), dim=-1) 
        # ipdb.set_trace()
        # Note: should be batched
        rfi = RFI(
            msa_masked,
            msa_full,
            indep.seq[None],
            indep.seq[None],
            xyz,
            sctors,
            indep.idx[None],
            indep.bond_feats[None],
            dist_matrix[None],
            indep.chirals[None],
            indep.atom_frames[None],
            t1d,
            t2d,
            xyz_t,
            alpha_t,
            mask_t,
            indep.same_chain[None],
            ~is_diffused,
            None,
            None,
            None)
        return rfi
    

def assert_has_coords(xyz, indep):
    assert len(xyz.shape) == 3
    missing_backbone = torch.isnan(xyz).any(dim=-1)[...,:3].any(dim=-1)
    prot_missing_bb = missing_backbone[~indep.is_sm]
    sm_missing_ca = torch.isnan(xyz).any(dim=-1)[...,1]
    try:
        assert not prot_missing_bb.any(), f'prot_missing_bb {prot_missing_bb}'
        assert not sm_missing_ca.any(), f'sm_missing_ca {sm_missing_ca}'
    except Exception as e:
        print(e)
        import ipdb
        ipdb.set_trace()


def has_coords(xyz, indep):
    assert len(xyz.shape) == 3
    missing_backbone = torch.isnan(xyz).any(dim=-1)[...,:3].any(dim=-1)
    prot_missing_bb = missing_backbone[~indep.is_sm]
    sm_missing_ca = torch.isnan(xyz).any(dim=-1)[...,1]
    try:
        assert not prot_missing_bb.any(), f'prot_missing_bb {prot_missing_bb}'
        assert not sm_missing_ca.any(), f'sm_missing_ca {sm_missing_ca}'
    except Exception as e:
        print(e)
        import ipdb
        ipdb.set_trace()


def pad_dim(x, dim, new_l, value=0):
    padding = [0]*2*x.ndim
    padding[2*dim] = new_l - x.shape[dim]
    padding = padding[::-1]
    return F.pad(x, pad=tuple(padding), value=value)

def write_traj(path, xyz_stack, seq, bond_feats, natoms=23, **kwargs):
    xyz23 = pad_dim(xyz_stack, 2, natoms)
    if bond_feats is not None:
        bond_feats = bond_feats[None]
    with open(path, 'w') as fh:
        for i, xyz in enumerate(xyz23):
            rf2aa.util.writepdb_file(fh, xyz, seq, bond_feats=bond_feats, modelnum=i, **kwargs)

def minifier(argument_map):
    argument_map['out_9'] = None
    argument_map['out_0'] = None
    argument_map['out_2'] = None
    argument_map['out_3'] = None
    argument_map['out_5'] = None
    argument_map['t2d'] = None


def adaptor_fix_bb_indep(out):
    """
    adapts the outputs of RF2-allatom phase 3 dataloaders into fixed bb outputs
    takes in a tuple with 22 items representing the RF2-allatom data outputs and returns an Indep dataclass.
    """
    assert len(out) == 25, f"found {len(out)} elements in RF2-allatom output"
    (seq, msa, msa_masked, msa_full, mask_msa, true_crds, atom_mask, idx_pdb, xyz_t, t1d, mask_t, xyz_prev,
        mask_prev, same_chain, unclamp, negative, atom_frames, bond_feats, dist_matrix, chirals, ch_label, symm_group,
         dataset_name, item, Ls) = out
    assert symm_group=="C1", f"example with {symm_group} found, symmetric training not set up for aa-diffusion"
    

    #remove permutation symmetry dimension if present
    natstack = true_crds.clone()
    maskstack = atom_mask.clone()
    if len(true_crds.shape) == 4 and len(atom_mask.shape) == 3:
        true_crds = true_crds[0]
        atom_mask = atom_mask[0]
    
    # our dataloaders return torch.zeros(L...) for atom frames and chirals when there are none, this updates it to use common shape 
    if torch.all(atom_frames == 0):
        atom_frames = torch.zeros((0,3,2))
    if torch.all(chirals == 0):
        chirals = torch.zeros((0,5))

    is_sm = rf2aa.util.is_atom(seq)

    is_protein, is_dna, is_rna  = mask_generator.get_polymer_type_masks(seq)

    is_n_terminus = msa_full[0, 0, :, MSAFULL_N_TERM].bool()
    is_c_terminus = msa_full[0, 0, :, MSAFULL_C_TERM].bool()
    terminus_type = torch.zeros(msa_masked.shape[2], dtype=int)
    terminus_type[is_n_terminus] = N_TERMINUS
    terminus_type[is_c_terminus] = C_TERMINUS

    indep = Indep(
        rf2aa.tensor_util.assert_squeeze(seq), # [L]
        true_crds[:,:NHEAVY], # [L, 14, 3]
        true_crds[:,:NHEAVY], # DJ- new xyz2 to keep track of orig AND diffused xyz
        natstack, 
        maskstack,
        Ls,
        idx_pdb,

        # SM specific
        bond_feats,
        chirals,
        atom_frames,
        same_chain,
        rf2aa.tensor_util.assert_squeeze(is_sm),
        terminus_type,

        # Nuc specific
        is_protein,
        is_dna,
        is_rna,
        )
    
    return indep, atom_mask, dataset_name

def pop_unoccupied(indep, atom_mask):
    """
    Inplace operation.
    Removes ligand atoms which are not present in the atom mask.
    Removes residues which do not have N,C,Ca in the atom mask.
    """
    pop = rf2aa.util.get_prot_sm_mask(atom_mask, indep.seq)
    pop_mask(indep, pop)
    atom_mask           = atom_mask[pop]
    return atom_mask

def pop_mask(indep, pop):
    n_atoms = indep.is_sm.sum()
    assertpy.assert_that(len(indep.atom_frames)).is_equal_to(n_atoms)

    N     = pop.sum()
    pop2d = pop[None,:] * pop[:,None]
    
    indep.seq           = indep.seq[pop]
    indep.xyz           = indep.xyz[pop]
    indep.xyz2          = indep.xyz2[pop]
    indep.natstack      = indep.natstack[:,pop]
    indep.maskstack     = indep.maskstack[:,pop]
    indep.idx           = indep.idx[pop]
    indep.bond_feats    = indep.bond_feats[pop2d].reshape(N,N)
    indep.same_chain    = indep.same_chain[pop2d].reshape(N,N)
    indep.is_sm         = indep.is_sm[pop]
    indep.terminus_type = indep.terminus_type[pop]
    indep.is_protein    = indep.is_protein[pop]
    indep.is_dna        = indep.is_dna[pop]
    indep.is_rna        = indep.is_rna[pop]

    if indep.chirals.numel():
        n_shift = (~pop).cumsum(dim=0)
        chiral_indices = indep.chirals[:,:-1]
        chiral_shift = n_shift[chiral_indices.long()]
        indep.chirals[:,:-1] = chiral_indices - chiral_shift

def centre(indep, is_diffused):
    xyz = indep.xyz
    #Centre unmasked structure at origin, as in training (to prevent information leak)
    if torch.sum(~is_diffused) != 0:
        motif_com=xyz[~is_diffused,1,:].mean(dim=0) # This is needed for one of the potentials
        xyz = xyz - motif_com
    elif torch.sum(~is_diffused) == 0:
        xyz = xyz - xyz[:,1,:].mean(dim=0)
    indep.xyz = xyz


def diffuse(conf, diffuser, indep, is_diffused, t):
    
    indep.xyz = add_fake_frame_legs(indep.xyz, indep.is_sm)

    # Process nucleic acid shit
    # ipdb.set_trace()
    if conf['preprocess']['mask_by_polymer_type']:
        if conf['preprocess']['num_atoms_na']:
            num_atoms_na = conf['preprocess']['num_atoms_na']
        else:
            num_atoms_na = 3

        # assert (num_atoms_na % 3) == 0

        num_frames_na = num_atoms_na//3


    if t == diffuser.T: 
        t_list = [t,t]
    else: 
        t_list = [t+1,t]

    indep_diffused_t = copy.deepcopy(indep)
    indep_diffused_tplus1 = copy.deepcopy(indep)
    kwargs = {
        'xyz'                     :indep.xyz,
        'seq'                     :indep.seq,
        'atom_mask'               :None,
        'diffusion_mask'          :~is_diffused,
        't_list'                  :t_list,
        'diffuse_sidechains'      :conf['preprocess']['sidechain_input'],
        'include_motif_sidechains':conf['preprocess']['motif_sidechain_input'],
        # 'include_na_sidechains':conf['preprocess']['na_sidechain_input'],
        'is_sm': indep.is_sm,
        'is_na': indep.is_na,
        'num_frames_na': num_frames_na
    }

    diffused_fullatoms, aa_masks, true_crds = diffuser.diffuse_pose(**kwargs)




    ############################################
    ########### New Self Conditioning ##########
    ############################################

    # JW noticed that the frames returned from the diffuser are not from a single noising trajectory
    # So we are going to take a denoising step from x_t+1 to get x_t and have their trajectories agree

    # Only want to do this process when we are actually using self conditioning training
    from diffusion import get_beta_schedule
    from inference.utils import get_next_ca, get_next_frames
    if conf['preprocess']['new_self_cond'] and t < 200: # Only can get t+1 if we are at t < 200

        tmp_x_t_plus1 = diffused_fullatoms[0]

        beta_schedule, _, alphabar_schedule = get_beta_schedule(
                                    T=conf['diffuser']['T'],
                                    b0=conf['diffuser']['b_0'],
                                    bT=conf['diffuser']['b_T'],
                                    schedule_type=conf['diffuser']['schedule_type'],
                                    inference=False)

        _, ca_deltas = get_next_ca(
                                    xt=tmp_x_t_plus1,
                                    px0=true_crds,
                                    t=t+1,
                                    diffusion_mask=~is_diffused,
                                    crd_scale=conf['diffuser']['crd_scale'],
                                    beta_schedule=beta_schedule,
                                    alphabar_schedule=alphabar_schedule,
                                    noise_scale=1)

        # Noise scale ca hard coded for now. Maybe can eventually be piped down from inference configs? - NRB
        assert not torch.isnan(ca_deltas).any()
        assert not torch.isnan(true_crds[:,1,:]).any()
        assert not torch.isnan(tmp_x_t_plus1[is_diffused,1,0]).any()
        assert not torch.isnan(tmp_x_t_plus1[:,1,0]).any()

        frames_next = get_next_frames(
                                    xt=tmp_x_t_plus1,
                                    px0=true_crds,
                                    t=t+1,
                                    diffuser=diffuser,
                                    so3_type=conf['diffuser']['so3_type'],
                                    diffusion_mask=~is_diffused,
                                    noise_scale=1) # Noise scale frame hard coded for now - NRB

        frames_next = torch.from_numpy(frames_next) + ca_deltas[:,None,:]  # translate
        
        tmp_x_t = torch.zeros_like(tmp_x_t_plus1)
        tmp_x_t[:,:3] = frames_next
        
        if conf['preprocess']['motif_sidechain_input']:
            tmp_x_t[~is_diffused,:] = tmp_x_t_plus1[~is_diffused]
        
        assert not torch.isnan(tmp_x_t[~is_diffused,1,0]).any()
        assert not torch.isnan(tmp_x_t[is_diffused,1,0]).any()
        assert not torch.isnan(tmp_x_t[:,1,0]).any()
        diffused_fullatoms[1] = tmp_x_t

    # indep_diffused_tplus1.xyz = diffused_fullatoms[0, :, :14]
    # indep_diffused_t.xyz = diffused_fullatoms[1, :, :14]
    indep_diffused_tplus1.xyz = diffused_fullatoms[0, :, :NHEAVY]
    indep_diffused_t.xyz = diffused_fullatoms[1, :, :NHEAVY]
    return (indep_diffused_tplus1, indep_diffused_t), t_list

def forward(model, rfi, **kwargs):
    rfi_dict = dataclasses.asdict(rfi)
    # ipdb.set_trace()
    return RFO(*model(**{**rfi_dict, **kwargs}))

def mask_indep(indep, is_diffused, polymer_type_masks=None):

    if polymer_type_masks:
        indep.seq[is_diffused * indep.is_protein] = MASKINDEX
        indep.seq[is_diffused * indep.is_dna] = DNAMASKINDEX
        indep.seq[is_diffused * indep.is_rna] = RNAMASKINDEX


    else:

        indep.seq[is_diffused] = MASKINDEX


# def get_xyz_t_t2d(atom_frames, is_sm, xyz):
#     '''
#     Params:
#         xyz: [T, L, 36, 3]
#         is_sm: [L]
#         atom_frames: [F, 3, 2]
#     '''
#     assertpy.assert_that(is_sm.ndim).is_equal_to(3) # [F, 3, 2]
#     assertpy.assert_that(atom_frames.ndim).is_equal_to(3) # [F, 3, 2]
#     assertpy.assert_that(xyz.ndim).is_equal_to(3) # [L, n_atoms, 3]
#     L, = is_sm.shape
#     assertpy.assert_that(xyz.shape).is_equal_to((L, 36, 3))
#     t2d, mask_t_2d_remade = util.get_t2d(
#         xyz, is_sm, atom_frames)
#     t2d = t2d[None] # Add batch dimension # [B,T,L,L,44]
#     return


# def self_cond_new(indep, rfi, rfo):
#     # RFI is already batched
#     B = 1
#     L = indep.xyz.shape[0]
#     rfi_sc = copy.deepcopy(rfi)
#     zeros = torch.zeros(B,1,L,36-3,3).float().to(rfi.xyz.device)
#     xyz_t = torch.cat((rfo.xyz[-1:], zeros), dim=-2) # [B,T,L,27,3]
#     ic(xyz_t[0].shape, rfi.atom_frames.shape)
#     t2d, mask_t_2d_remade = util.get_t2d(
#         xyz_t[0], indep.is_sm, rfi.atom_frames[0])
#     t2d = t2d[None] # Add batch dimension # [B,T,L,L,44]
#     rfi_sc.xyz_t[0,1] = xyz_t[0, 0, :, 1]
#     rfi_sc.t2d[0, 1] = t2d[0, 0]
#     return rfi_sc

# def self_cond(indep, rfi, rfo):
#     # RFI is already batched
#     B = 1
#     L = indep.xyz.shape[0]
#     rfi_sc = copy.deepcopy(rfi)
#     zeros = torch.zeros(B,1,L,36-3,3).float().to(rfi.xyz.device)
#     # cat BB prediction with sidechain zeros 
#     xyz_t = torch.cat((rfo.xyz[-1:], zeros), dim=-2) # [B,T,L,27,3]

#     t2d, mask_t_2d_remade = util.get_t2d(xyz_t[0], indep.is_sm, rfi.atom_frames[0])
#     t2d = t2d[None] # Add batch dimension # [B,T,L,L,44]
    
#     # SECOND template is the previous px0 (index 1)
#     # This spot has -1 confidence in t1d, marking it as SC template
#     rfi_sc.xyz_t[0,1] = xyz_t[0, 0, :, 1]

#     # add 69th feature to t2d (accomodation for 3-template version)
#     blank = (torch.ones((1,1,L,L,1))*-1).to(rfi.xyz.device)
#     t2d = torch.cat((t2d, blank), dim=-1)
#     rfi_sc.t2d[0, 1] = t2d[0, 0]

#     return rfi_sc


def self_cond(indep, rfi, rfo, twotemplate, threetemplate):
    # RFI is already batched
    B = 1
    L = indep.xyz.shape[0]
    rfi_sc = copy.deepcopy(rfi)
    zeros = torch.zeros(B,1,L,36-3,3).float().to(rfi.xyz.device) # zeros for sidechains 
    # cat BB prediction with sidechain zeros 
    xyz_t = torch.cat((rfo.xyz[-1:], zeros), dim=-2) # [B,T,L,36,3] 

    t2d, mask_t_2d_remade = util.get_t2d(xyz_t[0], indep.is_sm, rfi.atom_frames[0])
    
    t2d = t2d[None] # Add batch dimension # [B,T,L,L,44]

    if twotemplate and (not threetemplate):
        # Insert the previous px0 into the SECOND template spot (index 1)
        # This spot has -1 confidence in t1d, marking it as SC template
        rfi_sc.xyz_t[0,1] = xyz_t[0,0,:,1]
        rfi_sc.t2d[0,1] = t2d[0,0]

    elif twotemplate and threetemplate:
        rfi_sc.xyz_t[0,1] = xyz_t[0,0,:,1]

        # add 69th feature to t2d (accomodation for 3-template version)
        blank = (torch.ones((1,1,L,L,1))*-1).to(rfi.xyz.device)
        t2d = torch.cat((t2d, blank), dim=-1)
        rfi_sc.t2d[0, 1] = t2d[0, 0]
    
    else:
        raise RuntimeError('Unsure what should happen here - re-evaluate when hit. -DJ')

    return rfi_sc

def diagnose_xyz(xyz):
    '''Returns a string describing where the coordinates are NaN'''
    has_ca = torch.isnan(xyz[..., 1, :]).any()
    has_backbone = torch.isnan(xyz[..., :3, :]).any()
    # has_heavy = torch.isnan(xyz[..., :3, :]).any()
    return f'diagnosis: nan-CA: {has_ca}    nan-BB: {has_backbone}'

class AtomizeResidues:
    def __init__(
        self,
        indep,
        input_str_mask # dict
        ) -> None:

        self.indep_initial_copy = copy.deepcopy(indep)
        self.indep = copy.deepcopy(indep)
        self.input_str_mask = input_str_mask
        self.input_seq_mask = input_str_mask

        # bookkeeping for de-atomization
        self.has_been_atomized = False
        self.atomized_res = [] # integer-coded AA identities of atomized residues
        self.atomized_res_idx = [] # tensor indices of atomized residues (0-indexed contiguous)
        self.atomized_idx = [] # PDB indices of atomized residues
    
    def featurize_atomized_residues(
        self,
        atom_mask, 
    ):
        """
        this function takes outputs of the RF2aa dataloader and the generated masks and refeaturizes the example where the 
        portion of the structure that is provided as context is treated as atoms instead of residues. the mask will be updated so 
        that only the tip atoms of the context residues are context and the rest is diffused
        """
        is_motif = self.input_str_mask
        seq_atomize_all,  xyz_atomize_all, frames_atomize_all, chirals_atomize_all, res_idx_atomize_all = \
            self.generate_atomize_protein_features(is_motif, atom_mask)
        if not res_idx_atomize_all: # no residues atomized, just return the inputs
            return
        
        # update msa feats, xyz, mask, frames and chirals
        total_L = self.update_rf_features_w_atomized_features(seq_atomize_all, \
                                                            xyz_atomize_all, \
                                                            frames_atomize_all, 
                                                            chirals_atomize_all
                                                            ) 

        pop = self.pop_protein_feats(res_idx_atomize_all, total_L)
        #handle diffusion specific things such as masking
        # self.construct_diffusion_masks(is_motif, is_atom_motif_all, pop)
    
    def generate_atomize_protein_features(self, is_motif, atom_mask):
        """
        given a motif, generate "atomized" features for the residues in that motif
        skips residues that have any unresolved atoms or neighbors with unresolved atoms
        """
        L = self.indep.bond_feats.shape[0]
        orig_L = L
        is_protein_motif = is_motif * ~self.indep.is_sm
        allatom_mask = rf2aa.util.allatom_mask.clone() # 1 if that atom index exists for that residue and 0 if it doesnt
        allatom_mask[:, 14:] = False # no Hs

        # make xyz the right dimensions for atomization
        xyz = torch.full((L, rf2aa.chemical.NTOTAL, 3), np.nan).float()
        xyz[:, :14] = self.indep.xyz[:, :14]

        aa2long_ = [[x.strip() if x is not None else None for x in y] for y in rf2aa.chemical.aa2long]
        # iterate through residue indices in the structure to atomize
        seq_atomize_all = []
        xyz_atomize_all = []
        frames_atomize_all = [self.indep.atom_frames]
        chirals_atomize_all = [self.indep.chirals]
        res_idx_atomize_all = [] # indices of residues that end up getting atomized (0-indexed, contiguous)
        res_atomize_all = [] # residues (integer-coded) that end up getting atomized 
        idx_atomize_all = [] # PDB indices of residues that end up getting atomized
        # track the res_idx and absolute index of the previous C to draw peptide bonds between contiguous residues
        prev_res_idx = -2
        prev_C_index = -2 
        for res_idx in is_protein_motif.nonzero():
            residue = self.indep.seq[res_idx] # number representing the residue in the token list, can be used to index aa2long
            # check if all the atoms in the residue are resolved, if not dont atomize
            if not torch.all(allatom_mask[residue]==atom_mask[res_idx]):
                raise Exception(f'not all atoms resolved for {residue} at 0-indexed position {res_idx}')
            N_term = res_idx == 0
            C_term = res_idx == self.indep.idx.shape[0]-1
            N_resolved = False
            C_resolved = False

            C_resolved = (not C_term) and self.indep.idx[res_idx+1]-self.indep.idx[res_idx] == 1
            N_resolved = (not N_term) and self.indep.idx[res_idx]-self.indep.idx[res_idx-1] == 1

            seq_atomize, _, xyz_atomize, _, frames_atomize, bond_feats_atomize, C_index, chirals_atomize = \
                            rf2aa.util.atomize_protein(res_idx, self.indep.seq[None], xyz, atom_mask, n_res_atomize=1)
            
            natoms = seq_atomize.shape[0]
            # update the chirals to be after all the other residues
            chirals_atomize[:, :-1] += L

            seq_atomize_all.append(seq_atomize)
            xyz_atomize_all.append(xyz_atomize)
            frames_atomize_all.append(frames_atomize)
            chirals_atomize_all.append(chirals_atomize)
            res_idx_atomize_all.append(res_idx)
            res_atomize_all.append(residue)
            idx_atomize_all.append(self.indep.idx[res_idx])

            # update bond_feats every iteration, update all other features at the end 
            bond_feats_new = torch.zeros((L+natoms, L+natoms))
            bond_feats_new[:L, :L] = self.indep.bond_feats
            bond_feats_new[L:, L:] = bond_feats_atomize
            # add bond between protein and atomized N
            if N_resolved:
                bond_feats_new[res_idx-1, L] = 6 # protein (backbone)-atom bond 
                bond_feats_new[L, res_idx-1] = 6 # protein (backbone)-atom bond 
            # add bond between protein and C, assumes every residue is being atomized one at a time (eg n_res_atomize=1)
            if C_resolved:
                bond_feats_new[res_idx+1, L+int(C_index.numpy())] = 6 # protein (backbone)-atom bond 
                bond_feats_new[L+int(C_index.numpy()), res_idx+1] = 6 # protein (backbone)-atom bond 
            # handle drawing peptide bond between contiguous atomized residues
            if self.indep.idx[res_idx]-self.indep.idx[prev_res_idx] == 1 and prev_res_idx > -1:
                bond_feats_new[prev_C_index, L] = 1 # single bond
                bond_feats_new[L, prev_C_index] = 1 # single bond
            prev_res_idx = res_idx
            prev_C_index =  L+int(C_index.numpy())
            #update same_chain every iteration
            same_chain_new = torch.zeros((L+natoms, L+natoms))
            same_chain_new[:L, :L] = self.indep.same_chain
            residues_in_prot_chain = self.indep.same_chain[res_idx].squeeze().nonzero()

            same_chain_new[L:, residues_in_prot_chain] = 1
            same_chain_new[residues_in_prot_chain, L:] = 1
            same_chain_new[L:, L:] = 1

            self.indep.bond_feats = bond_feats_new
            self.indep.same_chain = same_chain_new
            L = self.indep.bond_feats.shape[0]

        # save atomized position info needed for deatomization
        self.atomized_res = res_atomize_all
        self.atomized_res_idx = res_idx_atomize_all
        self.atomized_idx = idx_atomize_all

        return seq_atomize_all, xyz_atomize_all, frames_atomize_all, chirals_atomize_all, res_idx_atomize_all

    def update_rf_features_w_atomized_features(self, \
                                        seq_atomize_all, \
                                        xyz_atomize_all, \
                                        frames_atomize_all, \
                                        chirals_atomize_all, 
                                        ):
        """
        adds the msa, xyz, frame and chiral features from the atomized regions to the rosettafold input tensors
        """
        # Handle MSA feature updates
        seq_atomize_all = torch.cat(seq_atomize_all)
        
        atomize_L = seq_atomize_all.shape[0]
        self.indep.seq = torch.cat((self.indep.seq, seq_atomize_all), dim=0)

        # handle coordinates, need to handle permutation symmetry
        orig_L, natoms = self.indep.xyz.shape[:2]
        total_L = orig_L + atomize_L
        xyz_atomize_all = rf2aa.util.cartprodcat(xyz_atomize_all)
        N_symmetry = xyz_atomize_all.shape[0]
        xyz = torch.full((N_symmetry, total_L, NTOTAL, 3), np.nan).float()
        xyz[:, :orig_L, :natoms, :] = self.indep.xyz.expand(N_symmetry, orig_L, natoms, 3)
        xyz[:, orig_L:, 1, :] = xyz_atomize_all
        xyz[:, orig_L:, :3, :] += rf2aa.chemical.INIT_CRDS[:3]
        #ignoring permutation symmetry for now, network should learn permutations at low T
        self.indep.xyz = xyz[0]
        
        #handle idx_pdb 
        last_res = self.indep.idx[-1]
        idx_atomize = torch.arange(atomize_L) + last_res
        self.indep.idx = torch.cat((self.indep.idx, idx_atomize))
        
        # handle sm specific features- atom_frames, chirals
        self.indep.atom_frames = torch.cat(frames_atomize_all)
        self.indep.chirals = torch.cat(chirals_atomize_all)
        self.indep.terminus_type = torch.cat((self.indep.terminus_type, torch.zeros(atomize_L)))
        return total_L

    def pop_protein_feats(self, res_idx_atomize_all, total_L):
        """
        after adding the atom information into the tensors, remove the associated protein sequence and template information
        """
        is_atomized_residue = torch.tensor(res_idx_atomize_all) # indices of residues that have been atomized and need other feats removed
        pop = torch.ones((total_L))
        pop[is_atomized_residue] = 0
        pop = pop.bool()
        self.indep.seq         = self.indep.seq[pop]
        self.indep.xyz         = self.indep.xyz[pop]
        self.indep.idx     = self.indep.idx[pop]
        self.indep.same_chain  = self.indep.same_chain[pop][:, pop]
        self.indep.bond_feats = self.indep.bond_feats[pop][:, pop].long()
        n_shift = (~pop).cumsum(dim=0)
        chiral_indices = self.indep.chirals[:,:-1]
        chiral_shift = n_shift[chiral_indices.long()]
        self.indep.chirals[:,:-1] = chiral_indices - chiral_shift
        self.indep.terminus_type = self.indep.terminus_type[pop]
        return pop

    def get_atomized_res_idx_from_res(self, expect_H=False):

        atom_idx_by_res = {}
        
        N_atoms = rf2aa.chemical.NHEAVYPROT
        ipdb.set_trace()
        if expect_H:
            N_atoms = rf2aa.chemical.NTOTAL
        atomized_mask = rf2aa.util.allatom_mask[torch.tensor(self.atomized_res, dtype=int)][:,:N_atoms]
        atomized_res_natoms = atomized_mask.sum(dim=1)
        N_atomized_res = len(self.atomized_res)
        N_atomized_atoms = sum(atomized_res_natoms)

        L = self.indep.seq.shape[0] # length of atomized features
        L_base = L - N_atomized_atoms # length of non-atomized region in atomized features
        L_new = L_base + N_atomized_res # length of deatomized features

        idx_nonatomized = np.setdiff1d(np.arange(L_new), self.atomized_res_idx)
        # map residue indices in atomized features to indices in deatomized features
        return dict(zip(idx_nonatomized, np.arange(L_base)))


    def get_atom_idx_by_res(self, expect_H=False):

        atom_idx_by_res = {}
        
        N_atoms = rf2aa.chemical.NHEAVYPROT
        ipdb.set_trace()
        if expect_H:
            N_atoms = rf2aa.chemical.NTOTAL

        atomized_mask = rf2aa.util.allatom_mask[torch.tensor(self.atomized_res, dtype=int)][:,:N_atoms]
        atomized_res_natoms = atomized_mask.sum(dim=1)
        N_atomized_res = len(self.atomized_res)
        assertpy.assert_that(len(self.atomized_res)).is_equal_to(len(self.atomized_res_idx))
        N_atomized_atoms = sum(atomized_res_natoms)

        L = self.indep.seq.shape[0] # length of atomized features
        L_base = L - N_atomized_atoms # length of non-atomized region in atomized features
        L_new = L_base + N_atomized_res # length of deatomized features

        # indices of non-atomized positions in deatomized features
        idx_nonatomized = np.setdiff1d(np.arange(L_new), self.atomized_res_idx)
        assert(len(idx_nonatomized)==L_base)

        # deatomize the previously atomized residues
        for i in range(N_atomized_res):
            res_idx = self.atomized_res_idx[i]
            # assumes atomized atoms were in standard order
            atom_idx_range = L_base + sum(atomized_res_natoms[:i]) + np.arange(atomized_res_natoms[i])
            atom_idx_by_res[res_idx.item()] = atom_idx_range
            # xyz[res_idx, :len(atom_idx_range)] = xyz_pred[atom_idx_range,1]
        return atom_idx_by_res


    def get_deatomized_features(self, seq_pred, xyz_pred, expect_H=False):
        """Converts previously atomized residues back into residue representation and
        returns features for PDB output. Does not update instance variables.

        NOTE: This only generates features needed for output, NOT for input into RF (i.e.
        chirals, frames, etc, are ignored here).

        Args:
            seq_pred: torch.Tensor (L, NAATOKENS), Sequence (1-hot) output by the network
            xyz_pred: torch.Tensor (L, N_atoms, 3), Coordinates output by the network,
            without the batch dimension. Should be the same shape as the input coordinates
            (i.e. `self.indep.xyz`)

        Returns:
            Deatomized sequence (1-hot), coordinates, PDB indices, bond & same_chain features.
            seq: torch.Tensor (L_new, NAATOKENS)
            xyz: torch.Tensor (L_new, N_atoms, 3)
            idx: torch.Tensor (L_new,)
            bond_feats: torch.Tensor (L_new, L_new)
            same_chain: torch.Tensor (L_new, L_New)
        """
        N_atoms = rf2aa.chemical.NHEAVYPROT
        if expect_H:
            N_atoms = rf2aa.chemical.NTOTAL
        assert(seq_pred.shape[0] == self.indep.seq.shape[0])
        
        # no atomization was done, just return unmodified features
        if len(self.atomized_res)==0:
            return seq_pred, xyz_pred, self.indep.bond_feats, self.indep.same_chain

        # assumes all heavy atoms are present
        atomized_mask = rf2aa.util.allatom_mask[torch.tensor(self.atomized_res)][:,:N_atoms]
        atomized_res_natoms = atomized_mask.sum(dim=1)
        N_atomized_res = len(self.atomized_res)
        N_atomized_atoms = sum(atomized_res_natoms)

        L = seq_pred.shape[0] # length of atomized features
        L_base = L - N_atomized_atoms # length of non-atomized region in atomized features
        L_new = L_base + N_atomized_res # length of deatomized features

        # deatomized features
        seq = torch.full((L_new,), UNKINDEX).long()
        seq = torch.nn.functional.one_hot(seq, num_classes=NAATOKENS) # (L_new, NAATOKENS)
        xyz = torch.full((L_new, rf2aa.chemical.NTOTAL, 3), np.nan).float()
        idx = torch.full((L_new,), np.nan).long()
        bond_feats = torch.zeros((L_new, L_new)).long()
        same_chain = torch.zeros((L_new, L_new)).long()

        # indices of non-atomized positions in deatomized features
        idx_nonatomized = np.setdiff1d(np.arange(L_new), self.atomized_res_idx)
        assert(len(idx_nonatomized)==L_base)

        # map residue indices in atomized features to indices in deatomized features
        idxmap = dict(zip(np.arange(L_base), idx_nonatomized))

        # copy over features of positions that were never atomized
        seq[idx_nonatomized] = seq_pred[:L_base].long()
        xyz[idx_nonatomized] = xyz_pred[:L_base]
        idx[idx_nonatomized] = self.indep.idx[:L_base]
        for i_src, i_dest in enumerate(idx_nonatomized):
            bond_feats[i_dest,idx_nonatomized] = self.indep.bond_feats[i_src, :L_base]
            same_chain[i_dest,idx_nonatomized] = self.indep.same_chain[i_src, :L_base].long()

        # residue indices of residue-atom bonds in atomized features
        idx_atomize_bonds = torch.where(self.indep.bond_feats==6)

        # deatomize the previously atomized residues
        for i in range(N_atomized_res):
            res_idx = self.atomized_res_idx[i] 
            seq[res_idx, self.atomized_res[i]] = 1
            idx[res_idx] = self.atomized_idx[i]

            # assumes atomized atoms were in standard order
            atom_idx_range = L_base + sum(atomized_res_natoms[:i]) + np.arange(atomized_res_natoms[i])
            xyz[res_idx, :len(atom_idx_range)] = xyz_pred[atom_idx_range,1] 

            # bonds between atoms of this atomized residue and nonatomized residues
            for i,j in zip(*idx_atomize_bonds):
                if i in atom_idx_range: # assume bond features are symmetrical
                    bond_feats[res_idx, idxmap[int(j)]] = 5
                    bond_feats[idxmap[int(j)], res_idx] = 5

        # assign atomized residues to whichever chain their atomized atoms were bonded to
        same_chain = rf2aa.util.same_chain_from_bond_feats(bond_feats | same_chain)

        return seq, xyz, idx, bond_feats, same_chain

    @staticmethod
    def choose_random_atom_motif(natoms, p=0.5):
        """
        selects each atom to be in the motif with a probability p 
        """
        return torch.rand((natoms)) > p

    def choose_sm_contact_motif(self, xyz_atomize):
        """
        chooses atoms to be the motif based on the atoms that are closest to the small molecule
        """
        dist = torch.cdist(self.indep.xyz[self.indep.is_sm, 1, :], xyz_atomize)
        closest_sm_atoms = torch.min(dist, dim=-2)[0][0] # min returns a tuple of values and indices, we want the values
        contacts = closest_sm_atoms < 4
        # if no atoms are closer than 4 angstroms, choose the closest three atoms
        if torch.all(contacts == 0):
            min_indices = torch.argsort(closest_sm_atoms)[:3]
            contacts[min_indices] = 1
        return contacts
    
    @staticmethod
    def choose_contiguous_atom_motif(bond_feats_atomize):
        """
        chooses a contiguous 3 or 4 atom motif
        """
        natoms = bond_feats_atomize.shape[0]
        # choose atoms to be given as the motif 
        is_atom_motif = torch.zeros((natoms),dtype=bool)
        bond_graph = nx.from_numpy_matrix(bond_feats_atomize.numpy())
        paths = rf2aa.util.find_all_paths_of_length_n(bond_graph, 2)
        paths.extend(rf2aa.util.find_all_paths_of_length_n(bond_graph, 3))
        chosen_path = random.choice(paths)
        is_atom_motif[torch.tensor(chosen_path)] = 1
        return is_atom_motif

    def return_input_tensors(self):
        return self.indep, self.input_str_mask, self.input_seq_mask

GP_BOND = 7
def transform_indep(indep, 
                    is_res_str_shown, 
                    is_atom_str_shown, 
                    use_guideposts, 
                    guidepost_placement='anywhere'):
    
    indep = copy.deepcopy(indep)
    use_atomize = is_atom_str_shown is not None and len(is_atom_str_shown) > 0
    is_diffused = is_masked_seq = ~is_res_str_shown
    atomizer    = None
    gp_to_ptn_idx0 = None

    if use_guideposts:
        mask_gp = is_res_str_shown.clone()
        mask_gp[indep.is_sm] = False
        if use_atomize:
            atomized_residues = list(is_atom_str_shown.keys())
            mask_gp[atomized_residues] = True
        if mask_gp.sum() == 0:
            return indep, is_diffused, is_masked_seq, atomizer, {}


        indep_length_pre_gp = indep.length()
        indep, is_diffused, gp_to_ptn_idx0 = gp.make_guideposts(indep, mask_gp)
        L_added_gp = indep.length() - indep_length_pre_gp
        if use_atomize:
            gp_from_ptn_idx0 = {v:k for k,v in gp_to_ptn_idx0.items()}
            is_atom_str_shown = {gp_from_ptn_idx0[k]: v for k,v in is_atom_str_shown.items()}
            # Remove redundancy
            is_diffused[list(is_atom_str_shown.keys())] = True
        is_masked_seq = is_diffused.clone()

    if use_atomize:
        is_ligand = indep.is_sm
        is_diffused[indep.is_sm] = False
        indep_length_pre_atomized = indep.length()
        to_atomize_seq = indep.seq[list(is_atom_str_shown.keys())]
        is_atom_str_shown = {k.item() if hasattr(k, 'item') else k :v for k,v in is_atom_str_shown.items()}
        indep, is_diffused, is_masked_seq, atomizer = atomize.atomize_and_mask(indep, ~is_diffused, is_atom_str_shown)
        assertpy.assert_that(indep.is_sm.sum()).is_equal_to(indep.atom_frames.shape[0])
        ligand_idx = atomize.atomized_indices_res(atomizer, is_ligand)
        if use_guideposts:
            # HACK: use is_masked_seq as gp_idx0.
            is_gp = ~is_masked_seq
            is_inter_gp = is_gp[None, :] != is_gp[:, None]
            is_inter_gp[ligand_idx] = False
            is_inter_gp[:,ligand_idx] = False
            indep.bond_feats[is_inter_gp] = GP_BOND

    return indep, is_diffused, is_masked_seq, atomizer, gp_to_ptn_idx0


def hetatm_names(pdb):
    d = defaultdict(list)
    with open(pdb) as f:
        for line in f.readlines():
            if line.startswith('HETATM'):
                lig_name = line[17:20].strip()
                atom_name = line[12:16].strip()
                element_name = line[76:78].strip()
                d[lig_name].append((atom_name, element_name))
    return d

def without_H(atom_elem_by_lig):
    ''' Drops Hs from a dictionary like {'LG1': [('CB', 'C'), ('H2', 'H')]}'''
    out = {}
    for lig, atom_names in atom_elem_by_lig.items():
        out[lig] = [(atom_name, element) for atom_name, element in atom_names if element != 'H']
    return out

def rename_ligand_atoms(ref_fn, out_fn):
    """Copies names of ligand residue and ligand heavy atoms from input pdb
    into output (design) pdb."""

    ref_atom_names_by_lig = hetatm_names(ref_fn)
    ref_atom_names_by_lig = without_H(ref_atom_names_by_lig)
    with open(out_fn) as f:
        lines = [line.strip() for line in f.readlines()]

    lines2 = []
    ligand_counters = defaultdict(lambda: 0)
    for line in lines:
        if line.startswith('HETATM'):
            lig_name = line[17:20].strip()
            element_name = line[76:78].strip()
            assertpy.assert_that(ref_atom_names_by_lig).contains(lig_name)
            assertpy.assert_that(element_name).is_not_equal_to('H')
            ref_atom_name, ref_element_name = ref_atom_names_by_lig[lig_name][ligand_counters[lig_name]]
            assertpy.assert_that(element_name.upper()).is_equal_to(ref_element_name.upper())
            ligand_counters[lig_name] += 1
            line = line[:12] + ref_atom_name.ljust(4, ' ') + line[16:]
            line = line[:76] + ref_element_name.rjust(2, ' ') + line[78:]
        if line.startswith('MODEL'):
            ligand_counters = defaultdict(lambda: 0)
        lines2.append(line)

    with open(out_fn,'w') as f:
        for line in lines2:
            print(line, file=f)

def randomly_rotate_frames(xyz):
    L, _, _ = xyz.shape
    R_rand = rotation_conversions.random_rotations(L, dtype=xyz.dtype)
    frame_origins = xyz[:,1:2,:]
    xyz_centered = xyz - frame_origins
    rotated = torch.einsum('lab,lib->...lia', R_rand, xyz_centered)
    rotated += frame_origins
    return rotated


def eye_frames2(xyz):
    """
    replaces frames in xyz with identity frames, this version simply using chemical.INIT_CRDS as frame.
    """
    L, _, _ = xyz.shape

    T = xyz[:,1,:] # CA coords

    init = chemical.INIT_CRDS[:3] # (3,3)
    init = init[None].expand(L,3,3) # (L,3,3), replaces N,CA,C such that frame is identity

    xyz[:,:3,:] = init + T[:,None,:].expand(L,3,3)

    return xyz


def eye_frames(xyz, _assert=False):
    L, _, _ = xyz.shape

    # find the R that would yield ID matrix when combined w/ current frames 
    R_orig, _ = util.rigid_from_3_points(xyz[None,:,0,:], xyz[None,:,1,:], xyz[None,:,2,:])
    R_orig = R_orig[0] # (L,3,3)
    R_inv = R_orig.transpose(-1,-2)

    # apply R_inv to current frames such that they now have ID Rs 
    ca = xyz[:,1:2,:]
    rotated = torch.einsum('lab,lib->lia', R_inv, xyz - ca) + ca 

    if _assert: 
        # make sure it worked 
        R_new, _ = util.rigid_from_3_points(rotated[None,:,0,:], rotated[None,:,1,:], rotated[None,:,2,:])
        R_new = R_new[0] # (L,3,3)
        
        eye = torch.eye(3, dtype=R_new.dtype, device=R_new.device).unsqueeze(0).expand(L,3,3)
        
        is_close = torch.tensor([torch.allclose(R_new[i], eye[i], atol=1e-5) for i in range(L)])
        assert is_close.all()

    return rotated
