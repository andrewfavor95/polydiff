import torch
import assertpy
import torch.nn.functional as F
import ipdb
import dataclasses
from icecream import ic
from assertpy import assert_that
from rf2aa.chemical import NAATOKENS, MASKINDEX, NTOTAL, NHEAVYPROT, UNKINDEX
import rf2aa.util
from rf2aa import parsers
from dataclasses import dataclass
from rf2aa.data_loader import MSAFeaturize, MSABlockDeletion, merge_a3m_homo, merge_a3m_hetero
from rf2aa.kinematics import xyz_to_c6d, c6d_to_bins, xyz_to_t2d, get_chirals
import rf2aa.tensor_util
import torch
import copy
import numpy as np
from kinematics import get_init_xyz
import chemical
from rf2aa.chemical import MASKINDEX
import util
import inference.utils
import networkx as nx
import itertools
import random


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
    seq: torch.Tensor # [L]
    xyz: torch.Tensor # [L, 36?, 3]
    idx: torch.Tensor

    # SM specific
    bond_feats: torch.Tensor
    chirals: torch.Tensor
    atom_frames: torch.Tensor
    same_chain: torch.Tensor
    is_sm: torch.Tensor
    terminus_type: torch.Tensor

    def write_pdb(self, path):
        ic(self.xyz.shape, self.seq.shape)
        # final_seq = torch.where(self.seq >= 20, 0, self.seq)
        # util.writepdb(path, self.xyz[:,:3], final_seq)
        seq = self.seq
        seq = torch.where(seq == 20, 0, seq)
        seq = torch.where(seq == 21, 0, seq)
        rf2aa.util.writepdb(path,
            torch.nan_to_num(self.xyz), seq, self.bond_feats)

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
    xyz: torch.Tensor         # [40, 1, L, 3, 3]
    alpha_s: torch.Tensor     # [40, 1, L, 20, 2]
    xyz_allatom: torch.Tensor # [1, L, 36, 3]
    lddt: torch.Tensor        # [1, 50, L]
    msa: torch.Tensor
    pair: torch.Tensor
    state: torch.Tensor

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

    for l in pdb_lines:
        if 'CONECT' not in l:
            continue
        ids = [int(e.strip()) for e in l[6:].split()]
        if all(i in hetatm_ids for i in ids):
            lines.append(l)
            continue
        if any(i in hetatm_ids for i in ids):
            raise Exception(f'line {l} references atom ids in the target ligand {ligand} and another atom')
    return lines


class Model:

    def __init__(self, conf):
        self.conf = conf
        self.NTOKENS = rf2aa.chemical.NAATOKENS
        self.atomizer = None

    def forward(self, rfi, **kwargs):
        # ipdb.set_trace()
        rfi_dict = dataclasses.asdict(rfi)
        # assert set(rfi_dict.keys()) - set()
        return RFO(*self.model(**{**rfi_dict, **kwargs}))

    def make_indep(self, pdb, parse_hetatm):
        # self.target_feats = iu.process_target(self.inf_conf.input_pdb, parse_hetatom=True, center=False)
        # init_protein_tmpl=False, init_ligand_tmpl=False, init_protein_xyz=False, init_ligand_xyz=False,
        #     parse_hetatm=False, n_cycle=10, random_noise=5.0)
        chirals = torch.Tensor()
        atom_frames = torch.zeros((0,3,2))

        xyz_prot, mask_prot, idx_prot, seq_prot = parsers.parse_pdb(pdb, seq=True)

        target_feats = inference.utils.parse_pdb(pdb)
        xyz_prot, mask_prot, idx_prot, seq_prot = target_feats['xyz'], target_feats['mask'], target_feats['idx'], target_feats['seq']
        xyz_prot[:,14:] = 0 # remove hydrogens
        mask_prot[:,14:] = False
        xyz_prot = torch.tensor(xyz_prot)
        mask_prot = torch.tensor(mask_prot)
        protein_L, nprotatoms, _ = xyz_prot.shape
        msa_prot = torch.tensor(seq_prot)[None].long()
        ins_prot = torch.zeros(msa_prot.shape).long()
        a3m_prot = {"msa": msa_prot, "ins": ins_prot}
        if parse_hetatm:
            with open(pdb, 'r') as fh:
                stream = [l for l in fh if "HETATM" in l or "CONECT" in l]
            ligand = self.conf.inference.ligand
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
        if parse_hetatm:
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
        if parse_hetatm:
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

        indep = Indep(
            seq,
            xyz,
            idx_pdb,
            # SM specific
            bond_feats,
            chirals,
            atom_frames,
            same_chain,
            is_sm,
            terminus_type)
        return indep


    def insert_contig(self, indep, contig_map, partial_T=False):
        o = copy.deepcopy(indep)

        # Insert small mol into contig_map
        all_chains = set(ch for ch,_ in contig_map.hal)
        # Not yet implemented due to index shifting
        assert_that(len(all_chains)).is_equal_to(1)
        next_unused_chain = next(e for e in contig_map.chain_order if e not in all_chains)
        n_sm = indep.is_sm.sum()
        is_sm_idx0 = torch.nonzero(indep.is_sm, as_tuple=True)[0].tolist()
        contig_map.ref_idx0.extend(is_sm_idx0)
        n_protein_hal = len(contig_map.hal)
        contig_map.hal_idx0 = np.concatenate((contig_map.hal_idx0, np.arange(n_protein_hal, n_protein_hal+n_sm)))
        max_hal_idx = max(i for _, i  in contig_map.hal)
        contig_map.hal.extend(zip([next_unused_chain]*n_sm, range(max_hal_idx+200,max_hal_idx+200+n_sm)))
        chain_id = np.array([c for c, _ in contig_map.hal])
        L_mapped = len(contig_map.hal)
        n_prot = L_mapped - n_sm
        L_in, NATOMS, _ = indep.xyz.shape
        o.xyz = torch.full((L_mapped, NATOMS, 3), np.nan)

        o.xyz[contig_map.hal_idx0] = indep.xyz[contig_map.ref_idx0]
        o.seq = torch.full((L_mapped,), MASKINDEX)
        o.seq[contig_map.hal_idx0] = indep.seq[contig_map.ref_idx0]
        o.is_sm = torch.full((L_mapped,), 0).bool()
        o.is_sm[contig_map.hal_idx0] = indep.is_sm[contig_map.ref_idx0]
        o.same_chain = torch.tensor(chain_id[None, :] == chain_id[:, None])
        o.xyz = get_init_xyz(o.xyz[None, None], o.is_sm).squeeze()

        o.bond_feats = torch.full((L_mapped, L_mapped), 0).long()
        o.bond_feats[:n_prot, :n_prot] = rf2aa.util.get_protein_bond_feats(n_prot)
        n_prot_ref = L_in-n_sm
        o.bond_feats[n_prot:, n_prot:] = indep.bond_feats[n_prot_ref:, n_prot_ref:]

        hal_by_ref_d = dict(zip(contig_map.ref_idx0, contig_map.hal_idx0))
        def hal_by_ref(ref):
            return hal_by_ref_d[ref]
        hal_by_ref = np.vectorize(hal_by_ref, otypes=[float])
        o.chirals[...,:-1] = torch.tensor(hal_by_ref(o.chirals[...,:-1]))

        o.idx = torch.tensor([i for _, i in contig_map.hal])

        o.terminus_type = torch.zeros(L_mapped)
        o.terminus_type[0] = N_TERMINUS
        o.terminus_type[n_prot-1] = C_TERMINUS

        # is_diffused = torch.
        is_diffused_prot = ~torch.from_numpy(contig_map.inpaint_str)
        is_diffused_sm = torch.zeros(n_sm).bool()
        is_diffused = torch.cat((is_diffused_prot, is_diffused_sm))
        if contig_map.contig_atoms is not None:
            is_atomized = torch.zeros(n_protein_hal+n_sm).bool()
            is_atomized[contig_map.atomize_indices] = 1
            input_str_mask = torch.logical_or(is_atomized, ~is_diffused) # 1 for motif, 0 for diffused

            self.atomizer = AtomizeResidues(o, input_str_mask)
            atom_mask = rf2aa.util.allatom_mask[o.seq]
            atom_mask[:, 14:] = False # no Hs
            self.atomizer.featurize_atomized_residues(atom_mask, contig_map.atomize_indices2atomname)
            o, input_str_mask, input_seq_mask = self.atomizer.return_input_tensors()
            is_diffused = ~input_str_mask # 1 for diffused, 0 for motif
            is_seq_masked = ~input_seq_mask
            o.xyz[is_diffused] = np.nan
            o.xyz = get_init_xyz(o.xyz[None, None], o.is_sm).squeeze()
        else:
            is_seq_masked = is_diffused

        # HACK.  ComputeAllAtom in the network requires N and C coords even for atomized residues,
	    # However, these have no semantic value.  TODO: Remove the network's reliance on these coordinates.
        sm_ca = o.xyz[o.is_sm, 1]
        o.xyz[o.is_sm,:3] = sm_ca[...,None,:]
        o.xyz[o.is_sm] += chemical.INIT_CRDS
        #ic(o.xyz)
        # To see the shapes of the indep struct with contig inserted
        # print(rf2aa.tensor_util.info(rf2aa.tensor_util.to_ordered_dict(o)))
        return o, is_diffused, is_seq_masked


    def prepro(self, indep, t, is_diffused):
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
        seq_cat_shifted = seq_one_hot.argmax(dim=-1)
        seq_cat_shifted[seq_cat_shifted>=MASKINDEX] -= 1
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

        ### xyz_t ###
        #############
        if self.conf.preprocess.sidechain_input:
            raise Exception('not implemented')
            xyz_t[torch.where(seq_one_hot == 21, True, False),3:,:] = float('nan')
        else:
            xyz_t[is_diffused,3:,:] = float('nan')
        #xyz_t[:,3:,:] = float('nan')

        assert_that(xyz_t.shape).is_equal_to((L,NHEAVYPROT,3))
        xyz_t=xyz_t[None, None]
        xyz_t = torch.cat((xyz_t, torch.full((1,1,L,NTOTAL-NHEAVYPROT,3), float('nan'))), dim=3)

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

        alpha, _, alpha_mask, _ = rf2aa.util.get_torsions(xyz_t.reshape(-1,L,rf2aa.chemical.NTOTAL,3), seq_tmp,
            rf2aa.util.torsion_indices, rf2aa.util.torsion_can_flip, rf2aa.util.reference_angles)
        alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
        alpha[torch.isnan(alpha)] = 0.0
        alpha = alpha.reshape(-1,L,rf2aa.chemical.NTOTALDOFS,2)
        alpha_mask = alpha_mask.reshape(-1,L,rf2aa.chemical.NTOTALDOFS,1)
        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(-1, L, 3*rf2aa.chemical.NTOTALDOFS) # [n,L,30]

        alpha_t = alpha_t.unsqueeze(1) # [n,I,L,30]



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
        mask_t = torch.ones(1,1,L,L).bool()
        sctors = torch.zeros((1,L,rf2aa.chemical.NTOTALDOFS,2))

        xyz = torch.squeeze(xyz_t, dim=0)

        # NO SELF COND
        xyz_t = torch.zeros(1,1,L,3)
        t2d = torch.zeros(1,1,L,L,68)

        #ic(xyz.shape)
        # ic(
        #     xyz[0, is_diffused][0][:,0], # nan 3:
        #     xyz[0, indep.is_sm][0][:,0], # nan 14:
        #     xyz[0, ~is_diffused * ~indep.is_sm][0][:,0], # nan 14:
        # )

        is_protein_motif = ~is_diffused * ~indep.is_sm
        # idx_diffused = torch.nonzero(is_diffused)
        # idx_protein_motif  = torch.nonzero(is_protein_motif)
        # idx_sm = torch.nonzero(indep.is_sm)

        # ic(
        #     idx_diffused,
        #     idx_protein_motif,
        #     idx_sm
        # )

        # xyz = torch.nan_to_num(xyz)
        xyz[0, is_diffused*~indep.is_sm,3:] = torch.nan
        xyz[0, indep.is_sm,14:] = 0
        xyz[0, is_protein_motif, 14:] = 0

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


def pad_dim(x, dim, new_l):
    padding = [0]*2*x.ndim
    padding[2*dim] = new_l - x.shape[dim]
    padding = padding[::-1]
    return F.pad(x, pad=tuple(padding), value=0)

def write_traj(path, xyz_stack, seq, bond_feats, **kwargs):
    xyz23 = pad_dim(xyz_stack, 2, 23)
    with open(path, 'w') as fh:
        for i, xyz in enumerate(xyz23):
            rf2aa.util.writepdb_file(fh, xyz, seq, bond_feats=bond_feats[None], modelnum=i, **kwargs)

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
    assert len(out) == 22, f"found {len(out)} elements in RF2-allatom output"
    (seq, msa, msa_masked, msa_full, mask_msa, true_crds, atom_mask, idx_pdb, xyz_t, t1d, mask_t, xyz_prev,
        mask_prev, same_chain, unclamp, negative, atom_frames, bond_feats, chirals, ch_label, dataset_name, item) = out
    #remove permutation symmetry dimension if present
    if len(true_crds.shape) == 4 and len(atom_mask.shape) == 3:
        true_crds = true_crds[0]
        atom_mask = atom_mask[0]
    
    # our dataloaders return torch.zeros(L...) for atom frames and chirals when there are none, this updates it to use common shape 
    ic(atom_frames)
    if torch.all(atom_frames == 0):
        atom_frames = torch.zeros((0,3,2))
    if torch.all(chirals == 0):
        chirals = torch.zeros((0,5))

    is_sm = rf2aa.util.is_atom(seq)

    is_n_terminus = msa_full[0, 0, :, MSAFULL_N_TERM].bool()
    is_c_terminus = msa_full[0, 0, :, MSAFULL_C_TERM].bool()
    terminus_type = torch.zeros(msa_masked.shape[2], dtype=int)
    terminus_type[is_n_terminus] = N_TERMINUS
    terminus_type[is_c_terminus] = C_TERMINUS

    indep = Indep(
        rf2aa.tensor_util.assert_squeeze(seq), # [L]
        true_crds[:,:14], # [L, 14, 3]
        idx_pdb,

        # SM specific
        bond_feats,
        chirals,
        atom_frames,
        same_chain,
        rf2aa.tensor_util.assert_squeeze(is_sm),
        terminus_type)
    return indep, atom_mask, dataset_name

def pop_unoccupied(indep, atom_mask):
    """
    Inplace operation.
    Removes ligand atoms which are not present in the atom mask.
    Removes residues which do not have N,C,Ca in the atom mask.
    """
    n_atoms = indep.is_sm.sum()
    assertpy.assert_that(len(indep.atom_frames)).is_equal_to(n_atoms)

    pop = rf2aa.util.get_prot_sm_mask(atom_mask, indep.seq)
    ic(atom_mask.shape, indep.seq.shape, pop.shape)
    ic(f'Removing {(~pop).sum()} residues / atoms')
    N     = pop.sum()
    pop2d = pop[None,:] * pop[:,None]

    indep.seq           = indep.seq[pop]
    indep.xyz           = indep.xyz[pop]
    indep.idx           = indep.idx[pop]
    indep.bond_feats    = indep.bond_feats[pop2d].reshape(N,N)
    indep.same_chain    = indep.same_chain[pop2d].reshape(N,N)
    indep.is_sm         = indep.is_sm[pop]
    indep.terminus_type = indep.terminus_type[pop]

    n_shift = (~pop).cumsum(dim=0)
    chiral_indices = indep.chirals[:,:-1]
    chiral_shift = n_shift[chiral_indices.long()]
    indep.chirals[:,:-1] = chiral_indices - chiral_shift
    atom_mask           = atom_mask[pop]
    return atom_mask

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
        'is_sm': indep.is_sm
    }
    diffused_fullatoms, aa_masks, true_crds = diffuser.diffuse_pose(**kwargs)
    ic(diffused_fullatoms.shape, indep.xyz.shape)

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
            tmp_x_t[~is_diffused,:] = tmp_x_t_plus1[~is_diffused.squeeze()]
        
        diffused_fullatoms[1] = tmp_x_t

    indep_diffused_tplus1.xyz = diffused_fullatoms[0, :, :14]
    indep_diffused_t.xyz = diffused_fullatoms[1, :, :14]
    return (indep_diffused_tplus1, indep_diffused_t), t_list

def forward(model, rfi, **kwargs):
    rfi_dict = dataclasses.asdict(rfi)
    return RFO(*model(**{**rfi_dict, **kwargs}))

def mask_indep(indep, is_diffused):
    indep.seq[is_diffused] = MASKINDEX

def self_cond(indep, rfi, rfo):
    # RFI is already batched
    B = 1
    L = indep.xyz.shape[0]
    rfi_sc = copy.deepcopy(rfi)
    zeros = torch.zeros(B,1,L,36-3,3).float().to(rfi.xyz.device)
    xyz_t = torch.cat((rfo.xyz[-1:], zeros), dim=-2) # [B,T,L,27,3]
    t2d, mask_t_2d_remade = util.get_t2d(
        xyz_t[0], indep.is_sm, rfi.atom_frames[0])
    t2d = t2d[None] # Add batch dimension # [B,T,L,L,44]
    rfi_sc.xyz_t = xyz_t[:,:,:,1]
    rfi_sc.t2d = t2d
    return rfi_sc


class AtomizeResidues:
    def __init__(
        self,
        indep,
        input_str_mask # dict
        ) -> None:

        self.indep = indep
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
        atomize_indices2atomnames=None
    ):
        """
        this function takes outputs of the RF2aa dataloader and the generated masks and refeaturizes the example where the 
        portion of the structure that is provided as context is treated as atoms instead of residues. the mask will be updated so 
        that only the tip atoms of the context residues are context and the rest is diffused
        """
        is_motif = self.input_str_mask
        seq_atomize_all,  xyz_atomize_all, frames_atomize_all, chirals_atomize_all, res_idx_atomize_all, is_atom_motif_all = \
            self.generate_atomize_protein_features(is_motif, atom_mask, atomize_indices2atomnames)
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
        self.construct_diffusion_masks(is_motif, is_atom_motif_all, pop)
    
    def generate_atomize_protein_features(self, is_motif, atom_mask, atomize_indices2atomnames):
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
        is_atom_motif_all = [] # chosen atoms within each atomized residue chosen as the motif
        # track the res_idx and absolute index of the previous C to draw peptide bonds between contiguous residues
        prev_res_idx = -2
        prev_C_index = -2 
        for res_idx in is_protein_motif.nonzero():
            residue = self.indep.seq[res_idx] # number representing the residue in the token list, can be used to index aa2long
            # check if all the atoms in the residue are resolved, if not dont atomize
            if not torch.all(allatom_mask[residue]==atom_mask[res_idx]):
                continue
            N_term = res_idx ==  0
            C_term = res_idx == self.indep.idx.shape[0]-1

            if N_term:
                C_resolved = self.indep.idx[res_idx+1]-self.indep.idx[res_idx] == 1
                N_resolved = True
            if C_term:
                N_resolved = self.indep.idx[res_idx]-self.indep.idx[res_idx-1] == 1
                C_resolved = True

            if not C_term and not N_term:
                C_resolved = self.indep.idx[res_idx+1]-self.indep.idx[res_idx] == 1
                N_resolved = self.indep.idx[res_idx]-self.indep.idx[res_idx-1] == 1
            if not (C_resolved and N_resolved):
                continue
            
            seq_atomize, _, xyz_atomize, _, frames_atomize, bond_feats_atomize, C_index, chirals_atomize = \
                            rf2aa.util.atomize_protein(res_idx, self.indep.seq[None], xyz, atom_mask, n_res_atomize=1)
            
            natoms = seq_atomize.shape[0]
            if atomize_indices2atomnames is not None: # atoms to keep as motif given during inference
                is_atom_motif = torch.zeros((natoms), dtype=bool)
                # get indices of the atom names chosen as motif by user
                is_atom_motif_indices = [aa2long_[residue].index(atom_name) for atom_name in atomize_indices2atomnames[res_idx.item()]]
                is_atom_motif[is_atom_motif_indices] = 1
            elif self.indep.is_sm.any():
                is_atom_motif = self.choose_sm_contact_motif(xyz_atomize)
            else:
                is_atom_motif = self.choose_contiguous_atom_motif(bond_feats_atomize)
            # update the chirals to be after all the other residues
            chirals_atomize[:, :-1] += L

            seq_atomize_all.append(seq_atomize)
            xyz_atomize_all.append(xyz_atomize)
            frames_atomize_all.append(frames_atomize)
            chirals_atomize_all.append(chirals_atomize)
            res_idx_atomize_all.append(res_idx)
            res_atomize_all.append(residue)
            idx_atomize_all.append(self.indep.idx[res_idx])
            is_atom_motif_all.append(is_atom_motif)

            # update bond_feats every iteration, update all other features at the end 
            bond_feats_new = torch.zeros((L+natoms, L+natoms))
            bond_feats_new[:L, :L] = self.indep.bond_feats
            bond_feats_new[L:, L:] = bond_feats_atomize
            # add bond between protein and atomized N
            if not N_term:
                bond_feats_new[res_idx-1, L] = 6 # protein (backbone)-atom bond 
                bond_feats_new[L, res_idx-1] = 6 # protein (backbone)-atom bond 
            # add bond between protein and C, assumes every residue is being atomized one at a time (eg n_res_atomize=1)
            if not C_term:
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

        return seq_atomize_all, xyz_atomize_all, frames_atomize_all, chirals_atomize_all, res_idx_atomize_all, is_atom_motif_all

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

    def get_deatomized_features(self, seq_pred, xyz_pred):
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
        assert(xyz_pred.shape == self.indep.xyz.shape)
        assert(seq_pred.shape[0] == self.indep.seq.shape[0])
        
        # no atomization was done, just return unmodified features
        if len(self.atomized_res)==0:
            return seq_pred, xyz_pred, self.indep.bond_feats, self.indep.same_chain

        # assumes all heavy atoms are present
        N_atoms = xyz_pred.shape[1]
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
        xyz = torch.full((L_new, N_atoms, 3), np.nan).float()
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


    def construct_diffusion_masks(self, is_motif, is_atom_motif_all, pop):
        """
        takes the original motif, the atom_motif and the residues that were atomized as input and constructs diffusion masks 
        the structure mask is True for atoms provided as context
        the sequence mask is True for all "atom" nodes (residues are prespecified but not their identity)
        """
        self.indep.is_sm = rf2aa.util.is_atom(self.indep.seq)
        is_atom_motif_all =  torch.cat(is_atom_motif_all)
        is_motif = torch.cat((is_motif, is_atom_motif_all))

        is_motif = is_motif[pop]
        self.input_str_mask = is_motif
        # unmask atom types for all atoms in atomized sidechains (these are deterministic)
        self.input_seq_mask = torch.logical_or(is_motif, self.indep.is_sm)
    
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
