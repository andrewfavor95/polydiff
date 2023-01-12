import torch
import ipdb
import dataclasses
from icecream import ic
from assertpy import assert_that
from rf2aa.chemical import NAATOKENS, MASKINDEX, NTOTAL, NHEAVYPROT
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
from rf2aa.chemical import MASKINDEX
import util

@dataclass
class Indep:
    seq: torch.Tensor
    xyz: torch.Tensor
    idx: torch.Tensor

    # SM specific
    bond_feats: torch.Tensor
    chirals: torch.Tensor
    atom_frames: torch.Tensor
    same_chain: torch.Tensor
    is_sm: torch.Tensor

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

@dataclass
class RFO:
    logits: torch.Tensor
    logits_aa: torch.Tensor
    logits_pae: torch.Tensor
    logits_pde: torch.Tensor
    xyz: torch.Tensor
    alpha_s: torch.Tensor
    xyz_allatom: torch.Tensor
    lddt: torch.Tensor
    msa: torch.Tensor
    pair: torch.Tensor
    state: torch.Tensor

    def get_seq_logits(self):
        return self.logits_aa.permute(0,2,1)
    
    def get_xyz(self):
        return self.xyz_allatom


class Model:

    def __init__(self, conf):
        self.conf = conf
        self.NTOKENS = rf2aa.chemical.NAATOKENS

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
        atom_frames = torch.zeros((1,0,3,2))

        xyz_prot, mask_prot, idx_prot, seq_prot = parsers.parse_pdb(pdb, seq=True)
        xyz_prot[:,14:] = 0 # remove hydrogens
        mask_prot[:,14:] = False
        xyz_prot = torch.tensor(xyz_prot)
        mask_prot = torch.tensor(mask_prot)
        protein_L, nprotatoms, _ = xyz_prot.shape
        msa_prot = torch.tensor(seq_prot)[None].long()
        ins_prot = torch.zeros(msa_prot.shape).long()
        a3m_prot = {"msa": msa_prot, "ins": ins_prot}
        if parse_hetatm:
            stream = [l for l in open(pdb) if "HETATM" in l or "CONECT" in l]
            mol, msa_sm, ins_sm, xyz_sm, mask_sm = parsers.parse_mol("".join(stream), filetype="pdb", string=True)
            a3m_sm = {"msa": msa_sm.unsqueeze(0), "ins": ins_sm.unsqueeze(0)}
            G = rf2aa.util.get_nxgraph(mol)
            atom_frames = rf2aa.util.get_atom_frames(msa_sm, G)
            N_symmetry, sm_L, _ = xyz_sm.shape
            Ls = [protein_L, sm_L]
            a3m = merge_a3m_hetero(a3m_prot, a3m_sm, Ls)
            msa = a3m['msa'].long()
            chirals = get_chirals(mol, xyz_sm[0]) + protein_L
        else:
            Ls = [msa_prot.shape[-1], 0]
            N_symmetry = 1
            msa = msa_prot

        xyz = torch.full((N_symmetry, sum(Ls), NTOTAL, 3), np.nan).float()
        mask = torch.full(xyz.shape[:-1], False).bool()
        xyz[:, :Ls[0], :nprotatoms, :] = xyz_prot.expand(N_symmetry, Ls[0], nprotatoms, 3)
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
            bond_feats[Ls[0]:, Ls[0]:] = rf2aa.util.get_bond_feats(mol, G)


        same_chain = torch.zeros((sum(Ls), sum(Ls))).long()
        same_chain[:Ls[0], :Ls[0]] = 1
        same_chain[Ls[0]:, Ls[0]:] = 1
        is_sm = torch.zeros(sum(Ls)).bool()
        is_sm[Ls[0]:] = True

        indep = Indep(
            seq,
            xyz,
            idx_pdb,
            # SM specific
            bond_feats,
            chirals,
            atom_frames,
            same_chain,
            is_sm)
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
        return o



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
        msa_masked[:,:,0,NAATOKENS*2+NINDEL*2] = 1.0
        msa_masked[:,:,-1,NAATOKENS*2+NINDEL*2+1] = 1.0

        ### msa_full ###
        ################
        msa_full = torch.zeros((1,1,L,NAATOKENS+NINDEL+NTERMINUS))
        msa_full[:,:,:,:NAATOKENS] = seq_one_hot[None, None]
        msa_full[:,:,0,NAATOKENS+NINDEL] = 1.0
        msa_full[:,:,-1,NAATOKENS+NINDEL+1] = 1.0

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
        mask_t = torch.ones(1,L,L).bool()
        sctors = torch.zeros((1,L,rf2aa.chemical.NTOTALDOFS,2))

        xyz = torch.squeeze(xyz_t, dim=0)

        # NO SELF COND
        xyz_t = torch.zeros(1,1,L,3)
        t2d = torch.zeros(1,1,L,L,68)

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
            indep.atom_frames,
            t1d,
            t2d,
            xyz_t,
            alpha_t,
            mask_t[None],
            indep.same_chain[None],
            ~is_diffused)
        return rfi
    
    def self_cond(self, indep, rfi, rfo):
        # RFI is already batched
        B = 1
        L = indep.xyz.shape[0]
        rfi_sc = copy.deepcopy(rfi)
        zeros = torch.zeros(B,1,L,36-3,3).float().to(rfi.xyz.device)
        xyz_t = torch.cat((rfo.xyz[-1:], zeros), dim=-2) # [B,T,L,27,3]
        t2d, mask_t_2d_remade = util.get_t2d(
            xyz_t[0], indep.is_sm, rfi.atom_frames)
        t2d = t2d[None] # Add batch dimension # [B,T,L,L,44]
        # xyz_t = xyz_t[...,1,:]
        rfi_sc.xyz_t = xyz_t[:,:,:,1]
        rfi_sc.t2d = t2d
        return rfi_sc
