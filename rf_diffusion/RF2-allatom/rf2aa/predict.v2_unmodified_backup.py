import logging
LOGGER = logging.getLogger(__name__)
import sys, os, json, pickle, glob, io
import time
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import copy

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)

import rf2aa.parsers as parsers
from rf2aa.RoseTTAFoldModel  import RoseTTAFoldModule
import rf2aa.util as util
from rf2aa.util import *
from rf2aa.loss import *
from collections import namedtuple, OrderedDict
from rf2aa.ffindex import *
from rf2aa.data_loader import MSAFeaturize, MSABlockDeletion, merge_a3m_homo, merge_a3m_hetero
from rf2aa.kinematics import xyz_to_c6d, c6d_to_bins, xyz_to_t2d, get_chirals
from rf2aa.util_module import XYZConverter
from rf2aa.chemical import NTOTAL, NTOTALDOFS, NAATOKENS, INIT_CRDS
from rf2aa.parsers import read_templates, parse_multichain_fasta, parse_mixed_fasta
from rf2aa.memory import mem_report
from rf2aa.symmetry import symm_subunit_matrix, find_symm_subs, update_symm_subs, get_symm_map, get_symmetry

from scipy.interpolate import Akima1DInterpolator

def get_args():
    import argparse
    DB = "/projects/ml/TrRosetta/pdb100_2022Apr19/pdb100_2022Apr19"

    parser = argparse.ArgumentParser(description="RoseTTAFold: Protein structure prediction with 3-track attentions on 1D, 2D, and 3D features")
    
    ######################
    ### Model Settings ###
    ######################
    parser.add_argument("-checkpoint",
        default='/projects/protein-DNA-binders/scripts/rfaa_tf/rf2aa/models/RF2_tf7_1156.pt',
        help="Path to model weights")

    parser.add_argument("-db", default=DB, required=False, help="HHsearch database [%s]"%DB)
    parser.add_argument("-no_extra_l1", dest='use_extra_l1', default='True', action='store_false',
            help="Turn off chirality and LJ grad inputs to SE3 layers (for backwards compatibility).")
    parser.add_argument("-no_atom_frames", dest='use_atom_frames', default='True', action='store_false',
            help="Turn off l1 features from atom frames in SE3 layers (for backwards compatibility).")
    parser.add_argument("-device", required=False, default="cuda:0", help="device to use for predictions (e.g. 'cpu' or 'cuda:0')")

    ################
    ### Symmetry ###
    ################
    parser.add_argument("-symm", required=False, default="C1", help="Model with symmetry")
    parser.add_argument("-symm_fit", required=False, default=False, action='store_true',
        help="Use beta method for 3D updates with symmetry")
    parser.add_argument("-symm_scale", required=False, default=1.0,
        help="When symm_fit is enabled, a scalefactor on translation versus rotation")
    parser.add_argument("-output_asu", default=False, action='store_true', 
        help="When predicting with symmetry, output only the predicted asymmetric unit?")

    ##############
    ### Inputs ###
    ##############
    parser.add_argument("-sequence", required=False, default=None, nargs='+', help="Sequence(s) to predict. Can be DNA, RNA, or protein.")
    parser.add_argument("-sequence_mode", required=False, default="auto", help="Mode to read sequence inputs. Default is 'auto'. Other options are 'dsdna','dsrna','dna','rna','prot','full'")
    parser.add_argument("-fasta", required=False, default=None, nargs='+', help="Fasta-formatted file(s) to predict. Can be DNA, RNA, and/or protein")
    parser.add_argument("-fasta_mode", required=False, default='multi', help="How to treat multiple sequence lines in a fasta file. Options are 'msa','single','multi'.")
    parser.add_argument("-no_auto_dsdna", dest="auto_dsdna", required=False, action="store_false", help="disable automatic double-stranded DNA creation for input DNA sequence or fasta")
    parser.add_argument("-i","--input_files", required=False, default=None, nargs='+', help="Files to use as input. Type will be auto-detected. Valid files are .pdb, .a3m, .fa, .mol2. Fields without a file type prefix will be treated as raw sequences.")
    parser.add_argument("-list", required=False, default=None, help="text file where each line will be treated as the input_files for a separate prediction")
    parser.add_argument("-name", required=False, default=None, help="Name for single prediction")
    parser.add_argument("-namelist", required=False, default=None, help="text file where each line will be the name of a prediction")
    parser.add_argument("-csv", required=False, default=None, help="csv file where columns specify input files and parameters; each row will be a separate prediction")
    parser.add_argument("-csv_subdelim", required=False, default=';', help="delimiter to separate multiple files in the same field of input csv, such as multiple pdbs to use")
    parser.add_argument("-silent_path", required=False, default=None, help="path to silent file to use as input")


    ###########################
    ### Prediction settings ###
    ###########################
    parser.add_argument("-n_cycle", required=False, default=10, type=int, help="number of recycles")
    parser.add_argument("-select_by", required=False, default="plddt", help="parameter to choose best recycle")
    parser.add_argument("-interface_mode", required=False, default="protein_vs_other", help="How to define interface. Options are: 'protein_vs_other', 'one_vs_rest', 'each_vs_all', 'all_together'")
    parser.add_argument("-binder_ch", required=False, default=0, type=int, help="0-indexed chain ID to treat as one side of interface with 'one_vs_rest' interface_mode")
    parser.add_argument("-tmpl_chains", required=False, default=[], nargs='+', help="0-indexed chain IDs (in order of given inputs) to provide templates for")
    parser.add_argument("-tmpl_types", required=False, default=[], nargs='+', help="Chain types to provide templates for (protein, nucleic, and/or atom)")
    parser.add_argument("-tmpl_conf", required=False, default=0.5, type=float, help="Confidence to provide to model for given templates")
    parser.add_argument("-init_chains", required=False, default=[], nargs='+', help="0-indexed chain IDs (in order of given inputs) to initialize prediction with")
    parser.add_argument("-init_types", required=False, default=[], nargs='+', help="Chain types to initialize prediction with (protein, nucleic, and/or atom)")


    ###############
    ### Outputs ###
    ###############
    parser.add_argument("-o","--out_dir", required=False, default="./", help="path to directory to place all outputs from prediction")
    parser.add_argument("-overwrite",default=False, action='store_true', help='overwrite existing predictions')
    parser.add_argument("-scorefile", required=False, default="rf2_scores.csv", help="filename for scores from prediction. Only the basename will be used; use --out_dir to specify directory for output")
    parser.add_argument("-scores", required=False, default=['plddt','i_pae','p_bind','tag'], nargs='+', help="names of scores to report in output. Options are 'pae','i_pae','plddt','p_bind','lddt','i_lddt','ca_rms','rms','l_rms','i_rms','prot_rms','na_rms','sm_rms'")
    parser.add_argument("-extra_scores", required=False, action='extend', dest='scores', nargs="+", help="Names of additional scores to report. Useful if you want to keep all the default scores.")
    parser.add_argument("-dump_extra", default=False, action='store_true',help="Save initial (pre-prediction) and final recycle structures")
    parser.add_argument("-dump_traj", default=False, action='store_true',help="Save RF2 folding trajectory")
    parser.add_argument("-dump_aux", default=False, action='store_true',help="Save RF2 raw probabilities")
    parser.add_argument("-save_all_recycles", default=False, action='store_true', help='Save structures and scores for all recycles, not just best one')
    parser.add_argument("-silent_output", default=None, help='Path to output silent file instead of pdb')

    args = parser.parse_args()

    return args

MAXLAT=256
MAXSEQ=2048

MODEL_PARAM ={
        "n_extra_block"   : 4,
        "n_main_block"    : 32,
        "n_ref_block"     : 4,
        "d_msa"           : 256,
        "d_pair"          : 192,
        "d_templ"         : 64,
        "n_head_msa"      : 8,
        "n_head_pair"     : 6,
        "n_head_templ"    : 4,
        "d_hidden"        : 32,
        "d_hidden_templ"  : 64,
        "p_drop"       : 0.0,
        "lj_lin"       : 0.7,
        'symmetrize_repeats': False,
        'repeat_length': float('nan'),
        'symmsub_k': float('nan'),
        'sym_method': float('nan'),
        'main_block': float('nan'),
        'copy_main_block_template': False
        }

SE3_param = {
        "num_layers"    : 1,
        "num_channels"  : 32,
        "num_degrees"   : 2,
        "l0_in_features": 64,
        "l0_out_features": 64,
        "l1_in_features": 3,
        "l1_out_features": 2,
        "num_edge_features": 64,
        "div": 4,
        "n_heads": 4
        }
SE3_ref_param = {
        "num_layers"    : 2,
        "num_channels"  : 32,
        "num_degrees"   : 2,
        "l0_in_features": 64,
        "l0_out_features": 64,
        "l1_in_features": 3,
        "l1_out_features": 2,
        "num_edge_features": 64,
        "div": 4,
        "n_heads": 4
        }
MODEL_PARAM['SE3_param'] = SE3_param
MODEL_PARAM['SE3_ref_param'] = SE3_ref_param

alphabets = {
        "prot" : np.array(list("ARNDCQEGHILKMFPSTWYV-X0000000000"), dtype='|S1').view(np.uint8),
        "dna"  : np.array(list("00000000000000000000-0ACGTD00000"), dtype='|S1').view(np.uint8),
        "rna"  : np.array(list("00000000000000000000-000000ACGUN"), dtype='|S1').view(np.uint8),
        "full" : np.array(list("ARNDCQEGHILKMFPSTWYV-Xacgtxbdhuy"), dtype='|S1').view(np.uint8)
        }

# compute expected value from binned lddt
def lddt_unbin(pred_lddt):
    # calculate lddt prediction loss
    nbin = pred_lddt.shape[1]
    bin_step = 1.0 / nbin
    lddt_bins = torch.linspace(bin_step, 1.0, nbin, dtype=pred_lddt.dtype, device=pred_lddt.device)

    pred_lddt = nn.Softmax(dim=1)(pred_lddt)
    return torch.sum(lddt_bins[None,:,None]*pred_lddt, dim=1)

def pae_unbin(pred_pae):
    # calculate pae loss
    nbin = pred_pae.shape[1]
    bin_step = 0.5
    pae_bins = torch.linspace(bin_step, bin_step*(nbin-1), nbin, dtype=pred_pae.dtype, device=pred_pae.device)

    pred_pae = nn.Softmax(dim=1)(pred_pae)
    return torch.sum(pae_bins[None,:,None,None]*pred_pae, dim=1)


class Molecule():
    """
    Molecule class, used to store sequence and structure of one or more biomolecules.
    Compatible with protein, DNA, RNA, and small molecules.
    Contains methods to compute auxiliary information.
    """
    def __init__(self, name, msa=None, ins=None, Ls=None, xyz=None, mask=None, 
            seq=None, symm='C1', mols=None
                ):
        """
        Molecule constructor. Minimal requirement is a sequence, as a list (seq) or tensor (msa) of numbers.
        msa  : N x L : one or more sequences as a tensor
        ins  : N x L : in sequences for an msa. Will be ignored if msa is not also provided.
        Ls   : NChains : list containing length of each molecule / chain
        xyz  : 1 x L x NTOTAL x 3 : tensor of cartesian coordinates for all atoms
        mask : 1 x L x NTOTAL : tensor of boolean values for all atoms
        seq : L : 1D tensor of numbers for residues IDs in the molecule
        symm: string encoding the symmetry group for this molecule
        """
#        print(f"processing molecule of name {name}")
#        LOGGER.debug(name, msa,  Ls, xyz, mask, seq, symm, mols)
#        LOGGER.debug(name, Ls, seq, symm, mols)

        self.name = name

        if (msa is None) and (seq is None):
            raise ValueError('Molecule object requires either msa or seq to determine sequence')

        if Ls is None:
            if xyz is not None:
                Ls = [xyz.shape[1]]
            elif msa is not None:
                Ls = [msa.shape[1]]
            else:
                Ls = [len(seq)]
        
        self.Ls = Ls
        self.L = sum(Ls)

        if msa is None:
            seq_full = torch.Tensor(seq).long()
            if len(Ls) == 1:
                msa = seq_full.unsqueeze(0)
            else:
                seq_s = tuple(( seq_full for _ in range(len(Ls)+1) ))
                msa = torch.stack(seq_s, dim=0)
                L0 = 0
                for i, L in enumerate(Ls):
                    L1 = L0 + L
                    msa[i+1, L0:L1] = 20
                    L0 = L1

            ins = torch.zeros_like(msa)

        self.msa = msa
        self.ins = ins
        self.a3m = {"msa" : msa, "ins" : ins}

        if seq is None:
            seq = msa[0]

        try: self.seq = torch.from_numpy(seq)
        except: self.seq = seq
        
        self.add_ch_types()
        self.symm = symm
        self.symmids,self.symmRs,self.symmmeta,self.symmoffset = symm_subunit_matrix(self.symm)
            
        if xyz is None:
            xyz = self.initialize_empty_xyz()
            mask = self.empty_mask()
        elif mask is None:
            mask = self.full_mask()

        self.xyz = xyz
        self.mask = mask
        
        if mols is None:
            mols = [None for _ in range(len(Ls))]

        self.mols = mols

        return None

    def get_params(self):
        return {
                'tag' : self.name,
                'Ls'  : self.Ls,
                'seq' : self.seq.tolist(),
                'ch_types' : self.ch_types,
                'symm': self.symm,
                }

    def add_extra_features(self, mode="protein_vs_other", binder_ch=0):
        self.add_idx()
        self.add_same_chain(mode=mode,binder_ch=binder_ch)
        self.add_bond_feats()
        self.add_dist_matrix()
        self.add_atom_frames()
        self.add_chirals()
        return None

    def add_ch_types(self):
        """Determine molecular type of each chain in this molecule object."""
        self.ch_types = []
        for i, l in enumerate(self.Ls):
            L0 = sum(self.Ls[:i])
            L1 = L0 + l
            subseq = self.seq[L0:L1]
            protein = util.is_protein(subseq)
            nucleic = util.is_nucleic(subseq)
            atom = util.is_atom(subseq)
            if protein.any():
                assert not nucleic.any(), \
                        f"Found nucleic residues in protein chain (idx {i} of molecule {self.name}, seq {subseq})"
                assert not atom.any(), \
                        f"Found atomized residues in protein chain (idx {i} of molecule {self.name})"
                self.ch_types.append("protein")

            elif nucleic.any():
                assert not protein.any(), \
                        f"Found protein residues in nucleic chain (idx {i} of molecule {self.name})"
                assert not atom.any(), \
                        f"Found atomized residues in nucleic chain (idx {i} of molecule {self.name})"
                self.ch_types.append("nucleic")

            elif atom.any():
                assert not protein.any(), \
                        f"Found protein residues in atomic chain (idx {i} of molecule {self.name})"
                assert not nucleic.any(), \
                        f"Found nucleic residues in atomic chain (idx {i} of molecule {self.name})"
                self.ch_types.append("atom")

            else:
                raise ValueError(f'Malformed sequence {subseq} \n for chain {i} of molecule {self.name}')

        return None

    def initialize_empty_xyz(self, SYMM_OFFSET_SCALE=1.0, random_noise=5.0):
        Pmask = util.is_protein(self.seq)
        NAmask = util.is_nucleic(self.seq)

        xyz_cloud = torch.zeros((1,self.L,NTOTAL,3))

        xyz_cloud[:,NAmask] = (
            INIT_NA_CRDS.reshape(1,1,NTOTAL,3).repeat(1,NAmask.sum(),1,1)
            + torch.rand(1,NAmask.sum(),1,3)*random_noise - random_noise/2
            + SYMM_OFFSET_SCALE*self.symmoffset*NAmask.sum()**(1/2)
        )

        xyz_cloud[:,Pmask] = (
            INIT_CRDS.reshape(1,1,NTOTAL,3).repeat(1,Pmask.sum(),1,1)
            + torch.rand(1,Pmask.sum(),1,3)*random_noise - random_noise/2
            + SYMM_OFFSET_SCALE*self.symmoffset*Pmask.sum()**(1/2)
        )

        return xyz_cloud

    def empty_mask(self):
        return torch.full( (1,self.L,NTOTAL), False )

    def full_mask(self):
        return torch.full( (1,self.L,NTOTAL), True )

    def empty_f1d(self):
        return torch.cat((
                torch.nn.functional.one_hot(
                    torch.full((1, self.L), 20).long(),
                    num_classes=NAATOKENS-1).float(), # all gaps (no mask token)
                torch.zeros((1, self.L, 1)).float()
            ), -1)

    def add_idx(self):
        self.idx = torch.arange(self.L)
        L0 = 0
        for l in self.Ls:
            L0 += l
            self.idx[L0:] += 100
        return None

    def add_same_chain(self, mode="protein_vs_other", binder_ch=0):
        if mode=="protein_vs_other" and not ( 
                "nucleic" in self.ch_types or "atom" in self.ch_types
                ):
            mode = "one_vs_rest"
        
        if mode=="protein_vs_other":
            Amask = is_protein(self.seq)
            Bmask = ~Amask
            Amask_2d = Amask[None,:] * Amask[:,None]
            Bmask_2d = Bmask[None,:] * Bmask[:,None]
            same_chain = torch.logical_or(Amask_2d, Bmask_2d)

        elif mode=="one_vs_rest":
            Amask = torch.zeros((self.L), dtype=torch.bool)
            L0 = sum(self.Ls[:binder_ch])
            L1 = L0 + self.Ls[binder_ch]
            Amask[L0:L1] = True
            Bmask = ~Amask
            Amask_2d = Amask[None,:] * Amask[:,None]
            Bmask_2d = Bmask[None,:] * Bmask[:,None]
            same_chain = torch.logical_or(Amask_2d, Bmask_2d)

        elif mode=="each_vs_all":
            same_chain = torch.zeros((self.L,self.L), dtype=torch.bool)
            L0 = 0
            for i, l in enumerate(self.Ls):
                imask = torch.zeros((self.L), dtype=torch.bool)
                L1 = L0 + l
                imask[L0:L1] = True
                imask_2d = imask[None,:] * imask[:,None]
                same_chain = torch.logical_or(same_chain,imask_2d)
                L0 = L1

        elif mode=="all_together":
            same_chain = torch.ones((self.L,self.L), dtype=torch.bool)

        else:
            raise ValueError(f"Invalid input for interface_mode: {mode}. Valid modes are 'protein_vs_other','one_vs_rest','each_vs_all', and 'all_together'")

        self.same_chain = same_chain
        return None

    def add_bond_feats(self):
        """Compute bond features for molecule.
           Note that atomized pdb residues are not handled here."""
        bond_feats = torch.zeros((self.L, self.L)).long()
        
        if 'atom' not in self.ch_types:
            self.bond_feats = get_protein_bond_feats_from_idx(self.L, self.idx)
            return None

        for i, l in enumerate(self.Ls):
            L0 = sum(self.Ls[:i])
            L1 = L0 + l

            if self.ch_types[i] == 'atom':
                assert (self.mols[i] is not None), f"no OBmol object found for atom chain {i} in {self.name}"
                feats_i = util.get_bond_feats(self.mols[i])

            else:
                feats_i = util.get_protein_bond_feats(l)

            bond_feats[L0:L1, L0:L1] = feats_i

        self.bond_feats = bond_feats
        return None

    def add_dist_matrix(self):
        atom_bonds = (self.bond_feats > 0)*(self.bond_feats<5)
        dist_matrix = scipy.sparse.csgraph.shortest_path(atom_bonds.long().numpy(), directed=False)
        # dist_matrix = torch.tensor(np.nan_to_num(dist_matrix, posinf=4.0)) # protein portion is inf and you don't want to mask it out
        self.dist_matrix = torch.from_numpy(dist_matrix).float()
        return None

    def add_atom_frames(self):
        if 'atom' not in self.ch_types:
            self.atom_frames = torch.zeros_like(self.msa[:,0])
            return None
        
        atom_frames_s = []
        for i, l in enumerate(self.Ls):
            L0 = sum(self.Ls[:i])
            L1 = L0 + l
            if self.ch_types[i] != 'atom':
                continue

            assert (self.mols[i] is not None), f"no OBmol object found for atom chain {i} in {self.name}"
            seq_i = self.seq[L0:L1]
            G = get_nxgraph(self.mols[i])
            atom_frames_i = get_atom_frames(seq_i, G)
            atom_frames_s.append(atom_frames_i)

        self.atom_frames = torch.cat(atom_frames_s, axis=0)
        return None

    def add_chirals(self):
        if 'atom' not in self.ch_types:
            self.chirals = torch.Tensor()
            return None

        chirals_s = []
        for i, l in enumerate(self.Ls):
            L0 = sum(self.Ls[:i])
            L1 = L0 + l
            if self.ch_types[i] != 'atom':
                continue

            assert (self.mols[i] is not None), f"no OBmol object found for atom chain {i} in {self.name}"
            chirals_i = get_chirals(self.mols[i], self.xyz[0,L0:L1,0,:])
            chirals_i[:,:-1] = chirals_i[:,:-1] + L0 # shift indices
            chirals_s.append(chirals_i)

        self.chirals = torch.cat(chirals_s, axis=0)
        return None

    def add_templates(self, template_chains=[], template_types=[], templ_conf=0.5):
        """Fills xyz_t for the Molecule based on specified chains or chain types.\
           Each chain becomes a separate template."""
        
        empty_xyz = self.initialize_empty_xyz()
        empty_mask = self.empty_mask()
        empty_f1d = self.empty_f1d()

        N = sum(1 for i in range(len(self.Ls)) if (self.ch_types[i] in template_types or i in template_chains))
        
        if N == 0:
            self.xyz_t = empty_xyz
            self.mask_t = empty_mask
            self.f1d_t = empty_f1d
            return None
        
        self.xyz_t = torch.cat(tuple(empty_xyz for _ in range(N)), 0)
        self.mask_t = torch.cat(tuple(empty_mask for _ in range(N)), 0)
        self.f1d_t = torch.cat(tuple(empty_f1d for _ in range(N)), 0)

        n = 0
        for i, l in enumerate(self.Ls):
            if (self.ch_types[i] not in template_types) and (i not in template_chains):
               continue
            
            L0 = sum(self.Ls[:i])
            L1 = L0 + l
        
            self.xyz_t[n,L0:L1,:,:] = self.xyz[:,L0:L1,:,:]
            self.mask_t[n,L0:L1,:] = torch.ones((1,l,NTOTAL),dtype=torch.bool)
            self.f1d_t[n,L0:L1] = torch.cat((
                torch.nn.functional.one_hot(self.msa[0,L0:L1], num_classes=NAATOKENS-1).float(),
                templ_conf*torch.ones((l,1)).float()
            ), -1)
            n += 1

        return None

    def add_init_state(self, init_chains=[], init_types=[]):
        """Fills xyz_prev for the Molecule based on specified chains or chain types.
            Each chain becomes a separate template."""

        self.xyz_prev = self.initialize_empty_xyz()[0]
        self.mask_prev = self.empty_mask()[0]

        if len(init_chains) + len(init_types) == 0:
            return None

        for i, l in enumerate(self.Ls):
            if (self.ch_types[i] not in init_types) and (i not in init_chains):
                continue

            L0 = sum(self.Ls[:i])
            L1 = L0 + l

            self.xyz_prev[L0:L1,:,:] = self.xyz[0,L0:L1,:,:]
            self.mask_prev[L0:L1,:]  = self.mask[0,L0:L1,:]

        return None

    def get_network_inputs(self, args):
        seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(
            self.msa, self.ins, p_mask=0.0, 
            params={'MAXLAT': MAXLAT, 'MAXSEQ': MAXSEQ, 'MAXCYCLE': args.n_cycle}, tocpu=True
        )
        self.device = args.device
        device = args.device

        self.xyz_t, symmsub = find_symm_subs(self.xyz_t[:,:],self.symmRs,self.symmmeta)
        Osub = symmsub.shape[0]
        self.mask_t = self.mask_t.repeat(1,Osub,1)
        self.f1d_t = self.f1d_t.repeat(1,Osub,1)
        self.atom_frames = self.atom_frames.repeat(Osub,1,1)

        seq = seq[None].to(device, non_blocking=True)
        msa = msa_seed_orig[None].to(device, non_blocking=True)
        msa_masked = msa_seed[None].to(device, non_blocking=True)
        msa_full = msa_extra[None].to(device, non_blocking=True)
        idx = self.idx[None].to(device, non_blocking=True)
        xyz_t = self.xyz_t[None].to(device, non_blocking=True)
        mask_t = self.mask_t[None].to(device, non_blocking=True)
        t1d = self.f1d_t[None].to(device, non_blocking=True)
        xyz_prev = self.xyz_prev[None].to(device, non_blocking=True)
        mask_prev = self.mask_prev[None].to(device, non_blocking=True)
        same_chain = self.same_chain[None].to(device, non_blocking=True).long()
        atom_frames = self.atom_frames[None].to(device, non_blocking=True)
        bond_feats = self.bond_feats[None].to(device, non_blocking=True).long()
        dist_matrix = self.dist_matrix[None].to(device, non_blocking=True)
        chirals = self.chirals[None].to(device, non_blocking=True)

        symmids = self.symmids.to(device)
        symmsub = symmsub.to(device)
        symmRs = self.symmRs.to(device)
        symmmeta = copy.deepcopy(self.symmmeta)
        subsymms, _ = symmmeta
        for i in range(len(subsymms)):
            subsymms[i] = subsymms[i].to(device)

        B, _, N, L = msa.shape

        seq_unmasked = msa[:, 0, 0, :] # (B, L)
        mask_t_2d = util.get_prot_sm_mask(mask_t, seq_unmasked[0]) # (B, T, L)
        mask_t_2d = mask_t_2d[:,:,None]*mask_t_2d[:,:,:,None] # (B, T, L, L)
        mask_t_2d = mask_t_2d.float() * same_chain.float()[:,None] # (ignore inter-chain region)
        mask_recycle = util.get_prot_sm_mask(mask_prev, seq_unmasked[0])
        mask_recycle = mask_recycle[:,:,None]*mask_recycle[:,None,:] # (B, L, L)
        mask_recycle = same_chain.float()*mask_recycle.float()
        mask_recycle = mask_recycle[0]

        xyz_t_frames = util.xyz_t_to_frame_xyz(xyz_t, seq_unmasked, atom_frames)
        t2d = xyz_to_t2d(xyz_t_frames, mask_t_2d)

        seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,L)

        self.xyz_converter = XYZConverter().to(device)
        alpha, _, alpha_mask, _ = self.xyz_converter.get_torsions(
            xyz_t.reshape(-1,L,NTOTAL,3),
            seq_tmp
        )
        alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
        alpha[torch.isnan(alpha)] = 0.0
        alpha = alpha.reshape(1,-1,Osub*L,NTOTALDOFS,2)
        alpha_mask = alpha_mask.reshape(1,-1,Osub*L,NTOTALDOFS,1)
        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, Osub*L, 3*NTOTALDOFS).to(device)
        alpha_prev = torch.zeros((1,L,NTOTALDOFS,2), device=device)

        return msa_masked, msa_full, seq, msa, xyz_prev, alpha_prev, idx, \
                bond_feats, dist_matrix, chirals, atom_frames, \
                t1d, t2d, xyz_t, alpha_t, mask_t_2d, same_chain,\
                mask_recycle, symmids, symmsub, symmRs, symmmeta

    def merge(self, m2, name=None, inplace=False):
        if inplace:
            self = merge_molecules([self, m2], name=name)
            return None
        else:
            return merge_molecules([self, m2], name=name)
            
           
def merge_molecules(molecules, name=None):
    
    if len(molecules) == 1:
        return molecules[0]

    if len(molecules) == 0:
        raise ValueError("Error: attempting to merge list of 0 molecules")
    
    # Values that must match: symm
    assert len(set(m.symm for m in molecules)) == 1, "error merging molecules: symmetry must match"
    
    symm = molecules[0].symm

    # Values to merge: name, Ls, ch_types, xyz, mask, msa, ins, seq, mols
    if name is None:
        name = '_'.join(m.name for m in molecules)

    # merge per-chain parameters
    Ls = [l for m in molecules for l in m.Ls]
    mols = [mol for m in molecules for mol in m.mols]

    # merge coordinate information
    xyz = torch.cat(tuple(m.xyz for m in molecules), dim=1)
    mask = torch.cat(tuple(m.mask for m in molecules), dim=1)

    # merge sequence information
    seq = torch.cat(tuple(m.seq for m in molecules), dim=0)
    
    a3m = molecules[0].a3m
    L0 = molecules[0].L
    for i in range(1,len(molecules)):
        L1 = molecules[i].L
        a3m = merge_a3m_hetero(a3m, molecules[i].a3m, [L0,L1])
        L0 += L1
    
    msa = a3m['msa']
    ins = a3m['ins']

    # make new molecule
    molecule = Molecule(
        name, msa=msa, ins=ins, Ls=Ls, xyz=xyz, mask=mask,
        seq=seq, symm=symm, mols=mols
    )

    return molecule

def molecule_from_a3m(a3mfile, args, name=None):
    try:
        fields = a3mfile.split(':')
        if len(fields) == 2:
            nmer = int(fields[0])
            a3mfile = fields[1]
        else:
            assert len(fields) == 1
            nmer = 1
    except:
        raise Exception(f"unable to parse a3m input {a3mfile}")

    msa, ins, Ls = parsers.parse_multichain_fasta(a3mfile, mode=args.sequence_mode)
    for _ in range(1,nmer):
        Ls = Ls + Ls
    msa, ins = torch.from_numpy(msa).long(), torch.from_numpy(ins).long()
    msa, ins = merge_a3m_homo(msa, ins, nmer)
        
    if name is None:
        name = '.'.join(os.path.basename(a3mfile).split('.')[:-1])
        if nmer > 1:
            name = name + f'_{nmer}mer'

    return Molecule(name, msa=msa, ins=ins, Ls=Ls)

def molecule_from_parse_mol(data, args, name=None):
    obmol, msa, ins, xyz, mask = data
    xyz_full = torch.zeros((1,len(msa),NTOTAL,3))
    xyz_full[0,:,1,:] = xyz[0]
    mask_full = torch.full((1,len(msa),NTOTAL),False)
    mask_full[0,:,0] = mask[0]
    return Molecule(name, msa=msa.unsqueeze(0), ins=ins.unsqueeze(0), xyz=xyz_full, mask=mask_full, mols=[obmol])

def molecule_from_mol2(mol2file, args, name=None):
    data = parse_mol(mol2file, filetype="mol2")
    return molecule_from_parse_mol(data, args, name=name)

def molecule_from_seq(sequence, args, name=None):
    """Generates molecule object for a given protein, DNA, or RNA sequence."""
    # autodetect if sequence fits as DNA, RNA, or protein
    mode = args.sequence_mode
    if mode=='auto':
        mode = util.detect_sequence_type(sequence)
        if mode[:2] == 'ds' and not args.auto_dsdna:
            mode = mode[2:]
    
    print(f"creating molecule from sequence {sequence} in mode {mode}")

    assert mode in ['dsdna','dsrna','dna','rna','protein','full','smiles'], f"{mode} is not a valid mode for molecule_from_seq().\nValid modes are:\n\t'dsdna','dsrna','dna','rna','protein','full'"
    
    if name is None:
        name = f"{sequence}"
    
    if mode == 'smiles':
        data = parsers.parse_mol(sequence, string=True, filetype='smi', find_automorphs=False)
        return molecule_from_parse_mol(data, args, name=name)

    for x in ['dna', 'rna', 'prot', 'full']:
        if x in mode:
            chtype = x
            alphabet = alphabets[x]
            break
    
    bps = {
            'dna' : {'A':'T','C':'G','G':'C','T':'A', 'D':'D'},
            'rna' : {'A':'U','C':'G','G':'C','U':'A', 'N':'N'}
    }

    if 'ds' in mode:
        Ls = [len(sequence), len(sequence)]
        sequence = sequence + ''.join([bps[chtype][b] for b in sequence][::-1])
    else:
        Ls = [len(sequence)]

    seq = np.array(list(sequence), dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        seq[seq == alphabet[i]] = i
    
    # mask anything that doesn't fit
    seq[seq > len(alphabet)] = 20

    return Molecule(name, seq=seq, Ls=Ls, symm=args.symm)

def molecule_from_pdb(pdbfile, args, name=None):
    """Wrapper for molecule_from_pdbstring. Opens pdb file and reads its content."""
    with open(pdbfile, 'r') as f:
        pdbstring = f.read()
    if name is None:
        name = os.path.basename(pdbfile)[:-4]
    
    return molecule_from_pdbstring(pdbstring, args, name=name)

def molecule_from_pdbstring(pdbstring, args, name):
    """Generates molecule object from a pdb string. Uses both ATOM and HETATM lines."""
    print(f"creating molecule from pdb string with name {name}")

    # Get protein and nucleic acid data from ATOM lines:
    seq, Ls, xyz, mask = parsers.read_pdbstring_for_molecule(pdbstring)

    if args.dump_extra:
        util.writepdb(f"{args.out_dir}/{name}_pdbatom.pdb", xyz[0], seq, chain_Ls=Ls)

    stream = [l for l in pdbstring.split('\n') if l[:6] == "HETATM" or l[:6]=="CONECT"]
    if len(stream) > 0:
        data = parsers.parse_mol("".join(stream), filetype="pdb", string=True)
        if len(seq) > 0:
            small_m = molecule_from_parse_mol(data, args, name='tmp_small')
            macro_m = Molecule('tmp_macro', seq=seq, Ls=Ls, xyz=xyz, mask=mask)
            m = merge_molecules([macro_m, small_m], name=name)
        else:
            m = molecule_from_parse_mol(data, args, name=name)
    else:
        m = Molecule(name, seq=seq, Ls=Ls, xyz=xyz, mask=mask)
    
    return m
    
def molecule_from_fasta(fastafile, args, name=None):
    """create a single molecule from a fasta file"""

    if args.fasta_mode == 'msa':
        # treat fasta file like an a3m file
        return molecule_from_a3m(fastafile, args, name=name)

    if args.fasta_mode == 'single':
        # treat entire sequence as a single chain
        with open(fastafile,'r') as f:
            seq = ''.join([l.strip() for l in f if l[0] != '>'])
        return molecule_from_seq(seq, args, name=name)

    if args.fasta_mode == 'multi':
        # treat each line as a separate chain
        with open(fastafile,'r') as f:
            m_s = []
            for l in f:
                if l[0] == '>': 
                    continue
                if len(l.strip()) == 0:
                    continue
                m_s.append(molecule_from_seq(l.strip(),args))
        
        if name is None:
            name = '.'.join(os.path.basename(fastafile).split('.')[:-1])
        return merge_molecules(m_s, name=name)

def molecule_from_a3m_and_pdb(files, args, name=name):
    a3mfiles = [x for x in files.split(',') if x.split('.')[-1] == 'a3m']
    pdbfiles = [x for x in files.split(',') if x.split('.')[-1] == 'pdb']
    assert len(a3mfiles) == len(pdbfiles) == 1, \
            f"Cannot use multiple files of same type for same molecule: {files}"
    a3mfile, pdbfile = a3mfiles[0], pdbfiles[0]
    msa, ins, _ = parsers.parse_a3m(a3mfile)
    msa, ins = torch.from_numpy(msa).long(), torch.from_numpy(ins).long()

    seq, Ls, xyz, mask = parsers.read_pdb_for_molecule(pdbfile)

    if len(seq) != len(msa[0]):
#        LOGGER.debug(seq, msa[0], len(seq), msa.shape)
        msa, ins, seq, Ls, xyz, mask = align_msa_and_pdb(msa, ins, seq, Ls, xyz, mask)
    else:
        assert (seq == msa[0]).all(), \
                f"Mismatch between sequence found in a3m and pdb: {files}, \npdb: {seq}\na3m: {msa[0]}"

    if name is None:
        name = '.'.join(os.path.basename(a3mfile).split('.')[:-1])
        name += '_' + '.'.join(os.path.basename(pdbfile).split('.')[:-1])
    
    stream = [l for l in open(pdbfile) if l[:6] == "HETATM" or l[:6]=="CONECT"]
    if len(stream)>0:
        data = parsers.parse_mol("".join(stream), filetype="pdb", string=True)
        small_m = molecule_from_parse_mol(data, args, name='tmp_small')
        macro_m = Molecule('tmp_macro', seq=seq, Ls=Ls, xyz=xyz, mask=mask, msa=msa, ins=ins)
        m = merge_molecules([macro_m, small_m], name=name)
    else:
        m = Molecule(name, seq=seq, Ls=Ls, xyz=xyz, mask=mask, msa=msa, ins=ins)
    return m

def align_msa_and_pdb(msa, ins, seq, Ls, xyz, mask):
    raise NotImplementedError("sorry, alignment of pdb and sequence files with different lengths not yet implemented")
    return None

def molecule_from_fasta_and_pdb(files, args, name=name):
    raise NotImplementedError("sorry, input with fasta and pdb file not yet implemented")
    return None

def molecule_from_seq_and_pdb(files, args, name=name):
    raise NotImplementedError("sorry, input with sequence and pdb file not yet implemented")
    return None

class Predictor():
    def __init__(self, args):
        # define model name
        self.device = args.device
        self.active_fn = nn.Softmax(dim=1)

        FFindexDB = namedtuple("FFindexDB", "index, data")
        self.ffdb = FFindexDB(read_index(args.db+'_pdb.ffindex'),
                              read_data(args.db+'_pdb.ffdata'))

        # define model & load model
        MODEL_PARAM['use_extra_l1'] = args.use_extra_l1
        MODEL_PARAM['use_atom_frames'] = args.use_atom_frames
        self.model = RoseTTAFoldModule(
            **MODEL_PARAM,
            aamask = util.allatom_mask.to(self.device),
            atom_type_index = util.atom_type_index.to(self.device),
            ljlk_parameters = util.ljlk_parameters.to(self.device),
            lj_correction_parameters = util.lj_correction_parameters.to(self.device),
            num_bonds = util.num_bonds.to(self.device),
            cb_len = util.cb_length_t.to(self.device),
            cb_ang = util.cb_angle_t.to(self.device),
            cb_tor = util.cb_torsion_t.to(self.device),
            fit=args.symm_fit,
            tscale=args.symm_scale
        ).to(self.device)
        checkpoint = torch.load(args.checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        self.xyz_converter = XYZConverter().to(self.device)

        # loss & final activation function
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.active_fn = nn.Softmax(dim=1)

        # move some global data to cuda device
        self.ti_dev = torsion_indices.to(self.device)
        self.ti_flip = torsion_can_flip.to(self.device)
        self.ang_ref = reference_angles.to(self.device)
        self.fi_dev = frame_indices.to(self.device)
        self.l2a = long2alt.to(self.device)
        self.aamask = allatom_mask.to(self.device)
        self.atom_type_index = atom_type_index.to(self.device)
        self.ljlk_parameters = ljlk_parameters.to(self.device)
        self.lj_correction_parameters = lj_correction_parameters.to(self.device)
        self.hbtypes = hbtypes.to(self.device)
        self.hbbaseatoms = hbbaseatoms.to(self.device)
        self.hbpolys = hbpolys.to(self.device)
        self.num_bonds = num_bonds.to(self.device),
        self.cb_len = cb_length_t.to(self.device),
        self.cb_ang = cb_angle_t.to(self.device),
        self.cb_tor = cb_torsion_t.to(self.device),


    def predict(self, molecule, args):

        msa_masked, msa_full, seq, msa, xyz_prev, alpha_prev, idx,\
            bond_feats, dist_matrix, chirals, atom_frames,\
            t1d, t2d, xyz_t, alpha_t, mask_t_2d, same_chain,\
            mask_recycle, symmids, symmsub, symmRs, symmmeta = molecule.get_network_inputs(args)

        start = time.time()
        if 'cuda' in self.device:
            torch.cuda.reset_peak_memory_stats()
        self.model.eval()
        all_pred = []
        all_results = []
        select_by = args.select_by
        xyz_prev_orig = xyz_prev.clone()

        with torch.no_grad():
            msa_prev = None
            pair_prev = None
            state_prev = None
            
            best_result = {
                "plddt" : -1,
                "plddt_full" : None,
                "xyz"   : None,
                "logit" : None,
                "aa"    : None,
                "pae"   : 999,
                "pae_full" : None,
                "i_pae" : 999,
                "p_bind": -1, 
            }

            print ("%7s %2s %7s %7s %7s %7s %7s"%(
                "",
                "",
                "PAE",
                "i_PAE",
                "p_bind",
                "plddt",
                "best_"+select_by
            ) )

            for i_cycle in range(args.n_cycle):
# input debugging
#                LOGGER.debug(
#                    msa_masked[:,i_cycle].shape,
#                    msa_full[:,i_cycle].shape,
#                    seq[:,i_cycle].shape,
#                    msa[:,i_cycle,0].shape,
#                    xyz_prev.shape,
#                    alpha_prev.shape,
#                    idx.shape,
#                    bond_feats.shape,
#                    dist_matrix.shape,
#                    chirals.shape,
#                    atom_frames.shape,
#                    t1d.shape,
#                    t2d.shape,
#                    xyz_t[...,1,:].shape,
#                    alpha_t.shape,
#                    mask_t_2d.shape,
#                    same_chain.shape,
#                )
#                if i_cycle > 0:
#                    LOGGER.debug(
#                        msa_prev.shape,
#                        pair_prev.shape,
#                        state_prev.shape
#                    )

                logit_s, logit_aa_s, logit_pae, logit_pde, \
                p_bind, pred_crds, alpha, \
                pred_allatom, pred_lddt_binned, \
                msa_prev, pair_prev, state_prev = self.model(
                    msa_masked[:,i_cycle],
                    msa_full[:,i_cycle],
                    seq[:,i_cycle],
                    msa[:,i_cycle,0],
                    xyz_prev,
                    alpha_prev,
                    idx,
                    bond_feats=bond_feats,
                    dist_matrix=dist_matrix,
                    chirals=chirals,
                    atom_frames=atom_frames,
                    t1d=t1d,
                    t2d=t2d,
                    xyz_t=xyz_t[...,1,:],
                    alpha_t=alpha_t,
                    mask_t=mask_t_2d,
                    same_chain=same_chain,
                    msa_prev=msa_prev,
                    pair_prev=pair_prev,
                    state_prev=state_prev,
                    mask_recycle=mask_recycle,
                    symmids=symmids,
                    symmsub=symmsub,
                    symmRs=symmRs,
                    symmmeta=symmmeta
                )
                
                N, L = msa[:,i_cycle,0].shape
                logit_aa = logit_aa_s.reshape(1,-1,N,L)[:,:,0].permute(0,2,1)
                xyz_prev = pred_allatom[-1].unsqueeze(0)
                mask_recycle = None
                
                all_pred.append(pred_crds)

                pred_lddt = lddt_unbin(pred_lddt_binned)
                pae = pae_unbin(logit_pae)

                result = {
                    "plddt" : pred_lddt.mean(),
                    "plddt_full" : pred_lddt.clone(),
                    "plddt_binned" : pred_lddt_binned.clone(),
                    "xyz"   : xyz_prev.clone(), 
                    "logit" : logit_s,
                    "aa"    : logit_aa,
                    "pae"   : pae.mean().cpu().numpy(),
                    "pae_full" : pae.clone(),
                    "i_pae" : pae[~same_chain.bool()].mean().cpu().numpy(),
                    "p_bind": p_bind,
                }
                all_results.append(result)

                select_high = ["p_bind","plddt"]
                select_low = ["pae", "i_pae"]
                if select_by in select_high:
                    better = ( result[select_by] > best_result[select_by] )
                elif select_by in select_low:
                    better = ( result[select_by] < best_result[select_by] )
                else:
                    raise ValueError(f"select_by criteria {select_by} is not valid. Options are {select_low + select_high}")

                if better:
                    best_result = result
                
                print ("%7s %2d %7.3f %7.3f %7.3f %7.3f %7.3f"%(
                    "RECYCLE",
                    i_cycle,
                    result["pae"],
                    result["i_pae"],
                    result["p_bind"],
                    result["plddt"],
                    best_result[select_by]                
                ) )

            prob_s = list()
            for logit in logit_s:
                prob = self.active_fn(logit.float()) # distogram
                prob = prob.reshape(-1, L, L) #.permute(1,2,0).cpu().numpy()
                prob = prob / (torch.sum(prob, dim=0)[None]+1e-8)
                prob_s.append(prob)

        end = time.time()
        if 'cuda' in self.device:
            max_mem = torch.cuda.max_memory_allocated()/1e9
            print ("max mem", max_mem)
        print ("runtime", end-start)
        
        best_result['prob_s'] = prob_s
        best_result['xyz_last'] = xyz_prev
        best_result['xyz_init'] = xyz_prev_orig
        best_result['all_pred'] = all_pred
        best_result['xyz_t'] = xyz_t

        return best_result, all_results 

    def output_pdbs(self, molecule, result, args, suffix="", num_interp=5):
        """Helper function to write out the desired pdbs from a prediction"""

        # make full complex
        best_xyz = result['xyz']
        best_lddt = result['plddt_full']
        O = molecule.symmids.shape[0]
        Lasu = molecule.L
        seq = molecule.seq
        bond_feats = molecule.bond_feats
        name = molecule.name
        
        if args.output_asu:
            best_xyzfull = best_xyz
            seq_full = seq.unsqueeze(0)
            best_lddtfull = best_lddt
            bond_featsfull = bond_feats
            Lsfull = molecule.Ls
        else:
            best_xyzfull = torch.zeros( (1,O*Lasu,NTOTAL,3),device=self.device )
            best_xyzfull[:,:Lasu] = best_xyz[:,:Lasu]
            seq_full = torch.zeros( (1,O*Lasu),dtype=torch.uint8, device=self.device )
            seq_full[:,:Lasu] = seq[:Lasu]
            best_lddtfull = torch.zeros( (1,O*Lasu),device=self.device )
            best_lddtfull[:,:Lasu] = best_lddt[:,:Lasu]
            bond_featsfull = torch.zeros( (1,O*Lasu,O*Lasu),device=self.device )
            bond_featsfull[:,:Lasu,:Lasu] = bond_feats[:Lasu,:Lasu]
            Lsfull = molecule.Ls*O

            for i in range(1,O):
                best_xyzfull[:,(i*Lasu):((i+1)*Lasu)] = torch.einsum('ij,braj->brai', symmRs[i], best_xyz[:,:Lasu])
                bond_featsfull[:,(i*Lasu):((i+1)*Lasu),(i*Lasu):((i+1)*Lasu)] = bond_feats[:Lasu,:Lasu]
                seq_full[:,(i*Lasu):((i+1)*Lasu)] = seq[:Lasu]
        
        # save files
        if args.silent_output is not None:
            # silent files are a special case for when you don't care about outputting anything else except the structures
            pdb_holder = io.StringIO()
            util.writepdb_file(pdb_holder, best_xyzfull[0], seq_full, bfacts=best_lddtfull[0].float(), bond_feats=None, chain_Ls=Lsfull)
            pdb_string = pdb_holder.getvalue()
            pdb_holder.close()
            
            pose = Pose()
            core.import_pose.pose_from_pdbstring(pose, pdb_string)
            struct = sfd_out.create_SilentStructOP()
            struct.fill_struct(pose, name)
            sfd_out.add_structure(struct)
            sfd_out.write_silent_struct(struct, args.silent_output)
            return None


        out_dir = args.out_dir
        util.writepdb(f"{out_dir}/{name}_pred.pdb", best_xyzfull[0], seq_full, 
                bfacts=best_lddtfull[0].float(), bond_feats=None, chain_Ls=Lsfull)
                #bfacts=best_lddtfull[0].float(), bond_feats=bond_featsfull, chain_Ls=Lsfull)

        if args.dump_extra:
            if 'recycle' not in name.split('_')[-1]:
                util.writepdb(f"{out_dir}/{name}_init.pdb", result['xyz_init'][0], seq,chain_Ls=molecule.Ls)
                util.writepdb(f"{out_dir}/{name}_last.pdb", result['xyz_last'][0], seq, chain_Ls=molecule.Ls)
                for i in range(result['xyz_t'].shape[1]):
                    util.writepdb(f"{out_dir}/{name}_tmpl{i}.pdb", result['xyz_t'][0,i], seq, chain_Ls=molecule.Ls, file_mode="a")

        if args.dump_aux:
            prob_s = [prob.permute(1,2,0).detach().cpu().numpy().astype(np.float16) for prob in result["prob_s"]]
            with open(f"{out_dir}/{name}_aux.pkl", 'wb') as outf:
                pickle.dump(dict(
                    dist = prob_s[0].astype(np.float16), \
                    omega = prob_s[1].astype(np.float16),\
                    theta = prob_s[2].astype(np.float16),\
                    phi = prob_s[3].astype(np.float16)
                ), outf)

        if args.dump_traj:
            all_pred = torch.cat([result["xyz_init"][0:1,None,:,:3]]+result["all_pred"], dim=0)
            is_prot = ~util.is_atom(seq)
            T = all_pred.shape[0]
            t = np.arange(T)
            L = molecule.L
            n_frames = num_interp*(T-1)+1
            Y = np.zeros((n_frames,L,3,3))
            for i_res in range(L):
                for i_atom in range(3):
                    for i_coord in range(3):
                        interp = Akima1DInterpolator(t,all_pred[:,0,i_res,i_atom,i_coord].detach().cpu().numpy())
                        Y[:,i_res,i_atom,i_coord] = interp(np.arange(n_frames)/num_interp)
            Y = torch.from_numpy(Y).float()

            # 1st frame is final pred so pymol renders bonds correctly
            util.writepdb(f"{out_dir}/{name}_traj.pdb", Y[-1], seq,
                modelnum=0, bond_feats=bond_feats.unsqueeze(0), file_mode="w")
            for i in range(Y.shape[0]):
                util.writepdb(f"{out_dir}/{name}_traj.pdb", Y[i], seq,
                    modelnum=i+1, bond_feats=bond_feats.unsqueeze(0), file_mode="a")

        return None

def get_losses(M, R, args):
    """Compute desired loss terms based on input and output structures
M = Molecule object used for prediction input
R = dictionary with results from model prediction
"""
    keys = args.scores
    xyz = M.xyz.to(M.device)
    mask = M.mask.to(M.device)
    idx = M.idx.unsqueeze(0).to(M.device)
    
    if not M.mask.any():
        for x in keys:
            if x in ['lddt','i_lddt','ca_rms','rms','l_rms']:
#                print(f"No template model found. Skipping calculation of {x}")
                keys.remove(x)

    if len(M.Ls) <= 1:
        for x in keys:
            if x in ['l_rms']:
#                print(f"No interface found. Skipping calculation of {x}")
                keys.remove(x)

    if 'lddt' in keys:
#        print("Calculating lddt")
        R['lddt'] = calc_allatom_lddt(R['xyz'], xyz[0], idx, mask)

    if 'i_lddt' in keys:
#        print("Calculating interface lddt")
        mask_2d = util.get_prot_sm_mask(M.mask, M.seq[None]).unsqueeze(0)
        mask_2d = mask_2d[:,:,None]*mask_2d[:,:,:,None]
        mask_2d = mask_2d.squeeze(0).to(M.device)
        same_chain = M.same_chain.unsqueeze(0).to(M.device)
        _, R['i_lddt'] = calc_allatom_lddt_loss(
                R['xyz'], xyz[0], R['plddt_binned'], idx, mask, 
                mask_2d, same_chain, interface=True
        )

    if 'ca_rms' in keys:
#        print("Calculating ca rmsd")
        R['ca_rms'], _, _, _ = calc_allatom_rmsd(R['xyz'].unsqueeze(0), xyz, mask)

    if 'rms' in keys:
#        print("Calculating allatom rmsd")
        R['rms'], _, _, _ = calc_allatom_rmsd(R['xyz'].unsqueeze(0), xyz, mask, allatom=True)

    if 'l_rms' in keys:
#        print("Calculating l_rms")
        # align by interface B-side, rms on interface A-side
        if args.interface_mode == 'protein_vs_other':
            A_resmask = is_protein(M.seq)
        elif args.interface_mode == 'one_vs_rest':
            ch = args.binder_chain
            L0 = sum(M.Ls[:ch])
            L1 = L0 + M.Ls[ch]
            A_resmask = torch.full((M.L),False)
            A_resmask[L0:L1] = True

        A_mask = M.mask.clone()
        B_mask = M.mask.clone()
        A_mask[:,~A_resmask,:] = False
        B_mask[:,A_resmask,:] = False
        R['l_rms'], _, _, _ = calc_allatom_rmsd(
                R['xyz'].unsqueeze(0), xyz, B_mask, 
                score_mask=A_mask, allatom=True
        )

    return R

def save_scores(molecule, result, args):
    out_dir = args.out_dir
    filename = os.path.basename(args.scorefile)
    keys = args.scores
    if 'tag' not in keys:
        keys.append('tag')
    
    in_values = molecule.get_params()
    all_values = {**result, **in_values}
    values = {key : all_values[key] for key in all_values if key in keys}

    valid_terms = sorted(all_values.keys())
    for key in keys:
        assert key in valid_terms, f"{key} is not available for reporting.\nValid terms are: {valid_terms}"

    outcsv = f'{out_dir}/{filename}'

    keys = sorted(keys)
    
    if not os.path.exists(outcsv):
        with open(outcsv,'w') as f:
            f.write(','.join(keys)+'\n')

    with open(outcsv,'a') as f:
        f.write(','.join(["%.3f"%values[key] if key != 'tag' else values[key] for key in keys])+'\n')

    return None


molecule_makers = OrderedDict({
        "seq"  : molecule_from_seq,
        "a3m"  : molecule_from_a3m,
        "afa"  : molecule_from_a3m,
        "pdb"  : molecule_from_pdb,
        "pdbstring": molecule_from_pdbstring,
        "fa"   : molecule_from_fasta,
        "fasta": molecule_from_fasta,
        "mol2" : molecule_from_mol2,
        "a3m,pdb" : molecule_from_a3m_and_pdb,
        "fasta,pdb" : molecule_from_fasta_and_pdb,
        "fa,pdb" : molecule_from_fasta_and_pdb,
        "seq,pdb" : molecule_from_seq_and_pdb,
        })

molecule_cache = OrderedDict()
MAX_CACHE = 10

def update_cache(filename, molecule):
    if filename in molecule_cache.keys():
        # remove and reinsert to move to end of order
        del molecule_cache[filename]
        molecule_cache[filename] = molecule
        return None

    if len(molecule_cache) > MAX_CACHE:
        molecule_cache.popitem(last=False)
    
    molecule_cache[filename] = molecule
    return None

def make_molecule(file_dict, args):
    m_s = []
    try:
        name = file_dict['name']
        name = ''.join(x if x not in "\/:*?<>|= ," else "_" for x in name)
        name = name.replace("'","").replace('#','')
        del file_dict['name']
    except:
        name = None
        del file_dict['name']

    for ftype in file_dict:
        try:
            maker = molecule_makers[ftype]
        except:
            raise IndexError(f"no molecule maker function found for file type {ftype}")
        for i in range(len(file_dict[ftype])):
            infile = file_dict[ftype][i]
            try:
                m = molecule_cache[infile]
            except:
                m = maker(infile, args, name=name)
            update_cache(infile, m)
            m_s.append(m)

    m = merge_molecules(m_s, name=name)

    m.add_extra_features(mode=args.interface_mode)
    m.add_templates(template_chains=args.tmpl_chains, template_types=args.tmpl_types, templ_conf=args.tmpl_conf)
    m.add_init_state(init_chains=args.init_chains, init_types=args.init_types)

    return m

def process_inputs(args):
    input_dicts = []
    
    ftypes = sorted(molecule_makers.keys())

    if args.silent_path is not None:
        sfd_in = SilentFileData(SilentFileOptions())
        sfd_in.read_file(args.silent_path)
        tags = list(sfd_in.tags())
        for tag in tags:
            input_dict = OrderedDict()
            pose = Pose()
            sfd_in.get_structure(tag).fill_pose(pose)
            converter = PoseToStructFileRepConverter()
            converter.init_from_pose(pose)
            pdbstring = create_pdb_contents_from_sfr(converter.sfr())

            for ftype in ftypes:
                input_dict[ftype] = []
            input_dict['pdbstring'].append(pdbstring)
            input_dict['name'] = tag
            input_dicts.append(input_dict)
        return input_dicts
        
    if args.list is not None:
        with open(args.list,'r') as f_list:
            lines = [line.strip() for line in f_list]
        NINPUTS = len(lines)
        for i in range(NINPUTS):
            input_dict = OrderedDict()
            fields = lines[i].split()
#            LOGGER.debug(fields)
            for ftype in ftypes:
                input_dict[ftype] = []
            for field in fields:
                if len(field.split('.')) == 1:
                    ftype = 'seq'
                else:
                    ftype = ','.join(sorted([x.split('.')[-1] for x in field.split(',')]))
                input_dict[ftype].append(field)

            input_dicts.append(input_dict)

    else:
        NINPUTS = 1
        input_dicts = [OrderedDict()]
        for ftype in ftypes:
            input_dicts[0][ftype] = []

    if args.namelist is not None:
        with open(args.namelist,'r') as f_list:
            names = [line.strip() for line in f_list]
        assert len(names) == NINPUTS, "if using -namelist, must provide one name per prediction"
    elif args.name is not None and NINPUTS == 1:
        names = [args.name]
    else:
        names = [None for _ in range(NINPUTS)]

    for i in range(NINPUTS):
        input_dicts[i]['name'] = names[i]
        
    if args.input_files is not None:
        for i in range(NINPUTS):
            fields = args.input_files
            for field in fields:
                if len(field.split('.')) == 1:
                    ftype = 'seq'
                else:
                    ftype = ','.join(sorted([x.split('.')[-1] for x in field.split(',')]))
                input_dicts[i][ftype].append(field)
    
    input_opts = OrderedDict()
    input_opts['seq'] = args.sequence
    input_opts['fasta'] = args.fasta

    for ftype in input_opts:
        if input_opts[ftype] is not None:
            for i in range(NINPUTS):
                input_dicts[i][ftype] += input_opts[ftype]

#    LOGGER.debug(input_dicts)
    return input_dicts

def silent_init():
    import sys
    sys.path.insert(0, '/net/software/pyrosetta3.10/versions/release-350/setup/')
    global Pose, init, core, create_pdb_contents_from_sfr, PoseToStructFileRepConverter, SilentFileData, SilentFileOptions
    from pyrosetta import Pose, init
    from pyrosetta.rosetta import core
    from pyrosetta.rosetta.core.io.pdb import create_pdb_contents_from_sfr
    from pyrosetta.rosetta.core.io.pose_to_sfr import PoseToStructFileRepConverter
    from pyrosetta.rosetta.core.io.silent import SilentFileData
    from pyrosetta.rosetta.core.io.silent import SilentFileOptions
    init( "-mute all -beta_nov16 -in:file:silent_struct_type binary" +
              " -holes:dalphaball /software/rosetta/DAlphaBall.gcc"  +
              " -use_terminal_residues true -mute basic.io.database core.scoring" +
              "@/projects/protein-DNA-binders/flags_and_weights/flags_RM8B_245" )


def main():
    args = get_args()
    if args.silent_output is not None or args.silent_path is not None:
        silent_init()

    if args.silent_output is not None:
        global sfd_out
        if os.path.dirname(args.silent_output) != '':
            os.makedirs(os.path.dirname(args.silent_output), exist_ok=True)
        if not os.path.isfile(args.silent_output):
            with open(args.silent_output, 'w') as f: f.write(util.DEFAULT_SILENT_HEADER)
        sfd_out = SilentFileData(args.silent_output, False, False, "binary", SilentFileOptions())

    else:
        os.makedirs(args.out_dir, exist_ok=True)

    input_dicts = process_inputs(args)
    try:
        pred = Predictor(args)
    except RuntimeError:
        print(f"Device {args.device} not found. Running on cpu.")
        args.device = 'cpu'
        pred = Predictor(args)
    for input_dict in input_dicts:
        molecule = make_molecule(input_dict, args)
        if os.path.exists(f"{args.out_dir}/{molecule.name}_pred.pdb") and not args.overwrite:
            continue
        
        best_result, all_results = pred.predict(molecule, args)

        pred.output_pdbs(molecule, best_result, args)
        best_result = get_losses(molecule, best_result, args)
        save_scores(molecule, best_result, args)
    
        if args.save_all_recycles:
            original_name = copy.deepcopy(molecule.name)
            for i, result in enumerate(all_results):
                molecule.name = original_name + f'_recycle{i}'
                if args.dump_extra:
                    pred.output_pdbs(molecule, result, args)
                result = get_losses(molecule, result, args)
                save_scores(molecule, result, args)

    return None
    
if __name__ == "__main__":
    main()
