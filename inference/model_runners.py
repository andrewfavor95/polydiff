import copy
from datetime import datetime
import torch
from assertpy import assert_that
import numpy as np
from omegaconf import DictConfig, OmegaConf
import data_loader
from icecream import ic

import rf2aa.chemical
from rf2aa.chemical import NAATOKENS, MASKINDEX, NTOTAL, NHEAVYPROT
import rf2aa.util
import rf2aa.data_loader
from rf2aa.util_module import XYZConverter
from rf2aa.RoseTTAFoldModel import RoseTTAFoldModule
from rf2aa.kinematics import xyz_to_c6d, c6d_to_bins, xyz_to_t2d, get_chirals
import rf2aa.parsers
import rf2aa.tensor_util
import aa_model
import dataclasses

from kinematics import get_init_xyz
from diffusion import Diffuser
import seq_diffusion
from contigs import ContigMap
from inference import utils as iu
from potentials.manager import PotentialManager
from inference import symmetry
import logging
import torch.nn.functional as nn
import util
import hydra
from hydra.core.hydra_config import HydraConfig
import os

import sys
sys.path.append('../') # to access RF structure prediction stuff 

# When you import this it causes a circular import due to the changes made in apply masks for self conditioning
# This import is only used for SeqToStr Sampling though so can be fixed later - NRB
# import data_loader 
import model_input_logger
from model_input_logger import pickle_function_call

TOR_INDICES  = util.torsion_indices
TOR_CAN_FLIP = util.torsion_can_flip
REF_ANGLES   = util.reference_angles

class Sampler:

    def __init__(self, conf: DictConfig):
        """Initialize sampler.
        Args:
            conf: Configuration.
        """
        self.initialized = False
        self.initialize(conf)
    
    def initialize(self, conf: DictConfig):
        self._log = logging.getLogger(__name__)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        needs_model_reload = not self.initialized or conf.inference.ckpt_path != self._conf.inference.ckpt_path

        # Assign config to Sampler
        self._conf = conf

        # Initialize inference only helper objects to Sampler
        self.ckpt_path = conf.inference.ckpt_path

        if needs_model_reload:
            # Load checkpoint, so that we can assemble the config
            self.load_checkpoint()
            assemble_config_from_chk(self._conf, self.ckpt)
            # Now actually load the model weights into RF
            self.model = self.load_model()
        else:
            assemble_config_from_chk(self._conf, self.ckpt)

        # self.initialize_sampler(conf)
        self.initialized=True

        # Assemble config from the checkpoint
        print(' ')
        print('-'*100)
        print(' ')
        print("WARNING: The following options are not currently implemented at inference. Decide if this matters.")
        print("Delete these in inference/model_runners.py once they are implemented/once you decide they are not required for inference -- JW")
        print(" -predict_previous")
        print(" -prob_self_cond")
        print(" -seqdiff_b0")
        print(" -seqdiff_bT")
        print(" -seqdiff_schedule_type")
        print(" -seqdiff")
        print(" -freeze_track_motif")
        print(" -use_motif_timestep")
        print(" ")
        print("-"*100)
        print(" ")
        # Initialize helper objects
        self.inf_conf = self._conf.inference
        self.contig_conf = self._conf.contigmap
        self.denoiser_conf = self._conf.denoiser
        self.ppi_conf = self._conf.ppi
        self.potential_conf = self._conf.potentials
        self.diffuser_conf = self._conf.diffuser
        self.preprocess_conf = self._conf.preprocess
        self.diffuser = Diffuser(**self._conf.diffuser)
        self.model_adaptor = aa_model.Model(self._conf)
        # Temporary hack
        self.model.assert_single_sequence_input = True
        self.model_adaptor.model = self.model

        # TODO: Add symmetrization RMSD check here
        if self._conf.seq_diffuser.seqdiff is None:
            ic('Doing AR Sequence Decoding')
            self.seq_diffuser = None

            assert(self._conf.preprocess.seq_self_cond is False), 'AR decoding does not make sense with sequence self cond'
            self.seq_self_cond = self._conf.preprocess.seq_self_cond

        elif self._conf.seq_diffuser.seqdiff == 'continuous':
            ic('Doing Continuous Bit Diffusion')

            kwargs = {
                     'T': self._conf.diffuser.T,
                     's_b0': self._conf.seq_diffuser.s_b0,
                     's_bT': self._conf.seq_diffuser.s_bT,
                     'schedule_type': self._conf.seq_diffuser.schedule_type,
                     'loss_type': self._conf.seq_diffuser.loss_type
                     }
            self.seq_diffuser = seq_diffusion.ContinuousSeqDiffuser(**kwargs)

            self.seq_self_cond = self._conf.preprocess.seq_self_cond

        else:
            sys.exit(f'Seq Diffuser of type: {self._conf.seq_diffuser.seqdiff} is not known')

        if self.inf_conf.symmetry is not None:
            self.symmetry = symmetry.SymGen(
                self.inf_conf.symmetry,
                self.inf_conf.model_only_neighbors,
                self.inf_conf.recenter,
                self.inf_conf.radius, 
            )
        else:
            self.symmetry = None


        self.converter = XYZConverter()
        self.target_feats = iu.process_target(self.inf_conf.input_pdb, parse_hetatom=False, center=False)
        self.chain_idx = None

        if self.diffuser_conf.partial_T:
            assert self.diffuser_conf.partial_T <= self.diffuser_conf.T
            self.t_step_input = int(self.diffuser_conf.partial_T)
        else:
            self.t_step_input = int(self.diffuser_conf.T)
        if self.inf_conf.ppi_design and self.inf_conf.autogenerate_contigs:
            self.ppi_conf.binderlen = ''.join(chain_idx[0] for chain_idx in self.target_feats['pdb_idx']).index('B')

        self.potential_manager = PotentialManager(self.potential_conf, 
                                                  self.ppi_conf, 
                                                  self.diffuser_conf, 
                                                  self.inf_conf)
        
        # Get recycle schedule    
        recycle_schedule = str(self.inf_conf.recycle_schedule) if self.inf_conf.recycle_schedule is not None else None
        self.recycle_schedule = iu.recycle_schedule(self.T, recycle_schedule, self.inf_conf.num_recycles)

    def process_target(self, pdb_path):
        assert not (self.inf_conf.ppi_design and self.inf_conf.autogenerate_contigs), "target reprocessing not implemented yet for these configuration arguments"
        self.target_feats = iu.process_target(self.inf_conf.input_pdb)
        self.chain_idx = None

    @property
    def T(self):
        '''
            Return the maximum number of timesteps
            that this design protocol will perform.

            Output:
                T (int): The maximum number of timesteps to perform
        '''
        return self.diffuser_conf.T
    
    def load_checkpoint(self) -> None:
        """Loads RF checkpoint, from which config can be generated."""
        self._log.info(f'Reading checkpoint from {self.ckpt_path}')
        print(f'loading {self.ckpt_path}')
        self.ckpt  = torch.load(
            self.ckpt_path, map_location=self.device)
        print(f'loaded {self.ckpt_path}')


    def load_model(self):
        """Create RosettaFold model from preloaded checkpoint."""

        # for all-atom str loss
        self.ti_dev = rf2aa.util.torsion_indices
        self.ti_flip = rf2aa.util.torsion_can_flip
        self.ang_ref = rf2aa.util.reference_angles
        self.fi_dev = rf2aa.util.frame_indices
        self.l2a = rf2aa.util.long2alt
        self.aamask = rf2aa.util.allatom_mask
        self.num_bonds = rf2aa.util.num_bonds
        self.atom_type_index = rf2aa.util.atom_type_index
        self.ljlk_parameters = rf2aa.util.ljlk_parameters
        self.lj_correction_parameters = rf2aa.util.lj_correction_parameters
        self.hbtypes = rf2aa.util.hbtypes
        self.hbbaseatoms = rf2aa.util.hbbaseatoms
        self.hbpolys = rf2aa.util.hbpolys
        self.cb_len = rf2aa.util.cb_length_t
        self.cb_ang = rf2aa.util.cb_angle_t
        self.cb_tor = rf2aa.util.cb_torsion_t

        # model_param.
        self.ti_dev = self.ti_dev.to(self.device)
        self.ti_flip = self.ti_flip.to(self.device)
        self.ang_ref = self.ang_ref.to(self.device)
        self.fi_dev = self.fi_dev.to(self.device)
        self.l2a = self.l2a.to(self.device)
        self.aamask = self.aamask.to(self.device)
        self.num_bonds = self.num_bonds.to(self.device)
        self.atom_type_index = self.atom_type_index.to(self.device)
        self.ljlk_parameters = self.ljlk_parameters.to(self.device)
        self.lj_correction_parameters = self.lj_correction_parameters.to(self.device)
        self.hbtypes = self.hbtypes.to(self.device)
        self.hbbaseatoms = self.hbbaseatoms.to(self.device)
        self.hbpolys = self.hbpolys.to(self.device)
        self.cb_len = self.cb_len.to(self.device)
        self.cb_ang = self.cb_ang.to(self.device)
        self.cb_tor = self.cb_tor.to(self.device)

        # HACK: TODO: save this in the model config
        self.loss_param = {'lj_lin': 0.75}
        model = RoseTTAFoldModule(
            **self._conf.model,
            aamask=self.aamask,
            atom_type_index=self.atom_type_index,
            ljlk_parameters=self.ljlk_parameters,
            lj_correction_parameters=self.lj_correction_parameters,
            num_bonds=self.num_bonds,
            cb_len = self.cb_len,
            cb_ang = self.cb_ang,
            cb_tor = self.cb_tor,
            lj_lin=self.loss_param['lj_lin'],
            assert_single_sequence_input=True,
            ).to(self.device)
        
        if self._conf.logging.inputs:
            pickle_dir = pickle_function_call(model, 'forward', 'inference', minifier=aa_model.minifier)
            print(f'pickle_dir: {pickle_dir}')
        model = model.eval()
        self._log.info(f'Loading checkpoint.')
        if not self._conf.inference.zero_weights:
            model.load_state_dict(self.ckpt[self._conf.inference.state_dict_to_load], strict=True)
        return model

    def construct_contig(self, target_feats):
        """Create contig from target features."""
        if self.inf_conf.ppi_design and self.inf_conf.autogenerate_contigs:
            seq_len = target_feats['seq'].shape[0]
            self.contig_conf.contigs = [f'{self.ppi_conf.binderlen}',f'B{self.ppi_conf.binderlen+1}-{seq_len}']
        self._log.info(f'Using contig: {self.contig_conf.contigs}')
        # self.contig_conf.contigs = ['']
        if self.contig_conf.contigs == 'whole':
            L = len(target_feats["pdb_idx"])
            self.contig_conf.contigs = [f'{L}-{L}']
        return ContigMap(target_feats, **self.contig_conf)

    def construct_denoiser(self, L, visible):
        """Make length-specific denoiser."""
        # TODO: Denoiser seems redundant. Combine with diffuser.
        denoise_kwargs = OmegaConf.to_container(self.diffuser_conf)
        denoise_kwargs.update(OmegaConf.to_container(self.denoiser_conf))
        aa_decode_steps = min(denoise_kwargs['aa_decode_steps'], denoise_kwargs['partial_T'] or 999)
        denoise_kwargs.update({
            'L': L,
            'diffuser': self.diffuser,
            'seq_diffuser': self.seq_diffuser,
            'potential_manager': self.potential_manager,
            'visible': visible,
            'aa_decode_steps': aa_decode_steps,
        })
        return iu.Denoise(**denoise_kwargs)

    def sample_init(self, return_forward_trajectory=False):
        """Initial features to start the sampling process.
        
        Modify signature and function body for different initialization
        based on the config.
        
        Returns:
            xt: Starting positions with a portion of them randomly sampled.
            seq_t: Starting sequence with a portion of them set to unknown.
        """

        # moved this here as should be updated each iteration of diffusion
        self.contig_map = self.construct_contig(self.target_feats)
        L = len(self.target_feats['pdb_idx'])

        indep_orig = aa_model.make_indep(self._conf.inference.input_pdb, self._conf.inference.ligand)
        indep, self.is_diffused, self.is_seq_masked = self.model_adaptor.insert_contig(indep_orig, self.contig_map)
        self.t_step_input = self._conf.diffuser.T
        if self.diffuser_conf.partial_T:
            mappings = self.contig_map.get_mappings()
            assert indep.xyz.shape[0] ==  L + torch.sum(indep.is_sm), f"there must be a coordinate in the input PDB for each residue implied by the contig string for partial diffusion.  length of input PDB != length of contig string: {indep.xyz.shape[0]} != {L+torch.sum(indep.is_sm)}"
            assert torch.all(self.is_diffused[indep.is_sm] == 0), f"all ligand atoms must be in the motif"
            assert (mappings['con_hal_idx0'] == mappings['con_ref_idx0']).all(), 'all positions in the input PDB must correspond to the same index in the output pdb'
            indep = indep_orig
        indep.seq[self.is_seq_masked] = rf2aa.chemical.MASKINDEX
        # Diffuse the contig-mapped coordinates 
        if self.diffuser_conf.partial_T:
            self.t_step_input = self.diffuser_conf.partial_T
            assert self.diffuser_conf.partial_T <= self.diffuser_conf.T
        t_list = np.arange(1, self.t_step_input+1)
        atom_mask = None
        seq_one_hot = None
        fa_stack, aa_masks, xyz_true = self.diffuser.diffuse_pose(
            indep.xyz,
            seq_one_hot,
            atom_mask,
            indep.is_sm,
            diffusion_mask=~self.is_diffused,
            t_list=t_list,
            diffuse_sidechains=self.preprocess_conf.sidechain_input,
            include_motif_sidechains=self.preprocess_conf.motif_sidechain_input)

        xT = fa_stack[-1].squeeze()[:,:14,:]
        xt = torch.clone(xT)
        indep.xyz = xt

        self.denoiser = self.construct_denoiser(len(self.contig_map.ref), visible=~self.is_diffused)
        if self.symmetry is not None:
            raise Exception('not implemented')
            xt, seq_t = self.symmetry.apply_symmetry(xt, seq_t)
        
        self.msa_prev = None
        self.pair_prev = None
        self.state_prev = None
        
        return indep

    def _preprocess(self, seq, xyz_t, t, repack=False):
        raise Exception('should not be called')
        
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
        L = seq.shape[0]
        T = self.T
        ppi_design = self.inf_conf.ppi_design
        binderlen = self.ppi_conf.binderlen
        target_res = self.ppi_conf.hotspot_res


        '''
        msa_full:   NSEQ,NINDEL,NTERMINUS,
        msa_masked: NSEQ,NSEQ,NINDEL,NINDEL,NTERMINUS
        '''
        NTERMINUS = 2
        NINDEL = 1
        ### msa_masked ###
        ##################
        msa_masked = torch.zeros((1,1,L,2*NAATOKENS+NINDEL*2+NTERMINUS))

        msa_masked[:,:,:,:NAATOKENS] = seq[None, None]
        msa_masked[:,:,:,NAATOKENS:2*NAATOKENS] = seq[None, None]
        if self._conf.inference.annotate_termini:
            msa_masked[:,:,0,NAATOKENS*2+NINDEL*2] = 1.0
            msa_masked[:,:,-1,NAATOKENS*2+NINDEL*2+1] = 1.0

        ### msa_full ###
        ################
        msa_full = torch.zeros((1,1,L,NAATOKENS+NINDEL+NTERMINUS))
        msa_full[:,:,:,:NAATOKENS] = seq[None, None]
        if self._conf.inference.annotate_termini:
            msa_full[:,:,0,NAATOKENS+NINDEL] = 1.0
            msa_full[:,:,-1,NAATOKENS+NINDEL+1] = 1.0

        ### t1d ###
        ########### 
        # NOTE: Not adjusting t1d last dim (confidence) from sequence mask

        # Here we need to go from one hot with 22 classes to one hot with 21 classes
        # If sequence is masked, it becomes unknown
        # t1d = torch.zeros((1,1,L,NAATOKENS-1))

        #seqt1d = torch.clone(seq)
        seq_cat_shifted = seq.argmax(dim=-1)
        seq_cat_shifted[seq_cat_shifted>=MASKINDEX] -= 1
        t1d = torch.nn.functional.one_hot(seq_cat_shifted, num_classes=NAATOKENS-1)
        t1d = t1d[None, None] # [L, NAATOKENS-1] --> [1,1,L, NAATOKENS-1]
        # for idx in range(L):
            
        #     if seqt1d[idx,MASKINDEX] == 1:
        #         seqt1d[idx, MASKINDEX-1] = 1
        #         seqt1d[idx,MASKINDEX] = 0
        # t1d[:,:,:,:NPROTAAS+1] = seqt1d[None,None,:,:NPROTAAS+1]
        
        # Str Confidence
        if self.inf_conf.autoregressive_confidence:
            # Set confidence to 1 where diffusion mask is True, else 1-t/T
            strconf = torch.zeros((L)).float()
            strconf[self.mask_str.squeeze()] = 1.
            strconf[~self.mask_str.squeeze()] = 1. - t/self.T
            strconf = strconf[None,None,...,None]
        else:
            #NOTE: DJ - I don't know what this does or why it's here
            strconf = torch.where(self.mask_str.squeeze(), 1., 0.)[None,None,...,None]

        t1d = torch.cat((t1d, strconf), dim=-1)
        
        # Seq Confidence
        if self.inf_conf.autoregressive_confidence:
            # Set confidence to 1 where diffusion mask is True, else 1-t/T
            seqconf = torch.zeros((L)).float()
            seqconf[self.mask_seq.squeeze()] = 1.
            seqconf[~self.mask_seq.squeeze()] = 1. - t/self.T
            seqconf = seqconf[None,None,...,None]
        else:
            #NOTE: DJ - I don't know what this does or why it's here
            seqconf = torch.where(self.mask_seq.squeeze(), 1., 0.)[None,None,...,None]
        
        # # Seqdiff confidence is only added in when d_t1d is greater than or equal to 23 - NRB
        # if self.preprocess_conf.d_t1d >= 23:
        #     t1d = torch.cat((t1d, seqconf), dim=-1)
            
        t1d = t1d.float()
        
        ### xyz_t ###
        #############
        if self.preprocess_conf.sidechain_input:
            raise Exception('not implemented')
            xyz_t[torch.where(seq == 21, True, False),3:,:] = float('nan')
        else:
            xyz_t[~self.mask_str.squeeze(),3:,:] = float('nan')
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
        idx = torch.tensor(self.contig_map.rf)[None]

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



        #put tensors on device
        msa_masked = msa_masked.to(self.device)
        msa_full = msa_full.to(self.device)
        seq = seq.to(self.device)
        xyz_t = xyz_t.to(self.device)
        idx = idx.to(self.device)
        t1d = t1d.to(self.device)
        # t2d = t2d.to(self.device)
        alpha_t = alpha_t.to(self.device)
        
        ### added_features ###
        ######################
        # NB the hotspot input has been removed in this branch. 
        # JW added it back in, using pdb indexing

        if self.preprocess_conf.d_t1d == 24: # add hotpot residues
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
        
        return msa_masked, msa_full, seq[None], torch.squeeze(xyz_t, dim=0), idx, t1d, t2d, xyz_t, alpha_t
        
    # def sample_step(self, *, t, seq_t, x_t, seq_init, final_step, return_extra=False):
    #     '''Generate the next pose that the model should be supplied at timestep t-1.

    #     Args:
    #         t (int): The timestep that has just been predicted
    #         seq_t (torch.tensor): (L,22) The sequence at the beginning of this timestep
    #         x_t (torch.tensor): (L,14,3) The residue positions at the beginning of this timestep
    #         seq_init (torch.tensor): (L,22) The initialized sequence used in updating the sequence.
            
    #     Returns:
    #         px0: (L,14,3) The model's prediction of x0.
    #         x_t_1: (L,14,3) The updated positions of the next step.
    #         seq_t_1: (L,22) The updated sequence of the next step.
    #         tors_t_1: (L, ?) The updated torsion angles of the next  step.
    #         plddt: (L, 1) Predicted lDDT of x0.
    #     '''
    #     out = self._preprocess(seq_t, x_t, t)
    #     msa_masked, msa_full, seq_in, xt_in, idx_pdb, t1d, t2d, xyz_t, alpha_t = self._preprocess(
    #         seq_t, x_t, t)

    #     N,L = msa_masked.shape[:2]

    #     if self.symmetry is not None:
    #         idx_pdb, self.chain_idx = self.symmetry.res_idx_procesing(res_idx=idx_pdb)

    #     # decide whether to recycle information between timesteps or not
    #     if self.inf_conf.recycle_between and t < self.diffuser_conf.aa_decode_steps:
    #         msa_prev = self.msa_prev
    #         pair_prev = self.pair_prev
    #         state_prev = self.state_prev
    #     else:
    #         msa_prev = None
    #         pair_prev = None
    #         state_prev = None

    #     with torch.no_grad():
    #         # So recycling is done a la training
    #         px0=xt_in
    #         for _ in range(self.recycle_schedule[t-1]):
    #             msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt = self.model(msa_masked,
    #                                 msa_full,
    #                                 seq_in,
    #                                 px0,
    #                                 idx_pdb,
    #                                 t1d=t1d,
    #                                 t2d=t2d,
    #                                 xyz_t=xyz_t,
    #                                 alpha_t=alpha_t,
    #                                 msa_prev = msa_prev,
    #                                 pair_prev = pair_prev,
    #                                 state_prev = state_prev,
    #                                 t=torch.tensor(t),
    #                                 return_infer=True,
    #                                 motif_mask=self.diffusion_mask.squeeze().to(self.device))

    #     self.msa_prev=msa_prev
    #     self.pair_prev=pair_prev
    #     self.state_prev=state_prev
    #     # prediction of X0 
    #     _, px0  = self.allatom(torch.argmax(seq_in, dim=-1), px0, alpha)
    #     px0    = px0.squeeze()[:,:14]
    #     #sampled_seq = torch.argmax(logits.squeeze(), dim=-1)
    #     seq_probs   = torch.nn.Softmax(dim=-1)(logits.squeeze()/self.inf_conf.softmax_T)
    #     sampled_seq = torch.multinomial(seq_probs, 1).squeeze() # sample a single value from each position 
        
    #     # grab only the query sequence prediction - adjustment for Seq2StrSampler
    #     sampled_seq = sampled_seq.reshape(N,L,-1)[0,0]

    #     # Process outputs.
    #     mask_seq = self.mask_seq

    #     pseq_0 = torch.nn.functional.one_hot(
    #         sampled_seq, num_classes=22).to(self.device)

    #     pseq_0[mask_seq.squeeze()] = seq_init[
    #         mask_seq.squeeze()].to(self.device)

    #     seq_t = torch.nn.functional.one_hot(
    #         seq_t, num_classes=22).to(self.device)

    #     self._log.info(
    #        f'Timestep {t}, current sequence: { rf2aa.chemical.seq2chars(torch.argmax(pseq_0, dim=-1).tolist())}')
        
    #     if t > final_step:
    #         x_t_1, seq_t_1, tors_t_1, px0 = self.denoiser.get_next_pose(
    #             xt=x_t,
    #             px0=px0,
    #             t=t,
    #             diffusion_mask=self.mask_str.squeeze(),
    #             seq_diffusion_mask=self.mask_seq.squeeze(),
    #             seq_t=seq_t,
    #             pseq0=pseq_0,
    #             diffuse_sidechains=self.preprocess_conf.sidechain_input,
    #             align_motif=self.inf_conf.align_motif,
    #             include_motif_sidechains=self.preprocess_conf.motif_sidechain_input
    #         )
    #     else:
    #         x_t_1 = torch.clone(px0).to(x_t.device)
    #         seq_t_1 = torch.clone(pseq_0)
    #         # Dummy tors_t_1 prediction. Not used in final output.
    #         tors_t_1 = torch.ones((self.mask_str.shape[-1], 10, 2))
    #         px0 = px0.to(x_t.device)
    #     if self.symmetry is not None:
    #         x_t_1, seq_t_1 = self.symmetry.apply_symmetry(x_t_1, seq_t_1)
    #     if return_extra:
    #         return px0, x_t_1, seq_t_1, tors_t_1, plddt, logits
    #     return px0, x_t_1, seq_t_1, tors_t_1, plddt

    def symmetrise_prev_pred(self, px0, seq_in, alpha):
        """
        Method for symmetrising px0 output, either for recycling or for self-conditioning
        """
        _,px0_aa = self.converter.compute_all_atom(torch.argmax(seq_in, dim=-1), px0, alpha)
        px0_sym,_ = self.symmetry.apply_symmetry(px0_aa.to('cpu').squeeze()[:,:14], torch.argmax(seq_in, dim=-1).squeeze().to('cpu'))
        px0_sym = px0_sym[None].to(self.device)
        return px0_sym

class NRBStyleSelfCond(Sampler):
    """
    Model Runner for self conditioning in the style attempted by NRB
    """

    def sample_step(self, t, indep, rfo):
        '''
        Generate the next pose that the model should be supplied at timestep t-1.
        Args:
            t (int): The timestep that has just been predicted
            seq_t (torch.tensor): (L,22) The sequence at the beginning of this timestep
            x_t (torch.tensor): (L,14,3) The residue positions at the beginning of this timestep
            seq_init (torch.tensor): (L,22) The initialized sequence used in updating the sequence.
        Returns:
            px0: (L,14,3) The model's prediction of x0.
            x_t_1: (L,14,3) The updated positions of the next step.
            seq_t_1: (L) The updated sequence of the next step.
            tors_t_1: (L, ?) The updated torsion angles of the next  step.
            plddt: (L, 1) Predicted lDDT of x0.
        '''

        rfi = self.model_adaptor.prepro(indep, t, self.is_diffused)
        rf2aa.tensor_util.to_device(rfi, self.device)
        seq_init = torch.nn.functional.one_hot(
                indep.seq, num_classes=rf2aa.chemical.NAATOKENS).to(self.device).float()
        seq_t = torch.clone(seq_init)
        seq_in = torch.clone(seq_init)
        # B,N,L = xyz_t.shape[:3]

        ##################################
        ######## Str Self Cond ###########
        ##################################
        self_cond = False
        if ((t < self.diffuser.T) and (t != self.diffuser_conf.partial_T)) and self._conf.inference.str_self_cond:
            self_cond=True
            rfi = aa_model.self_cond(indep, rfi, rfo)

        if self.symmetry is not None:
            idx_pdb, self.chain_idx = self.symmetry.res_idx_procesing(res_idx=idx_pdb)

        with torch.no_grad():
            if self.recycle_schedule[t-1] > 1:
                raise Exception('not implemented')
            for rec in range(self.recycle_schedule[t-1]):
		# This is the assertion we should be able to use, but the
		# network's ComputeAllAtom requires even atoms to have N and C coords.                
		# aa_model.assert_has_coords(rfi.xyz[0], indep)
                assert not rfi.xyz[0,:,:3,:].isnan().any(), f'{t}: {rfi.xyz[0,:,:3,:]}'
                rfo = self.model_adaptor.forward(
                                    rfi,
                                    return_infer=True,
                                    # **{model_input_logger.LOG_ONLY_KEY: {
                                    #     't':t,
                                    #     'output_prefix':self.output_prefix,
                                    # }}
                                    )

                if self.symmetry is not None and self.inf_conf.symmetric_self_cond:
                    raise Exception('not implemented')
                    px0 = self.symmetrise_prev_pred(px0=px0,seq_in=seq_in, alpha=alpha)[:,:,:3]

                # To permit 'recycling' within a timestep, in a manner akin to how this model was trained
                # Aim is to basically just replace the xyz_t with the model's last px0, and to *not* recycle the state, pair or msa embeddings
                if rec < self.recycle_schedule[t-1] -1:
                    raise Exception('not implemented')
                    zeros = torch.zeros(B,1,L,24,3).float().to(xyz_t.device)
                    xyz_t = torch.cat((px0.unsqueeze(1),zeros), dim=-2) # [B,T,L,27,3]

                    t2d   = xyz_to_t2d(xyz_t) # [B,T,L,L,44]

                    if self.seq_self_cond:
                        # Allow this model to also do sequence recycling

                        t1d[:,:,:,:20] = logits[:,None,:,:20]
                        t1d[:,:,:,20]  = 0 # Setting mask token to zero
        px0 = rfo.get_xyz()[:,:14]
        logits = rfo.get_seq_logits()
        seq_decoded = [rf2aa.chemical.num2aa[s] for s in rfi.seq[0]]

        if self.seq_diffuser is None:
            # Default method of decoding sequence
            seq_probs   = torch.nn.Softmax(dim=-1)(logits.squeeze()/self.inf_conf.softmax_T)
            sampled_seq = torch.multinomial(seq_probs, 1).squeeze() # sample a single value from each position

            pseq_0 = torch.nn.functional.one_hot(
                sampled_seq, num_classes=rf2aa.chemical.NAATOKENS).to(self.device).float()
            
            pseq_0[~self.is_seq_masked] = seq_init[~self.is_seq_masked].to(self.device) # [L,22]
        else:
            # Sequence Diffusion
            pseq_0 = logits.squeeze()
            pseq_0 = pseq_0[:,:20]

            pseq_0[self.mask_seq.squeeze()] = seq_init[self.mask_seq.squeeze(),:20].to(self.device)

            sampled_seq = torch.argmax(pseq_0, dim=-1)

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        self._log.info(
                f'{current_time}: Timestep {t}, current sequence: { rf2aa.chemical.seq2chars(torch.argmax(pseq_0, dim=-1).tolist())}')

        if t > self._conf.inference.final_step:
            x_t_1, seq_t_1, tors_t_1, px0 = self.denoiser.get_next_pose(
                xt=rfi.xyz[0,:,:14].cpu(),
                px0=px0,
                t=t,
                diffusion_mask=~self.is_diffused,
                seq_diffusion_mask=~self.is_diffused,
                seq_t=seq_t,
                pseq0=pseq_0,
                diffuse_sidechains=self.preprocess_conf.sidechain_input,
                align_motif=self.inf_conf.align_motif,
                include_motif_sidechains=self.preprocess_conf.motif_sidechain_input,
            )
        else:
            # Final step.
            px0 = px0.cpu()
            px0[~self.is_diffused] = indep.xyz[~self.is_diffused]
            x_t_1 = torch.clone(px0)
            seq_t_1 = pseq_0

            # Dummy tors_t_1 prediction. Not used in final output.
            tors_t_1 = torch.ones((self.is_diffused.shape[-1], 10, 2))

        px0 = px0.cpu()
        x_t_1 = x_t_1.cpu()
        seq_t_1 = seq_t_1.cpu()

        if self.symmetry is not None:
            x_t_1, seq_t_1 = self.symmetry.apply_symmetry(x_t_1, seq_t_1)

        return px0, x_t_1, seq_t_1, tors_t_1, None, rfo

def sampler_selector(conf: DictConfig):
    if conf.inference.model_runner == 'default':
        sampler = Sampler(conf)
    elif conf.inference.model_runner == 'legacy':
        sampler = T1d28T2d45Sampler(conf)
    elif conf.inference.model_runner == 'seq2str':
        sampler = Seq2StrSampler(conf)
    elif conf.inference.model_runner == 'JWStyleSelfCond':
        sampler = JWStyleSelfCond(conf)
    elif conf.inference.model_runner == 'NRBStyleSelfCond':
        sampler = NRBStyleSelfCond(conf)
    else:
        raise ValueError(f'Unrecognized sampler {conf.model_runner}')
    return sampler


def assemble_config_from_chk(conf, ckpt) -> None:
    """
    Function for loading model config from checkpoint directly.

    Takes:
        - config file

    Actions:
        - Replaces all -model and -diffuser items
        - Throws a warning if there are items in -model and -diffuser that aren't in the checkpoint
    
    This throws an error if there is a flag in the checkpoint 'config_dict' that isn't in the inference config.
    This should ensure that whenever a feature is added in the training setup, it is accounted for in the inference script.

    JW
    """
    
    # get overrides to re-apply after building the config from the checkpoint
    overrides = []
    if HydraConfig.initialized():
        overrides = HydraConfig.get().overrides.task
        ic(overrides)
    if 'config_dict' in ckpt.keys():
        print("Assembling -model, -diffuser and -preprocess configs from checkpoint")

        # First, check all flags in the checkpoint config dict are in the config file
        for cat in ['model','diffuser','seq_diffuser','preprocess']:
            #assert all([i in self._conf[cat].keys() for i in self.ckpt['config_dict'][cat].keys()]), f"There are keys in the checkpoint config_dict {cat} params not in the config file"
            for key in conf[cat]:
                if key == 'chi_type' and ckpt['config_dict'][cat][key] == 'circular':
                    ic('---------------------------------------------SKIPPPING CIRCULAR CHI TYPE')
                    continue
                try:
                    print(f"USING MODEL CONFIG: self._conf[{cat}][{key}] = {ckpt['config_dict'][cat][key]}")
                    conf[cat][key] = ckpt['config_dict'][cat][key]
                except:
                    print(f'WARNING: config {cat}.{key} is not saved in the checkpoint. Check that conf.{cat}.{key} = {conf[cat][key]} is correct')
        # add back in overrides again
        for override in overrides:
            if override.split(".")[0] in ['model','diffuser','seq_diffuser','preprocess']:
                print(f'WARNING: You are changing {override.split("=")[0]} from the value this model was trained with. Are you sure you know what you are doing?') 
                mytype = type(self._conf[override.split(".")[0]][override.split(".")[1].split("=")[0]])

                if mytype == bool:
                    # special treatment for bools because they are strings in override 
                    self._conf[override.split(".")[0]][override.split(".")[1].split("=")[0]] = override.split("=")[1].lower().strip() == 'true'
                else:
                    self._conf[override.split(".")[0]][override.split(".")[1].split("=")[0]] = mytype(override.split("=")[1])
    else:
        print('WARNING: Model, Diffuser and Preprocess parameters are not saved in this checkpoint. Check carefully that the values specified in the config are correct for this checkpoint')     

    print('self._conf:')
    ic(conf)