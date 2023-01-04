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
from rf2aa.util_module import ComputeAllAtomCoords
from rf2aa.RoseTTAFoldModel import RoseTTAFoldModule

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
            self.assemble_config_from_chk()
            # Now actually load the model weights into RF
            self.model = self.load_model()
        else:
            self.assemble_config_from_chk()

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


        self.allatom = ComputeAllAtomCoords().to(self.device)
        self.target_feats = iu.process_target(self.inf_conf.input_pdb, parse_hetatom=True, center=False)
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
        print('This is inf_conf.ckpt_path')
        print(self.ckpt_path)
        self.ckpt  = torch.load(
            self.ckpt_path, map_location=self.device)

    def assemble_config_from_chk(self) -> None:
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
        if 'config_dict' in self.ckpt.keys():
            print("Assembling -model, -diffuser and -preprocess configs from checkpoint")

            # First, check all flags in the checkpoint config dict are in the config file
            for cat in ['model','diffuser','seq_diffuser','preprocess']:
                #assert all([i in self._conf[cat].keys() for i in self.ckpt['config_dict'][cat].keys()]), f"There are keys in the checkpoint config_dict {cat} params not in the config file"
                for key in self._conf[cat]:
                    if key == 'chi_type' and self.ckpt['config_dict'][cat][key] == 'circular':
                        ic('---------------------------------------------SKIPPPING CIRCULAR CHI TYPE')
                        continue
                    try:
                        print(f"USING MODEL CONFIG: self._conf[{cat}][{key}] = {self.ckpt['config_dict'][cat][key]}")
                        self._conf[cat][key] = self.ckpt['config_dict'][cat][key]
                    except:
                        print(f'WARNING: config {cat}.{key} is not saved in the checkpoint. Check that conf.{cat}.{key} = {self._conf[cat][key]} is correct')
            # add back in overrides again
            for override in overrides:
                if override.split(".")[0] in ['model','diffuser','seq_diffuser','preprocess']:
                    print(f'WARNING: You are changing {override.split("=")[0]} from the value this model was trained with. Are you sure you know what you are doing?') 
                    mytype = type(self._conf[override.split(".")[0]][override.split(".")[1].split("=")[0]])
                    self._conf[override.split(".")[0]][override.split(".")[1].split("=")[0]] = mytype(override.split("=")[1])
        else:
            print('WARNING: Model, Diffuser and Preprocess parameters are not saved in this checkpoint. Check carefully that the values specified in the config are correct for this checkpoint')     

        print('self._conf:')
        ic(self._conf)

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
            pickle_dir = pickle_function_call(model, 'forward', 'inference')
            print(f'pickle_dir: {pickle_dir}')
        model = model.eval()
        self._log.info(f'Loading checkpoint.')
        model.load_state_dict(self.ckpt['model_state_dict'], strict=True)
        return model

    def construct_contig(self, target_feats):
        """Create contig from target features."""
        if self.inf_conf.ppi_design and self.inf_conf.autogenerate_contigs:
            seq_len = target_feats['seq'].shape[0]
            self.contig_conf.contigs = [f'{self.ppi_conf.binderlen}',f'B{self.ppi_conf.binderlen+1}-{seq_len}']
        self._log.info(f'Using contig: {self.contig_conf.contigs}')
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
        self.mask_seq = torch.from_numpy(self.contig_map.inpaint_seq)[None,:]
        self.mask_str = torch.from_numpy(self.contig_map.inpaint_str)[None,:]
         
        target_feats = self.target_feats
        contig_map = self.contig_map

        xyz_27 = target_feats['xyz_27']
        mask_27 = target_feats['mask_27']
        seq_orig = target_feats['seq']
        L_mapped = len(self.contig_map.ref)

        # Only protein diffusion right now
        self.atom_frames = torch.zeros((1,0,3,2)).to(self.device)
        self.is_sm = torch.zeros(L_mapped).bool().to(self.device)
        self.chirals = torch.Tensor().to(self.device)[None]
        self.bond_feats = rf2aa.data_loader.get_protein_bond_feats(L_mapped).long().to(self.device)[None]
        # B,L,L
        self.same_chain = torch.ones(1, L_mapped, L_mapped).bool().to(self.device)

        diffusion_mask = self.mask_str
        self.diffusion_mask = diffusion_mask
        # adjust size of input xt according to residue map 
        if self.diffuser_conf.partial_T:
            assert xyz_27.shape[0] == L_mapped, f"there must be a coordinate in the input PDB for each residue implied by the contig string for partial diffusion.  length of input PDB != length of contig string: {xyz_27.shape[0]} != {L_mapped}"
            assert contig_map.hal_idx0 == contig_map.ref_idx0, f'for partial diffusion there can be no offset between the index of a residue in the input and the index of the residue in the output, {contig_map.hal_idx0} != {contig_map.ref_idx0}'
            # Partially diffusing from a known structure
            xyz_mapped=xyz_27
            atom_mask_mapped = mask_27
        else:
            # Fully diffusing from points initialised at the origin
            # adjust size of input xt according to residue map
            xyz_mapped = torch.full((1,1,L_mapped,27,3), np.nan)
            xyz_mapped[:, :, contig_map.hal_idx0, ...] = xyz_27[contig_map.ref_idx0,...]
            xyz_motif_prealign = xyz_mapped.clone()
            motif_prealign_com = xyz_motif_prealign[0,0,:,1].mean(dim=0)
            self.motif_com = xyz_27[contig_map.ref_idx0,1].mean(dim=0)
            xyz_mapped = get_init_xyz(xyz_mapped, self.is_sm).squeeze()
            # adjust the size of the input atom map
            atom_mask_mapped = torch.full((L_mapped, 27), False)
            atom_mask_mapped[contig_map.hal_idx0] = mask_27[contig_map.ref_idx0]

        # Diffuse the contig-mapped coordinates 
        if self.diffuser_conf.partial_T:
            assert self.diffuser_conf.partial_T <= self.diffuser_conf.T
            self.t_step_input = int(self.diffuser_conf.partial_T)
        else:
            self.t_step_input = int(self.diffuser_conf.T)
        t_list = np.arange(1, self.t_step_input+1)

        # Moved this here so that the sequence diffuser has access to t_list
        # NOTE: This is where we switch from an integer sequence to a one-hot sequence - NRB
        if self.seq_diffuser is None:
            seq_t = torch.full((1,L_mapped), 21).squeeze()
            seq_t[contig_map.hal_idx0] = seq_orig[contig_map.ref_idx0]
            seq_t[~self.mask_seq.squeeze()] = 21

            seq_t    = torch.nn.functional.one_hot(seq_t, num_classes=rf2aa.chemical.NAATOKENS).float() # [L,22]
            seq_orig = torch.nn.functional.one_hot(seq_orig, num_classes=rf2aa.chemical.NAATOKENS).float() # [L,22]
        else:
            # Sequence diffusion
            # Noise sequence using seq diffuser
            seq_mapped = torch.full((1,L_mapped), 0).squeeze()
            seq_mapped[contig_map.hal_idx0] = seq_orig[contig_map.ref_idx0]

            diffused_seq_stack, seq_orig = self.seq_diffuser.diffuse_sequence( 
                    seq = seq_mapped,
                    diffusion_mask = self.mask_seq.squeeze(),
                    t_list = t_list
                    )

            seq_t = torch.clone(diffused_seq_stack[-1]) # [L,20]

            zeros = torch.zeros(L_mapped,2)
            seq_t = torch.cat((seq_t,zeros), dim=-1) # [L,22]

        fa_stack, aa_masks, xyz_true = self.diffuser.diffuse_pose(
            xyz_mapped,
            torch.clone(seq_t),  # TODO: Check if copy is needed.
            atom_mask_mapped.squeeze(),
            self.is_sm,
            diffusion_mask=diffusion_mask.squeeze(),
            t_list=t_list,
            diffuse_sidechains=self.preprocess_conf.sidechain_input,
            include_motif_sidechains=self.preprocess_conf.motif_sidechain_input)
        xT = fa_stack[-1].squeeze()[:,:14,:]
        xt = torch.clone(xT)

        if self.diffuser_conf.partial_T and self.seq_diffuser is None:
            is_motif = self.mask_seq.squeeze()
            is_shown_at_t = torch.tensor(aa_masks[-1])
            visible = is_motif | is_shown_at_t
            if self.diffuser_conf.partial_T:
                seq_t[visible] = seq_orig[visible]
        else:
            # Sequence diffusion
            visible = self.mask_seq.squeeze()

        self.denoiser = self.construct_denoiser(len(self.contig_map.ref), visible=visible)
        if self.symmetry is not None:
            xt, seq_t = self.symmetry.apply_symmetry(xt, seq_t)
        self._log.info(f'Sequence init: {rf2aa.chemical.seq2chars(torch.argmax(seq_t, dim=-1))}')
        
        if return_forward_trajectory:
            forward_traj = torch.cat([xyz_true[None], fa_stack[:,:,:]])
            if self.seq_diffuser is None:
                aa_masks[:, diffusion_mask.squeeze()] = True
                return xt, seq_t, forward_traj, aa_masks, seq_orig
            else:
                # Seq Diffusion
                return xt, seq_t, forward_traj, diffused_seq_stack, seq_orig
        
        self.msa_prev = None
        self.pair_prev = None
        self.state_prev = None
        # For the implicit ligand potential
        if self.potential_conf.guiding_potentials is not None:
            if any(list(filter(lambda x: "substrate_contacts" in x, self.potential_conf.guiding_potentials))):
                assert len(self.target_feats['xyz_het']) > 0, "If you're using the Substrate Contact potential, you need to make sure there's a ligand in the input_pdb file!"
                xyz_het = torch.from_numpy(self.target_feats['xyz_het'])
                xyz_motif_prealign = xyz_motif_prealign[0,0][self.diffusion_mask.squeeze()]
                motif_prealign_com = xyz_motif_prealign[:,1].mean(dim=0)
                xyz_het_com = xyz_het.mean(dim=0)
                for pot in self.potential_manager.potentials_to_apply: # fix this
                    pot.motif_substrate_atoms = xyz_het
                    pot.diffusion_mask = self.diffusion_mask.squeeze()
                    pot.xyz_motif = xyz_motif_prealign
                    pot.diffuser = self.diffuser
        return xt, seq_t

    def _preprocess(self, seq, xyz_t, t, repack=False):
        
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
        _,px0_aa = self.allatom(torch.argmax(seq_in, dim=-1), px0, alpha)
        px0_sym,_ = self.symmetry.apply_symmetry(px0_aa.to('cpu').squeeze()[:,:14], torch.argmax(seq_in, dim=-1).squeeze().to('cpu'))
        px0_sym = px0_sym[None].to(self.device)
        return px0_sym

class NRBStyleSelfCond(Sampler):
    """
    Model Runner for self conditioning in the style attempted by NRB
    """

    def sample_step(self, *, t, seq_t, x_t, seq_init, final_step):
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
        msa_masked, msa_full, seq_in, xt_in, idx_pdb, t1d, t2d, xyz_t, alpha_t = self._preprocess(
            seq_t, x_t, t)

        B,N,L = xyz_t.shape[:3]

        ##################################
        ######## Str Self Cond ###########
        ##################################
        self_cond = False
        if ((t < self.diffuser.T) and (t != self.diffuser_conf.partial_T)) and self._conf.inference.str_self_cond:
            self_cond=True
            ic('Providing Self Cond')
                
            zeros = torch.zeros(B,1,L,36-3,3).float().to(xyz_t.device)
            xyz_t = torch.cat((self.prev_pred.unsqueeze(1),zeros), dim=-2) # [B,T,L,27,3]

            t2d, mask_t_2d_remade = util.get_t2d(
                xyz_t[0], self.is_sm, self.atom_frames)
            t2d = t2d[None] # Add batch dimension # [B,T,L,L,44]

        else:
            xyz_t = torch.zeros_like(xyz_t)
            t2d = torch.zeros(B,1,L,L,68)
        t2d = t2d.to(self.device)

        ##################################
        ######## Seq Self Cond ###########
        ##################################
        if self.seq_self_cond:
            if t < self.diffuser.T:
                ic('Providing Seq Self Cond')
        
                t1d[:,:,:,:20] = self.prev_seq_pred # [B,T,L,d_t1d]
                t1d[:,:,:,20]  = 0 # Setting mask token to zero
        
            else:
                t1d[:,:,:,:21] = 0

        if self.symmetry is not None:
            idx_pdb, self.chain_idx = self.symmetry.res_idx_procesing(res_idx=idx_pdb)

        with torch.no_grad():
            px0=xt_in
            seq_scalar = torch.argmax(seq_in, dim=-1)
            mask_t_2d = torch.ones(1,1,L,L).bool().to(self.device)
            for rec in range(self.recycle_schedule[t-1]):
                alpha_prev = torch.zeros((B,L,rf2aa.chemical.NTOTALDOFS,2)).to(self.device, non_blocking=True)
                ic(px0.shape, xyz_t.shape)
                ic(px0[...,:3,:].mean((0,1,2)))
                if self_cond:
                    ic(xyz_t[...,1,:].mean((0,1,2)))
                msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt = self.model(
                                    msa_masked,
                                    msa_full,
                                    seq_scalar,
                                    seq_scalar,
                                    px0,
                                    alpha_prev,
                                    idx_pdb,
                                    bond_feats=self.bond_feats,
                                    chirals=self.chirals,
                                    atom_frames=self.atom_frames,
                                    t1d=t1d,
                                    t2d=t2d,
                                    xyz_t=xyz_t[...,1,:],
                                    alpha_t=alpha_t,
                                    mask_t=mask_t_2d,
                                    same_chain=self.same_chain,
                                    msa_prev = None,
                                    pair_prev = None,
                                    state_prev = None,
                                    return_infer=True,
                                    # check valence of is_motif
                                    is_motif=self.diffusion_mask[0].to(self.device))

                if self.symmetry is not None and self.inf_conf.symmetric_self_cond:
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

        # self.prev_seq_pred = torch.clone(logits.squeeze()[:,:20])
        self.prev_seq_pred = torch.zeros(L, NAATOKENS)
        self.prev_seq_pred[:,0] = 1.0
        self.prev_pred = torch.clone(px0)

        # prediction of X0
        _, px0  = self.allatom(torch.argmax(seq_in, dim=-1), px0, alpha)
        px0    = px0.squeeze()[:,:14]

        if self.seq_diffuser is None:
            # pseq0 = None
            # Default method of decoding sequence
            seq_probs   = torch.nn.Softmax(dim=-1)(logits.squeeze()/self.inf_conf.softmax_T)
            sampled_seq = torch.multinomial(seq_probs, 1).squeeze() # sample a single value from each position

            pseq_0 = torch.nn.functional.one_hot(
                sampled_seq, num_classes=rf2aa.chemical.NAATOKENS).to(self.device).float()

            pseq_0[self.mask_seq.squeeze()] = seq_init[self.mask_seq.squeeze()].to(self.device) # [L,22]
        else:
            # Sequence Diffusion
            pseq_0 = logits.squeeze()
            pseq_0 = pseq_0[:,:20]

            pseq_0[self.mask_seq.squeeze()] = seq_init[self.mask_seq.squeeze(),:20].to(self.device)

            sampled_seq = torch.argmax(pseq_0, dim=-1)

        self._log.info(
                f'Timestep {t}, current sequence: { rf2aa.chemical.seq2chars(torch.argmax(pseq_0, dim=-1).tolist())}')

        if t > final_step:
            x_t_1, seq_t_1, tors_t_1, px0 = self.denoiser.get_next_pose(
                xt=x_t,
                px0=px0,
                t=t,
                diffusion_mask=self.mask_str.squeeze(),
                seq_diffusion_mask=self.mask_seq.squeeze(),
                seq_t=seq_t,
                pseq0=pseq_0,
                diffuse_sidechains=self.preprocess_conf.sidechain_input,
                align_motif=self.inf_conf.align_motif,
                include_motif_sidechains=self.preprocess_conf.motif_sidechain_input
            )
            self._log.info(
                    f'Timestep {t}, input to next step: { rf2aa.chemical.seq2chars(torch.argmax(seq_t_1, dim=-1).tolist())}')
        else:
            x_t_1 = torch.clone(px0).to(x_t.device)
            seq_t_1 = pseq_0

            # Dummy tors_t_1 prediction. Not used in final output.
            tors_t_1 = torch.ones((self.mask_str.shape[-1], 10, 2))
            px0 = px0.to(x_t.device)

        if self.symmetry is not None:
            x_t_1, seq_t_1 = self.symmetry.apply_symmetry(x_t_1, seq_t_1)

        return px0, x_t_1, seq_t_1, tors_t_1, plddt
        # return px0, x_t_1, seq_t_1, tors_t_1, None

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
