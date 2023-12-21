import copy
import torch
from assertpy import assert_that
import numpy as np
from omegaconf import DictConfig, OmegaConf
import data_loader
from icecream import ic
import pickle 

import rf2aa.chemical
from rf2aa.chemical import NAATOKENS, MASKINDEX, NTOTAL, NHEAVY, NHEAVYPROT, NHEAVYNUC, DNAMASKINDEX, RNAMASKINDEX
import rf2aa.util
import rf2aa.data_loader
# from rf2aa.util_module import ComputeAllAtomCoords
from rf2aa.util_module import XYZConverter
from rf2aa.RoseTTAFoldModel import RoseTTAFoldModule
from rf2aa.kinematics import xyz_to_c6d, c6d_to_bins, xyz_to_t2d, get_chirals
import rf2aa.parsers
import rf2aa.tensor_util
from rf2aa.Track_module import update_symm_Rs
import aa_model
import dataclasses
import copy

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from kinematics import get_init_xyz
from diffusion import Diffuser
import seq_diffusion
from contigs import ContigMap, DNA_Duplex_Protein_Monomer
from inference import utils as iu
from potentials.manager import PotentialManager
from inference import symmetry
# from inference import motif_manager
import logging
import torch.nn.functional as nn
import util
import hydra
from hydra.core.hydra_config import HydraConfig
import os
import matplotlib.pyplot as plt 
from memory import mem_report
# from pdb import set_trace
from pdb import set_trace
REPORT_MEM=False

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

    def __init__(self, conf: DictConfig, preloaded_ckpts={}, prebuilt_models={}):
        """Initialize sampler.
        Args:
            conf: Configuration.
        """
        self.initialized = False
        self.preloaded_ckpts = preloaded_ckpts
        self.prebuilt_models = prebuilt_models
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


        # If we want to use polymer adjacency conditioning 
        # ("secondary structure" close contacts, following the NA definition)
        if conf.scaffoldguided.scaffoldguided is True:
            print('USING SS CONDITIONING MODEL: ')
            self.use_ss_guidance = True
            self.delta_dim_t2d = 3
        else:
            self.use_ss_guidance = False
            self.delta_dim_t2d = 0


        self.index_map_dict, self.length_init = iu.get_index_map_dict(self._conf.contigmap.contigs)

        # Set up ss-adj matrix
        # This will be used for t2d conditioning:
        if self.use_ss_guidance:

            # By default fall back to masking everything:
            # We initialize the ss_matrix as fully masked, then we modify it as we go
            self.target_ss_matrix = (2*torch.ones((self.length_init,self.length_init))).long()

            # first priority: check for basepair range specifications (best way to specify):
            if self._conf.scaffoldguided.target_ss_pairs is not None:
                # Here we add paired regions to the target ss matrix
                self.target_ss_matrix = iu.ss_pairs_to_matrix(
                                                    self._conf.scaffoldguided.target_ss_pairs, 
                                                    self.index_map_dict, 
                                                    self.target_ss_matrix,
                                                    ss_pair_ori_list=self._conf.scaffoldguided.target_ss_pair_ori,
                                                    ).long()

            # second priority: check for ss-string specification (slightly less precise):
            elif self._conf.scaffoldguided.target_ss_string is not None:
                # Or just replace it with full ss string
                self.target_ss_matrix = torch.from_numpy(iu.sstr_to_matrix(self._conf.scaffoldguided.target_ss_string, only_basepairs=True)).long()

            # Now we can add force loops, or triple+ contacts
            # Force loops:
            if self._conf.scaffoldguided.force_loops_list is not None:
                self.target_ss_matrix = iu.force_loops(self._conf.scaffoldguided.force_loops_list, self.index_map_dict, self.target_ss_matrix)

            # Force the multi-contact:
            if self._conf.scaffoldguided.force_multi_contacts is not None:
                self.target_ss_matrix = iu.force_multi_contacts(self._conf.scaffoldguided.force_multi_contacts, self.index_map_dict, self.target_ss_matrix)

            



            # # otherwise fall back to masking everything:
            # else:
            #     self.target_ss_matrix = (2*torch.ones(indep.same_chain.shape)).long()

            # Save the matrix image if we want:
            # if self._conf.scaffoldguided.save_ss_matrix_png and (t==self._conf.diffuser.T):
            if self._conf.scaffoldguided.save_ss_matrix_png:
                print('SAVING SS MATRIX PIC!')
                output_pic_filapath = self._conf.inference.output_prefix+'.png'
                output_dirpath = os.path.dirname(output_pic_filapath)

                if not (os.path.exists(output_dirpath) and os.path.isdir(output_dirpath)):
                    os.mkdir(output_dirpath)

                fig, ax = plt.subplots(1,1,figsize=(5,5), dpi=300)
                ax.imshow(np.array(self.target_ss_matrix))
                plt.tight_layout()
                plt.savefig(output_pic_filapath, bbox_inches='tight', dpi='figure')
                plt.close()

        else:
            self.target_ss_matrix = None




        # If we want to show which polymer class is at each position
        # Should really always be true, seriously
        # TO DO: add a fourth category for small molecules, maybe another for metals or someshit
        if conf.preprocess.show_poly is True:
            print('SHOWING POLYMER CLASS: ')
            self.show_poly_class = True
            self.delta_dim_t1d = 3
        else:
            self.show_poly_class = False
            self.delta_dim_t1d = 0


        if needs_model_reload:
            # Load checkpoint, so that we can assemble the config
            if self.preloaded_ckpts.get(self.ckpt_path, False):
                print('******* Using preloaded model for ', self.ckpt_path, ' *********')
                self.model = self.prebuilt_models[self.ckpt_path]
                self.ckpt = self.preloaded_ckpts[self.ckpt_path]
                self.assemble_config_from_chk()
            else:
                print('******* Loading model for ', self.ckpt_path, ' from disk *********')
                self.load_checkpoint()
                self.assemble_config_from_chk()
                # Now actually load the model weights into RF
                self.model = self.load_model()

                # add the model to the prebuilt models
                self.prebuilt_models[self.ckpt_path] = self.model
                self.preloaded_ckpts[self.ckpt_path] = self.ckpt  # Now we have access to these in run_inference.py 
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
        self.model_adaptor = aa_model.Model(self._conf)
        # Temporary hack
        self.model.assert_single_sequence_input = True
        self.model_adaptor.model = self.model
        # DJ additions 
        self.cur_rigid_tmplt = None 

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


        # self.allatom = ComputeAllAtomCoords().to(self.device)
        self.converter = XYZConverter() 

        if self.diffuser_conf.partial_T:
            assert self.diffuser_conf.partial_T <= self.diffuser_conf.T
            self.t_step_input = int(self.diffuser_conf.partial_T)
        else:
            self.t_step_input = int(self.diffuser_conf.T)

        # Set up motif-fitting stuffs
        self.rfmotif = None
        # if self.inf_conf.rfmotif:
        #     self.rfmotif = motif_manager.create_motif_manager(self._conf, device=self.device)
        #     # self.motif_fit_step_list = [50,30,20,15,10,6,3,1]
        #     if self._conf['rfmotif']['fit_at_t']:
        #         self.t_motif_fit = int(self._conf['rfmotif']['fit_at_t'])
        #     else:
        #         self.t_motif_fit = int(self.t_step_input)

        #     if self._conf['rfmotif']['motif_fit_tsteps']:
        #         self.motif_fit_tsteps = [int(t_i) for t_i in self._conf['rfmotif']['motif_fit_tsteps'][0].split(',') ]
        #     else:
        #         self.motif_fit_tsteps = [int(t_i) for t_i in range(1,self.t_motif_fit+1)]




        # Set up target feature stuffs
        self.target_feats = iu.process_target(self.inf_conf.input_pdb, parse_hetatom=False, center=False, inf_conf=self.inf_conf)
        self.chain_idx = None


        # # Set up ss-adj t2d conditioning:
        # if conf.scaffoldguided.scaffoldguided is True:
        #     print(f'    using ss-adjacency (basepair) string: {conf.scaffoldguided.target_na_ss}')
        #     print(f'    setting d_t2d to 72')
        #     self.full_ss_cond_d_t2d = 72
            
        #     if (conf.scaffoldguided.target_na_ss is not None):
        #         print(f'    using ss-adjacency (basepair) string: {conf.scaffoldguided.target_na_ss}')
        #         self.target_ss_matrix = torch.from_numpy(iu.sstr_to_matrix(conf.scaffoldguided.target_na_ss, only_basepairs=True)).long()

        #     else:
        #         ipdb.set_trace()
        #         print(f'    masking full ss-adjacency matrix')
        #         self.target_ss_matrix = (2*torch.ones(rfi_tp1_t[0].t2d.shape[2:4])).long()

        # else:
        #     self.target_ss_matrix = None

        # if conf.scaffoldguided.scaffoldguided is True:
        #     self.use_ss_guidance = True
        # else:
        #     self.use_ss_guidance = False


        # Set up ppi design stuffs
        if self.inf_conf.ppi_design and self.inf_conf.autogenerate_contigs:
            self.ppi_conf.binderlen = ''.join(chain_idx[0] for chain_idx in self.target_feats['pdb_idx']).index('B')


        # Set up potentials stuffs
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
            overrides = list( copy.deepcopy(HydraConfig.get().overrides.task ))

            if self._conf.inference.overrides:
                overrides.extend(self._conf.inference.overrides)

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
                    
                    if mytype == bool: 
                        # special treatment for bools because they are strings in override 
                        self._conf[override.split(".")[0]][override.split(".")[1].split("=")[0]] = override.split("=")[1].lower().strip() == 'true'
                    else:
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

        # other params
        binder_net = self._conf.inference.two_template

        # HACK: TODO: save this in the model config
        self.loss_param = {'lj_lin': 0.75}
        model = RoseTTAFoldModule(
            # symmetrize_repeats=None, 
            # repeat_length=None,
            # symmsub_k=None,
            # sym_method=None,
            # main_block=None,
            # copy_main_block_template=None,
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
            binder_net=binder_net).to(self.device)
        
        if self._conf.logging.inputs:
            pickle_dir = pickle_function_call(model, 'forward', 'inference', minifier=aa_model.minifier)
            print(f'pickle_dir: {pickle_dir}')
        model = model.eval()
        self._log.info(f'Loading checkpoint.')
        
        
        
        if not self._conf.inference.zero_weights:
            print('*'*80)
            print('LOADING MODEL WEIGHTS')        
            # Lenient loading with custom feedback
            for name, param in model.named_parameters():
                state_dict = self.ckpt['model_state_dict']
                if name in state_dict:
                    param_shape = param.shape
                    state_shape = state_dict[name].shape
                    if param_shape != state_shape:
                        print(f"Model Load Warning: Parameter '{name}' shape mismatch: Model has {param_shape}, State dict has {state_shape}")
                else:
                    print(f"Model Load Warning: Parameter '{name}' not found in state dict")

            model.load_state_dict(self.ckpt['model_state_dict'], strict=False)
        return model

    def construct_contig(self, target_feats):
        """Create contig from target features."""
        if self.inf_conf.ppi_design and self.inf_conf.autogenerate_contigs:
            seq_len = target_feats['seq'].shape[0]
            self.contig_conf.contigs = [f'{self.ppi_conf.binderlen}',f'B{self.ppi_conf.binderlen+1}-{seq_len}']
        self._log.info(f'Using contig: {self.contig_conf.contigs}')
        
        if self.inf_conf.refine: 
            L = len(self.target_feats['seq'].squeeze())
            self.contig_conf['contigs'] = [f'{L}-{L}']

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
        denoise_kwargs.pop('eucl_type')
        return iu.Denoise(**denoise_kwargs)
    def mask_indep(self, indep):

        ipdb.set_trace()

        return indep
    def sample_init(self, return_forward_trajectory=False):
        """Initial features to start the sampling process.
        
        Modify signature and function body for different initialization
        based on the config.
        
        Returns:
            xt: Starting positions with a portion of them randomly sampled.
            seq_t: Starting sequence with a portion of them set to unknown.
        """
        print('Two template option is ', self.inf_conf.two_template)

        # moved this here as should be updated each iteration of diffusion
        self.contig_map = self.construct_contig(self.target_feats)

        # indep = self.model_adaptor.make_indep(self._conf.inference.input_pdb, self._conf.inference.ligand, inference_config=self._conf.inference, contig_map=self.contig_map)
        indep = self.model_adaptor.make_indep(self._conf.inference.input_pdb, self._conf.inference.ligand)

        # check for subsymm template and add to indep if present
        if self.inf_conf.subsymm_template:
            indep.subsymm_seq        = self.target_feats['subsymm_seq'].to(self.device)
            indep.subsymm_xyz        = self.target_feats['subsymm_xyz'].to(self.device)
            indep.mask_t_2d_subsymm  = self.target_feats['mask_2d_subsymm'].to(self.device) if torch.is_tensor(self.target_feats['mask_2d_subsymm']) else None

        is_partial = self.diffuser_conf.partial_T is not None

        # ipdb.set_trace()
        # TEST TEST TEST
        # indep, is_diffused = DNA_Duplex_Protein_Monomer(indep, self.contig_conf, self.target_feats['pdb_idx'])
        # TEST TEST TEST

        indep, is_diffused = self.model_adaptor.insert_contig(indep, self.contig_map, partial_T=is_partial) 
        # ipdb.set_trace()
        

        # create a residue mask based on polymer type:
        # I think we want this to be on the gpu
        self.polymer_mask = torch.zeros(1, indep.seq.shape[0], NAATOKENS).to(self.device)
        self.polymer_mask[0, indep.is_protein,  0:22] = 1
        self.polymer_mask[0, indep.is_dna,     22:27] = 1
        self.polymer_mask[0, indep.is_rna,     27:32] = 1


        # set attribute for polymer mask tokens, and check that all positions are accounted for:
        # I think we want this to be on the cpu
        self.polymer_mask_resi = torch.full_like(is_diffused, -1, dtype=indep.seq.dtype)
        self.polymer_mask_resi[indep.is_protein] = MASKINDEX
        self.polymer_mask_resi[indep.is_dna] = DNAMASKINDEX
        self.polymer_mask_resi[indep.is_rna] = RNAMASKINDEX
        assert ~(self.polymer_mask_resi < 0).any(), "SOME POSITIONS DONT HAVE A VALID POLYMER TYPE ASSIGNMENT"
        

        # Control how many atoms we save in the pdb
        self.num_atoms_saved = self.inf_conf.num_atoms_saved
        
        

        
        self.is_diffused = is_diffused
        
        # if self.diffuser_conf.partial_T:
        #     # raise Exception('not implemented')

        # Diffuse the contig-mapped coordinates 
        if self.diffuser_conf.partial_T:
            assert self.diffuser_conf.partial_T <= self.diffuser_conf.T
            self.t_step_input = int(self.diffuser_conf.partial_T)
        else:
            self.t_step_input = int(self.diffuser_conf.T)


        t_list = np.arange(1, self.t_step_input+1)
        
        # save coordinates at this step just to double check
        # tmp_outdir = '/home/davidcj/projects/rf_diffusion_allatom/rf_diffusion/inputs/tmp/'
        # fp1 = os.path.join(tmp_outdir, 'indep_crds_before_diffusion.pdb')
        # util.writepdb(fp1, indep.xyz[:,:14,:], indep.seq)

        # alter the diffusion mask to diffuse everything if we have a symm_template 
        if self.inf_conf.subsymm_template is not None:
            print('Detected symmetric template - diffusing all atoms')

            old_is_diffused = is_diffused.clone() # save old mask 

            is_diffused = torch.ones_like(is_diffused)
            self.is_diffused = is_diffused
            # need to reset according to new is diffused mask 
            # indep.seq[self.is_diffused] = 21 # set any residues allowed to diffuse to masked
            indep.seq = torch.where(self.is_diffused, self.polymer_mask_resi, indep.seq)

            # create tensor denoting which residues should have perfect confidence
            # even though they may technically be diffused (moving)
            has_imperfect_t1d = old_is_diffused.clone()
            self.has_imperfect_t1d = has_imperfect_t1d

            # if rigid_symm_motif, don't diffuse it with diffuser but still allow movement with denoiser
            if (self.inf_conf.rigid_symm_motif) or (self.inf_conf.initial_rigid_motif):
                diffuser_is_diffused = torch.clone(old_is_diffused)
            else:
                # not rigid motif, diffuse all 
                diffuser_is_diffused = self.is_diffused.clone()
            
            self.diffuser_is_diffused = diffuser_is_diffused
        
        elif self.inf_conf.rigid_repeat_motif:
            print('Detected rigid repeat motif args:')
            print('Forward diffusing non-motif, reverse diffusing all.')
            # doing rigid drifting motif repeat scaffolding 

            old_is_diffused = is_diffused.clone()

            # denoiser sees everything as diffused (i.e. can move)
            is_diffused_denoiser = torch.ones_like(is_diffused)
            self.is_diffused = is_diffused_denoiser

            # need to reset according to new is diffused mask 
            # indep.seq[self.is_diffused] = 21 # set any residues allowed to diffuse to masked
            indep.seq = torch.where(self.is_diffused, self.polymer_mask_resi, indep.seq)

            # diffuser sees the motif as not diffused (i.e. can't move)
            # just for initialization 
            diffuser_is_diffused = torch.clone(old_is_diffused)
            self.diffuser_is_diffused = diffuser_is_diffused

        elif self.inf_conf.motif_only_2d:
            print('Detected motif only 2d option')
            print('Foward/reverse diffuse everything.')
            assert self._conf.inference.two_template and self._conf.inference.three_template

            self.is_diffused_orig = is_diffused.clone() # keep this for later 

            # denoiser sees all as being reverse diffused
            is_diffused_denoiser = torch.ones_like(is_diffused)
            self.is_diffused = is_diffused_denoiser
            # indep.seq[self.is_diffused] = 21

            if self.inf_conf.supply_motif_seq:
                indep.seq = torch.where(self.is_diffused_orig, self.polymer_mask_resi, indep.seq)
            else:
                indep.seq = torch.where(self.is_diffused, self.polymer_mask_resi, indep.seq)

            # diffuser will also diffuse everything
            diffuser_is_diffused = torch.clone(is_diffused_denoiser)
            self.diffuser_is_diffused = diffuser_is_diffused
        
        else:
            self.is_diffused = is_diffused
            diffuser_is_diffused = self.is_diffused.clone()

        atom_mask = None
        seq_one_hot = None
        # center_crds = not (self._conf.inference.internal_sym is not None) # don't center coords if doing symmetry
        center_crds = True # DJ- new version centers the particle at origin

        if self._conf.inference.internal_sym is not None:
            symmids, symmRs, symmeta, offset = symmetry.get_pointsym_meta(self._conf.inference.internal_sym)
        else:
            symmids, symmRs, symmeta, offset = None, None, None, None

        if not self.inf_conf.start_from_input:
            fa_stack, aa_masks, xyz_true = self.diffuser.diffuse_pose(
                indep.xyz,
                seq_one_hot,
                atom_mask,
                indep.is_sm,
                diffusion_mask=~diffuser_is_diffused,
                t_list=t_list,
                diffuse_sidechains=self.preprocess_conf.sidechain_input,
                include_motif_sidechains=self.preprocess_conf.motif_sidechain_input,
                center_crds=center_crds,
                symmRs=symmRs,
                motif_only_2d=self.inf_conf.motif_only_2d)
            
            # xT = fa_stack[-1].squeeze()[:,:14,:]
            xT = fa_stack[-1].squeeze()[:,:NHEAVY,:]
            # xT = fa_stack[-1].squeeze()[:,:self.inf_conf.num_atoms_modeled,:]
            xt = torch.clone(xT)
            indep.xyz = xt
        
        else:
            print('Starting from input coordinates instead of diffusing')
            # user wants to start from input coordinates - presumably already diffused
            aa_masks = None
            fa_stack = None
            xyz_true = None

            # xT = indep.xyz[:,:14,:]
            xT = indep.xyz[:,:NHEAVY,:]
            # xT = indep.xyz[:,:self.inf_conf.num_atoms_modeled,:]
            xt = torch.clone(xT) 
            indep.xyz = xt
    

        # # now save again after diffusion 
        # fp2 = os.path.join(tmp_outdir, 'indep_crds_after_diffusion.pdb')
        # util.writepdb(fp2, indep.xyz[:,:14,:], indep.seq)
        # sys.exit('Exiting early')

        if self.diffuser_conf.partial_T and self.seq_diffuser is None:
            is_motif = ~is_diffused 
            is_shown_at_t = torch.full_like(is_motif, False)
            visible = is_motif | is_shown_at_t
            if self.diffuser_conf.partial_T:
                # seq_t[visible] = seq_orig[visible]
                assert 0, 'NEED TO MODIFY TO INCLUDE NA TOKENS!'
                indep.seq = torch.full_like(indep.seq, 20)
        else:
            # Sequence diffusion
            visible = ~is_diffused

        self.denoiser = self.construct_denoiser(len(self.contig_map.ref), visible=visible)
        

        # symmetrize the inputs 
        self.symmids, self.symmRs, self.symmeta, self.cur_symmsub = None,None,None,None
        if self.symmetry is not None:
            assert self._conf.inference.internal_sym is None, 'cannot use both new (inference.internal_sym) and classic (inference.symmetry) symmetry simultaneously'
            # classic version
            xt, seq_t = self.symmetry.apply_symmetry(indep.xyz, indep.seq)
            
        # propogates the diffused system symmetrically 
        elif self._conf.inference.internal_sym is not None:
            assert self.symmetry is None, 'Cannot use both new (inference.internal_sym) and classic (inference.symmetry) symmetry simultaneously' 
            # new version, minimal representation of subunits 
            # find rotation matrices/metadata for symmetry 
            # symmids, symmRs, symmeta, offset = symmetry.get_pointsym_meta(self._conf.inference.internal_sym) # dj - moved this to above

            # if partial_T, offset should be directly opposite of the vector from 
            # the center of mass to the axis of symmetry 
            if self.diffuser_conf.partial_T and 'c' in self._conf.inference.internal_sym.lower():
                offset  = None 
                com     = torch.mean(indep.xyz[:,1:2,:], dim=0, keepdim=True)
                proj_xy = com - com[...,-1] # project onto xy plane by subtracting z coord
                print('WARNING: OFFSET ASSUMES SYMMETRY AXIS IS ALIGNED WITH Z AXIS')
                offset  = proj_xy / torch.norm(offset, dim=-1, keepdim=True) # normalize

            
            # check if motif scaffolding with rigid particle size 
            if self.is_diffused.sum() != len(self.is_diffused.flatten()):
                print('Detected motif scaffolding from contigs, offset is in the direction of the motif COM')
                # offset should be toward the COM of the motif
                offset = None 
                motif_com = torch.mean(indep.xyz[self.is_diffused], dim=0, keepdim=True)
                norm = torch.norm(motif_com, dim=-1, keepdim=True)
                offset = motif_com / norm

            # Check if C2/3/5 template going into I -- if True, offset in direction of first chain in template 
            if self.inf_conf.subsymm_template is not None:
                cond_a = self.inf_conf.subsymm_template is not None
                cond_b = self._conf.inference.internal_sym.lower() in ['i','icos','icosahedral']
                cond_c = self.target_feats['subsymm_symbol'].lower() in ['c2','c3','c5']

                if cond_a and cond_b and cond_c:
                    print('Detected C2/3/5 template going into I symmetry, offset is in the direction of the first chain in the template')
                    offset = None 
                    
                    # Offset combins two things
                    # 1. offset toward center of mass of first chain in template
                    # 2. offset away from origin in the direction of sym ax of subsymm template

                    # (1)
                    tmplt_xyz    = self.target_feats['subsymm_xyz']
                    tmplt_lasu   = self.target_feats['subsymm_lasu']
                    tmplt_xyz_A  = tmplt_xyz[:tmplt_lasu] # first chain of template
                    com_A        = torch.mean(tmplt_xyz_A[:,1], dim=0)
                    offset_chA   = com_A / torch.norm(com_A, dim=-1, keepdim=True)

                    # (2)
                    MAGIC_AXIS_OFFSET_SCALE = 3
                    tmplt_axis   = self.target_feats['subsymm_axis']
                    offset_tmplt = tmplt_axis / torch.norm(tmplt_axis, dim=-1) * MAGIC_AXIS_OFFSET_SCALE

                    # combine
                    offset = offset_chA + offset_tmplt


            # scale offset w.r.t ASU length 
            Lasu = indep.xyz.shape[0]
            self.Lasu = Lasu
            if 'c' in self._conf.inference.internal_sym.lower():
                offset *= (Lasu**(1/2))
            else:
                offset *= (Lasu**(1/3))

            # scale offset manually 
            offset *= self._conf.inference.offset_scale 
            indep.xyz[self.is_diffused] = indep.xyz[self.is_diffused] + offset
            
            # this is the step that duplicates starting coordinates 
            indep, symmsub  = symmetry.find_minimal_neighbors(indep, symmRs, symmeta)

            
            # for passing to RF fwd pass in self.sample_step()
            self.symmids        = symmids.to(self.device)
            self.symmRs         = symmRs.to(self.device)
            self.symmeta        = [[symmeta[0][0].to(self.device)], symmeta[1]]
            self.cur_symmsub    = symmsub.to(self.device)

            print('ENTERED SYMMETRY MODE*****************')

            # Now alter self.is_diffused to match new shapes 
            nneigh = len(symmsub)
            self.is_diffused = self.is_diffused.repeat(nneigh) # copy is_diffused for each subunit 

        # repeat proteins
        elif self._conf.model.symmetrize_repeats:
            Lasu     = self._conf.model.repeat_length 
            assert indep.xyz.shape[0] % Lasu == 0, 'Lasu must be a factor of the number of tokens but found %d and %d' % (Lasu, indep.xyz.shape[0])

            # indep = symmetry.propogate_repeat_features(indep, Lasu, main_block=self._conf.model.main_block)
            indep = symmetry.propogate_repeat_features2(indep, Lasu, self._conf.inference)
            # self.denoiser.decode_scheduler.visible[indep.is_sm] = True # all sm are visible/not diffused

            self.is_diffused = self.is_diffused.repeat(self._conf.inference.n_repeats)
            self.is_diffused_orig = self.is_diffused_orig.repeat(self._conf.inference.n_repeats)
        
        if return_forward_trajectory:
            forward_traj = torch.cat([xyz_true[None], fa_stack[:,:,:]])
            if self.seq_diffuser is None:
                # aa_masks[:, diffusion_mask.squeeze()] = True
                # return xt, forward_traj
                return indep, forward_traj
            else:
                raise Exception('not implemented')
                # Seq Diffusion
                return xt, seq_t, forward_traj, diffused_seq_stack, seq_orig
        
        self.msa_prev = None
        self.pair_prev = None
        self.state_prev = None
        
        # ic(indep.xyz.shape)
        # assert False
        print('Total AA modeled: ', indep.xyz.shape[0])
        # ipdb.set_trace()
        return indep

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
    def get_full_chunk_idx0(self, indep, con_hal_idx0):
        """
        Just a way to check for the presence of motif chunks,
        and potentially correct ij_visible param, in case we want unconditional.

        """
        abet = 'abcdefghijklmnopqrstuvwxyz'
        ### do we want to split chunks at chain breaks?
        if self.inf_conf['chainbreak_chunks']:
            full_complex_idx0 = torch.tensor(self.contig_map.rf, dtype=torch.int64)
        else:
            full_complex_idx0 = None

        # AF : break regions into the start stop indices based on both template breaks and chain breaks
        # this just gets the breaks by mask regions between motifs
        mask_breaks = iu.get_breaks(con_hal_idx0)
        templ_range_inds = iu.find_template_ranges(con_hal_idx0, return_inds=False)

        # add the breaks from the chain jumps
        if full_complex_idx0 is not None:
            chain_breaks = iu.get_breaks(full_complex_idx0)
            chain_range_inds = iu.find_template_ranges(full_complex_idx0, return_inds=True)
            # merge these into a list of sub-chunk tuples for template region locations
            chunk_range_inds = iu.merge_regions(templ_range_inds, chain_range_inds)
        else:
            chunk_range_inds = templ_range_inds

        # now we have the complete con_hal_idx0 including templates that are separated by chain breaks!
        true_con_hal_idx0 = torch.tensor([ind for start,end in chunk_range_inds for ind in range(start,end+1)])

        # Update ij_visible:
        if self._conf.inference.ij_visible is None:
            self._conf.inference.ij_visible = abet[:len(chunk_range_inds)]

        return true_con_hal_idx0, full_complex_idx0
        



    def _get_3template_masks(self, indep):
        """
        Gets is_protein_motif and t2d_is_revealed for 3template inference
        """

        # if indep.metadata.get('refinement'):
            # refine = True
            # ref_dict = indep.metadata['refinement']
        # else:
            # refine = False
        refine = False

        con_hal_idx0 = torch.from_numpy( self.contig_map.get_mappings()['con_hal_idx0'] )


        # Assume that SM input will always be motif!! 
        if indep.is_sm.any() and not refine:
            print('Detected small molecule in input - assuming it is a motif chunk.')
            # where is sm in hal? 
            where_is_sm = torch.where(indep.is_sm)[0]
            # add it to con_hal_idx0
            con_hal_idx0 = torch.cat([con_hal_idx0, where_is_sm], dim=0)

        is_protein_motif = ~indep.is_sm * ~self.is_diffused_orig 

        # Update con_hal_idx0 to include chain breaks
        # ( this also conveniently locks all motifs in ij_visible if it has a null value )
        if torch.any(is_protein_motif):
            con_hal_idx0, full_complex_idx0 = self.get_full_chunk_idx0(indep, con_hal_idx0)

        if refine: 
            # we can rely on src_con_hal to tell us where in THIS hal the motif goes 
            src_con_hal_idx0 = torch.from_numpy( ref_dict['con_hal_idx0'] )
            # src_con_ref_idx0 = torch.from_numpy( ref_dict['src_con_ref_idx0'] )
            
            assert is_protein_motif.sum() == 0
            is_protein_motif[src_con_hal_idx0] = True 
            con_hal_idx0 = src_con_hal_idx0



        if not torch.any(is_protein_motif):
            # no motifs, blank masks 
            L = len(is_protein_motif)
            mask_t2d = torch.zeros((L,L))
            return is_protein_motif, mask_t2d

        ### is_protein_motif ###
        ########################
        abet = 'abcdefghijklmnopqrstuvwxyz'
        abet = [a for a in abet]
        abet2num = {a:i for i,a in enumerate(abet)} 

        
        if self._conf.inference.motif_only_2d: 
            # # the entire protein is diffused
            # # trying to reconstruct motif from 2d only 

            if not self._conf.model.symmetrize_repeats:
                
                # asymmetric case 
                is_protein_motif = ~indep.is_sm * ~self.is_diffused_orig
                if refine: 
                    is_protein_motif[src_con_hal_idx0] = True

                is_motif = is_protein_motif.clone() | indep.is_sm # Assumes any small molecule is a motif chunk

                # t2d_is_revealed
                L = len(is_protein_motif)
                mask_t2d = torch.zeros((L,L))

                # User can use ij_visible argument
                ij_visible = self._conf.inference.ij_visible
                if refine: 
                    ij_visible = ref_dict['ij_visible']
                    # if we had ligand, remove last character from ij_visible 
                    if ref_dict['ligand']: 
                        print('WARNING: Popping detected ligand chunk from reference ij_visible')
                        ij_visible = ij_visible[:-1]

                assert ij_visible is not None, '3 template + motif_only_2d requires description of motif pairwise visibility'
                ij_visible = ij_visible.split('-') # e.g., [abc,de,df,...]
                ij_visible_int = [tuple([abet2num[a] for a in s]) for s in ij_visible]
                # mask_t2d, _ = iu.get_repeat_t2d_mask(L, con_hal_idx0, ij_visible_int, 1, supplied_full_contig=True)
                # mask_t2d, _ = iu.get_repeat_t2d_mask(L, con_hal_idx0, ij_visible_int, 1, full_complex_idx0, self.contig_map, supplied_full_contig=True)
                mask_t2d, _ = iu.get_repeat_t2d_mask(L, con_hal_idx0, self.contig_map, ij_visible_int, 1, supplied_full_contig=True)

            else:
                # repeat/symmetric case
                assert not refine, 'refine not yet implemented for symmetry/repeat' 
                assert type(self._conf.inference.n_repeats) is int        # must be present 
                is_protein_motif = ~indep.is_sm * ~self.is_diffused_orig  # should be appropriate length 


                if is_protein_motif.sum() == len(con_hal_idx0):
                    supplied_full_contig = True
                    print('Detected full contig supplied--------------')
                else: 
                    print('Detected ASU contig supplied--------------')
                    supplied_full_contig = False


                ### t2d_is_revealed ###
                n_repeat = self._conf.inference.n_repeats
                L = len(is_protein_motif)
                mask_t2d = iu.parse_ij_get_repeat_mask(self._conf.inference.ij_visible, L, n_repeat, con_hal_idx0, supplied_full_contig, full_complex_idx0)



                is_motif = is_protein_motif.clone() | indep.is_sm # Assumes any small molecule is a motif chunk
        else: 
            raise Exception('3D motif not implemented yet')
            # non-motif is diffused, motif given in 3d  
            assert self._conf.model.symmetrize_repeats, 'assumes repeat protein inferences for now'
            is_protein_motif = ~indep.is_sm * ~self.diffuser_is_diffused.repeat(self._conf.inference.n_repeats)

            ### t2d_is_revealed ###
            n_repeat = self._conf.inference.n_repeats
            L = len(is_protein_motif)
            mask_t2d = iu.parse_ij_get_repeat_mask(self._conf.inference.ij_visible, L, n_repeat, con_hal_idx0)

        # return is_protein_motif, mask_t2d
        return is_motif, mask_t2d

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

        twotemplate   = self.inf_conf.two_template
        threetemplate = self.inf_conf.three_template

        if self.rfmotif and (t in self.motif_fit_tsteps): 
            rfmotif = self.rfmotif
        else:
            rfmotif = None


        if (twotemplate and threetemplate):
            is_motif, t2d_is_revealed = self._get_3template_masks(indep)
        else:
            is_motif, t2d_is_revealed = None,None 

        if (self.inf_conf.t2d_pic_filename) and (t==self._conf['diffuser']['T']):
            if self.inf_conf.t2d_pic_filename.endswith('.png'):
                out_pic_name = self.inf_conf.t2d_pic_filename
            else:
                out_pic_name = self.inf_conf.t2d_pic_filename + '.png'

            fig, ax = plt.subplots(1,1,figsize=(5,5), dpi=300)
            ax.imshow(t2d_is_revealed)
            plt.tight_layout()
            # plt.show()
            plt.savefig(out_pic_name, bbox_inches='tight', dpi='figure')
            plt.close()



        # # Set up ss-adj matrix
        # # This will be used for t2d conditioning:
        # if self.use_ss_guidance:

        #     # first priority: check for basepair range specifications (best way to specify):
        #     if self._conf.scaffoldguided.target_ss_pairs is not None:
        #         self.target_ss_matrix = torch.from_numpy(iu.ss_pairs_to_matrix(self._conf.scaffoldguided.target_ss_pairs, self._conf.contigmap.contigs)).long()
        #         # assert 0, 'NOT IMPLEMENTED YET! TO DO!'

        #     # second priority: check for ss-string specification (slightly less precise):
        #     elif self._conf.scaffoldguided.target_ss_string is not None:
        #         self.target_ss_matrix = torch.from_numpy(iu.sstr_to_matrix(self._conf.scaffoldguided.target_ss_string, only_basepairs=True)).long()
            
        #     # otherwise fall back to masking everything:
        #     else:
        #         self.target_ss_matrix = (2*torch.ones(indep.same_chain.shape)).long()

        #     # Save the matrix image if we want:
        #     if self._conf.scaffoldguided.save_ss_matrix_png and (t==self._conf.diffuser.T):
        #         print('SAVING SS MATRIX PIC!')
        #         output_pic_filapath = self._conf.inference.output_prefix+'.png'
        #         output_dirpath = os.path.dirname(output_pic_filapath)

        #         if not (os.path.exists(output_dirpath) and os.path.isdir(output_dirpath)):
        #             os.mkdir(output_dirpath)

        #         fig, ax = plt.subplots(1,1,figsize=(5,5), dpi=300)
        #         ax.imshow(np.array(self.target_ss_matrix))
        #         plt.tight_layout()
        #         plt.savefig(output_pic_filapath, bbox_inches='tight', dpi='figure')
        #         plt.close()

        # else:
        #     self.target_ss_matrix = None





        # Set up polymer class template:
        # This will be used for t1d conditioning
        if self.show_poly_class:
            self.poly_class_vec = torch.zeros_like(indep.seq)
            self.poly_class_vec[indep.is_rna] = 2 # RNA has token 2
            self.poly_class_vec[indep.is_dna] = 1 # DNA has token 1
            self.poly_class_vec[indep.is_protein] = 0 # Protein has token 0 (add last to overwrite to default val)

        else:
            self.poly_class_vec = None




        if not self.inf_conf.subsymm_t1d_perfect: 
            # all AA that are diffused (according to contigs) have intermediate confidences
            # even if they are templated in T2D 
            rfi = self.model_adaptor.prepro(indep, 
                                            t, 
                                            self.is_diffused, 
                                            twotemplate=twotemplate,
                                            threetemplate=threetemplate,
                                            t2d_is_revealed=t2d_is_revealed,
                                            is_motif=is_motif, 
                                            polymer_type_masks=self.inf_conf.mask_seq_by_polymer,
                                            rfmotif=rfmotif
                                            )
                                            # target_ss_matrix=self.target_ss_matrix,
        else:
            # Though they are diffused, the AA being templated in T2d will have 
            # perfect confidence, while all else has 1-t/T
            raise Exception('Not a good option, results were poor.')
            rfi = self.model_adaptor.prepro(indep, 
                                            t, 
                                            self.has_imperfect_t1d, 
                                            twotemplate=twotemplate,
                                            threetemplate=threetemplate,
                                            t2d_is_revealed=t2d_is_revealed,
                                            is_motif=is_motif, 
                                            polymer_type_masks=self.inf_conf.mask_seq_by_polymer,
                                            rfmotif=rfmotif
                                            )
                                            # target_ss_matrix=self.target_ss_matrix,

        # Now we modify rfi to accomodate the new t2d features
        # Adding target ss_matrix if that is what we wanna do:
        if (self.target_ss_matrix is not None) and (twotemplate and threetemplate):
            ss_adj_templ_onehot = torch.nn.functional.one_hot(self.target_ss_matrix, num_classes=3)
            ss_adj_templ_onehot = ss_adj_templ_onehot.reshape(1, 1, *ss_adj_templ_onehot.shape).repeat(1,3,1,1,1)
            rfi.t2d = torch.cat((rfi.t2d, ss_adj_templ_onehot), dim=-1)


        # Now we modify rfi to accomodate the new t1d features
        # Adding polymer class labels if that is what we wanna do:
        if (self.poly_class_vec is not None) and (twotemplate and threetemplate):
            polymer_templ_onehot = torch.nn.functional.one_hot(self.poly_class_vec, num_classes=3)
            polymer_templ_onehot = polymer_templ_onehot.reshape(1, 1, *polymer_templ_onehot.shape).repeat(1,3,1,1)
            rfi.t1d = torch.cat((rfi.t1d, polymer_templ_onehot), dim=-1)



        rf2aa.tensor_util.to_device(rfi, self.device)
        seq_init = torch.nn.functional.one_hot(indep.seq, num_classes=rf2aa.chemical.NAATOKENS).to(self.device).float()
        seq_t    = torch.clone(seq_init)
        seq_in   = torch.clone(seq_init)
        # B,N,L = xyz_t.shape[:3]

        ##################################
        ######## Str Self Cond ###########
        ##################################
        self_cond = False
        cond_A = ((t < self.diffuser.T) and (t != self.diffuser_conf.partial_T)) and self._conf.inference.str_self_cond
        cond_B = not self.inf_conf.refine  # cannot self cond with refinement model
        if cond_A and cond_B:
        # if ((t < self.diffuser.T) and (t != self.diffuser_conf.partial_T)) and self._conf.inference.str_self_cond:
            # in the middle of the traj, so self condition on previous px0
            self_cond=True

            rfi = aa_model.self_cond(indep, 
                                    rfi, 
                                    rfo, 
                                    twotemplate=twotemplate, 
                                    threetemplate=threetemplate)
            """
            2template self conditioning: 

            First template (zeroth index in t2d, t1d, xyz_t): 
                Associated with xt, i.e., the current coordinates of the trajectory 

            Second template (first index in t2d, t1d, xyz_t):
                Associated with px0 from previous step, i.e., self conditioning
            """

        # Check for subsymmetric template
        # if exists, slice in the t2d from subsym template 
        if self.inf_conf.subsymm_template is not None:
            mask_t_2d_subsymm = indep.mask_t_2d_subsymm
            # xyz_subsymm       = indep.subsymm_xyz
            xyz_subsymm = self.target_feats['subsymm_xyz']
            
            # translate subsym along sym axis if not C1 to ensure correct 
            if self.target_feats['subsymm_symbol'].lower() != 'c1':
                xyz_subsymm += 5*self.target_feats['subsymm_axis']

            # Make new xyz_t monomer with true template embedded into dummy zeros 
            _,natom,_ = xyz_subsymm.shape
            xyz_t = torch.zeros((self.Lasu,natom,3)).to(self.device)
            mask_t = torch.zeros((self.Lasu,)).to(self.device).bool()

            con_ref_idx0 = self.contig_map.get_mappings()['con_ref_idx0']
            con_hal_idx0 = self.contig_map.get_mappings()['con_hal_idx0']
            # single chain embedded into zeros according to contigs 
            xyz_t[con_hal_idx0]  = xyz_subsymm[con_ref_idx0].to(self.device)
            mask_t[con_hal_idx0] = True

            # now get xyz_t for current subsymm/Rs being modelled 
            cur_Rs = self.symmRs[self.cur_symmsub]
            xyz_t  = torch.einsum('sji,lai->slaj',cur_Rs.transpose(-1,-2), xyz_t).squeeze()
            xyz_t  = xyz_t.reshape(len(cur_Rs)*self.Lasu,natom,3)
            mask_t = mask_t.repeat(len(cur_Rs))
            mask_t_asu = mask_t.clone()         # dj - new for rigid motif fitting 
            mask_t_asu[self.Lasu:] = False 

            # calculate T2d on (propogated) subsym template
            L                = xyz_t.shape[0]
            zeros            = torch.zeros(1,L,9,3).to(self.device)
            xyz_subsymm_full = torch.cat([xyz_t[None],zeros], dim=2)
            assert xyz_subsymm_full.shape == (1,L,36,3), f'Got shape: {xyz_subsymm_full.shape}'

            is_sm = indep.is_sm
            atom_frames = rfi.atom_frames[0]

            # get t2d for subsym template
            t2d_subsymm, _ = util.get_t2d(xyz_subsymm_full,
                                          is_sm, 
                                          atom_frames)
            
            # Now slice in the t2d from subsym template into t2d going into model 
            # Make sure to acknowledge placement of t2d chunks within contigs 
            
            # Template was > C1 
            if mask_t_2d_subsymm != None:

                # remap to modelled subunits
                # creates tensor that is (Nres,Nres)
                mask_t_2d_subsymm_applied = mask_t_2d_subsymm[:,self.cur_symmsub[:,None],self.cur_symmsub[None,:]]
                mask_t_2d_subsymm_applied = mask_t_2d_subsymm_applied.repeat_interleave(self.Lasu,dim=1).repeat_interleave(self.Lasu,dim=2)
                
                mask_t_2d = mask_t[:,None] * mask_t[None,:] # grabs only residues that are motifs in contigs
                mask_t_2d_subsymm_applied = mask_t_2d * mask_t_2d_subsymm_applied

                # make same shape as t2d tensors to apply in one fell swoop 
                mask_2d_final = mask_t_2d_subsymm_applied.unsqueeze(-1).expand_as(rfi.t2d)

                # Now slice in subsymm template
                rfi.t2d[mask_2d_final] = t2d_subsymm[:,None,...].expand_as(rfi.t2d)[mask_2d_final]

                # DJ - add in subsymm template to rfixyz_t because it's currently zeros 
                if self.inf_conf.input_xyz_t:
                    # xyz_t is shape (1,2,L,3), CA only
                    print('ADDING SUBSYMM TEMPLATE TO XYZ_T')
                    rfi.xyz_t[:,:,mask_t] = xyz_subsymm_full[:,None,mask_t,1,:] # CA only 
            
            else:
                # template was C1
                # eye matrix same shape as total chains being modelled - intra-chain only 
                mask_t_2d_subsymm_applied = torch.eye(self.symmRs.shape[0]).bool().to(self.device)[None]
                mask_t_2d_subsymm_applied = mask_t_2d_subsymm_applied.repeat_interleave(self.Lasu,dim=1).repeat_interleave(self.Lasu,dim=2)

                mask_t_2d = mask_t[:,None] * mask_t[None,:] # grabs only residues that are motifs in contigs
                mask_t_2d_subsymm_applied = mask_t_2d * mask_t_2d_subsymm_applied

                # make same shape as t2d tensors to apply in one fell swoop
                mask_2d_final = mask_t_2d_subsymm_applied.unsqueeze(-1).expand_as(rfi.t2d)

                # Now slice in subsymm template
                rfi.t2d[mask_2d_final] = t2d_subsymm[:,None,...].expand_as(rfi.t2d)[mask_2d_final]

        
        if self.symmetry is not None:
            idx_pdb = rfi.idx
            idx_pdb, self.chain_idx = self.symmetry.res_idx_procesing(res_idx=idx_pdb)

        # Model Forward
        with torch.no_grad():
            if self.recycle_schedule[t-1] > 1:
                raise Exception('not implemented')
            
            for rec in range(self.recycle_schedule[t-1]):
                # This is the assertion we should be able to use, but the
                # network's ComputeAllAtom requires even atoms to have N and C coords.                
                # aa_model.assert_has_coords(rfi.xyz[0], indep)
                assert not rfi.xyz[0,:,:3,:].isnan().any(), f'{t}: {rfi.xyz[0,:,:3,:]}'
             
                # rfo = self.model_adaptor.forward(rfi, return_infer=True, **({model_input_logger.LOG_ONLY_KEY: {'t':t, 'output_prefix':self.output_prefix,}} if self._conf.logging.inputs else {}))
                kwargs = {model_input_logger.LOG_ONLY_KEY: {'t':t, 'output_prefix':self.output_prefix,}} if self._conf.logging.inputs else {}
                kwargs.update({'symmids':self.symmids,
                               'symmRs' :self.symmRs,
                               'symmeta':self.symmeta,
                               'symmsub':self.cur_symmsub}) # None by default - see self.sample_init()
                kwargs.update({'t':t}) # added for symm fitting in RF - model needs to know timestep 
                if self.inf_conf.p2p_crop > -1:
                    kwargs.update({'p2p_crop':self.inf_conf.p2p_crop})
                
                if REPORT_MEM:
                    print('MEM REPORT LINE 916 model runners')
                    mem_report() 
                    print('*'*50+'\n\n')

                # Update arguments passed to RoseTTAfold module
                # So that it can run forward pass with correct number of dimensions
                if (twotemplate and threetemplate):
                    kwargs.update({'ss_adj_conditioned':self.use_ss_guidance})
                    kwargs.update({'poly_class_conditioned':self.show_poly_class})

                # Now we will add rfmotif as an item in the kwarg dict
                # kwargs.update({'rfmotif':rfmotif})

                # debugging 
                # tmp_out = vars(rfi)
                # for key in tmp_out.keys():
                #     if torch.is_tensor(tmp_out[key]):
                #         tmp_out[key] = tmp_out[key].cpu().numpy()
                # with open('rfi_yes_motif_nosymm.pkl','wb') as f:
                #     pickle.dump(tmp_out,f)
                # sys.exit('Exiting for debugging')

                if self.inf_conf.refine: 
                    N_cycle = self.inf_conf.refine_recycles
                else: 
                    N_cycle = 1

                with torch.cuda.amp.autocast(True):
                    # rfo = self.model_adaptor.forward(rfi, N_cycle=N_cycle, rfmotif=rfmotif, return_infer=True, **kwargs)
                    # set_trace()
                    rfo = self.model_adaptor.forward(rfi, N_cycle=N_cycle, return_infer=True, **kwargs)

                print('********* SUCCESSFULL MODEL FORWARD *******')
                self.cur_symmsub = rfo.symmsub
                
                # Symmsubs may have changed, so need to update Xt to match model predicted symmsubs
                if self.inf_conf.internal_sym is not None:
                    xt_asu = rfi.xyz.squeeze(dim=0)[:self.Lasu]
                    cur_Rs = self.symmRs[self.cur_symmsub]
                    s = len(cur_Rs)
                    updated_xt = torch.einsum('sji,lai->slaj', cur_Rs.transpose(-1,-2), xt_asu)
                    updated_xt = updated_xt.reshape(s*self.Lasu, -1, 3)
                    rfi.xyz = updated_xt.unsqueeze(0)

                if REPORT_MEM:
                    print('MEM REPORT LINE 920 MODEL RUNNERS')
                    mem_report()
                    print('*'*50+'\n\n')
                
                # sys.exit('debugging')
                if False: 
                #if self.symmetry is not None and self.inf_conf.symmetric_self_cond:
                    print('WARNING: SYMMETRIZED SELF COND NOT OCCURING - NOT IMPLEMENTED')
                    print('WARNING: DJ has not validated symmetric self cond in all atom')
                    px0 = self.symmetrise_prev_pred(px0=rfo.xyz_allatom[:,:14], seq_in=rfo.seq, alpha=alpha)[:,:,:3]


                # To permit 'recycling' within a timestep, in a manner akin to how this model was trained
                # Aim is to basically just replace the xyz_t with the model's last px0, and to *not* recycle the state, pair or msa embeddings
                if rec < self.recycle_schedule[t-1] -1:
                    raise Exception('not implemented')
                    zeros = torch.zeros(B,1,L,24,3).float().to(xyz_t.device)
                    xyz_t = torch.cat((px0.unsqueeze(1),zeros), dim=-2) # [B,T,L,27,3]

                    t2d   = xyz_to_t2d(xyz_t) # [B,T,L,L,44]

                    if self.seq_self_cond:
                        # Allow this model to also do sequence recycling
                        assert 0, 'MODIFY THIS LINE TO INCLUDE NA TOKENS!'
                        t1d[:,:,:,:20] = logits[:,None,:,:20]
                        t1d[:,:,:,20]  = 0 # Setting mask tokens to zero

        # px0         = rfo.get_xyz()[:,:14]
        # px0         = rfo.get_xyz()[:,:23]
        # px0         = rfo.get_xyz()[:,:NHEAVY]
        px0         = rfo.get_xyz()[:,:self.inf_conf.num_atoms_modeled]
        logits      = rfo.get_seq_logits()
        seq_decoded = [rf2aa.chemical.num2aa[s] for s in rfi.seq[0]]

        logits = logits.float()
        px0    = px0.float()
        # Modify logits if we want to use polymer masks:
        if self._conf.inference['mask_seq_by_polymer']:
            # logit_penalty = 1e7
            logit_penalty = 1e12
            logits = logits - logit_penalty*(torch.ones_like(self.polymer_mask)-self.polymer_mask)
            # set_trace()

        if self.seq_diffuser is None:
            # Default method of decoding sequence
            seq_probs   = torch.nn.Softmax(dim=-1)(logits.squeeze()/self.inf_conf.softmax_T)
            sampled_seq = torch.multinomial(seq_probs, 1).squeeze() # sample a single value from each position

            pseq_0 = torch.nn.functional.one_hot(
                sampled_seq, num_classes=rf2aa.chemical.NAATOKENS).to(self.device).float()

            pseq_0[~self.is_diffused] = seq_init[~self.is_diffused].to(self.device) # [L,22]
        else:
            # Sequence Diffusion
            assert 0, "IF WE DOING IT THIS WAY, NEED TO CHANGE TO ACCOMODATE NA TOKENS"
            pseq_0 = logits.squeeze()
            pseq_0 = pseq_0[:,:20]

            pseq_0[self.mask_seq.squeeze()] = seq_init[self.mask_seq.squeeze(),:20].to(self.device)

            sampled_seq = torch.argmax(pseq_0, dim=-1)

        self._log.info(
                f'Timestep {t}, current sequence: { rf2aa.chemical.seq2chars(torch.argmax(pseq_0, dim=-1).tolist())}')

        # doing rigid motif symm scaffolding
        if self._conf.inference.rigid_symm_motif:
            if self.cur_rigid_tmplt is None:
                self.cur_rigid_tmplt = xyz_subsymm_full # xyz of propogated motif 

            rigid_symm_motif_kwargs = {'xyz_template'   : self.cur_rigid_tmplt.squeeze(dim=0),
                                       'motif_mask'     : mask_t,
                                       'symmRs'         : self.symmRs,
                                       'symmsub'        : self.cur_symmsub}
            rigid_repeat_motif_kwargs = {}
        
        # doing rigid repeat motif symm scaffolding
        elif self._conf.inference.rigid_repeat_motif: 
            print('ENTERING RIGID REPEAT MOTIF')
            if self.cur_rigid_tmplt is None: 
                # keep track of the current rigid motif - in indep[~self.diffuser_is_diffused]
                self.cur_rigid_tmplt = indep.xyz
            
            is_motif = torch.cat([~self.diffuser_is_diffused]*self._conf.inference.n_repeats, dim=0)
            rigid_repeat_motif_kwargs = {'xyz_template'         : self.cur_rigid_tmplt,
                                         'is_motif'             : is_motif,
                                         'enforce_repeat_fit'   : self._conf.inference.enforce_repeat_fit,
                                         'fit_optim_steps'      : self._conf.inference.rigid_fit_optim_steps,
                                         'repeat_length'        : self._conf.model.repeat_length}
            rigid_symm_motif_kwargs = {}
        else:
            rigid_symm_motif_kwargs = {}
            rigid_repeat_motif_kwargs = {}

        
        ### Can also do the repeat protein motif fitting kwargs here 
        if self._conf.inference.rigid_repeat_motif:
            if self.cur_rigid_tmplt is None: 
                pass

        if t > self._conf.inference.final_step:
            # ipdb.set_trace()
            # x_t_1, seq_t_1, tors_t_1, px0, cur_rigid_tmplt = self.denoiser.get_next_pose(
            #     xt=rfi.xyz[0,:,:14].cpu(),
            #     px0=px0,
            #     t=t,
            #     diffusion_mask=~self.is_diffused,
            #     seq_diffusion_mask=~self.is_diffused,
            #     seq_t=seq_t,
            #     pseq0=pseq_0,
            #     diffuse_sidechains=self.preprocess_conf.sidechain_input,
            #     align_motif=self.inf_conf.align_motif,
            #     include_motif_sidechains=self.preprocess_conf.motif_sidechain_input,
            #     rigid_symm_motif_kwargs=rigid_symm_motif_kwargs,
            #     rigid_repeat_motif_kwargs=rigid_repeat_motif_kwargs,
            #     origin_before_update=self._conf.inference.origin_before_update,
            #     rfmotif=rfmotif,
            # )
            x_t_1, seq_t_1, tors_t_1, px0, cur_rigid_tmplt = self.denoiser.get_next_pose(
                xt=rfi.xyz[0,:,:self._conf.inference.num_atoms_modeled].cpu(),
                px0=px0,
                t=t,
                diffusion_mask=~self.is_diffused,
                seq_diffusion_mask=~self.is_diffused,
                seq_t=seq_t,
                pseq0=pseq_0,
                diffuse_sidechains=self.preprocess_conf.sidechain_input,
                align_motif=self.inf_conf.align_motif,
                include_motif_sidechains=self.preprocess_conf.motif_sidechain_input,
                rigid_symm_motif_kwargs=rigid_symm_motif_kwargs,
                rigid_repeat_motif_kwargs=rigid_repeat_motif_kwargs,
                origin_before_update=self._conf.inference.origin_before_update,
                num_atoms_modeled=self._conf.inference.num_atoms_modeled,
                rfmotif=rfmotif,
            )
            # x_t_1, seq_t_1, tors_t_1, px0, cur_rigid_tmplt = self.denoiser.get_next_pose(
            #     xt=rfi.xyz[0,:,:NHEAVY].cpu(),
            #     px0=px0,
            #     t=t,
            #     diffusion_mask=~self.is_diffused,
            #     seq_diffusion_mask=~self.is_diffused,
            #     seq_t=seq_t,
            #     pseq0=pseq_0,
            #     diffuse_sidechains=self.preprocess_conf.sidechain_input,
            #     align_motif=self.inf_conf.align_motif,
            #     include_motif_sidechains=self.preprocess_conf.motif_sidechain_input,
            #     rigid_symm_motif_kwargs=rigid_symm_motif_kwargs,
            #     rigid_repeat_motif_kwargs=rigid_repeat_motif_kwargs,
            #     origin_before_update=self._conf.inference.origin_before_update,
            #     num_atoms_modeled=self._conf.inference.num_atoms_modeled,
            #     rfmotif=rfmotif,
            # )
            self.cur_rigid_tmplt = cur_rigid_tmplt
        else:
            # ipdb.set_trace()
            px0 = px0.cpu()
            px0[~self.is_diffused] = indep.xyz[~self.is_diffused]
            x_t_1 = torch.clone(px0)
            seq_t_1 = pseq_0

            # Dummy tors_t_1 prediction. Not used in final output.
            tors_t_1 = torch.ones((self.is_diffused.shape[-1], 10, 2))
            # ipdb.set_trace()


        if self._conf.inference.internal_sym is not None:
            # Re-symmetrize after stochastic denoising step 
            fake_indep = copy.deepcopy(indep)  # dummy indep, just to pass current set of crds to get neighbor list 
            fake_indep.xyz = x_t_1.to(device=self.symmRs.device)
            _, symmsub  = symmetry.find_minimal_neighbors(fake_indep, self.symmRs, self.symmeta)

            # x_t_1 = update_symm_Rs(x_t_1.to(self.symmRs.device)[None], self.Lasu, symmsub, self.symmRs, fit_symm=False).squeeze(0)
            if symmsub.shape[0] > 1:
                x_t_1 = update_symm_Rs(x_t_1.to(self.symmRs.device)[None], self.Lasu, symmsub, self.symmRs, recenter_particle=self._conf.model.allow_particle_recenter).squeeze(0)

        px0 = px0.cpu()
        x_t_1 = x_t_1.cpu()
        seq_t_1 = seq_t_1.cpu()

        if self.symmetry is not None:
            x_t_1, seq_t_1 = self.symmetry.apply_symmetry(x_t_1, seq_t_1)
        
        if REPORT_MEM:
            print('MEM REPORT END OF MODEL_RUNNERS.SAMPLE_STEP')
            mem_report()
        # ipdb.set_trace()
        return px0, x_t_1, seq_t_1, tors_t_1, None, rfo

def sampler_selector(conf: DictConfig, preloaded_ckpts={}, preloaded_models={}):
    if conf.inference.model_runner == 'default':
        sampler = Sampler(conf)
    elif conf.inference.model_runner == 'legacy':
        sampler = T1d28T2d45Sampler(conf)
    elif conf.inference.model_runner == 'seq2str':
        sampler = Seq2StrSampler(conf)
    elif conf.inference.model_runner == 'JWStyleSelfCond':
        sampler = JWStyleSelfCond(conf)
    elif conf.inference.model_runner == 'NRBStyleSelfCond':
        sampler = NRBStyleSelfCond(conf, preloaded_ckpts, preloaded_models)
    else:
        raise ValueError(f'Unrecognized sampler {conf.model_runner}')
    return sampler
