import sys, os
import shutil
import collections
# Insert the se3 transformer version packaged with RF2-allatom before anything else, so it doesn't end up in the python module cache.
script_dir = os.path.dirname(os.path.realpath(__file__))
aa_se3_path = os.path.join(script_dir, 'RF2-allatom/rf2aa/SE3Transformer')
sys.path.insert(0, aa_se3_path)
import copy
import dataclasses
import time
import pickle 
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from datetime import date
from contextlib import ExitStack
import time 
import torch
import torch.nn as nn
import ipdb
from torch.utils import data
import math 

#rf2_allatom = __import__('RF2-allatom')
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RF2-allatom'))
import rf2aa.chemical
import rf2aa.data_loader
import rf2aa.util
import rf2aa.loss
import rf2aa.tensor_util
from rf2aa.tensor_util import assert_equal
from rf2aa.util_module import XYZConverter
from rf2aa.util import get_frames
from rf2aa.RoseTTAFoldModel import RoseTTAFoldModule
from rf2aa.loss import compute_general_FAPE, mask_unresolved_frames
import loss_aa
import run_inference
import aa_model
import atomize
from data_loader import (
    get_train_valid_set, loader_pdb, loader_fb, loader_complex, loader_pdb_fixbb, loader_fb_fixbb, loader_complex_fixbb, loader_cn_fixbb, default_dataset_configs,
    #Dataset, DatasetComplex, 
    DistilledDataset, DistributedWeightedSampler
)
import error

from kinematics import xyz_to_c6d, c6d_to_bins2, xyz_to_t2d, xyz_to_bbtor, get_init_xyz
import loss 
from loss import *
import util
from scheduler import get_stepwise_decay_schedule_with_warmup

import rotation_conversions as rot_conv

#added for inpainting training
from icecream import ic
from apply_masks import mask_inputs
import random
import model_input_logger
from model_input_logger import pickle_function_call

# added for diffusion training 
from diffusion import Diffuser
from seq_diffusion import ContinuousSeqDiffuser, DiscreteSeqDiffuser

# added for logging git diff
import subprocess

# distributed data parallel
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

#torch.autograd.set_detect_anomaly(True)

global N_EXAMPLE_PER_EPOCH
global DEBUG 
global WANDB
USE_AMP = False

N_PRINT_TRAIN = 1 
#BATCH_SIZE = 1 * torch.cuda.device_count()

# num structs per epoch
# must be divisible by #GPUs
#N_EXAMPLE_PER_EPOCH = 25600*2


def add_weight_decay(model, l2_coeff):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        #if len(param.shape) == 1 or name.endswith(".bias"):
        if "norm" in name or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_coeff}]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class EMA(nn.Module):
    def __init__(self, model, decay):
        super().__init__()
        self.decay = decay

        self.model = model
        self.shadow = deepcopy(self.model)

        for param in self.shadow.parameters():
            param.detach_()

    @torch.no_grad()
    def update(self):
        if not self.training:
            print("EMA update should only be called during training", file=stderr, flush=True)
            return

        model_params = OrderedDict(self.model.named_parameters())
        shadow_params = OrderedDict(self.shadow.named_parameters())

        # check if both model contains the same set of keys
        assert model_params.keys() == shadow_params.keys()

        for name, param in model_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            if param.requires_grad:
                shadow_params[name].sub_((1. - self.decay) * (shadow_params[name] - param))

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.shadow.named_buffers())

        # check if both model contains the same set of keys
        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            # buffers are copied
            shadow_buffers[name].copy_(buffer)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.model(*args, **kwargs)
        else:
            return self.shadow(*args, **kwargs)

def get_datetime():
    return str(date.today()) + '_' + str(time.time())

def no_batch_collate_fn(data):
    assert len(data) == 1
    return data[0]

class Trainer():
    def __init__(self, model_name='BFF', ckpt_load_path=None,
                 n_epoch=100, lr=1.0e-4, l2_coeff=1.0e-2, port=None, interactive=False,
                 model_param={}, loader_param={}, loss_param={}, batch_size=1, accum_step=1, 
                 maxcycle=4, diffusion_param={}, preprocess_param={}, outdir=None, wandb_prefix='',
                 metrics=None, zero_weights=False, log_inputs=False, n_write_pdb=100,
                 reinitialize_missing_params=False, verbose_checks=False, saves_per_epoch=None, resume=False,
                 grad_clip=0.2, polymer_focus=None, polymer_frac_cutoff=0.25):

        self.model_name = model_name #"BFF"
        self.ckpt_load_path = ckpt_load_path
        self.n_epoch = n_epoch
        self.init_lr = lr
        self.l2_coeff = l2_coeff
        self.port = port
        self.interactive = interactive
        self.outdir = outdir
        self.zero_weights=zero_weights
        self.metrics=metrics or []
        self.log_inputs=log_inputs
        self.n_write_pdb = n_write_pdb
        self.reinitialize_missing_params = reinitialize_missing_params
        self.rate_deque = collections.deque(maxlen=200)
        self.verbose_checks=verbose_checks
        self.resume=resume
        self.grad_clip=grad_clip

        self.polymer_focus=polymer_focus
        self.polymer_frac_cutoff=polymer_frac_cutoff

        self.outdir = self.outdir or f'./train_session{get_datetime()}'
        ic(self.outdir)
        if os.path.isdir(self.outdir) and not DEBUG:
            sys.exit('EXITING: self.outdir already exists. Dont clobber')
        os.makedirs(self.outdir, exist_ok=True)
        #
        self.model_param = model_param
        self.loader_param = loader_param
        self.valid_param = deepcopy(loader_param)
        self.valid_param['MINTPLT'] = 1
        self.valid_param['SEQID'] = 150.0
        self.loss_param = loss_param
        ic(self.loss_param)
        self.ACCUM_STEP = accum_step
        self.batch_size = batch_size

        self.diffusion_param = diffusion_param
        self.preprocess_param = preprocess_param
        self.wandb_prefix=wandb_prefix
        self.saves_per_epoch=saves_per_epoch

        # For diffusion
        diff_kwargs = {'T'              :diffusion_param['diff_T'],
                       'b_0'            :diffusion_param['diff_b0'],
                       'b_T'            :diffusion_param['diff_bT'],
                       'min_b'          :diffusion_param['diff_min_b'],
                       'max_b'          :diffusion_param['diff_max_b'],
                       'min_sigma'      :diffusion_param['diff_min_sigma'],
                       'max_sigma'      :diffusion_param['diff_max_sigma'],
                       'schedule_type'  :diffusion_param['diff_schedule_type'],
                       'so3_schedule_type' : diffusion_param['diff_so3_schedule_type'],
                       'so3_type'       :diffusion_param['diff_so3_type'],
                       'chi_type'       :diffusion_param['diff_chi_type'],
                       'aa_decode_steps':diffusion_param['aa_decode_steps'],
                       'crd_scale'      :diffusion_param['diff_crd_scale']}
        
        self.diffuser = Diffuser(**diff_kwargs)
        self.schedule = self.diffuser.eucl_diffuser.beta_schedule
        self.alphabar_schedule = self.diffuser.eucl_diffuser.alphabar_schedule

        # For Sequence Diffusion
        seq_diff_type = diffusion_param['seqdiff']
        self.seq_diff_type = seq_diff_type
        seqdiff_kwargs = {'T'              : diffusion_param['diff_T'], # Use same T as for str diff
                          's_b0'           : diffusion_param['seqdiff_b0'],
                          's_bT'           : diffusion_param['seqdiff_bT'],
                          'schedule_type'  : diffusion_param['seqdiff_schedule_type'],
                          'loss_type'      : diffusion_param['seqdiff_loss_type']
                         }

        if not seq_diff_type:
            print('Training with autoregressive sequence decoding')
            self.seq_diffuser = None

        elif seq_diff_type == 'uniform':
            print('Training with discrete sequence diffusion')
            seqdiff_kwargs['rate_matrix'] = 'uniform'
            seqdiff_kwargs['lamda'] = diffusion_param['seqdiff_lambda']

            self.seq_diffuser = DiscreteSeqDiffuser(**seqdiff_kwargs)

        elif seq_diff_type == 'continuous':
            print('Training with continuous sequence diffusion')

            self.seq_diffuser = ContinuousSeqDiffuser(**seqdiff_kwargs)

        else: 
            print(f'Sequence diffusion with type {seq_diff_type} is not implemented')
            raise NotImplementedError()

        # for all-atom str loss
        self.ti_dev = rf2aa.util.torsion_indices
        self.ti_flip = rf2aa.util.torsion_can_flip
        self.ang_ref = rf2aa.util.reference_angles
        self.l2a = rf2aa.util.long2alt
        self.aamask = rf2aa.util.allatom_mask
        self.num_bonds = rf2aa.util.num_bonds
        self.ljlk_parameters = rf2aa.util.ljlk_parameters
        self.lj_correction_parameters = rf2aa.util.lj_correction_parameters
        
        # create a loss schedule - sigmoid (default) if use tschedule, else empty dict
        constant_schedule = not loss_param['use_tschedule']
        loss_names = loss_param['scheduled_losses']
        schedule_type = loss_param['scheduled_types']
        schedule_params = loss_param['scheduled_params']
        self.loss_schedules = loss.get_loss_schedules(diff_kwargs['T'], loss_names=loss_names, schedule_types=schedule_type, schedule_params=schedule_params, constant=constant_schedule)
        self.loss_param.pop('use_tschedule')
        self.loss_param.pop('scheduled_losses')
        self.loss_param.pop('scheduled_types')
        self.loss_param.pop('scheduled_params')
        print('These are the loss names which have t_scheduling activated')
        print(self.loss_schedules.keys())

        self.hbtypes = rf2aa.util.hbtypes
        self.hbbaseatoms = rf2aa.util.hbbaseatoms
        self.hbpolys = rf2aa.util.hbpolys

        # module torsion -> allatom
        # self.compute_allatom_coords = ComputeAllAtomCoords()

        #self.diffuser.get_allatom = self.compute_allatom_coords

        # loss & final activation function
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.active_fn = nn.Softmax(dim=1)

        self.maxcycle = maxcycle
        
        print (model_param, loader_param, loss_param)
        
        # Assemble "Config" for inference
        self.diff_kwargs = diff_kwargs
        self.seqdiff_kwargs = seqdiff_kwargs
        self.assemble_config()
        ic(self.config_dict) 

        self.assemble_train_args()

    def assemble_config(self) -> None:
        config_dict = {}
        config_dict['model'] = self.model_param
        
        #rename diffusion params to match config
        #infer_names=dict(zip([i for i in self.diffusion_param.keys()],[i[5:] if i[:5] == 'diff_' else i for i in self.diffusion_param.keys()]))
        #config_dict['diffuser'] = {infer_names[k]: v for k, v in self.diffusion_param.items()}
        config_dict['diffuser'] = self.diff_kwargs
        config_dict['seq_diffuser'] = self.seqdiff_kwargs
        # Add seq_diff_type
        config_dict['seq_diffuser']['seqdiff'] = self.seq_diff_type
        config_dict['preprocess'] = self.preprocess_param
        ic(self.preprocess_param)
        self.config_dict = config_dict

    def assemble_train_args(self) -> None:

        # preprocess and model param are saved in config dict
        # and so are not saved here

        # Hack to pickle wrapped functions
        loader_param_pickle_safe = copy.deepcopy(self.loader_param)
        loader_param_pickle_safe['DIFF_MASK_PROBS'] = {k.__name__:v for k,v in loader_param_pickle_safe['DIFF_MASK_PROBS'].items()}
        diffusion_param_pickle_safe = copy.deepcopy(self.diffusion_param)
        diffusion_param_pickle_safe['diff_mask_probs'] = {k.__name__:v for k,v in diffusion_param_pickle_safe['diff_mask_probs'].items()}
        # ic(pickle_safe)
        self.training_arguments = {

            'ckpt_load_path': self.ckpt_load_path,
            'interactive': self.interactive,
            'n_epoch': self.n_epoch,
            'learning_rate': self.init_lr,
            'l2_coeff': self.l2_coeff,
            'port': self.port,

            'epoch_size': N_EXAMPLE_PER_EPOCH,
            'batch_size': self.batch_size,
            'accum_step': self.ACCUM_STEP,
            'maxcycle': self.maxcycle,
            'wandb_prefix': self.wandb_prefix,
            'metrics': self.metrics,
            'zero_weights': self.zero_weights,
            'log_inputs': self.log_inputs,

            'diffusion_param': diffusion_param_pickle_safe,
            'loader_param': loader_param_pickle_safe,
            'loss_param': self.loss_param

        }
        torch.save(self.training_arguments, 'tmp.out')

    def calc_loss(self, logit_s, label_s,
                  logit_aa_s, label_aa_s, mask_aa_s, logit_exp,
                  pred_in, pred_tors, true, mask_crds, mask_BB, mask_2d, same_chain,
                  pred_lddt, idx, dataset, chosen_task, t, xyz_in, diffusion_mask,
                  seq_diffusion_mask, seq_t, is_sm, unclamp=False, negative=False,
                  w_dist=1.0, w_aa=1.0, w_str=1.0, w_all=0.5, w_exp=1.0,
                  w_lddt=1.0, w_blen=1.0, w_bang=1.0, w_lj=0.0, w_hb=0.0,
                  lj_lin=0.75, use_H=False, w_disp=0.0, w_motif_disp=0.0, 
                  w_ax_ang=0.0, w_frame_dist=0.0, eps=1e-6, w_motif_fape=0.0,
                  w_nonmotif_fape=0.0, norm_fape=10.0, clamp_fape=10.0,
                  backprop_non_displacement_on_given=False, masks_1d={},
                  atom_frames=None, w_ligand_intra_fape=1.0, w_prot_lig_inter_fape=1.0):
        
        # assuming all bad crds have been popped
        assert not torch.isnan(pred_in).any()
        assert not torch.isnan(true).any()

        gpu = pred_in.device


        #NB t is 1-indexed
        t_idx = t-1
 
        # dictionary for keeping track of losses 
        loss_dict = {}

        B, L = true.shape[:2]
        seq = label_aa_s[:,0].clone()
        assert (B==1) # fd - code assumes a batch size of 1

        is_atom         = rf2aa.util.is_atom(seq).squeeze().to(device=pred_in.device)
        is_motif        = masks_1d['input_str_mask'].to(device=is_atom.device) # (L,), True if it's a motif TOKEN (protein or sm)
        is_sm_motif     = is_motif & is_atom    # (L,), True if it's a sm motif
        is_prot_motif   = is_motif & ~is_atom   # (L,), True if it's a protein motif

        loss_s = list()
        tot_loss = 0.0
        
        # get tscales
        c6d_tscale      = self.loss_schedules.get('c6d',[1]*(t))[t_idx]
        aa_tscale       = self.loss_schedules.get('aa_cce',[1]*(t))[t_idx]
        disp_tscale     = self.loss_schedules.get('displacement',[1]*(t))[t_idx]
        lddt_loss_tscale = self.loss_schedules.get('lddt_loss',[1]*(t))[t_idx]
        bang_tscale     = self.loss_schedules.get('bang',[1]*(t))[t_idx]
        blen_tscale     = self.loss_schedules.get('blen',[1]*(t))[t_idx]
        exp_tscale      = self.loss_schedules.get('exp',[1]*(t))[t_idx]
        lj_tscale       = self.loss_schedules.get('lj',[1]*(t))[t_idx]
        hb_tscale       = self.loss_schedules.get('lj',[1]*(t))[t_idx]
        str_tscale      = self.loss_schedules.get('w_str',[1]*(t))[t_idx]
        w_all_tscale    = self.loss_schedules.get('w_all',[1]*(t))[t_idx]

        # tot_loss += (1.0-w_all)*w_str*tot_str
        loss_weights = {
            'displacement'              : w_disp*disp_tscale,
            'motif_displacement'        : w_motif_disp*disp_tscale,
            'aa_cce'                    : w_aa*aa_tscale,
            'frame_sqL2'                : w_frame_dist*disp_tscale,
            'axis_angle'                : w_ax_ang*disp_tscale,
            'tot_str'                   : (1.0 - w_all*w_all_tscale)*w_str,
            'motif_fape'                : w_motif_fape*disp_tscale,
            'nonmotif_fape'             : w_nonmotif_fape*disp_tscale,
            'ligand_intra_fape'         : w_ligand_intra_fape*disp_tscale,
            'prot_lig_inter_fape'       : w_prot_lig_inter_fape*disp_tscale,
        }

        for i in range(4):
            loss_weights[f'c6d_{i}'] = w_dist*c6d_tscale

        # Displacement prediction loss between xyz prev and xyz_true
        if unclamp:
            disp_loss = calc_displacement_loss(pred_in, true, gamma=0.99, d_clamp=None)
        else:
            disp_loss = calc_displacement_loss(pred_in, true, gamma=0.99, d_clamp=10.0)
 
        loss_dict['displacement'] = disp_loss
 
        # Displacement prediction loss between xyz prev and xyz_true for only motif region.
        if diffusion_mask.any():
            motif_disp_loss = calc_displacement_loss(pred_in[:,:,diffusion_mask], 
                                                     true[:,diffusion_mask], 
                                                     gamma=0.99, d_clamp=None)
            
            loss_dict['motif_displacement'] = motif_disp_loss


 
        if backprop_non_displacement_on_given:
            pred = pred_in
        else:
            pred = torch.clone(pred_in)
            pred[:,:,diffusion_mask] = pred_in[:,:,diffusion_mask].detach()


        # set up frames
        # (B,L,natom,3)
        
        frames, frame_mask = get_frames(pred_in[-1,:,...], mask_crds, seq, self.fi_dev, atom_frames)

        # sys.exit()
        # update frames and frames_mask to only include BB frames (have to update both for compatibility with compute_general_FAPE)
        frames_BB = frames.clone()
        frames_BB[..., 1:, :, :] = 0        # all frames except BB frames are set to zero
        frame_mask_BB = frame_mask.clone()
        frame_mask_BB[...,1:] =False
        
        # c6d loss
        for i in range(4):
            # schedule factor for c6d 
            # syntax is if it's not in the scheduling dict, loss has full weight (i.e., 1x)

            loss = self.loss_fn(logit_s[i], label_s[...,i]) # (B, L, L)

            if i == 0: 
                mask_2d_ = mask_2d
            else:
                # apply anglegram loss only when both residues have valid BB frames (i.e. not metal ions, and not examples with unresolved atoms in frames)
                _, bb_frame_good = mask_unresolved_frames(frames_BB, frame_mask_BB, mask_crds) # (1, L, nframes)
                bb_frame_good = bb_frame_good[...,0] # (1,L)
                loss_mask_2d = bb_frame_good & bb_frame_good[...,None]
                mask_2d_ = mask_2d & loss_mask_2d

            loss = (mask_2d_*loss).sum() / (mask_2d_.sum() + eps)
            loss_s.append(loss[None].detach())

            loss_dict[f'c6d_{i}'] = loss.clone()
        
        if not self.seq_diffuser is None:
            raise Exception('not implemented')
            if self.seq_diffuser.continuous_seq():
                # Continuous Analog Bit Diffusion
                # Leave the shape of logit_aa_s as [L,21] so the model can learn to predict zero at 21st entry
                logit_aa_s = logit_aa_s.squeeze() # [L,21]
                logit_aa_s = logit_aa_s.transpose(0,1) # [L,21]

                label_aa_s = label_aa_s.squeeze() # [L]

                loss = self.seq_diffuser.loss(seq_true=label_aa_s, seq_pred=logit_aa_s, diffusion_mask=~seq_diffusion_mask)
                tot_loss += w_aa*loss # Not scaling loss by timestep
            else:
                # Discrete Diffusion 

                # Reshape logit_aa_s from [B,21,L] to [B,L,20]. 20 aa since seq diffusion cannot handle gap character
                p_logit_aa_s = logit_aa_s[:,:20].transpose(1,2) # [B,L,21]

                intseq_t = torch.argmax(seq_t, dim=-1)
                loss, loss_aux, loss_vb = self.seq_diffuser.loss(x_t=intseq_t, x_0=seq, p_logit_x_0=p_logit_aa_s, t=t, diffusion_mask=seq_diffusion_mask)
                tot_loss += w_aa*loss # Not scaling loss by timestep
                
                loss_dict['loss_aux'] = float(loss_aux.detach())
                loss_dict['loss_vb']  = float(loss_vb.detach())
        else:
            # Classic Autoregressive Sequence Prediction
            loss = self.loss_fn(logit_aa_s, label_aa_s.reshape(B, -1))
            loss = loss * mask_aa_s.reshape(B, -1)
            loss = loss.sum() / (mask_aa_s.sum() + 1e-8)

        loss_s.append(loss[None].detach())

        loss_dict['aa_cce'] = loss.clone()

        ######################################
        #### squared L2 loss on rotations ####
        ###################################### 
        I,B,L = pred.shape[:3]
        N_pred, Ca_pred, C_pred = pred[:,:,:,0], pred[:,:,:,1], pred[:,:,:,2]
        N_true, Ca_true, C_true = true[:,:,0], true[:,:,1], true[:,:,2]
        
        # get predicted frames 
        R_pred,_ = rigid_from_3_points(N_pred.reshape(I*B,L,3), 
                                       Ca_pred.reshape(I*B,L,3), 
                                       C_pred.reshape(I*B,L,3))
        
        R_pred = R_pred.reshape(I,B,L,3,3)
        # get true frames 
        R_true,_ = rigid_from_3_points(N_true, Ca_true, C_true)


        # calculate frame distance loss 
        loss_frame_dist = loss_aa.frame_distance_loss(R_pred, R_true.squeeze(), is_sm) # NOTE: loss calc assumes batch size 1 due to squeeze 
        loss_dict['frame_sqL2'] = loss_frame_dist.clone()


        # convert to axis angle representation and calculate axis-angle loss 
        # axis_angle_pred = rot_conv.matrix_to_axis_angle(R_pred)
        # axis_angle_true = rot_conv.matrix_to_axis_angle(R_true)
        # ax_ang_loss = axis_angle_loss(axis_angle_pred, axis_angle_true)
        ax_ang_loss = 0 
        assert w_ax_ang == 0.0, 'Axis angle loss turned off by DJ'
        # append to dictionary  
        loss_dict['axis_angle'] = ax_ang_loss

        #######################
        ##### FAPE losses #####
        #######################

        # Structural loss
        tot_str, str_loss = calc_str_loss(pred, true, mask_2d, same_chain, negative=negative,
                                              A=10.0, d_clamp=None if unclamp else 10.0, gamma=1.0)
    
        
        # FAPE on protein motif (non SM) residues -- has smaller normalizer and smaller clamp distance
        t2d_is_revealed = masks_1d['t2d_is_revealed']
        t2d_is_revealed_protein_only = t2d_is_revealed.clone()
        # mask out all (i,j) pairs where i or j is a small molecule
        t2d_is_revealed_protein_only[is_sm,:] = False 
        t2d_is_revealed_protein_only[:,is_sm] = False
        
        if is_prot_motif.sum() > 0:
            had_prot_motif = True
            tot_motif_fape, _ = calc_str_loss(pred, true, t2d_is_revealed_protein_only.to(device=pred.device), same_chain, negative=negative,
                                                A=norm_fape/2, d_clamp=None if unclamp else clamp_fape/2, gamma=1.0)
        else: 
            had_prot_motif = False
            tot_motif_fape = float('nan')
        
        # FAPE on non-motif residues
        tot_nonmotif_fape, _ = calc_str_loss(pred, true, ~t2d_is_revealed.to(device=pred.device), same_chain, negative=negative,
                                             A=norm_fape, d_clamp=None if unclamp else clamp_fape, gamma=1.0)
        
        ### Ligand Intra FAPE ###
        dclamp_sm = 4.0
        Z_sm = 4.0
        dclamp = clamp_fape
        Z = norm_fape 
        res_mask = ~((mask_crds[:,:,:3].sum(dim=-1) < 3.0) * ~(rf2aa.util.is_atom(seq)))
    
        # create 2d masks for intrachain and interchain fape calculations
        nframes = frame_mask.shape[-1]
        frame_atom_mask_2d_allatom = torch.einsum('bfn,bra->bfnra', frame_mask_BB, mask_crds).bool() # B, L, nframes, L, natoms
        frame_atom_mask_2d = frame_atom_mask_2d_allatom[:, :, :, :, :3]
        frame_atom_mask_2d_intra_allatom = frame_atom_mask_2d_allatom * same_chain[:, :,None, :, None].bool().expand(-1,-1,nframes,-1, rf2aa.chemical.NTOTAL)
        frame_atom_mask_2d_intra = frame_atom_mask_2d_intra_allatom[:, :, :, :, :3]
        different_chain = ~same_chain.bool()
        frame_atom_mask_2d_inter = frame_atom_mask_2d*different_chain[:, :,None, :, None].expand(-1,-1,nframes,-1, 3)

        # FAPE within ligand
        sm_res_mask = (rf2aa.util.is_atom(seq)*res_mask).squeeze()
        
        # DJ - fix masks so invalid sm atoms are not scored 
        frame_atom_mask_2d_intra[...,sm_res_mask, 0] = False
        frame_atom_mask_2d_intra[...,sm_res_mask, 2] = False
        mask_crds[...,sm_res_mask, 0] = False
        mask_crds[...,sm_res_mask, 2] = False

        

        if rf2aa.util.is_atom(seq).sum() > 0: 
            had_sm = True
            l_fape_sm_intra, _, _ = compute_general_FAPE(
                    pred[:,sm_res_mask[None],:,:3],
                    true[:,sm_res_mask,:3,:3],
                    atom_mask           = mask_crds[:,sm_res_mask, :3],
                    frames              = frames_BB[:,sm_res_mask],
                    frame_mask          = frame_mask_BB[:,sm_res_mask],
                    frame_atom_mask_2d  = frame_atom_mask_2d_intra[:, sm_res_mask][:, :, :, sm_res_mask],
                    dclamp=dclamp_sm,
                    Z=Z_sm
                )
            # l_fape_sm_intra is (N,) shape, need to apply weighted sum
            l_fape_sm_intra = weighted_decay_sum(l_fape_sm_intra, gamma=0.99) # in ./loss.py 
        
        else: 
            had_sm = False 
            l_fape_sm_intra = float('nan') # will skip adding later 

        #########################################################
        # FAPE protein --> ligand and ligand --> protein

        if (had_sm and had_prot_motif): 

            # inter_p_sm_atom_mask = torch.ones_like(sm_res_mask).bool()
            # inter_p_sm_frame_mask = torch.ones_like(sm_res_mask).bool()
            sm_prot_inter_frame_atom_is_scored = torch.zeros((L,nframes,L,rf2aa.chemical.NTOTAL), device=pred.device).bool()
            
            # Cursed scatter of True's into 2D mask
            # sm_prot_inter_frame_atom_is_scored[is_sm_motif, 0, is_prot_motif, :3] = True
            score_2d = sm_prot_inter_frame_atom_is_scored
            a,b,c,d = score_2d.shape
            # SM frames against protin BB atoms 
            for i in range(a):                          
                if is_sm_motif[i]:                  
                    for k in range(c):
                        if is_prot_motif[k]: 
                            # set i,0,k,:3 -->True 
                            score_2d[i,0,k,:3] = True   
            
            # Prot frames against SM atoms (only available atom is first index)
            #sm_prot_inter_frame_atom_is_scored[is_prot_motif, 0, is_sm_motif, 1:2] = True
            for i in range(a): 
                if is_prot_motif[i]:
                    for k in range(c):
                        if is_sm_motif[k]:
                            # set i,0,k,1:2 --> True 
                            score_2d[i,0,k,1:2] = True 


            l_fape_prot_sm_inter, _, _ = compute_general_FAPE(
                pred[...,:3,:].squeeze(),
                true[...,:3,:],
                atom_mask           = mask_crds[...,:3],
                frames              = frames_BB,
                frame_mask          = frame_mask_BB,
                frame_atom_mask_2d  = sm_prot_inter_frame_atom_is_scored[None,...,:3],
                dclamp              = dclamp, 
                Z                   = Z)
            
            l_fape_prot_sm_inter= weighted_decay_sum(l_fape_prot_sm_inter, gamma=0.99) # in ./loss.py

        else: 
            l_fape_prot_sm_inter = float('nan')


        # Add FAPE losses to loss dict 
        loss_s.append(str_loss)

        loss_dict['tot_str']             = tot_str
        loss_dict['motif_fape']          = tot_motif_fape
        loss_dict['nonmotif_fape']       = tot_nonmotif_fape
        loss_dict['ligand_intra_fape']   = l_fape_sm_intra
        loss_dict['prot_lig_inter_fape'] = l_fape_prot_sm_inter

        ########################
        # Done with FAPE calcs #
        ########################


        # Report RMSD on motif chunks 
        is_protein_motif = masks_1d['input_str_mask']
        is_protein_motif[is_atom] = False # protein only motif
        is_not_protein_motif = ~is_protein_motif
        is_not_protein_motif[is_atom] = False # protein only non-motif 

        motif_rmsd      = calc_discontiguous_motif_rmsd(pred.detach(), true.detach(), is_protein_motif)
        non_motif_rmsd  = calc_discontiguous_motif_rmsd(pred.detach(), true.detach(), ~is_protein_motif)
        ligand_rmsd     = calc_discontiguous_motif_rmsd(pred.detach(), true.detach(), is_atom.to(is_protein_motif.device))
        
        # check if there was just a single ligand atom
        if rf2aa.util.is_atom(seq).sum() <= 1:
            ligand_rmsd = float('nan')

        loss_dict['motif_rmsd']         = motif_rmsd
        loss_dict['non_motif_rmsd']     = non_motif_rmsd
        loss_dict['ligand_rmsd']        = ligand_rmsd
        loss_weights['motif_rmsd']      = 0.0
        loss_weights['non_motif_rmsd']  = 0.0
        loss_weights['ligand_rmsd']     = 0.0
        
        
        tot_loss = rf2aa.util.resolve_loss_summation(tot_loss, 
                                                     loss_dict, 
                                                     loss_weights, 
                                                     had_sm, 
                                                     had_prot_motif)
                    
            

        # tot_loss += 0.0 * (pred_tors.mean() + pred_lddt.mean())

        # # AllAtom loss
        # # get ground-truth torsion angles
        # true_tors, true_tors_alt, tors_mask, tors_planar = util.get_torsions(true, seq, self.ti_dev, self.ti_flip, self.ang_ref, mask_in=mask_crds)
        # # masking missing residues as well
        # tors_mask *= mask_BB[...,None] # (B, L, 10)

        # # get alternative coordinates for ground-truth
        # true_alt = torch.zeros_like(true)
        # true_alt.scatter_(2, self.l2a[seq,:,None].repeat(1,1,1,3), true)
        
        # natRs_all, _n0 = self.compute_allatom_coords(seq, true[...,:3,:], true_tors)
        # natRs_all_alt, _n1 = self.compute_allatom_coords(seq, true_alt[...,:3,:], true_tors_alt)
        # predTs = pred[-1,...]
        # predRs_all, pred_all = self.compute_allatom_coords(seq, predTs, pred_tors[-1]) 

        # #  - resolve symmetry
        # xs_mask = self.aamask[seq] # (B, L, 27)
        # xs_mask[0,:,14:]=False # (ignore hydrogens except lj loss)
        # xs_mask *= mask_crds # mask missing atoms & residues as well
        # natRs_all_symm, nat_symm = resolve_symmetry(pred_all[0], natRs_all[0], true[0], natRs_all_alt[0], true_alt[0], xs_mask[0])
        # #frame_mask = torch.cat( [torch.ones((L,1),dtype=torch.bool,device=tors_mask.device), tors_mask[0]], dim=-1 )
        # frame_mask = torch.cat( [mask_BB[0][:,None], tors_mask[0,:,:8]], dim=-1 ) # only first 8 torsions have unique frames

        # # allatom fape and torsion angle loss
        # if negative: # inter-chain fapes should be ignored for negative cases
        #     L1 = same_chain[0,0,:].sum()
        #     frame_maskA = frame_mask.clone()
        #     frame_maskA[L1:] = False
        #     xs_maskA = xs_mask.clone()
        #     xs_maskA[0, L1:] = False
        #     l_fape_A = compute_FAPE(
        #         predRs_all[0,frame_maskA][...,:3,:3], 
        #         predRs_all[0,frame_maskA][...,:3,3], 
        #         pred_all[xs_maskA][...,:3], 
        #         natRs_all_symm[frame_maskA][...,:3,:3], 
        #         natRs_all_symm[frame_maskA][...,:3,3], 
        #         nat_symm[xs_maskA[0]][...,:3],
        #         eps=1e-4)
        #     frame_maskB = frame_mask.clone()
        #     frame_maskB[:L1] = False
        #     xs_maskB = xs_mask.clone()
        #     xs_maskB[0,:L1] = False
        #     l_fape_B = compute_FAPE(
        #         predRs_all[0,frame_maskB][...,:3,:3], 
        #         predRs_all[0,frame_maskB][...,:3,3], 
        #         pred_all[xs_maskB][...,:3], 
        #         natRs_all_symm[frame_maskB][...,:3,:3], 
        #         natRs_all_symm[frame_maskB][...,:3,3], 
        #         nat_symm[xs_maskB[0]][...,:3],
        #         eps=1e-4)
        #     fracA = float(L1)/len(same_chain[0,0])
        #     l_fape = fracA*l_fape_A + (1.0-fracA)*l_fape_B
        # else:
        #     l_fape = compute_FAPE(
        #         predRs_all[0,frame_mask][...,:3,:3], 
        #         predRs_all[0,frame_mask][...,:3,3], 
        #         pred_all[xs_mask][...,:3], 
        #         natRs_all_symm[frame_mask][...,:3,:3], 
        #         natRs_all_symm[frame_mask][...,:3,3], 
        #         nat_symm[xs_mask[0]][...,:3],
        #         eps=1e-4)
        # l_tors = torsionAngleLoss(
        #     pred_tors,
        #     true_tors,
        #     true_tors_alt,
        #     tors_mask,
        #     tors_planar,
        #     eps = 1e-10)
        
        # # torsion timestep scheduling taken care of by w_all scheduling 
        # tot_loss += w_all*w_str*(l_fape+l_tors)
        # loss_s.append(l_fape[None].detach())
        # loss_s.append(l_tors[None].detach())

        # loss_dict['fape'] = float(l_fape.detach())
        # loss_dict['tors'] = float(l_tors.detach())

        # predicted lddt loss

        # lddt_loss, ca_lddt = calc_lddt_loss(pred[:,:,:,1].detach(), true[:,:,1], pred_lddt, idx, mask_BB, mask_2d, same_chain, negative=negative)
        # tot_loss += w_lddt*lddt_loss*lddt_loss_tscale
        # loss_s.append(lddt_loss.detach()[None])
        # loss_s.append(ca_lddt.detach())
    
        # loss_dict['ca_lddt'] = float(ca_lddt[-1].detach())
        # loss_dict['lddt_loss'] = float(lddt_loss.detach())
        
        # # allatom lddt loss
        # true_lddt = calc_allatom_lddt(pred_all[0,...,:14,:3], nat_symm[...,:14,:3], xs_mask[0,...,:14], idx[0], same_chain[0], negative=negative)
        # loss_s.append(true_lddt[None].detach())
        # loss_dict['allatom_lddt'] = float(true_lddt.detach())
        # #loss_s.append(true_lddt.mean()[None].detach())
        
        # # bond geometry

        # blen_loss, bang_loss = calc_BB_bond_geom(pred[-1], true, mask_BB)
        # if w_blen > 0.0:
        #     tot_loss += w_blen*blen_loss*blen_tscale
        # if w_bang > 0.0:
        #     tot_loss += w_bang*bang_loss*bang_tscale

        # loss_dict['blen'] = float(blen_loss.detach())
        # loss_dict['bang'] = float(bang_loss.detach())

        # # lj potential
        # lj_loss = calc_lj(
        #     seq[0], pred_all[0,...,:3], 
        #     self.aamask, same_chain[0], 
        #     self.ljlk_parameters, self.lj_correction_parameters, self.num_bonds,
        #     lj_lin=lj_lin, use_H=use_H, negative=negative)

        # if w_lj > 0.0:
        #     tot_loss += w_lj*lj_loss*lj_tscale

        # loss_dict['lj'] = float(lj_loss.detach())

        # # hbond [use all atoms not just those in native]
        # hb_loss = calc_hb(
        #     seq[0], pred_all[0,...,:3], 
        #     self.aamask, self.hbtypes, self.hbbaseatoms, self.hbpolys)
        # if w_hb > 0.0:
        #     tot_loss += w_hb*hb_loss*hb_tscale

        # loss_s.append(torch.stack((blen_loss, bang_loss, lj_loss, hb_loss)).detach())

        # loss_dict['hb'] = float(hb_loss.detach())
        
        loss_dict['total_loss'] = float(tot_loss.detach())
        loss_dict = rf2aa.tensor_util.cpu(loss_dict)

        return tot_loss, loss_dict, loss_weights

    def calc_acc(self, prob, dist, idx_pdb, mask_2d, return_cnt=False):
        B = idx_pdb.shape[0]
        L = idx_pdb.shape[1] # (B, L)
        seqsep = torch.abs(idx_pdb[:,:,None] - idx_pdb[:,None,:]) + 1
        mask = seqsep > 24
        mask = torch.triu(mask.float())
        mask *= mask_2d
        #
        cnt_ref = dist < 20
        cnt_ref = cnt_ref.float() * mask
        #
        cnt_pred = prob[:,:20,:,:].sum(dim=1) * mask
        #
        top_pred = torch.topk(cnt_pred.view(B,-1), L)
        kth = top_pred.values.min(dim=-1).values
        tmp_pred = list()
        for i_batch in range(B):
            tmp_pred.append(cnt_pred[i_batch] > kth[i_batch])
        cnt_pred = torch.stack(tmp_pred, dim=0)
        cnt_pred = cnt_pred.float()*mask
        #
        condition = torch.logical_and(cnt_pred==cnt_ref, cnt_ref==torch.ones_like(cnt_ref))
        n_good = condition.float().sum()
        n_total = (cnt_ref == torch.ones_like(cnt_ref)).float().sum() + 1e-9
        n_total_pred = (cnt_pred == torch.ones_like(cnt_pred)).float().sum() + 1e-9
        prec = n_good / n_total_pred
        recall = n_good / n_total
        F1 = 2.0*prec*recall / (prec+recall+1e-9)
        if return_cnt:
            return torch.stack([prec, recall, F1]), cnt_pred, cnt_ref

        return torch.stack([prec, recall, F1])

    def load_model(self, model, optimizer, scheduler, scaler, model_name, rank, suffix='last', resume_train=False):

        #chk_fn = "models/%s_%s.pt"%(model_name, suffix)
        #assert not (self.ckpt_load_path is None )
        chk_fn = self.ckpt_load_path

        loaded_epoch = 0
        best_valid_loss = 999999.9
        if self.zero_weights:
            return 0, best_valid_loss
        if not os.path.exists(chk_fn):
            raise Exception(f'no model found at path: {chk_fn}, pass -zero_weights if you intend to train the model with no initialization and no starting weights')
        print('*** FOUND MODEL CHECKPOINT ***')
        print('Located at ',chk_fn)

        ic(rank)
        if isinstance(rank, str):
            # For CPU debugging
            map_location = {"cuda:%d"%0: "cpu"}
        else:
            map_location = {"cuda:%d"%0: "cuda:%d"%rank}
        checkpoint = torch.load(chk_fn, map_location=map_location)
        rename_model = False
        # Set to false for faster loading when debugging
        cautious = True
        for m, weight_state in [
            (model.module.model, checkpoint['final_state_dict']),
            (model.module.shadow, checkpoint['model_state_dict']),
        ]:
            if self.reinitialize_missing_params:
                model_state = m.state_dict()
                if cautious:
                    new_chk = {}
                    for param in model_state:
                        if param not in weight_state:
                            print ('missing',param)
                            rename_model=True
                        elif (weight_state[param].shape == model_state[param].shape):
                            new_chk[param] = weight_state[param]
                        else:
                            print (
                                'wrong size',param,
                                weight_state[param].shape,
                                model_state[param].shape )

                else:
                    new_chk = weight_state
                m.load_state_dict(new_chk, strict=False)
            else:
                m.load_state_dict(weight_state, strict=True)

        if resume_train and (not rename_model):
            print (' ... loading optimization params')
            loaded_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                #print (' ... loading scheduler params')
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                scheduler.last_epoch = loaded_epoch + 1
            #if 'best_loss' in checkpoint:
            #    best_valid_loss = checkpoint['best_loss']
        return loaded_epoch, best_valid_loss

    def checkpoint_fn(self, model_name, description):
        if not os.path.exists(f"{self.outdir}/models"):
            os.mkdir(f"{self.outdir}/models")
        name = "%s_%s.pt"%(model_name, description)
        return os.path.join(f"{self.outdir}/models", name)
    
    # main entry function of training
    # 1) make sure ddp env vars set
    # 2) figure out if we launched using slurm or interactively
    #   - if slurm, assume 1 job launched per GPU
    #   - if interactive, launch one job for each GPU on node
    def run_model_training(self, world_size):
        if ('MASTER_ADDR' not in os.environ or os.environ['MASTER_ADDR'] == ''):
            ic('setting master_addr')
            os.environ['MASTER_ADDR'] = 'localhost' # multinode requires this set in submit script
        if ('MASTER_PORT' not in os.environ):
            os.environ['MASTER_PORT'] = '%d'%self.port

        if (not self.interactive and "SLURM_NTASKS" in os.environ and "SLURM_PROCID" in os.environ):
            world_size = int(os.environ["SLURM_NTASKS"])
            rank = int (os.environ["SLURM_PROCID"])
            print ("Launched from slurm", rank, world_size)
            self.train_model(rank, world_size)
        else:
            print ("Launched from interactive")
            world_size = torch.cuda.device_count()
            ic(world_size)
            if world_size <= 1:
                self.train_model(0, 1)
            else:
                mp.spawn(self.train_model, args=(world_size,), nprocs=world_size, join=True)

    def train_model(self, rank, world_size, return_setup=False):
        #print ("running ddp on rank %d, world_size %d"%(rank, world_size))
        
        # save git diff from most recent commit
        gitdiff_fn = open(f'{self.outdir}/git_diff.txt','w')
        git_diff = subprocess.Popen(["git diff"], cwd = os.getcwd(), shell = True, stdout = gitdiff_fn, stderr = subprocess.PIPE)
        print('Saved git diff between current state and last commit')


        if WANDB and rank == 0:
            print('initializing wandb')
            resume = None
            name='_'.join([self.wandb_prefix, self.outdir.replace('./','')])
            id = None
            if self.resume:
                name=None
                id=self.resume
                resume='must'
            wandb.init(
                    project="motif_scaffold_na",
                    entity="bakerlab", 
                    name=name,
                    id=id,
                    resume=resume)
            print(f'{wandb.run.id=}')

            all_param = {}
            all_param.update(self.loader_param)
            all_param.update(self.model_param)
            all_param.update(self.loss_param)
            all_param.update(self.diffusion_param)

            # wandb.config.update(all_param)
            wandb.config = all_param
            wandb.save(os.path.join(os.getcwd(), self.outdir, 'git_diff.txt'))
        ic(os.environ['MASTER_ADDR'], rank, world_size, torch.cuda.device_count())
        print(f'{rank=} {world_size=} initializing process group')
        dist.init_process_group(backend="gloo", world_size=world_size, rank=rank)
        print(f'{rank=} {world_size=} initialized process group')
        if torch.cuda.device_count():
            gpu = rank % torch.cuda.device_count()
            torch.cuda.set_device("cuda:%d"%gpu)
        else:
            gpu = 'cpu'
        
        self.n_train = N_EXAMPLE_PER_EPOCH


        dataset_configs, homo = default_dataset_configs(self.loader_param, debug=DEBUG)
        
        print('Making train sets')
        ic(self.config_dict)
        train_set = DistilledDataset(dataset_configs, 
                                     self.loader_param, self.diffuser, self.seq_diffuser, self.ti_dev, self.ti_flip, self.ang_ref,
                                     self.diffusion_param, self.preprocess_param, self.model_param, self.config_dict, homo)
        #get proportion of seq2str examples
        if 'seq2str' in self.loader_param['TASK_NAMES']:
            p_seq2str = self.loader_param['TASK_P'][self.loader_param['TASK_NAMES'].index('seq2str')]
        else:
            p_seq2str = 0

        train_sampler = DistributedWeightedSampler(dataset_configs,
                                                   dataset_options=self.loader_param['DATASETS'],
                                                   dataset_prob=self.loader_param['DATASET_PROB'],
                                                   num_example_per_epoch=N_EXAMPLE_PER_EPOCH,
                                                   num_replicas=world_size, rank=rank, replacement=True)
        
        print('THIS IS LOAD PARAM GOING INTO DataLoader inits')
        print(LOAD_PARAM)

        train_loader = data.DataLoader(train_set, sampler=train_sampler, batch_size=self.batch_size, collate_fn=no_batch_collate_fn, **LOAD_PARAM)

        # move some global data to cuda device
        self.ti_dev = self.ti_dev.to(gpu)
        self.ti_flip = self.ti_flip.to(gpu)
        self.ang_ref = self.ang_ref.to(gpu)
        self.l2a = self.l2a.to(gpu)
        self.aamask = self.aamask.to(gpu)
        #self.compute_allatom_coords = self.compute_allatom_coords.to(gpu)

        self.num_bonds = self.num_bonds.to(gpu)
        self.ljlk_parameters = self.ljlk_parameters.to(gpu)
        self.lj_correction_parameters = self.lj_correction_parameters.to(gpu)
        self.hbtypes = self.hbtypes.to(gpu)
        self.hbbaseatoms = self.hbbaseatoms.to(gpu)
        self.hbpolys = self.hbpolys.to(gpu)

        
        #self.diffuser.get_allatom = self.compute_allatom_coords 
        
        ## JW I have changed this so we always just put Lx22 sequence into the embedding
        #print(f'Using onehot sequence (Lx22) input for model')
        #self.model_param['input_seq_onehot'] = True
        
        # define model
        print('Making model...')
        ic(self.model_param)
        # Set to zero in rf_diffusion
        # model_param.pop('d_time_emb')
        # model_param.pop('d_time_emb_proj')
        # Unused
        # model_param.pop('use_motif_timestep')

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
        self.ti_dev = self.ti_dev.to(gpu)
        self.ti_flip = self.ti_flip.to(gpu)
        self.ang_ref = self.ang_ref.to(gpu)
        self.fi_dev = self.fi_dev.to(gpu)
        self.l2a = self.l2a.to(gpu)
        self.aamask = self.aamask.to(gpu)
        #self.compute_allatom_coords = self.compute_allatom_coords.to(gpu)
        self.num_bonds = self.num_bonds.to(gpu)
        self.atom_type_index = self.atom_type_index.to(gpu)
        self.ljlk_parameters = self.ljlk_parameters.to(gpu)
        self.lj_correction_parameters = self.lj_correction_parameters.to(gpu)
        self.hbtypes = self.hbtypes.to(gpu)
        self.hbbaseatoms = self.hbbaseatoms.to(gpu)
        self.hbpolys = self.hbpolys.to(gpu)
        self.cb_len = self.cb_len.to(gpu)
        self.cb_ang = self.cb_ang.to(gpu)
        self.cb_tor = self.cb_tor.to(gpu)

        ddp_model, optimizer, scheduler, scaler, loaded_epoch = self.init_model(gpu)

        if return_setup:
            return ddp_model, train_loader, optimizer, scheduler, scaler
        
        #valid_pdb_sampler.set_epoch(0)
        #valid_homo_sampler.set_epoch(0)
        #valid_compl_sampler.set_epoch(0)
        #valid_neg_sampler.set_epoch(0)
        #valid_tot, valid_loss, valid_acc = self.valid_pdb_cycle(ddp_model, valid_pdb_loader, rank, gpu, world_size, loaded_epoch)
        #_, _, _ = self.valid_pdb_cycle(ddp_model, valid_homo_loader, rank, gpu, world_size, loaded_epoch, header="Homo")
        #_, _, _ = self.valid_ppi_cycle(ddp_model, valid_compl_loader, valid_neg_loader, rank, gpu, world_size, loaded_epoch)
        for epoch in range(loaded_epoch+1, self.n_epoch+1):
            train_sampler.set_epoch(epoch)
            
            print('Just before calling train cycle...')
            train_tot, train_loss, train_acc = self.train_cycle(ddp_model, train_loader, optimizer, scheduler, scaler, rank, gpu, world_size, epoch)
            #valid_tot, valid_loss, valid_acc = self.valid_pdb_cycle(ddp_model, valid_pdb_loader, rank, gpu, world_size, epoch)
            #_, _, _ = self.valid_pdb_cycle(ddp_model, valid_homo_loader, rank, gpu, world_size, epoch, header="Homo")
            #_, _, _ = self.valid_ppi_cycle(ddp_model, valid_compl_loader, valid_neg_loader, rank, gpu, world_size, epoch)
            
            #valid_tot, valid_loss, valid_acc = self.valid_cycle(ddp_model, valid_loader, rank, gpu, world_size, epoch)
            if rank == 0: # save model
                self.save_model(epoch, ddp_model, optimizer, scheduler, scaler)

        dist.destroy_process_group()

    def save_model(self, suffix, ddp_model, optimizer, scheduler, scaler):
        #save every epoch     
        model_path = self.checkpoint_fn(self.model_name, str(suffix))
        print(f'saving model to {model_path}')
        torch.save({'epoch': suffix,
                    #'model_state_dict': ddp_model.state_dict(),
                    'model_state_dict': ddp_model.module.shadow.state_dict(),
                    'final_state_dict': ddp_model.module.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'config_dict':self.config_dict,
                    'training_arguments': self.training_arguments},
                    model_path)

    def init_model(self, device):
        model = RoseTTAFoldModule(
            symmetrize_repeats=None, 
            repeat_length=None,
            symmsub_k=None,
            sym_method=None,
            main_block=None,
            copy_main_block_template=None,
            **self.model_param,
            aamask=self.aamask,
            atom_type_index=self.atom_type_index,
            ljlk_parameters=self.ljlk_parameters,
            lj_correction_parameters=self.lj_correction_parameters,
            num_bonds=self.num_bonds,
            cb_len = self.cb_len,
            cb_ang = self.cb_ang,
            cb_tor = self.cb_tor,
            lj_lin=self.loss_param['lj_lin']
            ).to(device)
        
        if self.log_inputs:
            pickle_dir, self.pickle_counter = pickle_function_call(model, 'forward', 'training', minifier=aa_model.minifier)
            print(f'pickle_dir: {pickle_dir}')
        if self.verbose_checks:
            model.verbose_checks = True
        model = EMA(model, 0.999)
        print('Instantiating DDP')
        ddp_model = model
        if torch.cuda.device_count():
            ddp_model = DDP(model, device_ids=[device], find_unused_parameters=False, broadcast_buffers=False)
        else:
            ddp_model = DDP(model, find_unused_parameters=False)
        # if rank == 0:
        #     print ("# of parameters:", count_parameters(ddp_model))
        
        # define optimizer and scheduler
        opt_params = add_weight_decay(ddp_model, self.l2_coeff)
        #optimizer = torch.optim.Adam(opt_params, lr=self.init_lr)
        optimizer = torch.optim.AdamW(opt_params, lr=self.init_lr)
        #scheduler = get_stepwise_decay_schedule_with_warmup(optimizer, 1000, 10000, 0.95) # For initial round of training
        #scheduler = get_stepwise_decay_schedule_with_warmup(optimizer, 100, 10000, 0.95) # Trialled using this in diffusion training
        scheduler = get_stepwise_decay_schedule_with_warmup(optimizer, 0, 10000, 0.95) # for fine-tuning
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
       
        # load model
        print('About to load model...')
        loaded_epoch, best_valid_loss = self.load_model(ddp_model, optimizer, scheduler, scaler, 
                                                       self.model_name, device, resume_train=False)

        print('Done loading model')

        return ddp_model, optimizer, scheduler, scaler, loaded_epoch

        if loaded_epoch >= self.n_epoch:
            DDP_cleanup()
            return

    def train_cycle(self, ddp_model, train_loader, optimizer, scheduler, scaler, rank, gpu, world_size, epoch):

        print('Entering self.train_cycle')
        # Turn on training mode
        ddp_model.train()
        
        # clear gradients
        optimizer.zero_grad()

        start_time = time.time()
        
        counter = 0
        
        print('About to enter train loader loop')
        for loader_out in train_loader:
            

            indep, rfi_tp1_t, chosen_dataset, item, little_t, is_diffused, chosen_task, atomizer, masks_1d, item_context = loader_out
            context_msg = f'rank: {rank}: {item_context}'
            # ipdb.set_trace()
            
            if indep.seq.shape[0] <= 3:
                # skip SM examples w/ too few atoms
                continue 

            # If we want to filter data inputs based on polymer content:
            if self.polymer_focus:
                passes_content_check = util.polymer_content_check(indep.seq, self.polymer_focus,  self.polymer_frac_cutoff)
                # Skip if example doesn't have enough of our focus polymer
                if not passes_content_check:
                    continue




            with error.context(context_msg):
                rfi_tp1, rfi_t = rfi_tp1_t

                N_cycle = np.random.randint(1, self.maxcycle+1) # number of recycling

                # Defensive assertions
                assert little_t > 0
                assert N_cycle == 1, 'cycling not implemented'
                i_cycle = N_cycle-1
                assert(( self.preprocess_param['prob_self_cond'] == 0 ) ^ \
                    ( self.preprocess_param['str_self_cond'] or self.preprocess_param['seq_self_cond'] )), \
                    'prob_self_cond must be > 0 for str_self_cond or seq_self_cond to be active'

                # Checking whether this example was of poor quality and the dataloader just returned None - NRB
                if indep.seq.shape[0] == 0:
                    ic('Train cycle received bad example, skipping')
                    continue

                # Save trues for writing pdbs later.
                xyz_prev_orig = rfi_tp1.xyz[0]
                seq_unmasked = indep.seq[None]

                # for saving pdbs
                seq_original = torch.clone(indep.seq)

                # transfer inputs to device
                B, _, L, _ = rfi_t.msa_latent.shape
                rf2aa.tensor_util.to_device(rfi_t, gpu)

                counter += 1 


                # get diffusion_mask for the displacement loss
                diffusion_mask     = ~is_diffused

                unroll_performed = False

                # Some percentage of the time, provide the model with the model's prediction of x_0 | x_t+1
                # When little_t == T should not unroll as we cannot go back further in time.
                step_back = not (little_t == self.config_dict['diffuser']['T']) and (torch.tensor(self.preprocess_param['prob_self_cond']) > torch.rand(1))
                if step_back:
                    unroll_performed = True
                    rf2aa.tensor_util.to_device(rfi_tp1, gpu)

                    # Take 1 step back in time to get the training example to feed to the model
                    # For this model evaluation msa_prev, pair_prev, and state_prev are all None and i_cycle is
                    # constant at 0
                    if DEBUG: 
                        # torch.save(rfi_tp1, 'rfi_tp1_bugfixed.pt')
                        pass 

                    with torch.no_grad():
                        with ddp_model.no_sync():
                            with torch.cuda.amp.autocast(enabled=USE_AMP):
                                # ipdb.set_trace()
                                rfo = aa_model.forward(
                                        ddp_model,
                                        rfi_tp1,
                                        use_checkpoint=False,
                                        return_raw=False
                                        )
                                # ipdb.set_trace()
                                rfi_t = aa_model.self_cond(indep, rfi_t, rfo,
                                                        twotemplate=self.preprocess_param['twotemplate'], 
                                                        threetemplate=self.preprocess_param['threetemplate'],
                                        )
                                # xyz_prev_orig = rfi_t.xyz[0,:,:14].clone()
                                # ipdb.set_trace()
                                xyz_prev_orig = rfi_t.xyz[0,:,:rf2aa.chemical.NHEAVY].clone()
                if DEBUG:
                    # torch.save(rfi_t, 'rfi_t_bugfixed.pt')
                    # sys.exit('exiting early')
                    # rfo_dict = dataclasses.asdict(rfo)
                    # for k,v in rfo_dict.items():
                    #     if torch.is_tensor(v):
                    #         ic(k, torch.isnan(v).sum())
                    # sys.exit('debugging')
                    pass

                with ExitStack() as stack:
                    if counter%self.ACCUM_STEP != 0:
                        stack.enter_context(ddp_model.no_sync())
                    with torch.cuda.amp.autocast(enabled=USE_AMP):
                        rfo = aa_model.forward(
                                        ddp_model,
                                        rfi_t,
                                        # N_cycle=N_cycle,
                                        use_checkpoint=True,
                                        **({model_input_logger.LOG_ONLY_KEY: {'t':int(little_t), 'item': item}} if self.log_inputs else {}))

                        logit_s, logit_aa_s, logits_pae, logits_pde, p_bind, pred_crds, alphas, px0_allatom, pred_lddts, _, _, _, _ = rfo.unsafe_astuple()
                        is_diffused = is_diffused.to(gpu)
                        indep.seq = indep.seq.to(gpu)
                        indep.is_sm = indep.is_sm.to(gpu)
                        indep.xyz = indep.xyz.to(gpu)
                        indep.same_chain = indep.same_chain.to(gpu)
                        indep.idx = indep.idx.to(gpu)
                        true_crds = torch.zeros((1,L,36,3)).to(gpu)
                        indep.natstack = indep.natstack.to(gpu)
                        indep.maskstack = indep.maskstack.to(gpu)

                        # true_crds[0,:,:14,:] = indep.xyz[:,:14,:]
                        true_crds[0,:,:rf2aa.chemical.NHEAVY,:] = indep.xyz[:,:rf2aa.chemical.NHEAVY,:]
                        mask_crds = ~torch.isnan(true_crds).any(dim=-1)
                        if all([len(a) > 0 for a in indep.Ls]): 
                            # we have prot and sm
                            true_crds, mask_crds = resolve_equiv_natives_asmb(pred_crds[-1], 
                                                                              indep.natstack[None], 
                                                                              indep.maskstack[None], 
                                                                              indep.Ls)
                        
                        else: 
                            # only prot or only sm 
                            true_crds, mask_crds = resolve_equiv_natives(pred_crds[-1], 
                                                                         indep.natstack[None], 
                                                                         indep.maskstack[None])

                        

                        # mask_crds[:,~is_diffused,:] = False
                        assert not (~is_diffused).any(), 'There is a non-diffused token but assumption is motif_only_2d'

                        ### Dj - commented next lines out, they are not used in calc loss + assertion fails w/ sm only
                        # mask_BB = ~indep.is_sm[None]
                        # mask_2d = mask_BB[:,None,:] * mask_BB[:,:,None] # ignore pairs having missing residues
                        # assert torch.sum(mask_2d) > 0, "mask_2d is blank"
                        # mask_BB = None 
                        # mask_2d = None
                        mask_disto_1d = torch.ones_like(indep.seq).bool()[None]
                        mask_disto_2d = mask_disto_1d[:,None,:] * mask_disto_1d[:,:,None]
                        mask_BB = mask_disto_1d
                        mask_2d = mask_disto_2d

                        true_crds_frame = rf2aa.util.xyz_to_frame_xyz(true_crds, indep.seq[None], indep.atom_frames[None])
                        c6d = rf2aa.kinematics.xyz_to_c6d(true_crds_frame)
                        negative = torch.tensor([False])
                        c6d = rf2aa.kinematics.c6d_to_bins(c6d, indep.same_chain[None], negative=negative)

                        label_aa_s = indep.seq[None, None]
                        mask_aa_s = is_diffused[None, None]
                        same_chain = indep.same_chain[None]
                        seq_diffusion_mask = torch.ones(L).bool()
                        seq_t = torch.nn.functional.one_hot(indep.seq, 80)[None].float()
                        xyz_t = rfi_t.xyz[None]
                        unclamp = torch.tensor([False])

                        # Useful logging
                        if False:
                            is_protein_motif = ~is_diffused * ~indep.is_sm
                            idx_diffused = torch.nonzero(is_diffused)
                            idx_protein_motif  = torch.nonzero(is_protein_motif)
                            idx_sm = torch.nonzero(indep.is_sm)

                        seq_diffusion_mask[:] = True
                        # mask_crds[:] = False
                        true_crds[:,indep.is_protein,rf2aa.chemical.NHEAVYPROT:] = 0
                        true_crds[:,indep.is_na,rf2aa.chemical.NHEAVYNUC:] = 0
                        xyz_t[:] = 0
                        seq_t[:] = 0

                        # get atomized frames for general FAPE computation in calc_loss
                        loss, loss_dict, loss_weights = self.calc_loss(
                                logit_s, 
                                c6d,
                                logit_aa_s, 
                                label_aa_s, 
                                mask_aa_s, 
                                None,
                                pred_crds, 
                                alphas, 
                                true_crds, 
                                mask_crds,
                                mask_BB, 
                                mask_2d, 
                                same_chain,
                                pred_lddts, 
                                indep.idx[None], 
                                chosen_dataset, 
                                chosen_task, 
                                diffusion_mask=diffusion_mask,
                                seq_diffusion_mask=seq_diffusion_mask, 
                                seq_t=seq_t, 
                                xyz_in=xyz_t, 
                                is_sm=indep.is_sm, 
                                unclamp=unclamp,
                                negative=negative, 
                                t=int(little_t), 
                                masks_1d=masks_1d,
                                atom_frames=rfi_t.atom_frames,
                                **self.loss_param)


                        # Loss dict containing losses scaled by their weights 
                        weighted_loss_dict = dict()
                        for k,v in loss_dict.items(): 
                            w = loss_weights.get(k,1)
                            weighted_loss_dict[k] = (loss_dict[k]*w)

                        weighted_loss_dict = rf2aa.tensor_util.cpu(weighted_loss_dict)
                        
                        # Force all model parameters to participate in loss. Truly a cursed workaround.
                        loss += 0.0 * (logits_pae.mean() + logits_pde.mean() + alphas.mean() + pred_lddts.mean() + p_bind.mean())
                    loss = loss / self.ACCUM_STEP

                    if DEBUG and False: 
                        report_gradient_norms(loss_dict, ddp_model, scaler, optimizer, int(little_t))

                    if gpu != 'cpu':
                        print(f'DEBUG: {rank=} {gpu=} {counter=} size: {indep.xyz.shape[0]} {torch.cuda.max_memory_reserved(gpu) / 1024**3:.2f} GB reserved {torch.cuda.max_memory_allocated(gpu) / 1024**3:.2f} GB allocated {torch.cuda.get_device_properties(gpu).total_memory / 1024**3:.2f} GB total')
                    if not torch.isnan(loss):
                        with torch.autograd.set_detect_anomaly(True):
                            scaler.scale(loss).backward()
                    else:
                        msg = f'NaN loss encountered, skipping: {context_msg}'
                        if not DEBUG:
                            print(msg)
                        else:
                            raise Exception(msg)


                    if counter%self.ACCUM_STEP == 0:
                        # gradient clipping
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), self.grad_clip) # default is 0.2 
                        scaler.step(optimizer)
                        scale = scaler.get_scale()
                        scaler.update()
                        skip_lr_sched = (scale != scaler.get_scale())
                        optimizer.zero_grad()
                        if not skip_lr_sched:
                            scheduler.step()
                        ddp_model.module.update() # apply EMA
                
                ## check parameters with no grad
                #if rank == 0:
                #    for n, p in ddp_model.named_parameters():
                #        if p.grad is None and p.requires_grad is True:
                #            print('Parameter not used:', n, p.shape)  # prints unused parameters. Remove them from your model
                

                log_metrics = (counter % N_PRINT_TRAIN == 0) and (rank == 0)
                if log_metrics:
                    max_mem = torch.cuda.max_memory_allocated()/1e9
                    train_time  = time.time() - start_time
                    
                    #### Writing losses to stdout ####
                    if 'diff' in chosen_task:
                        task_str = f'diff_t{int(little_t)}'
                    else:
                        task_str = chosen_task
                    
                    outstr = f"Local {task_str} | {chosen_dataset[0]}: [{epoch}/{self.n_epoch}] Batch: [{counter*self.batch_size*world_size}/{self.n_train}] Time: {train_time} Loss dict: "

                    str_stack = []
                    for k in sorted(list(loss_dict.keys())):
                        str_stack.append(f'{k}--{round( float(loss_dict[k]), 4)}')
                    outstr += '  '.join(str_stack)
                    sys.stdout.write(outstr+'\n')

                     #### Writing WEIGHTED losses to stdout #### 
                    w_outstr = f"Local WEIGHTED {task_str} | {chosen_dataset[0]}: [{epoch}/{self.n_epoch}] Batch: [{counter*self.batch_size*world_size}/{self.n_train}] Time: {train_time} Loss dict: "
                    wstr_stack = [] 
                    for k in sorted(list(loss_dict.keys())):
                        w = loss_weights.get(k,1)
                        wstr_stack.append(f'{k}--{round(float(loss_dict[k]*w), 4)}')
                    w_outstr += '  '.join(wstr_stack)
                    sys.stdout.write(w_outstr+'\n')

                    
                    if rank == 0:
                        loss_dict.update({'t':little_t, 'total_examples':epoch*self.n_train+counter*world_size, 'dataset':chosen_dataset[0], 'task':chosen_task[0]})
                        metrics = {}
                        for m in self.metrics:
                            with torch.no_grad():
                                if hasattr(m, 'accepts_indep') and m.accepts_indep:
                                    rf2aa.tensor_util.to_device(indep, 'cpu')
                                    metrics.update(m(indep, pred_crds[-1, 0].cpu(), is_diffused.cpu()))
				# Currently broken
                                # metrics.update(m(logit_s, c6d,
                                # logit_aa_s, label_aa_s, mask_aa_s, None,
                                # pred_crds, alphas, true_crds, mask_crds,
                                # mask_BB, mask_2d, same_chain,
                                # pred_lddts, indep.idx[None], chosen_dataset, chosen_task, diffusion_mask=diffusion_mask,
                                # seq_diffusion_mask=seq_diffusion_mask, seq_t=seq_t, xyz_in=xyz_t, unclamp=unclamp,
                                # negative=negative, t=int(little_t), **self.loss_param)
                                # **self.loss_param))
                        loss_dict['metrics'] = metrics
                        loss_dict['loss_weights'] = loss_weights
                        if WANDB:
                            wandb.log(loss_dict)
                    sys.stdout.flush()
                
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
		# TODO: Use these when writing output PDBs
                # logits_argsort = torch.argsort(logit_aa_s, dim=1, descending=True)
                # top1_sequence = (logits_argsort[:, :1])
                # top1_sequence = torch.clamp(top1_sequence, 0,19)

                # if self.diffusion_param['seqdiff'] == 'continuous':
                #     top1_sequence = torch.argmax(logit_aa_s[:,:20,:], dim=1)

                n_processed = self.batch_size*world_size * counter
                save_pdb = np.random.randint(0,self.n_write_pdb) == 0
                if save_pdb:
                    (L,) = indep.seq.shape
                    pdb_dir = os.path.join(self.outdir, 'training_pdbs')
                    os.makedirs(pdb_dir, exist_ok=True)
                    prefix = f'{pdb_dir}/epoch_{epoch}_{n_processed}_{chosen_task}_{chosen_dataset}_t_{int( little_t )}'

                    rf2aa.tensor_util.to_device(indep, 'cpu')

                    pred_xyz = xyz_prev_orig.clone()
                    # ipdb.set_trace()
                    pred_xyz[:,:3] = pred_crds[-1, 0]
                    # ipdb.set_trace()

                    for suffix, xyz in [
                        ('input', xyz_prev_orig),
                        ('pred', pred_xyz),
                        ('true', indep.xyz),
                    ]:
                        indep_write = copy.deepcopy(indep)
                        # indep_write.xyz[:,:14] = xyz[:,:14]
                        indep_write.xyz[:,:rf2aa.chemical.NHEAVY] = xyz[:,:rf2aa.chemical.NHEAVY]
                        if atomizer:
                            indep_write = atomize.deatomize(atomizer, indep_write)
                        indep_write.write_pdb(f'{prefix}_{suffix}.pdb')

                    indep_true = indep
                    motif_deatomized = None
                    if atomizer:
                        indep_true = atomize.deatomize(atomizer, indep_true)
                        motif_deatomized = atomize.convert_atomized_mask(atomizer, ~is_diffused)

                    rfi_t_writeable = copy.deepcopy(rfi_t)
                    rf2aa.tensor_util.to_device(rfi_t_writeable, torch.device('cpu'))

                    with open(f'{prefix}_info.pkl', 'wb') as fh:
                        pickle.dump({
                            'motif': motif_deatomized,
                            'masks_1d': masks_1d,
                            'idx': indep_true.idx,
                            'is_sm': indep_true.is_sm,
                            'rfi_t': rfi_t_writeable,
                        }, fh)

                    if self.log_inputs:
                        shutil.copy(self.pickle_counter.last_pickle, f'{prefix}_input_pickle.pkl')
                    print(f'writing training PDBs with prefix: {prefix}')

                # Expected epoch time logging
                if rank == 0:
                    elapsed_time = time.time() - start_time
                    mean_rate = n_processed / elapsed_time
                    expected_epoch_time = int(self.n_train / mean_rate)
                    m, s = divmod(expected_epoch_time, 60)
                    h, m = divmod(m, 60)
                    print(f'Expected time per epoch of size ({self.n_train}) (h:m:s) based off {n_processed} measured pseudo batch times: {h:d}:{m:02d}:{s:.0f}')

                if self.saves_per_epoch and rank==0:
                    n_processed_next = self.batch_size*world_size * (counter+1)
                    n_fractionals = np.arange(1, self.saves_per_epoch) / self.saves_per_epoch
                    for fraction in n_fractionals:
                        save_before_n = fraction * self.n_train
                        if n_processed <= save_before_n and n_processed_next > save_before_n:
                            self.save_model(f'{epoch - 1 + fraction:.2f}', ddp_model, optimizer, scheduler, scaler)
                            break

        # TODO(fix or delete)
        # write total train loss
        # train_tot /= float(counter * world_size)
        # train_loss /= float(counter * world_size)
        # train_acc  /= float(counter * world_size)

        # dist.all_reduce(train_tot, op=dist.ReduceOp.SUM)
        # dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        # dist.all_reduce(train_acc, op=dist.ReduceOp.SUM)
        # train_tot = train_tot.cpu().detach()
        # train_loss = train_loss.cpu().detach().numpy()
        # train_acc = train_acc.cpu().detach().numpy()
        train_tot = train_loss = train_acc = -1

        if rank == 0:
            
            train_time = time.time() - start_time
            sys.stdout.write("Train: [%04d/%04d] Batch: [%05d/%05d] Time: %16.1f | total_loss: %8.4f \n"%(\
                    epoch, self.n_epoch, self.n_train, self.n_train, train_time, train_tot, \
                    ))
            sys.stdout.flush()

            
        return train_tot, train_loss, train_acc


def make_trainer(args, model_param, loader_param, loss_param, diffusion_param, preprocess_param):
    
    global N_EXAMPLE_PER_EPOCH
    global DEBUG 
    global WANDB
    global LOAD_PARAM
    global LOAD_PARAM2
    # set epoch size 
    N_EXAMPLE_PER_EPOCH = args.epoch_size 
    ic(N_EXAMPLE_PER_EPOCH)

    # set global debug and wandb params 
    if args.debug:
        DEBUG = True 
        WANDB = False 
        # loader_param['DATAPKL'] = 'subsampled_dataset.pkl'
        # loader_param['DATAPKL_AA'] = 'subsampled_all-atom-dataset.pkl'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
        print('set os.environ variables')
        ic.configureOutput(includeContext=True)
    else:
        DEBUG = False 
        WANDB = True 
    if not args.wandb:
        WANDB = False
    
    # set load params based on debug
    global LOAD_PARAM
    global LOAD_PARAM2

    max_workers = 8 if preprocess_param['prob_self_cond'] == 0 else 0

    LOAD_PARAM = {'shuffle': False,
              'num_workers': max_workers if not DEBUG else 0,
              'pin_memory': True}
    LOAD_PARAM2 = {'shuffle': False,
              'num_workers': max_workers if not DEBUG else 0,
              'pin_memory': True}


    # set random seed
    run_inference.seed_all(args.seed)
    #np.random.seed(args.seed)

    mp.freeze_support()
    train = Trainer(model_name=args.model_name,
                    ckpt_load_path=args.ckpt_load_path,
                    interactive=args.interactive,
                    n_epoch=args.num_epochs, lr=args.lr, l2_coeff=1.0e-2,
                    port=args.port, model_param=model_param, loader_param=loader_param, 
                    loss_param=loss_param, 
                    batch_size=args.batch_size,
                    accum_step=args.accum,
                    maxcycle=args.maxcycle,
                    diffusion_param=diffusion_param,
                    preprocess_param=preprocess_param,
                    wandb_prefix=args.wandb_prefix,
                    metrics=args.metric,
                    zero_weights=args.zero_weights,
                    log_inputs=args.log_inputs,
                    n_write_pdb=args.n_write_pdb,
                    reinitialize_missing_params=args.reinitialize_missing_params,
                    verbose_checks=args.verbose_checks,
                    saves_per_epoch=args.saves_per_epoch,
                    outdir=args.out_dir,
                    resume=args.resume,
                    grad_clip=args.grad_clip,
                    polymer_focus=args.polymer_focus,
                    polymer_frac_cutoff=args.polymer_frac_cutoff,
                    )
    return train

if __name__ == "__main__":
    from arguments import get_args
    all_args = get_args()
    if not all_args[0].debug:
        import wandb
    train = make_trainer(*all_args)
    train.run_model_training(torch.cuda.device_count())
