import sys, os
import time
import pickle 
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from datetime import date 
import time 
import torch
import torch.nn as nn
from torch.utils import data
from data_loader import (
    get_train_valid_set, loader_pdb, loader_fb, loader_complex, loader_pdb_fixbb, loader_fb_fixbb, loader_complex_fixbb, loader_cn_fixbb,
    Dataset, DatasetComplex, DistilledDataset, DistributedWeightedSampler
)

from kinematics import xyz_to_c6d, c6d_to_bins2, xyz_to_t2d, xyz_to_bbtor, get_init_xyz
from RoseTTAFoldModel  import RoseTTAFoldModule
import loss 
from loss import *
from util import *
from util_module import ComputeAllAtomCoords
from scheduler import get_stepwise_decay_schedule_with_warmup

import rotation_conversions as rot_conv

#added for inpainting training
from icecream import ic
from apply_masks import mask_inputs
import random
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

class Trainer():
    def __init__(self, model_name='BFF', ckpt_load_path=None,
                 n_epoch=100, lr=1.0e-4, l2_coeff=1.0e-2, port=None, interactive=False,
                 model_param={}, loader_param={}, loss_param={}, batch_size=1, accum_step=1, 
                 maxcycle=4, diffusion_param={}, preprocess_param={}, outdir=f'./train_session{get_datetime()}', wandb_prefix='',
                 metrics=None, zero_weights=False, log_inputs=False):

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

        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir, exist_ok=True)
        else:
            pass
            #sys.exit('EXITING: self.outdir already exists. Dont clobber')
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
        seqdiff_kwargs = {'T'             : diffusion_param['diff_T'], # Use same T as for str diff
                          's_b0'           : diffusion_param['seqdiff_b0'],
                          's_bT'           : diffusion_param['seqdiff_bT'],
                          'schedule_type' : diffusion_param['seqdiff_schedule_type']
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
        self.ti_dev = torsion_indices
        self.ti_flip = torsion_can_flip
        self.ang_ref = reference_angles
        self.l2a = long2alt
        self.aamask = allatom_mask
        self.num_bonds = num_bonds
        self.ljlk_parameters = ljlk_parameters
        self.lj_correction_parameters = lj_correction_parameters
        
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

        self.hbtypes = hbtypes
        self.hbbaseatoms = hbbaseatoms
        self.hbpolys = hbpolys

        # module torsion -> allatom
        self.compute_allatom_coords = ComputeAllAtomCoords()

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
    def assemble_config(self) -> None:
        config_dict = {}
        config_dict['model'] = self.model_param
        
        #rename diffusion params to match config
        #infer_names=dict(zip([i for i in self.diffusion_param.keys()],[i[5:] if i[:5] == 'diff_' else i for i in self.diffusion_param.keys()]))
        #config_dict['diffuser'] = {infer_names[k]: v for k, v in self.diffusion_param.items()}
        config_dict['diffuser'] = self.diff_kwargs
        config_dict['seq_diffuser'] = self.seqdiff_kwargs
        # Add seq_diff_type
        config_dict['seq_diffuser']['seq_diff_type'] = self.seq_diff_type
        config_dict['preprocess'] = self.preprocess_param
        self.config_dict = config_dict

    def calc_loss(self, logit_s, label_s,
                  logit_aa_s, label_aa_s, mask_aa_s, logit_exp,
                  pred_in, pred_tors, true, mask_crds, mask_BB, mask_2d, same_chain,
                  pred_lddt, idx, dataset, chosen_task, t, xyz_in, diffusion_mask,
                  seq_diffusion_mask, seq_t, unclamp=False, negative=False,
                  w_dist=1.0, w_aa=1.0, w_str=1.0, w_all=0.5, w_exp=1.0,
                  w_lddt=1.0, w_blen=1.0, w_bang=1.0, w_lj=0.0, w_hb=0.0,
                  lj_lin=0.75, use_H=False, w_disp=0.0, w_motif_disp=0.0, w_ax_ang=0.0, w_frame_dist=0.0, eps=1e-6, backprop_non_displacement_on_given=False):

        #NB t is 1-indexed
        t_idx = t-1
 
        # dictionary for keeping track of losses 
        loss_dict = {}

        B, L = true.shape[:2]
        seq = label_aa_s[:,0].clone()
        assert (B==1) # fd - code assumes a batch size of 1

        loss_s = list()
        tot_loss = 0.0
        
        # get tscales
        c6d_tscale = self.loss_schedules.get('c6d',[1]*(t))[t_idx]
        aa_tscale = self.loss_schedules.get('aa_cce',[1]*(t))[t_idx]
        disp_tscale = self.loss_schedules.get('displacement',[1]*(t))[t_idx]
        lddt_loss_tscale = self.loss_schedules.get('lddt_loss',[1]*(t))[t_idx]
        bang_tscale = self.loss_schedules.get('bang',[1]*(t))[t_idx]
        blen_tscale = self.loss_schedules.get('blen',[1]*(t))[t_idx]
        exp_tscale = self.loss_schedules.get('exp',[1]*(t))[t_idx]
        lj_tscale = self.loss_schedules.get('lj',[1]*(t))[t_idx]
        hb_tscale = self.loss_schedules.get('lj',[1]*(t))[t_idx]
        str_tscale = self.loss_schedules.get('w_str',[1]*(t))[t_idx]
        w_all_tscale = self.loss_schedules.get('w_all',[1]*(t))[t_idx]

        # Displacement prediction loss between xyz prev and xyz_true
        if unclamp:
            disp_loss = calc_displacement_loss(pred_in, true, gamma=0.99, d_clamp=None)
        else:
            disp_loss = calc_displacement_loss(pred_in, true, gamma=0.99, d_clamp=10.0)
 
        tot_loss += w_disp*disp_loss*disp_tscale
        loss_dict['displacement'] = float(disp_loss.detach())
 
        # Displacement prediction loss between xyz prev and xyz_true for only motif region.
        if diffusion_mask.any():
            motif_disp_loss = calc_displacement_loss(pred_in[:,:,diffusion_mask], true[:,diffusion_mask], gamma=0.99, d_clamp=None)
            tot_loss += w_motif_disp*motif_disp_loss*disp_tscale
            loss_dict['motif_displacement'] = float(motif_disp_loss.detach())
 
        if backprop_non_displacement_on_given:
            pred = pred_in
        else:
            pred = torch.clone(pred_in)
            pred[:,:,diffusion_mask] = pred_in[:,:,diffusion_mask].detach()

        # c6d loss
        for i in range(4):
            # schedule factor for c6d 
            # syntax is if it's not in the scheduling dict, loss has full weight (i.e., 1x)

            loss = self.loss_fn(logit_s[i], label_s[...,i]) # (B, L, L)
            loss = (mask_2d*loss).sum() / (mask_2d.sum() + eps)
            tot_loss += w_dist*loss*c6d_tscale
            loss_s.append(loss[None].detach())

            loss_dict[f'c6d_{i}'] = float(loss.detach())

        if not self.seq_diffuser is None:
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
            tot_loss += w_aa*loss*aa_tscale

        loss_s.append(loss[None].detach())

        loss_dict['aa_cce'] = float(loss.detach())

        loss = nn.BCEWithLogitsLoss()(logit_exp, mask_BB.float())
        tot_loss += w_exp*loss*exp_tscale

        loss_s.append(loss[None].detach())

        loss_dict['exp_resolved'] = float(loss.detach())

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
        loss_frame_dist = frame_distance_loss(R_pred, R_true.squeeze()) # NOTE: loss calc assumes batch size 1 due to squeeze 
        loss_dict['frame_sqL2'] = float(loss_frame_dist.detach())
        tot_loss += w_frame_dist*loss_frame_dist*disp_tscale #NOTE: scheduled same as coordinate displacement loss 

        # convert to axis angle representation and calculate axis-angle loss 
        axis_angle_pred = rot_conv.matrix_to_axis_angle(R_pred)
        axis_angle_true = rot_conv.matrix_to_axis_angle(R_true)
        ax_ang_loss = axis_angle_loss(axis_angle_pred, axis_angle_true)
        
        # append to dictionary  
        loss_dict['axis_angle'] = float(ax_ang_loss.detach())
        tot_loss += w_ax_ang*ax_ang_loss*disp_tscale #NOTE: scheduled same as coordinate displacement loss 

        
        # Calculate displacement on xt-1 backcalculated from px0 and x0.
        # Currently not backpropable
        
        xt1_squared_disp, xt1_disp = track_xt1_displacement(true, pred, xyz_in,
                t, diffusion_mask, self.schedule, self.alphabar_schedule)

        loss_dict['xt1_displacement'] = xt1_disp
        loss_dict['xt1_squared_displacement'] = xt1_squared_disp

        # Structural loss
        if unclamp:
            tot_str, str_loss = calc_str_loss(pred, true, mask_2d, same_chain, negative=negative,
                                              A=10.0, d_clamp=None, gamma=1.0)
        else:
            tot_str, str_loss = calc_str_loss(pred, true, mask_2d, same_chain, negative=negative,
                                              A=10.0, d_clamp=10.0, gamma=1.0)
        
        # dj - str loss timestep scheduling: 
        # scale w_all to keep the contributions of BB/ALLatom fape summing to 1.0
        w_all = w_all*w_all_tscale

        tot_loss += (1.0-w_all)*w_str*tot_str
        loss_s.append(str_loss)
        
        #loss_dict['str_loss'] = float(str_loss.detach())
        loss_dict['tot_str'] = float(tot_str.detach())

        # AllAtom loss
        # get ground-truth torsion angles
        true_tors, true_tors_alt, tors_mask, tors_planar = get_torsions(true, seq, self.ti_dev, self.ti_flip, self.ang_ref, mask_in=mask_crds)
        # masking missing residues as well
        tors_mask *= mask_BB[...,None] # (B, L, 10)

        # get alternative coordinates for ground-truth
        true_alt = torch.zeros_like(true)
        true_alt.scatter_(2, self.l2a[seq,:,None].repeat(1,1,1,3), true)
        
        natRs_all, _n0 = self.compute_allatom_coords(seq, true[...,:3,:], true_tors)
        natRs_all_alt, _n1 = self.compute_allatom_coords(seq, true_alt[...,:3,:], true_tors_alt)
        predTs = pred[-1,...]
        predRs_all, pred_all = self.compute_allatom_coords(seq, predTs, pred_tors[-1]) 

        #  - resolve symmetry
        xs_mask = self.aamask[seq] # (B, L, 27)
        xs_mask[0,:,14:]=False # (ignore hydrogens except lj loss)
        xs_mask *= mask_crds # mask missing atoms & residues as well
        natRs_all_symm, nat_symm = resolve_symmetry(pred_all[0], natRs_all[0], true[0], natRs_all_alt[0], true_alt[0], xs_mask[0])
        #frame_mask = torch.cat( [torch.ones((L,1),dtype=torch.bool,device=tors_mask.device), tors_mask[0]], dim=-1 )
        frame_mask = torch.cat( [mask_BB[0][:,None], tors_mask[0,:,:8]], dim=-1 ) # only first 8 torsions have unique frames

        # allatom fape and torsion angle loss
        if negative: # inter-chain fapes should be ignored for negative cases
            L1 = same_chain[0,0,:].sum()
            frame_maskA = frame_mask.clone()
            frame_maskA[L1:] = False
            xs_maskA = xs_mask.clone()
            xs_maskA[0, L1:] = False
            l_fape_A = compute_FAPE(
                predRs_all[0,frame_maskA][...,:3,:3], 
                predRs_all[0,frame_maskA][...,:3,3], 
                pred_all[xs_maskA][...,:3], 
                natRs_all_symm[frame_maskA][...,:3,:3], 
                natRs_all_symm[frame_maskA][...,:3,3], 
                nat_symm[xs_maskA[0]][...,:3],
                eps=1e-4)
            frame_maskB = frame_mask.clone()
            frame_maskB[:L1] = False
            xs_maskB = xs_mask.clone()
            xs_maskB[0,:L1] = False
            l_fape_B = compute_FAPE(
                predRs_all[0,frame_maskB][...,:3,:3], 
                predRs_all[0,frame_maskB][...,:3,3], 
                pred_all[xs_maskB][...,:3], 
                natRs_all_symm[frame_maskB][...,:3,:3], 
                natRs_all_symm[frame_maskB][...,:3,3], 
                nat_symm[xs_maskB[0]][...,:3],
                eps=1e-4)
            fracA = float(L1)/len(same_chain[0,0])
            l_fape = fracA*l_fape_A + (1.0-fracA)*l_fape_B
        else:
            l_fape = compute_FAPE(
                predRs_all[0,frame_mask][...,:3,:3], 
                predRs_all[0,frame_mask][...,:3,3], 
                pred_all[xs_mask][...,:3], 
                natRs_all_symm[frame_mask][...,:3,:3], 
                natRs_all_symm[frame_mask][...,:3,3], 
                nat_symm[xs_mask[0]][...,:3],
                eps=1e-4)
        l_tors = torsionAngleLoss(
            pred_tors,
            true_tors,
            true_tors_alt,
            tors_mask,
            tors_planar,
            eps = 1e-10)
        
        # torsion timestep scheduling taken care of by w_all scheduling 
        tot_loss += w_all*w_str*(l_fape+l_tors)
        loss_s.append(l_fape[None].detach())
        loss_s.append(l_tors[None].detach())

        loss_dict['fape'] = float(l_fape.detach())
        loss_dict['tors'] = float(l_tors.detach())

        # predicted lddt loss

        lddt_loss, ca_lddt = calc_lddt_loss(pred[:,:,:,1].detach(), true[:,:,1], pred_lddt, idx, mask_BB, mask_2d, same_chain, negative=negative)
        tot_loss += w_lddt*lddt_loss*lddt_loss_tscale
        loss_s.append(lddt_loss.detach()[None])
        loss_s.append(ca_lddt.detach())
    
        loss_dict['ca_lddt'] = float(ca_lddt[-1].detach())
        loss_dict['lddt_loss'] = float(lddt_loss.detach())
        
        # allatom lddt loss
        true_lddt = calc_allatom_lddt(pred_all[0,...,:14,:3], nat_symm[...,:14,:3], xs_mask[0,...,:14], idx[0], same_chain[0], negative=negative)
        loss_s.append(true_lddt[None].detach())
        loss_dict['allatom_lddt'] = float(true_lddt.detach())
        #loss_s.append(true_lddt.mean()[None].detach())
        
        # bond geometry

        blen_loss, bang_loss = calc_BB_bond_geom(pred[-1], true, mask_BB)
        if w_blen > 0.0:
            tot_loss += w_blen*blen_loss*blen_tscale
        if w_bang > 0.0:
            tot_loss += w_bang*bang_loss*bang_tscale

        loss_dict['blen'] = float(blen_loss.detach())
        loss_dict['bang'] = float(bang_loss.detach())

        # lj potential
        lj_loss = calc_lj(
            seq[0], pred_all[0,...,:3], 
            self.aamask, same_chain[0], 
            self.ljlk_parameters, self.lj_correction_parameters, self.num_bonds,
            lj_lin=lj_lin, use_H=use_H, negative=negative)

        if w_lj > 0.0:
            tot_loss += w_lj*lj_loss*lj_tscale

        loss_dict['lj'] = float(lj_loss.detach())

        # hbond [use all atoms not just those in native]
        hb_loss = calc_hb(
            seq[0], pred_all[0,...,:3], 
            self.aamask, self.hbtypes, self.hbbaseatoms, self.hbpolys)
        if w_hb > 0.0:
            tot_loss += w_hb*hb_loss*hb_tscale

        loss_s.append(torch.stack((blen_loss, bang_loss, lj_loss, hb_loss)).detach())

        loss_dict['hb'] = float(hb_loss.detach())
        
        loss_dict['total_loss'] = float(tot_loss.detach())

        return tot_loss, torch.cat(loss_s, dim=0), loss_dict

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

        if DEBUG:
            chk_fn='debug'

        loaded_epoch = -1
        best_valid_loss = 999999.9
        if not os.path.exists(chk_fn):
            if self.zero_weights:
                return -1, best_valid_loss
            raise Exception(f'no model found at path: {chk_fn}, pass -zero_weights if you intend to train the model with no initialization and no starting weights')
        print('*** FOUND MODEL CHECKPOINT ***')
        print('Located at ',chk_fn)

        map_location = {"cuda:%d"%0: "cuda:%d"%rank}
        checkpoint = torch.load(chk_fn, map_location=map_location)
        rename_model = False
        new_chk = {}
        ctr=0
        for param in model.module.model.state_dict():
            #print('On param ',ctr,' of ',len(model.module.model.state_dict()))
            ctr += 1

            if param not in checkpoint['model_state_dict']:
                print ('missing',param)
                rename_model=True
            elif (checkpoint['model_state_dict'][param].shape == model.module.model.state_dict()[param].shape):
                new_chk[param] = checkpoint['model_state_dict'][param]
            else:
                print (
                    'wrong size',param,
                    checkpoint['model_state_dict'][param].shape,
                     model.module.model.state_dict()[param].shape )

        model.module.model.load_state_dict(new_chk, strict=False)
        model.module.shadow.load_state_dict(new_chk, strict=False)

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
            if world_size == 1:
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
            wandb.init(
                    project="fancy-pants ",
                    entity="bakerlab", 
                    name='_'.join([self.wandb_prefix, self.outdir.replace('./','')]))

            all_param = {}
            all_param.update(self.loader_param)
            all_param.update(self.model_param)
            all_param.update(self.loss_param)
            all_param.update(self.diffusion_param)

            wandb.config = all_param
            wandb.save(os.path.join(os.getcwd(), self.outdir, 'git_diff.txt'))
        gpu = rank % torch.cuda.device_count()
        ic(os.environ['MASTER_ADDR'])
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
        torch.cuda.set_device("cuda:%d"%gpu)

        #define dataset & data loader
        print('Getting train/valid set...')
        pdb_items, fb_items, compl_items, neg_items, cn_items, valid_pdb, valid_homo, valid_compl, valid_neg, valid_cn, homo = get_train_valid_set(self.loader_param)
        pdb_IDs, pdb_weights, pdb_dict = pdb_items
        fb_IDs, fb_weights, fb_dict = fb_items
        compl_IDs, compl_weights, compl_dict = compl_items
        neg_IDs, neg_weights, neg_dict = neg_items
        cn_IDs, cn_weights, cn_dict = cn_items
        
        self.n_train = N_EXAMPLE_PER_EPOCH
        self.n_valid_pdb = len(valid_pdb.keys())
        self.n_valid_pdb = (self.n_valid_pdb // world_size)*world_size
        self.n_valid_homo = len(valid_homo.keys())
        self.n_valid_homo = (self.n_valid_homo // world_size)*world_size
        self.n_valid_compl = len(valid_compl.keys())
        self.n_valid_compl = (self.n_valid_compl // world_size)*world_size
        self.n_valid_neg = len(valid_neg.keys())
        self.n_valid_neg = (self.n_valid_neg // world_size)*world_size
        self.n_valid_cn = len(valid_cn.keys())
        self.n_valid_neg = (self.n_valid_neg // world_size)*world_size

        # ic(type(pdb_items))
        # ic(type(fb_items))
        # ic(type(compl_items))
        # ic(type(neg_items))

        # ic(type(pdb_dict))
        # ic(type(fb_dict))
        # ic(type(compl_dict))
        # ic(type(neg_dict))

        # ic(type(pdb_IDs))
        # ic(type(fb_IDs))
        # ic(type(compl_IDs))
        # ic(type(neg_IDs))



        if (rank==0):
            print ('Loaded',
                len(valid_pdb.keys()),'monomers,',
                len(valid_homo.keys()),'homomers,',
                len(valid_compl.keys()),'heteromers, and',
                len(valid_neg.keys()),'negative heteromer'
            )
            print ('Using',
                self.n_valid_pdb,'monomers,',
                self.n_valid_homo,'homomers,',
                self.n_valid_compl,'heteromers, and',
                self.n_valid_neg,'negative heteromers',
            )
        
        print('Making train sets')
        train_set = DistilledDataset(pdb_IDs, loader_pdb, loader_pdb_fixbb, pdb_dict,
                                     compl_IDs, loader_complex, loader_complex_fixbb, compl_dict,
                                     neg_IDs, loader_complex, neg_dict,
                                     fb_IDs, loader_fb, loader_fb_fixbb, fb_dict,
                                     cn_IDs, None, loader_cn_fixbb, cn_dict, # None is a placeholder as we don't currently have a loader_cn
                                     homo, self.loader_param, self.diffuser, self.seq_diffuser, self.ti_dev, self.ti_flip, self.ang_ref, 
                                     self.diffusion_param, self.preprocess_param, self.model_param)

        valid_pdb_set = Dataset(list(valid_pdb.keys())[:self.n_valid_pdb],
                                loader_pdb, valid_pdb,
                                self.loader_param, homo, p_homo_cut=-1.0)
        valid_homo_set = Dataset(list(valid_homo.keys())[:self.n_valid_homo],
                                loader_pdb, valid_homo,
                                self.loader_param, homo, p_homo_cut=2.0)
        valid_compl_set = DatasetComplex(list(valid_compl.keys())[:self.n_valid_compl],
                                         loader_complex, valid_compl,
                                         self.loader_param, negative=False)
        valid_neg_set = DatasetComplex(list(valid_neg.keys())[:self.n_valid_neg],
                                        loader_complex, valid_neg,
                                        self.loader_param, negative=True)
 
        #get proportion of seq2str examples
        if 'seq2str' in self.loader_param['TASK_NAMES']:
            p_seq2str = self.loader_param['TASK_P'][self.loader_param['TASK_NAMES'].index('seq2str')]
        else:
            p_seq2str = 0

        train_sampler = DistributedWeightedSampler(train_set, pdb_weights, compl_weights, neg_weights, fb_weights, cn_weights, p_seq2str,
                                                   dataset_options=self.loader_param['DATASETS'],
                                                   dataset_prob=self.loader_param['DATASET_PROB'],
                                                   num_example_per_epoch=N_EXAMPLE_PER_EPOCH,
                                                   num_replicas=world_size, rank=rank, replacement=True)

        valid_pdb_sampler = data.distributed.DistributedSampler(valid_pdb_set, num_replicas=world_size, rank=rank)
        valid_homo_sampler = data.distributed.DistributedSampler(valid_homo_set, num_replicas=world_size, rank=rank)
        valid_compl_sampler = data.distributed.DistributedSampler(valid_compl_set, num_replicas=world_size, rank=rank)
        valid_neg_sampler = data.distributed.DistributedSampler(valid_neg_set, num_replicas=world_size, rank=rank)
        
        print('THIS IS LOAD PARAM GOING INTO DataLoader inits')
        print(LOAD_PARAM)
        train_loader = data.DataLoader(train_set, sampler=train_sampler, batch_size=self.batch_size, **LOAD_PARAM)
        valid_pdb_loader = data.DataLoader(valid_pdb_set, sampler=valid_pdb_sampler, **LOAD_PARAM)
        valid_homo_loader = data.DataLoader(valid_homo_set, sampler=valid_homo_sampler, **LOAD_PARAM2)
        valid_compl_loader = data.DataLoader(valid_compl_set, sampler=valid_compl_sampler, **LOAD_PARAM)
        valid_neg_loader = data.DataLoader(valid_neg_set, sampler=valid_neg_sampler, **LOAD_PARAM)

        # move some global data to cuda device
        self.ti_dev = self.ti_dev.to(gpu)
        self.ti_flip = self.ti_flip.to(gpu)
        self.ang_ref = self.ang_ref.to(gpu)
        self.l2a = self.l2a.to(gpu)
        self.aamask = self.aamask.to(gpu)
        self.compute_allatom_coords = self.compute_allatom_coords.to(gpu)

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
        model = RoseTTAFoldModule(**self.model_param, d_t1d=self.preprocess_param['d_t1d'], d_t2d=self.preprocess_param['d_t2d'], T=self.diffusion_param['diff_T']).to(gpu)
        if self.log_inputs:
            pickle_dir = pickle_function_call(model, 'forward', 'training')
            print(f'pickle_dir: {pickle_dir}')

        model = EMA(model, 0.999)
        print('Instantiating DDP')
        ddp_model = DDP(model, device_ids=[gpu], find_unused_parameters=False)
        if rank == 0:
            print ("# of parameters:", count_parameters(ddp_model))
        
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
                                                       self.model_name, gpu, resume_train=False)

        print('Done loading model')

        if loaded_epoch >= self.n_epoch:
            DDP_cleanup()
            return

        if return_setup:
            return ddp_model, train_loader, optimizer, scheduler, scaler
        
        #valid_pdb_sampler.set_epoch(0)
        #valid_homo_sampler.set_epoch(0)
        #valid_compl_sampler.set_epoch(0)
        #valid_neg_sampler.set_epoch(0)
        #valid_tot, valid_loss, valid_acc = self.valid_pdb_cycle(ddp_model, valid_pdb_loader, rank, gpu, world_size, loaded_epoch)
        #_, _, _ = self.valid_pdb_cycle(ddp_model, valid_homo_loader, rank, gpu, world_size, loaded_epoch, header="Homo")
        #_, _, _ = self.valid_ppi_cycle(ddp_model, valid_compl_loader, valid_neg_loader, rank, gpu, world_size, loaded_epoch)
        for epoch in range(loaded_epoch+1, self.n_epoch):
            train_sampler.set_epoch(epoch)
            valid_pdb_sampler.set_epoch(epoch)
            valid_homo_sampler.set_epoch(epoch)
            valid_compl_sampler.set_epoch(epoch)
            valid_neg_sampler.set_epoch(epoch)
            
            print('Just before calling train cycle...')
            train_tot, train_loss, train_acc = self.train_cycle(ddp_model, train_loader, optimizer, scheduler, scaler, rank, gpu, world_size, epoch)
            #valid_tot, valid_loss, valid_acc = self.valid_pdb_cycle(ddp_model, valid_pdb_loader, rank, gpu, world_size, epoch)
            #_, _, _ = self.valid_pdb_cycle(ddp_model, valid_homo_loader, rank, gpu, world_size, epoch, header="Homo")
            #_, _, _ = self.valid_ppi_cycle(ddp_model, valid_compl_loader, valid_neg_loader, rank, gpu, world_size, epoch)
            
            #valid_tot, valid_loss, valid_acc = self.valid_cycle(ddp_model, valid_loader, rank, gpu, world_size, epoch)

            if rank == 0: # save model
                """
                if valid_tot < best_valid_loss:
                    best_valid_loss = valid_tot
                    torch.save({'epoch': epoch,
                                #'model_state_dict': ddp_model.state_dict(),
                                'model_state_dict': ddp_model.module.shadow.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'scaler_state_dict': scaler.state_dict(),
                                'best_loss': best_valid_loss,
                                'train_loss': train_loss,
                                'train_acc': train_acc,
                                'valid_loss': valid_loss,
                                'valid_acc': valid_acc},
                                self.checkpoint_fn(self.model_name, 'best'))
                """
                #save every epoch     
                torch.save({'epoch': epoch,
                            #'model_state_dict': ddp_model.state_dict(),
                            'model_state_dict': ddp_model.module.shadow.state_dict(),
                            'final_state_dict': ddp_model.module.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'scaler_state_dict': scaler.state_dict(),
                            'train_loss': train_loss,
                            'train_acc': train_acc,
                            'valid_loss': 999.9,
                            'valid_acc': 999.9,
                            'best_loss': 999.9,
                            'config_dict':self.config_dict},
                            self.checkpoint_fn(self.model_name, str(epoch)))
                
        dist.destroy_process_group()

    def train_cycle(self, ddp_model, train_loader, optimizer, scheduler, scaler, rank, gpu, world_size, epoch):

        print('Entering self.train_cycle')
        # Turn on training mode
        ddp_model.train()
        
        # clear gradients
        optimizer.zero_grad()

        start_time = time.time()
        
        # For intermediate logs
        local_tot = 0.0
        local_loss = None
        local_acc = None
        train_tot = 0.0
        train_loss = None
        train_acc = None

        counter = 0
        
        print('About to enter train loader loop')
        for loader_out in train_loader:

            seq,\
            msa,\
            msa_masked,\
            msa_full,\
            mask_msa,\
            true_crds,\
            mask_crds,\
            idx_pdb,\
            xyz_t,\
            t1d,\
            t2d,\
            alpha_t,\
            xyz_prev,\
            same_chain,\
            unclamp,\
            negative,\
            masks_1d,\
            chosen_task,\
            chosen_dataset,\
            little_t = loader_out

            '''
                Current Dimensions:
                
                seq (torch.tensor)        : [B,n,I,L,22] noised one hot sequence

                msa (torch.tensor)        : [B,I,N_long,L]
                
                msa_masked (torch.tensor) : [B,n,I,N_short,L,48] 
                
                msa_full (torch.tensor)   : [B,n,I,N_long,L,25]

                mask_msa (torch.tensor)   : [B,n,N_short,L] The msa mask at t 

                true_crds (torch.tensor)  : [B,L,27,3]

                mask_crds (torch.tensor)  : [B,L,27]

                idx_pdb (torch.tensor)    : [B,L]
                
                xyz_t (torch.tensor)      : [B,n,T,L,27,3] Noised true coordinates at t+1 and t
                
                t1d (torch.tensor)        : [B,n,T,L,33] Template 1D features at t+1 and t

                t2d (torch.tensor)        : [B,n,T,L,L,44] Template 2D features at t+1 and t
                
                alpha_t (torch.tensor)    : [B,n,T,L,30]

                xyz_prev (torch.tensor)   : [B,n,L,27,3]
                
                ...

                little_t (torch.tensor)   : [n] The timesteps t+1 and t
            '''
            assert all([i > 0 for i in little_t])

            # Checking whether this example was of poor quality and the dataloader just returned None - NRB
            if seq.shape[1] == 0:
                ic('Train cycle received bad example, skipping')
                continue

            #ic(seq.shape)
            #ic(msa.shape)
            #ic(msa_masked.shape)
            #ic(msa_full.shape)
            #ic(mask_msa.shape)
            #ic(true_crds.shape)
            #ic(mask_crds.shape)
            #ic(idx_pdb.shape)
            #ic(xyz_t.shape)
            #ic(t1d.shape)
            #ic(xyz_prev.shape)
            #ic(same_chain)
            # ic(unclamp)
            # ic(negative)
            # ic(masks_1d)
            # ic(chosen_task)
            # ic(chosen_dataset)
            # ic(atom_mask)

            # torch.save(torch.clone(xyz_t), 'xyz_t.pt')
            # torch.save(torch.clone(seq), 'seq.pt')
            # torch.save(torch.clone(atom_mask), 'atom_mask.pt')



            # for saving pdbs
            seq_original = torch.clone(seq)

            # Do diffusion + apply masks 
            start = time.time()

            # for saving pdbs
            seq_masked = torch.clone(seq)
            xyz_t_in = torch.clone(xyz_t)

            # transfer inputs to device
            B, _, N, L = msa.shape

            idx_pdb = idx_pdb.to(gpu, non_blocking=True) # (B, L)
            true_crds = true_crds.to(gpu, non_blocking=True) # (B, L, 27, 3)
            mask_crds = mask_crds.to(gpu, non_blocking=True) # (B, L, 27)
            same_chain = same_chain.to(gpu, non_blocking=True)

            xyz_t = xyz_t.to(gpu, non_blocking=True)
            t1d = t1d.to(gpu, non_blocking=True)
            xyz_t = xyz_t.to(gpu, non_blocking=True)
            xyz_prev = xyz_prev.to(gpu, non_blocking=True)
            seq = seq.to(gpu, non_blocking=True)
            msa = msa.to(gpu, non_blocking=True)
            msa_masked = msa_masked.to(gpu, non_blocking=True)
            msa_full = msa_full.to(gpu, non_blocking=True)
            mask_msa = mask_msa.to(gpu, non_blocking=True)
            xyz_prev = xyz_prev.to(gpu, non_blocking=True)

            counter += 1 

            N_cycle = np.random.randint(1, self.maxcycle+1) # number of recycling

            # get diffusion_mask for the displacement loss
            diffusion_mask     = masks_1d['input_str_mask'].squeeze()
            seq_diffusion_mask = masks_1d['input_seq_mask'].squeeze().to(gpu, non_blocking=True)

            unroll_performed = False

            assert(( self.preprocess_param['prob_self_cond'] == 0 ) ^ \
                   ( self.preprocess_param['str_self_cond'] or self.preprocess_param['seq_self_cond'] )), \
                  'prob_self_cond must be > 0 for str_self_cond or seq_self_cond to be active'

            # Some percentage of the time, provide the model with the model's prediction of x_0 | x_t+1
            # When little_t[0] == little_t[1] we are at t == T so we should not unroll
            if not (little_t[0] == little_t[1]) and (torch.tensor(self.preprocess_param['prob_self_cond']) > torch.rand(1)):

                unroll_performed = True

                # Take 1 step back in time to get the training example to feed to the model
                # For this model evaluation msa_prev, pair_prev, and state_prev are all None and i_cycle is
                # constant at 0
                with torch.no_grad():
                    with ddp_model.no_sync():
                        with torch.cuda.amp.autocast(enabled=USE_AMP):

                            if self.preprocess_param['seq_self_cond']:
                                # When doing sequence self-conditioning we give the model the perturbed seq and the model's prediction of the seq
                                # These are in two lines in the MSA and in this order: [model pred, perturbed true]
                                # Since this is the step before self conditioning, we just make the model pred all zeros

                                use_msa_masked = torch.stack((msa_masked[:,0,:,0],msa_masked[:,0,:,0]), dim=2) # [B,I,N_short,L,48]
                                use_msa_masked[:,:,0] = 0

                                use_msa_full = torch.stack((msa_full[:,0,:,0],msa_full[:,0,:,0]), dim=2) # [B,I,N_long,L,25]
                                use_msa_full[:,:,0] = 0

                            else:
                                # When doing no sequence self-conditioning we an MSA with depth == 1 - NRB
                                use_msa_masked = msa_masked[:,0]
                                use_msa_full   = msa_full[:,0]

                            # Select timestep t = t+1
                            use_seq        = seq[:,0]
                            use_xyz_prev   = xyz_prev[:,0] # grab t entry, could also make this None when not self conditioning
                            use_t1d        = t1d[:,0]
                            use_t2d        = t2d[:,0] # May want to make this zero for self cond
                            use_xyz_t      = xyz_t[:,0]
                            use_alpha_t    = alpha_t[:,0]

                            _, sequence_logits, _, px0_xyz, _, _ = ddp_model(
                                                                       use_msa_masked[:,0],
                                                                       use_msa_full[:,0],
                                                                       use_seq[:,0],
                                                                       use_xyz_prev,
                                                                       idx_pdb,
                                                                       t=little_t[0],
                                                                       t1d=use_t1d,
                                                                       t2d=use_t2d,
                                                                       xyz_t=use_xyz_t,
                                                                       alpha_t=use_alpha_t,
                                                                       msa_prev=None,
                                                                       pair_prev=None,
                                                                       state_prev=None,
                                                                       return_raw=False,
                                                                       motif_mask=diffusion_mask
                                                                       )

            # From here on out we will operate with t = t
            seq        = seq[:,1]
            t1d        = t1d[:,1]
            alpha_t    = alpha_t[:,1]
            little_t   = little_t[1]
            xyz_prev   = xyz_prev[:,1]

            # Only provide previous xyz - NRB
            msa_prev   = None
            pair_prev  = None
            state_prev = None

            if self.preprocess_param['str_self_cond']:
                if unroll_performed:
                    # When doing self conditioning training, the model only receives px0_xyz information through template features
                    # Turn model's prediction of x0 structure into xyz_t and t2d
                    I,B,L = px0_xyz.shape[:3]

                    assert(px0_xyz.shape[-2] == 3) # No sidechains allowed

                    px0_xyz = px0_xyz[-1] # Only grab final prediction
                    
                    zeros = torch.zeros(B,1,L,24,3).float().to(px0_xyz.device)
                    xyz_t = torch.cat((px0_xyz.unsqueeze(1),zeros), dim=-2) # [B,T,L,27,3]
                    t2d   = xyz_to_t2d(xyz_t) # [B,T,L,L,44]
                else:
                    xyz_t = torch.zeros_like(xyz_t[:,1])
                    t2d   = torch.zeros_like(t2d[:,1])
            else:
                # Default behavior, no str self conditioning
                xyz_t = xyz_t[:,1]
                t2d   = t2d[:,1]

            if self.preprocess_param['seq_self_cond']:
                if unroll_performed:
                    # Removing prediction of mask character
                    pred_seq = sequence_logits[0,:20,:].permute(1,0) # [L,20]
                    pred_seq = pred_seq[:L] # Only take first L characters, not N*L

                    ################
                    ## msa_masked ##
                    ################
                    msa_masked = torch.stack((msa_masked[:,1,:,0],msa_masked[:,1,:,0]), dim=2) # [B,I,N_short,L,48]

                    msa_masked[:,:,0]         = 0 # Zero this out before we reassign it - NRB
                    msa_masked[:,:,0,:,:20]   = pred_seq
                    msa_masked[:,:,0,:,22:42] = pred_seq

                    ################
                    ### msa_full ###
                    ################
                    msa_full = torch.stack((msa_full[:,1,:,0],msa_full[:,1,:,0]), dim=2) # [B,I,N_long,L,25]

                    msa_full[:,:,0]       = 0 # Zero this out before we reassign it - NRB
                    msa_full[:,:,0,:,:20] = pred_seq
                     
                else:
                    msa_masked_zeros = torch.zeros_like(msa_masked[:,1,:,0]).to(msa_masked.device)
                    msa_masked       = torch.stack((msa_masked_zeros, msa_masked[:,1,:,0]), dim=2) # [B,I,N_short,L,48]

                    msa_full_zeros   = torch.zeros_like(msa_full[:,1,:,0]).to(msa_full.device)
                    msa_full         = torch.stack((msa_full_zeros, msa_full[:,1,:,0]), dim=2) # [B,I,N_long,L,25]

                # Always use mask_msa as the input version but stacked in a different dimension
                # [B,n,N_short,L] to [B,N_short,L] since N_short of input is always size 1, now n becomes N_short
                mask_msa = mask_msa.squeeze(2) # [B,2,L]

            else:
                # Default behavior, no seq self conditioning
                msa_masked = msa_masked[:,1]
                msa_full   = msa_full[:,1]
                mask_msa   = mask_msa[:,1]

            with torch.no_grad():
                for i_cycle in range(N_cycle-1):
                    with ddp_model.no_sync():
                        with torch.cuda.amp.autocast(enabled=USE_AMP):
                            msa_prev, pair_prev, xyz_prev, state_prev, alpha = ddp_model(
                                                                      msa_masked[:,0],
                                                                      msa_full[:,0],
                                                                      seq[:,0],
                                                                      xyz_prev, 
                                                                      idx_pdb,
                                                                      t=little_t,
                                                                      t1d=t1d,
                                                                      t2d=t2d,
                                                                      xyz_t=xyz_t,
                                                                      alpha_t=alpha_t,
                                                                      msa_prev=msa_prev,
                                                                      pair_prev=pair_prev,
                                                                      state_prev=state_prev,
                                                                      return_raw=True,
                                                                      motif_mask=diffusion_mask,
                                                                      i_cycle=i_cycle,
                                                                      n_cycle=N_cycle,
                                                                      )

                            #_, xyz_prev = self.compute_allatom_coords(seq[:,i_cycle], xyz_prev, alpha)

            i_cycle = N_cycle-1

            if counter%self.ACCUM_STEP != 0:
                with ddp_model.no_sync():
                    with torch.cuda.amp.autocast(enabled=USE_AMP):
                        logit_s, logit_aa_s, logit_exp, pred_crds, alphas, pred_lddts = ddp_model(msa_masked[:,0],
                                                                   msa_full[:,0],
                                                                   seq[:,0], xyz_prev,
                                                                   idx_pdb,
                                                                   t=little_t,
                                                                   t1d=t1d, t2d=t2d,
                                                                   xyz_t=xyz_t, alpha_t=alpha_t,
                                                                   msa_prev=msa_prev,
                                                                   pair_prev=pair_prev,
                                                                   state_prev=state_prev,
                                                                   use_checkpoint=True,
                                                                   motif_mask=diffusion_mask,
                                                                   i_cycle=i_cycle,
                                                                   n_cycle=N_cycle,
                                                                   )

                        # find closest homo-oligomer pairs
                        true_crds, mask_crds = resolve_equiv_natives(pred_crds[-1], true_crds, mask_crds)
                        mask_crds[:,~masks_1d['loss_str_mask'][0],:] = False
                        # processing labels for distogram orientograms
                        mask_BB = ~(mask_crds[:,:,:3].sum(dim=-1) < 3.0) # ignore residues having missing BB atoms for loss calculation
                        mask_2d = mask_BB[:,None,:] * mask_BB[:,:,None] # ignore pairs having missing residues
                        mask_2d = mask_2d*masks_1d['loss_str_mask_2d'].to(mask_2d.device)
                        assert torch.sum(mask_2d) > 0, "mask_2d is blank"
                        c6d, _ = xyz_to_c6d(true_crds)
                        c6d = c6d_to_bins2(c6d, same_chain, negative=negative)

                        prob = self.active_fn(logit_s[0]) # distogram
                        acc_s = self.calc_acc(prob, c6d[...,0], idx_pdb, mask_2d)

                        if self.preprocess_param['seq_self_cond']:
                            # logit_aa_s is [B,21,N*L] and we only want the first L
                            logit_aa_s = logit_aa_s[:,:,:L] # [B,21,N*L]
                    
                        loss, loss_s, loss_dict = self.calc_loss(logit_s, c6d,
                                logit_aa_s, msa[:, i_cycle], mask_msa[:,i_cycle], logit_exp,
                                pred_crds, alphas, true_crds, mask_crds,
                                mask_BB, mask_2d, same_chain,
                                pred_lddts, idx_pdb, chosen_dataset[0], chosen_task[0], diffusion_mask=diffusion_mask,
                                seq_diffusion_mask=seq_diffusion_mask, seq_t=seq[:,0], xyz_in=xyz_t, unclamp=unclamp, 
                                negative=negative, t=int(little_t), **self.loss_param)
                    loss = loss / self.ACCUM_STEP

                    if not torch.isnan(loss):

                        scaler.scale(loss).backward()
            else:
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    logit_s, logit_aa_s, logit_exp, pred_crds, alphas, pred_lddts = ddp_model(msa_masked[:,0],
                                                               msa_full[:,0],
                                                               seq[:,0], xyz_prev,
                                                               idx_pdb,
                                                               t=little_t,
                                                               t1d=t1d, t2d=t2d,
                                                               xyz_t=xyz_t, alpha_t=alpha_t,
                                                               msa_prev=msa_prev,
                                                               pair_prev=pair_prev,
                                                               state_prev=state_prev,
                                                               use_checkpoint=True,
                                                               motif_mask=diffusion_mask)
                        
                    # find closest homo-oligomer pairs
                    true_crds, mask_crds = resolve_equiv_natives(pred_crds[-1], true_crds, mask_crds)
                    mask_crds[:,~masks_1d['loss_str_mask'][0],:] = False
                    # processing labels for distogram orientograms
                    mask_BB = ~(mask_crds[:,:,:3].sum(dim=-1) < 3.0) # ignore residues having missing BB atoms for loss calculation
                    mask_2d = mask_BB[:,None,:] * mask_BB[:,:,None] # ignore pairs having missing residues
                    mask_2d = mask_2d*masks_1d['loss_str_mask_2d'].to(mask_2d.device)
                    c6d, _ = xyz_to_c6d(true_crds)
                    c6d = c6d_to_bins2(c6d, same_chain, negative=negative)

                    prob = self.active_fn(logit_s[0]) # distogram
                    acc_s = self.calc_acc(prob, c6d[...,0], idx_pdb, mask_2d)

                    if self.preprocess_param['seq_self_cond']:
                        # logit_aa_s is [B,21,N*L] and we only want the first L
                        logit_aa_s = logit_aa_s[:,:,:L] # [B,21,L]
                    
                    loss, loss_s, loss_dict = self.calc_loss(logit_s, c6d,
                                logit_aa_s, msa[:, i_cycle], mask_msa[:,i_cycle], logit_exp,
                                pred_crds, alphas, true_crds, mask_crds,
                                mask_BB, mask_2d, same_chain,
                                pred_lddts, idx_pdb, chosen_dataset[0], chosen_task[0], diffusion_mask=diffusion_mask,
                                seq_diffusion_mask=seq_diffusion_mask, seq_t=seq[:,0], xyz_in=xyz_t, unclamp=unclamp,
                                t=int(little_t), negative=negative, **self.loss_param)
                loss = loss / self.ACCUM_STEP
                if not torch.isnan(loss):
                    scaler.scale(loss).backward()
                    
                # gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 0.2)
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
            

            local_tot += loss.detach()*self.ACCUM_STEP
            if local_loss == None:
                local_loss = torch.zeros_like(loss_s.detach())
                local_acc  = torch.zeros_like(acc_s.detach())
            local_loss  += loss_s.detach()
            local_acc  += acc_s.detach()
            
            train_tot += loss.detach()*self.ACCUM_STEP
            if train_loss == None:
                train_loss = torch.zeros_like(loss_s.detach())
                train_acc  = torch.zeros_like(acc_s.detach())
            train_loss += loss_s.detach()
            train_acc  += acc_s.detach()
            
            if counter % N_PRINT_TRAIN == 0:
                if rank == 0:
                    max_mem = torch.cuda.max_memory_allocated()/1e9
                    train_time  = time.time() - start_time
                    local_tot  /= float(N_PRINT_TRAIN)
                    local_loss /= float(N_PRINT_TRAIN)
                    local_acc  /= float(N_PRINT_TRAIN)
                    
                    local_tot = local_tot.cpu().detach()
                    local_loss = local_loss.cpu().detach().numpy()
                    local_acc = local_acc.cpu().detach().numpy()

                    if 'diff' in chosen_task[0]:
                        task_str = f'diff_t{int(little_t)}'
                    else:
                        task_str = chosen_task[0]
                    

                    outstr = f"Local {task_str} | {chosen_dataset[0]}: [{epoch}/{self.n_epoch}] Batch: [{counter*self.batch_size*world_size}/{self.n_train}] Time: {train_time} Loss dict: "

                    str_stack = []
                    for k in sorted(list(loss_dict.keys())):
                        str_stack.append(f'{k}--{round( float(loss_dict[k]), 4)}')
                    outstr += '  '.join(str_stack)
                    sys.stdout.write(outstr+'\n')
                    
                    if WANDB and rank == 0:
                        loss_dict.update({'t':little_t, 'total_examples':epoch*self.n_train+counter*world_size, 'dataset':chosen_dataset[0], 'task':chosen_task[0]})
                        metrics = {}
                        for m in self.metrics:
                            with torch.no_grad():
                                metrics.update(m(logit_s, c6d,
                                    logit_aa_s, msa[:, i_cycle], mask_msa[:,i_cycle], logit_exp,
                                    pred_crds, alphas, true_crds, mask_crds,
                                    mask_BB, mask_2d, same_chain,
                                    pred_lddts, idx_pdb, chosen_dataset[0], chosen_task[0], diffusion_mask, t=little_t, unclamp=unclamp, negative=negative,
                                **self.loss_param))
                        loss_dict['metrics'] = metrics
                        wandb.log(loss_dict)

                    sys.stdout.flush()
                    local_tot = 0.0
                    local_loss = None 
                    local_acc = None 
                torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            logits_argsort = torch.argsort(logit_aa_s, dim=1, descending=True)
            top1_sequence = (logits_argsort[:, :1])
            top1_sequence = torch.clamp(top1_sequence, 0,19)

            if diffusion_param['seqdiff'] == 'continuous':
                top1_sequence = torch.argmax(logit_aa_s[:,:20,:], dim=1)

            # Also changed to allow for one-hot sequence input
            seq_original = torch.argmax(seq_original[:,:,:,:,:20], dim=-1)
            seq_masked = torch.argmax(seq_masked[:,:,:,:,:20], dim=-1)

            clamped = top1_sequence

            if chosen_task[0] != 'seq2str' and np.random.randint(0,100) == 0:
                if not os.path.isdir(f'./{self.outdir}/training_pdbs/'):
                    os.makedirs(f'./{self.outdir}/training_pdbs')
                writepdb(f'{self.outdir}/training_pdbs/test_epoch_{epoch}_{counter}_{chosen_task[0]}_{chosen_dataset[0]}_t_{int( little_t )}pred.pdb',pred_crds[-1,0,:,:3,:],top1_sequence[0,:])
                writepdb(f'{self.outdir}/training_pdbs/test_epoch_{epoch}_{counter}_{chosen_task[0]}_{chosen_dataset[0]}_t_{int( little_t )}true.pdb',true_crds[0,:,:3,:],torch.clamp(seq_original[0,0,:],0,19))
                writepdb(f'{self.outdir}/training_pdbs/test_epoch_{epoch}_{counter}_{chosen_task[0]}_{chosen_dataset[0]}_t_{int( little_t )}input.pdb',xyz_t[0,:,:,:3,:],torch.clamp(seq_masked[0,0,:],0,19))
                with open(f'{self.outdir}/training_pdbs/test_epoch_{epoch}_{counter}_{chosen_task[0]}_{chosen_dataset[0]}_t_{int( little_t )}pred_input.txt','w') as f:
                    f.write(str(masks_1d['input_str_mask'][0].cpu().detach().numpy())+'\n')
                    f.write(str(masks_1d['input_seq_mask'][0].cpu().detach().numpy())+'\n') 
                    f.write(str(t1d[:,:,:,-1].cpu().detach().numpy()))
        # write total train loss
        train_tot /= float(counter * world_size)
        train_loss /= float(counter * world_size)
        train_acc  /= float(counter * world_size)

        dist.all_reduce(train_tot, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_acc, op=dist.ReduceOp.SUM)
        train_tot = train_tot.cpu().detach()
        train_loss = train_loss.cpu().detach().numpy()
        train_acc = train_acc.cpu().detach().numpy()

        if counter % 1000 == 0:
            # Dump the model's weights every 1000 examples
            torch.save({'epoch': epoch,
                #'model_state_dict': ddp_model.state_dict(),
                'model_state_dict': ddp_model.module.shadow.state_dict(),
                'final_state_dict': ddp_model.module.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'valid_loss': 999.9,
                'valid_acc': 999.9,
                'best_loss': 999.9},
                self.checkpoint_fn(self.model_name, 'intermediate'))

        if rank == 0:
            
            train_time = time.time() - start_time
            sys.stdout.write("Train: [%04d/%04d] Batch: [%05d/%05d] Time: %16.1f | total_loss: %8.4f | %s | %.4f %.4f %.4f\n"%(\
                    epoch, self.n_epoch, self.n_train, self.n_train, train_time, train_tot, \
                    " ".join(["%8.4f"%l for l in train_loss]),\
                    train_acc[0], train_acc[1], train_acc[2]))
            sys.stdout.flush()

            
        return train_tot, train_loss, train_acc

    def valid_pdb_cycle(self, ddp_model, valid_loader, rank, gpu, world_size, epoch, header='PDB'):
        valid_tot = 0.0
        valid_loss = None
        valid_acc = None
        valid_lddt = None
        counter = 0
        
        start_time = time.time()
        
        with torch.no_grad(): # no need to calculate gradient
            ddp_model.eval() # change it to eval mode
            for seq, msa, msa_masked, msa_full, mask_msa, true_crds, mask_crds, idx_pdb, xyz_t, t1d, xyz_prev, same_chain, unclamp, negative in valid_loader:
                # transfer inputs to device
                B, _, N, L = msa.shape

                idx_pdb = idx_pdb.to(gpu, non_blocking=True) # (B, L)
                true_crds = true_crds.to(gpu, non_blocking=True) # (B, L, 27, 3)
                mask_crds = mask_crds.to(gpu, non_blocking=True) # (B, L, 27)
                same_chain = same_chain.to(gpu, non_blocking=True)

                xyz_t = xyz_t.to(gpu, non_blocking=True)
                t1d = t1d.to(gpu, non_blocking=True)
                
                xyz_prev = xyz_prev.to(gpu, non_blocking=True)

                seq = seq.to(gpu, non_blocking=True)
                msa = msa.to(gpu, non_blocking=True)
                msa_masked = msa_masked.to(gpu, non_blocking=True)
                msa_full = msa_full.to(gpu, non_blocking=True)
                mask_msa = mask_msa.to(gpu, non_blocking=True)
                
                # processing template features
                t2d = xyz_to_t2d(xyz_t)
                # get torsion angles from templates
                seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,L)
                alpha, _, alpha_mask, _ = get_torsions(xyz_t.reshape(-1,L,27,3), seq_tmp, self.ti_dev, self.ti_flip, self.ang_ref)
                alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
                alpha[torch.isnan(alpha)] = 0.0
                alpha = alpha.reshape(B,-1,L,10,2)
                alpha_mask = alpha_mask.reshape(B,-1,L,10,1)
                alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(B, -1, L, 30)
                # processing template coordinates
                xyz_t = get_init_xyz(xyz_t)
                xyz_prev = get_init_xyz(xyz_prev[:,None]).reshape(B, L, 27, 3)

                counter += 1

                N_cycle = self.maxcycle # number of recycling

                msa_prev = None
                pair_prev = None
                state_prev = None
                lddt_s = list()
                for i_cycle in range(N_cycle-1): 
                    msa_prev, pair_prev, xyz_prev, state_prev, alpha = ddp_model(msa_masked[:,i_cycle],
                                                              msa_full[:,i_cycle],
                                                              seq[:,i_cycle], xyz_prev, 
                                                              idx_pdb,
                                                              t1d=t1d, t2d=t2d,
                                                              xyz_t=xyz_t, alpha_t=alpha_t,
                                                              msa_prev=msa_prev,
                                                              pair_prev=pair_prev,
                                                              state_prev=state_prev,
                                                              return_raw=True)
                    #_, xyz_prev = self.compute_allatom_coords(seq[:,i_cycle], xyz_prev, alpha)
                
                    true_crds_i, mask_crds_i = resolve_equiv_natives(xyz_prev, true_crds, mask_crds)                    
                    # processing labels for distogram orientograms
                    mask_BB = ~(mask_crds_i[:,:,:3].sum(dim=-1) < 3.0) # ignore residues having missing BB atoms for loss calculation
                    mask_2d = mask_BB[:,None,:] * mask_BB[:,:,None] # ignore pairs having missing residues

                    lddt = calc_lddt(xyz_prev[:,:,1,:][None], true_crds_i[:,:,1,:], mask_BB, mask_2d, same_chain)
                    lddt_s.append(lddt.detach())

                i_cycle = N_cycle-1
                logit_s, logit_aa_s, logit_exp, pred_crds, alphas, pred_lddts = ddp_model(msa_masked[:,i_cycle],
                                                           msa_full[:,i_cycle],
                                                           seq[:,i_cycle], xyz_prev,
                                                           idx_pdb,
                                                           t1d=t1d, t2d=t2d,
                                                           xyz_t=xyz_t, alpha_t=alpha_t,
                                                           msa_prev=msa_prev,
                                                           pair_prev=pair_prev,
                                                           state_prev=state_prev,
                                                           use_checkpoint=False)
                
                true_crds, mask_crds = resolve_equiv_natives(pred_crds[-1], true_crds, mask_crds)
                mask_BB = ~(mask_crds[:,:,:3].sum(dim=-1) < 3.0)
                mask_2d = mask_BB[:,None,:] * mask_BB[:,:,None]

                c6d, _ = xyz_to_c6d(true_crds)
                c6d = c6d_to_bins2(c6d, same_chain, negative=negative)

                prob = self.active_fn(logit_s[0]) # distogram
                acc_s = self.calc_acc(prob, c6d[...,0], idx_pdb, mask_2d)

                loss, loss_s, loss_dict = self.calc_loss(logit_s, c6d,
                        logit_aa_s, msa[:, i_cycle], mask_msa[:,i_cycle], logit_exp,
                        pred_crds, alphas, true_crds, mask_crds,
                        mask_BB, mask_2d, same_chain,
                        pred_lddts, idx_pdb, t=little_t, unclamp=unclamp, negative=negative,
                        **self.loss_param)
                lddt = calc_lddt(pred_crds[-1:,:,:,1,:], true_crds[:,:,1,:], mask_BB, mask_2d, same_chain)
                lddt_s.append(lddt.detach())
                lddt_s = torch.cat(lddt_s)
                
                valid_tot += loss.detach()
                if valid_loss == None:
                    valid_loss = torch.zeros_like(loss_s.detach())
                    valid_acc = torch.zeros_like(acc_s.detach())
                    valid_lddt = torch.zeros_like(lddt_s.detach())
                valid_loss += loss_s.detach()
                valid_acc += acc_s.detach()
                valid_lddt += lddt_s.detach()

            
        valid_tot /= float(counter*world_size)
        valid_loss /= float(counter*world_size)
        valid_acc /= float(counter*world_size)
        valid_lddt /= float(counter*world_size)
        
        dist.all_reduce(valid_tot, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_lddt, op=dist.ReduceOp.SUM)
       
        valid_tot = valid_tot.cpu().detach().numpy()
        valid_loss = valid_loss.cpu().detach().numpy()
        valid_acc = valid_acc.cpu().detach().numpy()
        valid_lddt = valid_lddt.cpu().detach().numpy()
        
        if rank == 0:
            
            train_time = time.time() - start_time
            sys.stdout.write("%s: [%04d/%04d] Batch: [%05d/%05d] Time: %16.1f | total_loss: %8.4f | %s | %s | %.4f %.4f %.4f\n"%(\
                    header, epoch, self.n_epoch, counter*world_size, counter*world_size, 
                    train_time, valid_tot, \
                    " ".join(["%8.4f"%l for l in valid_loss]),\
                    " ".join(["%8.4f"%l for l in valid_lddt]),\
                    valid_acc[0], valid_acc[1], valid_acc[2])) 
            sys.stdout.flush()
        return valid_tot, valid_loss, valid_acc
    
    def valid_ppi_cycle(self, ddp_model, valid_pos_loader, valid_neg_loader, rank, gpu, world_size, epoch, verbose=False):
        valid_tot = 0.0
        valid_loss = None
        valid_acc = None
        valid_lddt = None
        valid_inter = None
        counter = 0
        
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        
        start_time = time.time()
        
        with torch.no_grad(): # no need to calculate gradient
            ddp_model.eval() # change it to eval mode
            for seq, msa, msa_masked, msa_full, mask_msa, true_crds, mask_crds, idx_pdb, xyz_t, t1d, xyz_prev, same_chain, unclamp, negative in valid_pos_loader:
                # transfer inputs to device
                B, _, N, L = msa.shape

                idx_pdb = idx_pdb.to(gpu, non_blocking=True) # (B, L)
                true_crds = true_crds.to(gpu, non_blocking=True) # (B, L, 27, 3)
                mask_crds = mask_crds.to(gpu, non_blocking=True) # (B, L, 27)
                same_chain = same_chain.to(gpu, non_blocking=True)

                xyz_t = xyz_t.to(gpu, non_blocking=True)
                t1d = t1d.to(gpu, non_blocking=True)
                
                xyz_prev = xyz_prev.to(gpu, non_blocking=True)

                seq = seq.to(gpu, non_blocking=True)
                msa = msa.to(gpu, non_blocking=True)
                msa_masked = msa_masked.to(gpu, non_blocking=True)
                msa_full = msa_full.to(gpu, non_blocking=True)
                mask_msa = mask_msa.to(gpu, non_blocking=True)
                
                # processing labels for distogram orientograms
                mask_BB = ~(mask_crds[:,:,:3].sum(dim=-1) < 3.0) # ignore residues having missing BB atoms for loss calculation
                mask_2d = mask_BB[:,None,:] * mask_BB[:,:,None] # ignore pairs having missing residues
                c6d, _ = xyz_to_c6d(true_crds)
                c6d = c6d_to_bins2(c6d, same_chain, negative=negative)

                # processing template features
                t2d = xyz_to_t2d(xyz_t)
                # get torsion angles from templates
                seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,L)
                alpha, _, alpha_mask, _ = get_torsions(xyz_t.reshape(-1,L,27,3), seq_tmp, self.ti_dev, self.ti_flip, self.ang_ref)
                alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
                alpha[torch.isnan(alpha)] = 0.0
                alpha = alpha.reshape(B,-1,L,10,2)
                alpha_mask = alpha_mask.reshape(B,-1,L,10,1)
                alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(B, -1, L, 30)
                # processing template coordinates
                xyz_t = get_init_xyz(xyz_t)
                xyz_prev = get_init_xyz(xyz_prev[:,None]).reshape(B, L, 27, 3)

                counter += 1

                N_cycle = self.maxcycle # number of recycling

                msa_prev = None
                pair_prev = None
                state_prev = None
                lddt_s = list()
                for i_cycle in range(N_cycle-1): 
                    msa_prev, pair_prev, xyz_prev, state_prev, alpha = ddp_model(msa_masked[:,i_cycle],
                                                              msa_full[:,i_cycle],
                                                              seq[:,i_cycle], xyz_prev, 
                                                              idx_pdb,
                                                              t1d=t1d, t2d=t2d,
                                                              xyz_t=xyz_t, alpha_t=alpha_t,
                                                              msa_prev=msa_prev,
                                                              pair_prev=pair_prev,
                                                              state_prev=state_prev,
                                                              return_raw=True)
                    #_, xyz_prev = self.compute_allatom_coords(seq[:,i_cycle], xyz_prev, alpha)
                    lddt = calc_lddt(xyz_prev[:,:,1,:][None], true_crds[:,:,1,:], mask_BB, mask_2d, same_chain)
                    lddt_s.append(lddt.detach())

                i_cycle = N_cycle-1
                logit_s, logit_aa_s, logit_exp, pred_crds, alphas, pred_lddts = ddp_model(msa_masked[:,i_cycle],
                                                           msa_full[:,i_cycle],
                                                           seq[:,i_cycle], xyz_prev,
                                                           idx_pdb,
                                                           t1d=t1d, t2d=t2d,
                                                           xyz_t=xyz_t, alpha_t=alpha_t,
                                                           msa_prev=msa_prev,
                                                           pair_prev=pair_prev,
                                                           state_prev=state_prev,
                                                           use_checkpoint=False)
                prob = self.active_fn(logit_s[0]) # distogram
                acc_s, cnt_pred, cnt_ref = self.calc_acc(prob, c6d[...,0], idx_pdb, mask_2d, return_cnt=True)
                
                # inter-chain contact prob
                cnt_pred = cnt_pred * (1-same_chain).float()
                cnt_ref = cnt_ref * (1-same_chain).float()
                max_prob = cnt_pred.max()
                if max_prob > 0.5:
                    if (cnt_ref > 0).any():
                        TP += 1.0
                    else:
                        FP += 1.0
                else:
                    if (cnt_ref > 0).any():
                        FN += 1.0
                    else:
                        TN += 1.0
                inter_s = torch.tensor([TP, FP, TN, FN], device=prob.device).float()

                loss, loss_s, loss_dict = self.calc_loss(logit_s, c6d,
                        logit_aa_s, msa[:, i_cycle], mask_msa[:,i_cycle], logit_exp,
                        pred_crds, alphas, true_crds, mask_crds,
                        mask_BB, mask_2d, same_chain,
                        pred_lddts, idx_pdb, t=little_t, unclamp=unclamp, negative=negative,
                        **self.loss_param)
                lddt = calc_lddt(pred_crds[-1:,:,:,1,:], true_crds[:,:,1,:], mask_BB, mask_2d, same_chain)
                lddt_s.append(lddt.detach())
                lddt_s = torch.cat(lddt_s)
                
                valid_tot += loss.detach()
                if valid_loss == None:
                    valid_loss = torch.zeros_like(loss_s.detach())
                    valid_acc = torch.zeros_like(acc_s.detach())
                    valid_lddt = torch.zeros_like(lddt_s.detach())
                    valid_inter = torch.zeros_like(inter_s.detach())
                valid_loss += loss_s.detach()
                valid_acc += acc_s.detach()
                valid_lddt += lddt_s.detach()
                valid_inter += inter_s.detach()

            
        valid_tot /= float(counter*world_size)
        valid_loss /= float(counter*world_size)
        valid_acc /= float(counter*world_size)
        valid_lddt /= float(counter*world_size)
        
        dist.all_reduce(valid_tot, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_lddt, op=dist.ReduceOp.SUM)
       
        valid_tot = valid_tot.cpu().detach().numpy()
        valid_loss = valid_loss.cpu().detach().numpy()
        valid_acc = valid_acc.cpu().detach().numpy()
        valid_lddt = valid_lddt.cpu().detach().numpy()
        
        if rank == 0:
            
            train_time = time.time() - start_time
            sys.stdout.write("Hetero: [%04d/%04d] Batch: [%05d/%05d] Time: %16.1f | total_loss: %8.4f | %s | %s | %.4f %.4f %.4f\n"%(\
                    epoch, self.n_epoch, counter*world_size, counter*world_size, train_time, valid_tot, \
                    " ".join(["%8.4f"%l for l in valid_loss]),\
                    " ".join(["%8.4f"%l for l in valid_lddt]),\
                    valid_acc[0], valid_acc[1], valid_acc[2])) 
            sys.stdout.flush()
        
        valid_tot = 0.0
        valid_loss = None
        valid_acc = None
        counter = 0

        start_time = time.time()
        
        with torch.no_grad(): # no need to calculate gradient
            ddp_model.eval() # change it to eval mode
            for seq, msa, msa_masked, msa_full, mask_msa, true_crds, mask_crds, idx_pdb, xyz_t, t1d, xyz_prev, same_chain, unclamp, negative in valid_neg_loader:
                # transfer inputs to device
                B, _, N, L = msa.shape

                idx_pdb = idx_pdb.to(gpu, non_blocking=True) # (B, L)
                true_crds = true_crds.to(gpu, non_blocking=True) # (B, L, 27, 3)
                mask_crds = mask_crds.to(gpu, non_blocking=True) # (B, L, 27)
                same_chain = same_chain.to(gpu, non_blocking=True)

                xyz_t = xyz_t.to(gpu, non_blocking=True)
                t1d = t1d.to(gpu, non_blocking=True)
                
                xyz_prev = xyz_prev.to(gpu, non_blocking=True)

                seq = seq.to(gpu, non_blocking=True)
                msa = msa.to(gpu, non_blocking=True)
                msa_masked = msa_masked.to(gpu, non_blocking=True)
                msa_full = msa_full.to(gpu, non_blocking=True)
                mask_msa = mask_msa.to(gpu, non_blocking=True)
                
                # processing labels for distogram orientograms
                mask_BB = ~(mask_crds[:,:,:3].sum(dim=-1) < 3.0) # ignore residues having missing BB atoms for loss calculation
                mask_2d = mask_BB[:,None,:] * mask_BB[:,:,None] # ignore pairs having missing residues
                c6d, _ = xyz_to_c6d(true_crds)
                c6d = c6d_to_bins2(c6d, same_chain, negative=negative)

                # processing template features
                t2d = xyz_to_t2d(xyz_t)
                # get torsion angles from templates
                seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,L)
                alpha, _, alpha_mask, _ = get_torsions(xyz_t.reshape(-1,L,27,3), seq_tmp, self.ti_dev, self.ti_flip, self.ang_ref)
                alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
                alpha[torch.isnan(alpha)] = 0.0
                alpha = alpha.reshape(B,-1,L,10,2)
                alpha_mask = alpha_mask.reshape(B,-1,L,10,1)
                alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(B, -1, L, 30)
                # processing template coordinates
                xyz_t = get_init_xyz(xyz_t)
                xyz_prev = get_init_xyz(xyz_prev[:,None]).reshape(B, L, 27, 3)

                counter += 1

                N_cycle = self.maxcycle # number of recycling

                msa_prev = None
                pair_prev = None
                state_prev = None
                for i_cycle in range(N_cycle-1): 
                    msa_prev, pair_prev, xyz_prev, state_prev, alpha = ddp_model(msa_masked[:,i_cycle],
                                                              msa_full[:,i_cycle],
                                                              seq[:,i_cycle], xyz_prev, 
                                                              idx_pdb,
                                                              t1d=t1d, t2d=t2d,
                                                              xyz_t=xyz_t, alpha_t=alpha_t,
                                                              msa_prev=msa_prev,
                                                              pair_prev=pair_prev,
                                                              state_prev=state_prev,
                                                              return_raw=True)
                    #_, xyz_prev = self.compute_allatom_coords(seq[:,i_cycle], xyz_prev, alpha)

                i_cycle = N_cycle-1
                logit_s, logit_aa_s, logit_exp, pred_crds, alphas, pred_lddts = ddp_model(msa_masked[:,i_cycle],
                                                           msa_full[:,i_cycle],
                                                           seq[:,i_cycle], xyz_prev,
                                                           idx_pdb,
                                                           t1d=t1d, t2d=t2d,
                                                           xyz_t=xyz_t, alpha_t=alpha_t,
                                                           msa_prev=msa_prev,
                                                           pair_prev=pair_prev,
                                                           state_prev=state_prev,
                                                           use_checkpoint=False)
                prob = self.active_fn(logit_s[0]) # distogram
                acc_s, cnt_pred, cnt_ref = self.calc_acc(prob, c6d[...,0], idx_pdb, mask_2d, return_cnt=True)
                
                # inter-chain contact prob
                cnt_pred = cnt_pred * (1-same_chain).float()
                cnt_ref = cnt_ref * (1-same_chain).float()
                max_prob = cnt_pred.max()
                if max_prob > 0.5:
                    if (cnt_ref > 0).any():
                        TP += 1.0
                    else:
                        FP += 1.0
                else:
                    if (cnt_ref > 0).any():
                        FN += 1.0
                    else:
                        TN += 1.0
                inter_s = torch.tensor([TP, FP, TN, FN], device=prob.device).float()

                loss, loss_s, loss_dict = self.calc_loss(logit_s, c6d,
                        logit_aa_s, msa[:, i_cycle], mask_msa[:,i_cycle], logit_exp,
                        pred_crds, alphas, true_crds, mask_crds,
                        mask_BB, mask_2d, same_chain,
                        pred_lddts, idx_pdb, t=little_t, unclamp=unclamp, negative=negative,
                        **self.loss_param)
                
                valid_tot += loss.detach()
                if valid_loss == None:
                    valid_loss = torch.zeros_like(loss_s.detach())
                    valid_acc = torch.zeros_like(acc_s.detach())
                valid_loss += loss_s.detach()
                valid_acc += acc_s.detach()
                valid_inter += inter_s.detach()

            
        valid_tot /= float(counter*world_size)
        valid_loss /= float(counter*world_size)
        valid_acc /= float(counter*world_size)
        
        dist.all_reduce(valid_tot, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_inter, op=dist.ReduceOp.SUM)
       
        valid_tot = valid_tot.cpu().detach().numpy()
        valid_loss = valid_loss.cpu().detach().numpy()
        valid_acc = valid_acc.cpu().detach().numpy()
        valid_inter = valid_inter.cpu().detach().numpy()
        
        if rank == 0:
            TP, FP, TN, FN = valid_inter 
            prec = TP/(TP+FP+1e-4)
            recall = TP/(TP+FN+1e-4)
            F1 = 2*TP/(2*TP+FP+FN+1e-4)
            
            train_time = time.time() - start_time
            sys.stdout.write("PPI: [%04d/%04d] Batch: [%05d/%05d] Time: %16.1f | total_loss: %8.4f | %s | %.4f %.4f %.4f | %.4f %.4f %.4f\n"%(\
                    epoch, self.n_epoch, counter*world_size, counter*world_size, train_time, valid_tot, \
                    " ".join(["%8.4f"%l for l in valid_loss]),\
                    valid_acc[0], valid_acc[1], valid_acc[2],\
                    prec, recall, F1))
            sys.stdout.flush()
        return valid_tot, valid_loss, valid_acc
    

if __name__ == "__main__":
    from arguments import get_args
    args, model_param, loader_param, loss_param, diffusion_param, preprocess_param = get_args()
    
    # set epoch size 
    global N_EXAMPLE_PER_EPOCH
    N_EXAMPLE_PER_EPOCH = args.epoch_size 

    # set global debug and wandb params 
    global DEBUG 
    global WANDB
    if args.debug:
        DEBUG = True 
        WANDB = False 
    else:
        DEBUG = False 
        WANDB = True 
        import wandb
    
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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
                    log_inputs=args.log_inputs)
    train.run_model_training(torch.cuda.device_count())
