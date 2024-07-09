import numpy as np
import os
import sys
from itertools import combinations
from omegaconf import DictConfig
from kinematics import xyz_to_t2d
import torch
import torch.nn.functional as nn
# from util import get_torsions
from diffusion import get_beta_schedule, get_aa_schedule, get_chi_betaT
from diff_util import get_aa_schedule, th_interpolate_angles, th_min_angle
from icecream import ic
from scipy.spatial.transform import Rotation as scipy_R
from scipy.spatial.transform import Slerp

# from util import torsion_indices as TOR_INDICES
# from util import torsion_can_flip as TOR_CAN_FLIP
# from util import reference_angles as REF_ANGLES
# from util import rigid_from_3_points
from rf2aa.util import torsion_indices as TOR_INDICES 
from rf2aa.util import torsion_can_flip as TOR_CAN_FLIP
from rf2aa.util import reference_angles as REF_ANGLES
from rf2aa.util import rigid_from_3_points

# from util_module import ComputeAllAtomCoords
from rf2aa.util_module import XYZConverter

from potentials.manager import PotentialManager
from rf2aa.chemical import NAATOKENS, aa2num, aa2long, atomnum2atomtype, NTOTAL, CHAIN_GAP, to1letter, NHEAVY, NHEAVYPROT, NHEAVYNUC, NTOTALDOFS
import util
import random
import logging
import string 
import hydra
import rf2aa.chemical
from rf2aa.kinematics import normQ, Qs2Rs
import aa_model
import parsers
import matplotlib.pyplot as plt
from pdb import set_trace
from icecream import ic 
ic(util.__file__)

from . import symmetry 
###########################################################
#### Functions which can be called outside of Denoiser ####
###########################################################

# These functions behave exactly the same as before but now do not rely on class fields from the Denoiser

# Useful thing for keeping track of chain indices:
abet = 'abcdefghijklmnopqrstuvwxyz'
abet = [a for a in abet]
abet2num = {a:i for i,a in enumerate(abet)} 
num2abet = {i:a for i,a in enumerate(abet)} 

contact_type_to_int = {
                'protein_protein': 0, 
                'dna_dna': 1, 
                'rna_rna': 2, 
                'protein_dna': 3, 
                'dna_protein': 3, 
                'protein_rna': 4, 
                'rna_protein': 4, 
                'dna_rna': 5, 
                'rna_dna': 5, 
                }

def slerp_update(r_t, r_0, t, mask=0):
    """slerp_update uses SLERP to update the frames at time t to the
    predicted frame for t=0

    Args:
        R_t, R_0: rotation matrices of shape [3, 3]
        t: time step
        mask: set to 1 / True to skip update.

    Returns:
        slerped rotation for time t-1 of shape [3, 3]
    """
    # interpolate FRAMES between one and next 
    if not mask:
        key_rots = scipy_R.from_matrix(np.stack([r_t, r_0], axis=0))
    else:
        key_rots = scipy_R.from_matrix(np.stack([r_t, r_t], axis=0))

    key_times = [0,1]

    interpolator = Slerp(key_times, key_rots)
    alpha = np.array([1/t])
    
    # grab the interpolated FRAME 
    interp_frame  = interpolator(alpha)
    
    # constructed rotation matrix which when applied YIELDS interpolated frame 
    interp_rot = (interp_frame.as_matrix().squeeze() @ np.linalg.inv(r_t.squeeze()) )[None,...]

    return interp_rot

def get_next_frames(xt, px0, t, diffuser, so3_type, diffusion_mask, noise_scale=1.):
    """get_next_frames gets updated frames using either SLERP or the IGSO(3) + score_based reverse diffusion.
    

    based on self.so3_type use slerp or score based update.

    SLERP xt frames towards px0, by factor 1/t
    Rather than generating random rotations (as occurs during forward process), calculate rotation between xt and px0
   
    Args:
        xt: noised coordinates of shape [L, 14, 3]
        px0: prediction of coordinates at t=0, of shape [L, 14, 3]
        t: integer time step
        diffuser: Diffuser object for reverse igSO3 sampling
        so3_type: The type of SO3 noising being used ('igso3', or 'slerp')
        diffusion_mask: of shape [L] of type bool, True means not to be
            updated (e.g. mask is true for motif residues)
        noise_scale: scale factor for the noise added (IGSO3 only)
    
    Returns:
        backbone coordinates for step x_t-1 of shape [L, 3, 3]
    """
    N_0  = px0[None,:,0,:]
    Ca_0 = px0[None,:,1,:]
    C_0  = px0[None,:,2,:]

    R_0, Ca_0 = rigid_from_3_points(N_0, Ca_0, C_0)

    N_t  = xt[None, :, 0, :]
    Ca_t = xt[None, :, 1, :]
    C_t  = xt[None, :, 2, :]

    R_t, Ca_t = rigid_from_3_points(N_t, Ca_t, C_t)

    R_0 = scipy_R.from_matrix(R_0.squeeze().numpy())
    R_t = scipy_R.from_matrix(R_t.squeeze().numpy())

    # Sample next frame for each residue
    all_rot_transitions = []
    for i in range(len(xt)):
        r_0 = R_0[i].as_matrix()
        r_t = R_t[i].as_matrix()
        mask_i = diffusion_mask[i]

        if so3_type == "igso3":
            r_t_next = diffuser.so3_diffuser.reverse_sample(r_t, r_0, t,
                    mask=mask_i, noise_level=noise_scale)[None,...]
            interp_rot =  r_t_next @ (r_t.T)
        elif so3_type == "slerp":
            interp_rot = slerp_update(r_t, r_0, t, diffusion_mask[i])
        else:
            assert False, "so3 diffusion type %s not implemented"%so3_type

        all_rot_transitions.append(interp_rot)

    all_rot_transitions = np.stack(all_rot_transitions, axis=0)

    # Apply the interpolated rotation matrices to the coordinates
    next_crds   = np.einsum('lrij,laj->lrai', all_rot_transitions, xt[:,:3,:] - Ca_t.squeeze()[:,None,...].numpy()) + Ca_t.squeeze()[:,None,None,...].numpy()

    # (L,3,3) set of backbone coordinates with slight rotation 
    return next_crds.squeeze(1)


def get_mu_xt_x0(xt, px0, t, beta_schedule, alphabar_schedule, eps=1e-6):
    """
    Given xt, predicted x0 and the timestep t, give mu of x(t-1)
    Assumes t is 0 indexed
    """
    #sigma is predefined from beta. Often referred to as beta tilde t
    t_idx = t-1
    sigma = ((1-alphabar_schedule[t_idx-1])/(1-alphabar_schedule[t_idx]))*beta_schedule[t_idx]

    xt_ca = xt[:,1,:]
    px0_ca = px0[:,1,:]

    a = ((torch.sqrt(alphabar_schedule[t_idx-1] + eps)*beta_schedule[t_idx])/(1-alphabar_schedule[t_idx]))
    b = ((torch.sqrt(1-beta_schedule[t_idx] + eps)*(1-alphabar_schedule[t_idx-1]))/(1-alphabar_schedule[t_idx]))

    mu = a*px0_ca + b*xt_ca

    return mu, sigma


def get_next_ca(xt, px0, t, diffusion_mask, crd_scale, beta_schedule, alphabar_schedule, noise_scale=1.):
    """
    Given full atom x0 prediction (xyz coordinates), diffuse to x(t-1)
    
    Parameters:
        
        xt (L, 14/27, 3) set of coordinates
        
        px0 (L, 14/27, 3) set of coordinates

        t: time step. Note this is zero-index current time step, so are generating t-1    

        logits_aa (L x 20 ) amino acid probabilities at each position

        seq_schedule (L): Tensor of bools, True is unmasked, False is masked. For this specific t

        diffusion_mask (torch.tensor, required): Tensor of bools, True means NOT diffused at this residue, False means diffused 

        noise_scale: scale factor for the noise being added

    """
    # get_allatom = ComputeAllAtomCoords().to(device=xt.device)
    converter = XYZConverter().to(device=xt.device)

    L = len(xt)

    # bring to origin after global alignment (when don't have a motif) or replace input motif and bring to origin, and then scale 
    px0 = px0 * crd_scale
    xt = xt * crd_scale

    # get mu(xt, x0)
    mu, sigma = get_mu_xt_x0(xt, px0, t, beta_schedule=beta_schedule, alphabar_schedule=alphabar_schedule)

    sampled_crds = torch.normal(mu, torch.sqrt(sigma*noise_scale))
    delta = sampled_crds - xt[:,1,:] #check sign of this is correct

    if not diffusion_mask is None:
        # calculate the mean displacement between the current motif and where 
        # RoseTTAFold thinks it should go 
        # print('Got motif delta')
        # motif_delta = (px0[diffusion_mask,:3,...] - xt[diffusion_mask,:3,...]).mean(0).mean(0)

        delta[diffusion_mask,...] = 0
        # delta[diffusion_mask,...] = motif_delta

    out_crds = xt + delta[:, None, :]

    return out_crds/crd_scale, delta/crd_scale

class DecodeSchedule():
    """
    Class for managing AA decoding schedule stuff
    """

    def __init__(self, L, visible, aa_decode_steps=40, mode='distance_based'):

        # only distance based or logit based for now
        assert mode in ['distance_based','delta_prob_based']# , 'uniform_linear', 'ar_fixbb']
        self.mode = mode

        self.visible = visible

        # start as all high - only matters when a residue is being decoded
        # at which point we will know the true T
        self.T = torch.full((L,), 999)

        # number of residues being decoded on each step
        if aa_decode_steps > 0:
            tmp = np.array(list(range((~self.visible).sum())))
            np.random.shuffle(tmp)
            ndecode_per_step = np.array_split(tmp, aa_decode_steps)
            np.random.shuffle(ndecode_per_step)
            self.ndecode_per_step = [len(a) for a in ndecode_per_step]


    def get_next_idx(self, cur_indices, dmap, ):
        """
        Given indices being currently sampled and a distance map, return one more index which is allowed to
        be sampled at the same time as cur indices

        Parameters:

            cur_indices (list, required): indices of residues also being decoded this step

            dmap (torch.tensor, required): (L,L) distance map of CA's
        """
        L = dmap.shape[0]
        options = torch.arange(L)[~self.visible] # indices of non-decoded residues

        # find the index with the largest average distance from all decoded residues
        #mean_distances = dmap[cur_indices,options]
        d_rows    = dmap[cur_indices]
        d_columns = d_rows[:,options]
        mean_distances = d_columns.mean(dim=0)


        #mean_distances = mean_distances.mean(dim=0)


        best_idx_local  = torch.argmax(mean_distances) # index within options tensor
        best_idx_global = options[best_idx_local]      # the option itself

        return best_idx_global


    # def get_decode_positions(self, t_idx, px0):
    def get_decode_positions(self, t_idx, px0, seq_t, seq_px0):
        """
        Returns the next (0-indexed) positions to decode for this timestep
        """
        L = px0.shape[0]
        assert t_idx < len( self.ndecode_per_step ) # taken care of outside this class in sampling loop

        N = self.ndecode_per_step[t_idx]
        decode_list = []

        if self.mode == 'distance_based':
            # perform dynamic distance based sampling
            ca   = px0[:,1,:]
            dmap = torch.sqrt( (ca[None,:] - ca[:,None]).square().sum(dim=-1) + 1e-6 )

            for i in range(N):
                if i == 0:
                    # sample a random residue which hasn't been decoded yet
                    first_idx = np.random.choice(torch.arange(L)[~self.visible])
                    decode_list.append(int(first_idx))
                    self.visible[first_idx] = True
                    self.T[first_idx] = t_idx + 1
                    continue

                # given already sampled indices, get another
                decode_idx = self.get_next_idx(decode_list,dmap)

                decode_list.append(int(decode_idx))
                self.visible[decode_idx] = True # set this now because get_next_idx depends on it
                self.T[decode_idx] = t_idx+1    # now that we know this residue is decoded, set its big T value

        elif self.mode == 'delta_prob_based':
            # Perform delta probability based sampling
            current_probs = torch.gather(seq_px0.cpu(), 1, seq_t.view(-1, 1))
            dprob = seq_px0.cpu() - current_probs
            print('AFAV IN PROGRESS IMPLEMENTING!!!!')
            print('AFAV IN PROGRESS IMPLEMENTING!!!!')
            print('AFAV IN PROGRESS IMPLEMENTING!!!!')
            print('AFAV IN PROGRESS IMPLEMENTING!!!!')
            print('AFAV IN PROGRESS IMPLEMENTING!!!!')
            print('AFAV IN PROGRESS IMPLEMENTING!!!!')
            raise NotImplementedError(f'Andrew needs to finish using torch.multinomial to implement this option')
            set_trace()

    

        return decode_list


    @property
    def idx2steps(self):
        return self.T

class Denoise():
    """
    Class for getting x(t-1) from predicted x0 and x(t)
    Strategy:
        Ca coordinates: Rediffuse to x(t-1) from predicted x0
        Frames: SLERP 1/t of the way to x0 prediction
        Torsions: 1/t of the way to the x0 prediction

    """
    def __init__(self,
                 T,
                 L,
                 diffuser,
                 visible,
                 seq_diffuser=None,
                 b_0=0.001,
                 b_T=0.1,
                 min_b=1.0,
                 max_b=12.5,
                 min_sigma=0.05,
                 max_sigma=1.5,     # DJ- simply put this in as dummy 
                 noise_level=0.5,
                 schedule_type='cosine',
                 so3_schedule_type='linear',
                 schedule_kwargs={},
                 so3_type='slerp',
                 chi_type='interp',
                 noise_scale_ca=1.,
                 noise_scale_frame=0.5,
                 noise_scale_torsion=1.,
                 crd_scale=1/15,
                 aa_decode_steps=100,
                 potential_manager=None,
                 softmax_T=1e-5,
                 partial_T=None,
                 seq_decode_mode='distance_based', # option to update sequence by logit difference
                 ):
        """
        
        Parameters:
            noise_level: scaling on the noise added (set to 0 to use no noise,
                to 1 to have full noise)
            
        """
        self.T = T
        self.L = L 
        self.diffuser = diffuser
        self.seq_diffuser = seq_diffuser
        self.b_0 = b_0
        self.b_T = b_T
        self.noise_level = noise_level
        self.schedule_type = schedule_type
        self.so3_type = so3_type
        self.chi_type = chi_type
        self.crd_scale = crd_scale
        self.noise_scale_ca = noise_scale_ca
        self.noise_scale_frame = noise_scale_frame
        self.noise_scale_torsion = noise_scale_torsion
        self.aa_decode_steps=aa_decode_steps
        self.potential_manager = potential_manager
        self.seq_decode_mode = seq_decode_mode
        self._log = logging.getLogger(__name__)


        self.schedule, self.alpha_schedule, self.alphabar_schedule = get_beta_schedule(self.T, self.b_0, self.b_T, self.schedule_type, inference=True)

        # dynamic schedule dictionaries for chi angles 
        max_chi_T = 160
        chi_b_0 = 0.001
        chi_abar_T=1e-3
        assert max_chi_T >= aa_decode_steps # assume never decoding for more than 100 steps, change if otherwise
        self.chi_beta_T = get_chi_betaT(max_chi_T, chi_b_0, chi_abar_T, method='cosine') # precalculate chi beta schedules for dynamic T


        # amino acid decoding schedule 
        #out = get_aa_schedule(T,L,nsteps=aa_decode_steps)
        #self.aa_decode_times, self.decode_order, self.idx2steps, self.aa_mask_stack = out

        if seq_diffuser is None: self.decode_scheduler = DecodeSchedule(L, visible, aa_decode_steps, mode=self.seq_decode_mode)

    @property
    def idx2steps(self):
        return self.decode_scheduler.idx2steps.numpy()

    @staticmethod 
    def get_dynamic_mu_sigma(chi_t, chi_0, T, t, chi_beta_T, beta_0, schedule_type='cosine'):
        """
        Given currente chis, prediction of chi0, dynamic T, current t, and chi_beta_T schedules, 
        sample new angles 
        
        Need to make this faster probably.
        """
        t_idx = t-1
        assert t > 1 # t must be 2+ to do this
        assert len(chi_t) == len(chi_0)
        
        max_t = max(chi_beta_T.keys())
        betas_t = torch.full_like(chi_t, float('nan'))
        abars_t = torch.full_like(chi_t, float('nan'))
        abars_t_minus1 = torch.full_like(chi_t, float('nan')) 
        
        for i,_ in enumerate(chi_t):
            T_for_this_chi = T[i]
            
            # make sure T is in domain of chi_beta_T
            # if we choose max T, it's clearly a masked residue and doesn't matter anyway 
            cur_T = min(T_for_this_chi, max_t)

            if t <= cur_T: # it's a valid position to find a beta for 
                # get custom schedules for this amino acid based on its T 
                beta_T = chi_beta_T[int(cur_T)]

                beta_schedule, alpha_schedule, abar_schedule = get_beta_schedule(cur_T, 
                                                                                 beta_0, 
                                                                                 beta_T, 
                                                                                 schedule_type)
                
                betas_t[i] = beta_schedule[t_idx]
                abars_t[i] = abar_schedule[t_idx]
                abars_t_minus1[i] = abar_schedule[t_idx-1]

            else: # it's a position who will be masked so beta doesn't matter 
                betas_t[i] = 0
                abars_t[i] = 0
                abars_t_minus1[i] = 0


        
        # Now that we have abars and betas, create mu and sigma for all angles 
        variance = ((1 - abars_t_minus1)/(1-abars_t))*betas_t
        
        a = (torch.sqrt(abars_t_minus1)*betas_t)/(1-abars_t)*chi_0
        b = (torch.sqrt(1-betas_t)*(1-abars_t_minus1)/(1-abars_t))*chi_t
        
        mean = a+b
        
        return mean, variance

    def reveal_residues(self, seq_t, seq_px0, px0, t):
        '''
        Reveal some amino acids in the sequence according to schedule and predictions

        seq_t (torch.tensor): [L] Integer sequence

        seq_px0 (torch.tensor): [L] Model prediction of sequence

        px0 (torch.tensor): [L,14,3] Predicted set of full atom crds

        t (int): Current timestep

        '''
        next_seq = torch.clone(seq_t)
        
        if t <= self.aa_decode_steps:
            t_idx = t-1
            # decode_positions           = self.decode_scheduler.get_decode_positions(t_idx, px0)
            decode_positions           = self.decode_scheduler.get_decode_positions(t_idx, px0, seq_t, seq_px0)
            replacement                = seq_px0[decode_positions]
            replacement                = replacement.to(next_seq.device)
            
            next_seq[decode_positions] = replacement

        return next_seq

    def get_next_torsions(self, xt, px0, seq_t, seq_logit_px0, t, diffusion_mask=None, noise_scale=1., get_seq_next=True):
        """
        Samples the next chi angles and amino acid identities for the pose 

        xt (torch.tensor, required): L,14,3 or L,27,3 set of fullatom crds 

        px0 (torch.tensor, required): Predicted set of full atom crds 

        seq_t (torch.tensor, required): Current sequence in the pose 

        pseq_0 (torch.tensor, required): predicted sequence logits for the x0 prediction

        t (int, required): timestep 

        noise_scale: scale factor for the noise added (only applies to "wrapped_normal")

        """
        converter = XYZConverter().to(device=xt.device)

        L = len(xt)
        # argmaxed sequence logits for x0 prediction
        seq_px0 = torch.argmax(seq_logit_px0, dim=-1) 

        ### initialize full atom reps to calculate torsions 
        RAD = True # use radians for angle stuff 
        # build representation with num_atoms=rf2aa.chemical.NTOTAL
        if not xt.shape[1] == rf2aa.chemical.NTOTAL:
            xt_full = torch.full((L,rf2aa.chemical.NTOTAL,3),np.nan).float()
            # xt_full[:,:14,:] = xt[:,:14]
            xt_full[:,:NHEAVY,:] = xt[:,:NHEAVY]

            
        if not px0.shape[1] == rf2aa.chemical.NTOTAL:
            px0_full = torch.full((L,rf2aa.chemical.NTOTAL,3),np.nan).float()
            # px0_full[:,:14,:] = px0[:,:14]
            px0_full[:,:NHEAVY,:] = px0[:,:NHEAVY]



        # there is no situation where we should have any NaN BB crds here  
        mask = torch.full((L, rf2aa.chemical.NTOTAL), False)
        # set_trace()
        # mask[:,:14] = True 
        mask[:,:NHEAVY] = True 

        ### Calcualte torsions and interpolate between them 

        # Xt torsions
        # torsions_sincos_xt, torsions_alt_sincos_xt, tors_mask_xt, tors_planar_xt = get_torsions(xt_full[None], 
        #                                                                                         seq_t[None], 
        #                                                                                         TOR_INDICES, 
        #                                                                                         TOR_CAN_FLIP, 
        #                                                                                         REF_ANGLES)
        # torsions_sincos_xt, torsions_alt_sincos_xt, tors_mask_xt, tors_planar_xt = XYZConverter.get_torsions(xt_full[None], 
        #                                                                                                     seq_t[None], 
        #                                                                                                     TOR_INDICES, 
        #                                                                                                     TOR_CAN_FLIP, 
        #                                                                                                     REF_ANGLES)


        torsions_sincos_xt, torsions_alt_sincos_xt, tors_mask_xt, tors_planar_xt = converter.get_torsions(xt_full[None], 
                                                                                                            seq_t[None])


        torsions_angle_xt     = torch.atan2(torsions_sincos_xt[...,1],
                                            torsions_sincos_xt[...,0]).squeeze()
        torsions_angle_xt_alt = torch.atan2(torsions_alt_sincos_xt[...,1],
                                            torsions_alt_sincos_xt[...,0]).squeeze()



        # pX0 torsions - uses same seq_t to calculate 
        # torsions_sincos_px0, torsions_alt_sincos_px0, tors_mask_px0, tors_planar_px0 = get_torsions(px0_full[None], 
        #                                                                                             seq_t[None], 
        #                                                                                             TOR_INDICES, 
        #                                                                                             TOR_CAN_FLIP, 
        #                                                                                             REF_ANGLES)
        # torsions_sincos_px0, torsions_alt_sincos_px0, tors_mask_px0, tors_planar_px0 = XYZConverter.get_torsions(px0_full[None], 
        #                                                                                                         seq_t[None], 
        #                                                                                                         TOR_INDICES, 
        #                                                                                                         TOR_CAN_FLIP, 
        #                                                                                                         REF_ANGLES)

        torsions_sincos_px0, torsions_alt_sincos_px0, tors_mask_px0, tors_planar_px0 = converter.get_torsions(px0_full[None], 
                                                                                                                seq_t[None])

        torsions_angle_px0     = torch.atan2(torsions_sincos_px0[...,1],
                                             torsions_sincos_px0[...,0]).squeeze()

        # Take the first one as ground truth --> don't need alt for px0 
        # torsions_angle_alt_px0 = torch.atan2(torsions_alt_sincos_px0[...,1],
        #                                      torsions_alt_sincos_px0[...,0]).squeeze()


        # calculate the minimum angle between both options in xt and the angles in px0
        a = th_min_angle(torsions_angle_xt*180/np.pi,     torsions_angle_px0*180/np.pi, radians=False)*np.pi/180
        b = th_min_angle(torsions_angle_xt_alt*180/np.pi, torsions_angle_px0*180/np.pi, radians=False)*np.pi/180
        condition = torch.abs(a) < torch.abs(b)

        torsions_xt_mindiff = torch.where(condition, torsions_angle_xt, torsions_angle_xt_alt)
        mindiff = torch.where(condition, a, b)

        # these are in radians now 
        start,end = torsions_xt_mindiff.flatten(), torsions_angle_px0.flatten()


        if self.chi_type == 'interp':
            # everybody who is being diffused is going 1/t of the way to px0
            # so interpolate over t steps and use the first interpolation point 
            diff_steps = torch.full_like(start, t)
            angle_interpolations = th_interpolate_angles(start*180/np.pi, end*180/np.pi, t, n_diffuse=diff_steps, radians=False)*np.pi/180
            angle_interpolations = angle_interpolations.reshape(L,NTOTALDOFS,t)
            # angle_interpolations = angle_interpolations.reshape(L,10,t)
            

            # don't allow movement where diffusion mask is True 
            if diffusion_mask is not None:
                angle_interpolations[diffusion_mask,:,:] = end.reshape(L,NTOTALDOFS,-1)[diffusion_mask,:,:1].repeat(1,1,t)
                # angle_interpolations[diffusion_mask,:,:] = end.reshape(L,10,-1)[diffusion_mask,:,:1].repeat(1,1,t)
            # grab interpolated torsions and build new full atom pose 
            # index 0 is the current angle
            # next_torsions = torsions_angle_px0
            temp=0
            if t > 1:
                next_torsions = angle_interpolations[:,:,1]
            elif t == 1:
                next_torsions = torsions_angle_px0
            else:
                # t is 0
                raise RuntimeError('Cannot operate on t=0 input, lowest is t=1')


        elif self.chi_type == 'wrapped_normal':
            # sample from wrapped normal distributions using dynamic aa schedule 
            # Instead of deterministic interpolation, sample from p(Xt-1|Xt,X0)

            #T      = torch.from_numpy(self.idx2steps).repeat(10,1).T # L,10 repeat of idx2steps 
            # T      = self.decode_scheduler.idx2steps.repeat(10,1).T # L,10 repeat of idx2steps
            T      = self.decode_scheduler.idx2steps.repeat(NTOTALDOFS,1).T # L,10 repeat of idx2steps
            T_flat = T.flatten()

            # sample mean angles and stddevs based on 
            # start angle, end angle, t and dynamic T's 

            mu, sigma = self.get_dynamic_mu_sigma(chi_t=start, 
                                                  chi_0=end, 
                                                  T=T_flat, 
                                                  t=t, 
                                                  chi_beta_T=self.chi_beta_T,
                                                  beta_0=0.001)

            # now sample from wrapped normal with scheduled mean and variance to get new angles 
            # next_torsions = torch.normal(mu, sigma*noise_scale).reshape(L,10) % (2*np.pi)
            next_torsions = torch.normal(mu, sigma*noise_scale).reshape(L,NTOTALDOFS) % (2*np.pi)


            if diffusion_mask is not None:
                # replace the sampled torsions with the torsions from the motif 
                # next_torsions[diffusion_mask,...] = torsions_angle_xt[diffusion_mask,...]
                next_torsions[diffusion_mask,...] = torsions_xt_mindiff[diffusion_mask,...]

        else:
            raise NotImplementedError(f'chi_type {self.chi_type} not implemented yet.')

        next_torsions_trig = torch.stack([torch.cos(next_torsions),
                                          torch.sin(next_torsions)], dim=-1)
        
        if get_seq_next:
            next_seq = self.reveal_residues(seq_t, seq_px0, px0, t)
        else:
            next_seq = None

        return next_torsions_trig, next_seq


    def align_to_xt_motif(self, px0, xT, diffusion_mask, eps=1e-6):
        """
        Need to align px0 to motif in xT. This is to permit the swapping of residue positions in the px0 motif for the true coordinates.
        First, get rotation matrix from px0 to xT for the motif residues.
        Second, rotate px0 (whole structure) by that rotation matrix
        Third, centre at origin
        """

        #if True:
        #    return px0
        def rmsd(V,W, eps=0):
            # First sum down atoms, then sum down xyz
            N = V.shape[-2]
            return np.sqrt(np.sum((V-W)*(V-W), axis=(-2,-1)) / N + eps)

        assert xT.shape[1] == px0.shape[1], f'xT has shape {xT.shape} and px0 has shape {px0.shape}'

        L,n_atom,_ = xT.shape # A is number of atoms
        atom_mask = ~torch.isnan(px0)
        #convert to numpy arrays
        px0 = px0.cpu().detach().numpy()
        xT = xT.cpu().detach().numpy()
        diffusion_mask = diffusion_mask.cpu().detach().numpy()

        #1 centre motifs at origin and get rotation matrix
        px0_motif = px0[diffusion_mask,:3].reshape(-1,3)
        xT_motif  =  xT[diffusion_mask,:3].reshape(-1,3)
        px0_motif_mean = np.copy(px0_motif.mean(0)) #need later
        xT_motif_mean  = np.copy(xT_motif.mean(0))

        # center at origin
        px0_motif  = px0_motif-px0_motif_mean
        xT_motif   = xT_motif-xT_motif_mean

        # A = px0_motif
        # B = xT_motif 
        A = xT_motif
        B = px0_motif

        C = np.matmul(A.T, B)

        # compute optimal rotation matrix using SVD
        U,S,Vt = np.linalg.svd(C)


        # ensure right handed coordinate system
        d = np.eye(3)
        d[-1,-1] = np.sign(np.linalg.det(Vt.T@U.T))

        # construct rotation matrix
        R = Vt.T@d@U.T

        # get rotated coords
        rB = B@R

        # calculate rmsd
        rms = rmsd(A,rB)
        self._log.info(f'Sampled motif RMSD: {rms:.2f}')

        #2 rotate whole px0 by rotation matrix
        atom_mask = atom_mask.cpu()
        px0[~atom_mask] = 0 #convert nans to 0
        px0 = px0.reshape(-1,3) - px0_motif_mean
        px0_ = px0 @ R
        # xT_motif_out = xT_motif.reshape(-1,3)
        # xT_motif_out = (xT_motif_out @ R ) + px0_motif_mean
        # ic(xT_motif_out.shape)
        # xT_motif_out = xT_motif_out.reshape((diffusion_mask.sum(),3,3))


        #3 put in same global position as xT
        px0_ = px0_ + xT_motif_mean
        px0_ = px0_.reshape([L,n_atom,3])
        px0_[~atom_mask] = float('nan')
        return torch.Tensor(px0_)
        # return torch.tensor(xT_motif_out)


    def get_potential_gradients(self, seq, xyz, diffusion_mask ):
        '''
        This could be moved into potential manager if desired - NRB

        Function to take a structure (x) and get per-atom gradients used to guide diffusion update

        Inputs:

            seq (torch.tensor, required): [L,?] The current sequence. TODO determine sequence representation being used
            xyz (torch.tensor, required): [L,27,3] Coordinates at which the gradient will be computed

        Outputs:

            Ca_grads (torch.tensor): [L,3] The gradient at each Ca atom
        '''

        if self.potential_manager == None or self.potential_manager.is_empty(): return torch.zeros(xyz.shape[0], 3)

        use_Cb = False

        # seq.requires_grad = True
        xyz.requires_grad = True

        if not seq.grad is None: seq.grad.zero_()
        if not xyz.grad is None: xyz.grad.zero_()

        current_potential = self.potential_manager.compute_all_potentials(seq, xyz)
        current_potential.backward()

        # Since we are not moving frames, Cb grads are same as Ca grads - NRB
        # Need access to calculated Cb coordinates to be able to get Cb grads though
        Ca_grads = xyz.grad[:,1,:]

        if not diffusion_mask == None:
            Ca_grads[diffusion_mask,:] = 0

        # check for NaN's 
        if torch.isnan(Ca_grads).any():
            print('WARNING: NaN in potential gradients, replacing with zero grad.')
            Ca_grads[:] = 0

        return Ca_grads

    def get_next_pose(self,
                      xt,
                      px0,
                      t,
                      diffusion_mask,
                      seq_diffusion_mask,
                      seq_t,
                      pseq0,
                      diffuse_sidechains,
                      fix_motif=True,
                      align_motif=True,
                      include_motif_sidechains=True,
                      rigid_symm_motif_kwargs={},
                      rigid_repeat_motif_kwargs={},
                      origin_before_update=False,
                      num_atoms_modeled=14,
                      rfmotif=None,
                      update_seq_t=None):
                      # origin_before_update=False):
        """
        Wrapper function to take px0, xt and t, and to produce xt-1
        First, aligns px0 to xt
        Then gets coordinates, frames and torsion angles

        Parameters:
            
            xt (torch.tensor, required): Current coordinates at timestep t: [L, 14|27, 3]

            px0 (torch.tensor, required): Prediction of x0 : [L, 14|27, 3]

            t (int, required): timestep t

            diffusion_mask (torch.tensor, required): Mask for structure diffusion

            seq_diffusion_mask (torch.tensor, required): Mask for sequence diffusion

            seq_t (torch.tensor, required): [L,22] Sequence at the current timestep 

            pseq0 (torch.tensor, required): AR decoding: [L,22] Seq Diff: [L,20] Model's prediction of sequence

            diffuse_sidechains (bool): Do diffusive sidechain prediction

            fix_motif (bool): Fix the motif structure

            align_motif (bool): Align the model's prediction of the motif to the input motif

            include_motif_sidechains (bool): Provide sidechains of the fixed motif to the model
        """
        if origin_before_update:
            COM_ALL = xt[:,1,:].mean(0)
            xt = xt - COM_ALL
            px0 = px0 - COM_ALL.to(px0.device)

        # get_allatom = ComputeAllAtomCoords().to(device=xt.device)
        # converter = XYZConverter().to(device=xt.device)
        converter = XYZConverter().to(device=xt.device)
        L,n_atom = xt.shape[:2]
        if diffuse_sidechains:
            assert (xt.shape[1]  == 14) or (xt.shape[1]  == 27)
            assert (px0.shape[1] == 14) or (px0.shape[1] == 27)# need full atom rep for torsion calculations   

        pseq0 = pseq0.to(seq_t.device)

        #align to motif
        # DJ - I altered self.align_to_xt_motif to do a different alignment
        #      Commenting both out for now. Next line was using original functionality,
        #      Line after that uses new functionality which aligns xt motif to px0 motif
        if align_motif and diffusion_mask.any():
            px0 = self.align_to_xt_motif(px0, xt, diffusion_mask)
        # xT_motif_aligned = self.align_to_xt_motif(px0, xt, diffusion_mask)

        px0=px0.to(xt.device)
        # Now done with diffusion mask. if fix motif is False, just set diffusion mask to be all True, and all coordinates can diffuse
        if not fix_motif:
            diffusion_mask[:] = False
        
        # get the next set of CA coordinates 
        _, ca_deltas = get_next_ca(xt, px0, t, diffusion_mask,
                crd_scale=self.crd_scale, beta_schedule=self.schedule, alphabar_schedule=self.alphabar_schedule, noise_scale=self.noise_scale_ca)
        
        # get the next set of backbone frames (coordinates)
        frames_next = get_next_frames(xt, px0, t, diffuser=self.diffuser,
                so3_type=self.so3_type, diffusion_mask=diffusion_mask, noise_scale=self.noise_scale_frame)

        # Apply gradient step from guiding potentials
        # This can be moved to below where the full atom representation is calculated to allow for potentials involving sidechains
        grad_ca    = self.get_potential_gradients(seq_t.clone(), xt.clone(), diffusion_mask=diffusion_mask)
        ca_deltas += self.potential_manager.get_guide_scale(t) * grad_ca
        
        # add the delta to the new frames 
        frames_next = torch.from_numpy(frames_next) + ca_deltas[:,None,:]  # translate

        # rigid-body fitting of motif for symmetry
        if len(rigid_symm_motif_kwargs) > 0:
            frames_next, next_rigid_tmplt = fit_rigid_motif_symm(frames_next, **rigid_symm_motif_kwargs)
        
        # rigid-body fitting of motif for repeats
        elif len(rigid_repeat_motif_kwargs) > 0:
            print('Getting next frames for repeat motif')
            frames_next, next_rigid_tmplt = fit_rigid_motif_repeat(frames_next, **rigid_repeat_motif_kwargs)
        else:
            next_rigid_tmplt = None

        
        if diffuse_sidechains:
            if self.seq_diffuser:
                raise NotImplementedError('Sidechain diffusion and sequence diffusion cannot be performed at the same time')
            seq_t = torch.argmax(seq_t, dim=-1).cpu() # [L]
            pseq0 = torch.argmax(pseq0, dim=-1).cpu() # [L]
            # torsions_next, seq_next = self.get_next_torsions(xt, px0, seq_t, pseq0, t, diffusion_mask, noise_scale = self.noise_scale_torsion)
            torsions_next, seq_next = self.get_next_torsions(xt, px0, seq_t, pseq0, t, diffusion_mask=diffusion_mask, noise_scale = self.noise_scale_torsion)
            # build full atom representation with the new torsions but the current seq
            # _, fullatom_next =  get_allatom(seq_t[None], frames_next[None], torsions_next[None])
            _,fullatom_next = converter.compute_all_atom(seq_t[None], frames_next[None], torsions_next[None])
            # _,fullatom_next = self.converter.compute_all_atom(seq_t[None], frames_next[None], torsions_next[None])
            seq_next = torch.nn.functional.one_hot(
                    seq_next, num_classes=rf2aa.chemical.NAATOKENS).float()

        else:
            fullatom_next = torch.full_like(xt,float('nan')).unsqueeze(0)
            fullatom_next[:,:,:3] = frames_next[None]
            # This is never used so just make it a fudged tensor - NRB
            torsions_next = torch.zeros(1,1)
            if self.seq_diffuser:
                seq_next = self.seq_diffuser.get_next_sequence(seq_t[:,:rf2aa.chemical.NNAPROTAAS], pseq0, t, seq_diffusion_mask) # [L,32]
                zeros = torch.zeros(L, rf2aa.chemical.NAATOKENS-rf2aa.chemical.NNAPROTAAS).to(device=seq_next.device) # [L,80-32]
                seq_next = torch.cat((seq_next, zeros), dim=-1) # [L,80]

                # AF: experimental... how we handle torsions and full atom coords I guess.
                if update_seq_t is not None:
                    # seq_t = torch.argmax(seq_t, dim=-1).cpu() # [L]
                    # pseq0 = torch.argmax(pseq0, dim=-1).cpu() # [L]
                    # torsions_next, _ = self.get_next_torsions(xt, px0, seq_t, pseq0, t, diffusion_mask=diffusion_mask, noise_scale = self.noise_scale_torsion)
                    torsions_next, _ = self.get_next_torsions(xt, px0, torch.argmax(seq_t, dim=-1).cpu(), torch.argmax(pseq0, dim=-1).cpu(), t, 
                                                diffusion_mask=diffusion_mask, 
                                                noise_scale = self.noise_scale_torsion
                                                )
                    _, fullatom_next = converter.compute_all_atom(torch.argmax(seq_t, dim=-1).cpu()[None], frames_next[None].to(torch.float32), torsions_next[None])


            elif update_seq_t == 'decode_seq':
                seq_t = torch.argmax(seq_t, dim=-1).cpu() # [L]
                
                # Set next sequence to random max prob at randomly selected positions:
                if self.seq_decode_mode=='delta_prob_based': # for this we need to keep pseq0 as probabilities
                    # set_trace()
                    seq_next = self.reveal_residues(seq_t, pseq0, px0, t) 
                    pseq0 = torch.argmax(pseq0, dim=-1).cpu() # [L] # convert to max prob after using prob to sample seq
                else: # otherwise (default) just use max prob before finding next seq
                    pseq0 = torch.argmax(pseq0, dim=-1).cpu() # [L]
                    seq_next = self.reveal_residues(seq_t, pseq0, px0, t)
                    
                # Update atoms and torsions:
                torsions_next, _ = self.get_next_torsions(xt, px0, seq_t, pseq0, t, diffusion_mask=diffusion_mask, noise_scale = self.noise_scale_torsion, get_seq_next=False)
                _,fullatom_next = converter.compute_all_atom(seq_t[None], frames_next[None].to(torch.float32), torsions_next[None])
                
                # convert seq_next to onehot
                seq_next = torch.nn.functional.one_hot(seq_next, num_classes=rf2aa.chemical.NAATOKENS).float()


            elif update_seq_t == 'show_full_seq':
                seq_t = torch.argmax(seq_t, dim=-1).cpu() # [L]
                pseq0 = torch.argmax(pseq0, dim=-1).cpu() # [L]

                # Update atoms and torsions:
                torsions_next, _ = self.get_next_torsions(xt, px0, seq_t, pseq0, t, diffusion_mask=diffusion_mask, noise_scale = self.noise_scale_torsion)
                _,fullatom_next = converter.compute_all_atom(
                    seq_t[None], frames_next[None].to(torch.float32), torsions_next[None])

                # Set next sequence directly to max prob resi per position:
                seq_next = torch.nn.functional.one_hot(pseq0, num_classes=rf2aa.chemical.NAATOKENS).float()


            else:
                if self.aa_decode_steps > 0:
                    seq_t = torch.argmax(seq_t, dim=-1).cpu() # [L]
                    pseq0 = torch.argmax(pseq0, dim=-1).cpu() # [L]
                    seq_next = self.reveal_residues(seq_t, pseq0, px0, t)
                    # print(seq_next)
                    seq_next = torch.nn.functional.one_hot(
                            seq_next, num_classes=rf2aa.chemical.NAATOKENS).float()


                else:
                    seq_next = seq_t

        


        # print(pseq0)
        if include_motif_sidechains:
            # fullatom_next[:,diffusion_mask,:14] = xt[None,diffusion_mask]
            fullatom_next[:,diffusion_mask,:num_atoms_modeled] = xt[None,diffusion_mask]

        if origin_before_update:
            fullatom_next = fullatom_next + COM_ALL
            px0 = px0 + COM_ALL.to(px0.device)
            # print('Successful px0 origin before update')
            
        # return fullatom_next.squeeze()[:,:14,:], seq_next, torsions_next, px0, next_rigid_tmplt
        return fullatom_next.squeeze()[:,:num_atoms_modeled,:], seq_next, torsions_next, px0, next_rigid_tmplt
    

def fit_rigid_motif_symm(frames_next, motif_mask, xyz_template, symmRs, symmsub, TSCALE=1.0, **kwargs):
        """
        Takes in updated frames which may have perturbed the motif and fits a rigid body transformation 
        of the motif to the updated coordinates. 
        """
        xyz_template_clone = xyz_template.clone()

        # a coordinate-based error 
        def dist_error_comp(R0,T0,frames_next_motif,xyz_template,TSCALE):
            template_COM = xyz_template.mean(dim=0)
            template_corr = torch.einsum('ij,rj->ri', R0, xyz_template-template_COM) + template_COM + TSCALE*T0[None,None,:]
            loss  = torch.abs(frames_next_motif-template_corr).mean()

            return loss

        def Q2R(Q):
            Qs = torch.cat((torch.ones((1),device=Q.device),Q),dim=-1)
            Qs = normQ(Qs)
            return Qs2Rs(Qs[None,:]).squeeze(0)

        # grab only the motif within the updated frames
        frames_next_motif = frames_next.clone()[:,1,:].to(device=xyz_template.device)
        frames_next_motif = frames_next_motif[motif_mask,:]
        
        # grab only the motif within the template 
        xyz_template_ca = xyz_template[:,1,:]
        xyz_template_ca = xyz_template_ca[motif_mask,:]
        xyz_template_asu = xyz_template[motif_mask,:]

        if symmRs is not None:
            # only grab ASU 
            frames_next_motif = frames_next_motif[:(frames_next_motif.shape[0] // len(symmsub)),:]
            xyz_template_ca   = xyz_template_ca[:(xyz_template_ca.shape[0] // len(symmsub)),:]
            xyz_template_asu  = xyz_template_asu[:(xyz_template_asu.shape[0] // len(symmsub)),:]


        with torch.enable_grad():
            T0 = torch.zeros(3,device=xyz_template.device).requires_grad_(True)
            Q0 = torch.zeros(3,device=xyz_template.device).requires_grad_(True)

        lbfgs = torch.optim.LBFGS([T0,Q0],
                    history_size=10,
                    max_iter=4,
                    line_search_fn="strong_wolfe")
        
        def closure():
            lbfgs.zero_grad()
            loss = dist_error_comp(Q2R(Q0), T0, frames_next_motif, xyz_template_ca, TSCALE)
            loss.backward()
            return loss
        
        # fit the (ASU) motif to the updated coordinates
        for e in range(12):
            loss = lbfgs.step(closure)

        # apply the fitted transformation to the motif
        template_COM         = xyz_template.mean(dim=0)
        updated_template_asu = torch.einsum('ij,brj->bri', Q2R(Q0), xyz_template_asu-template_COM) + template_COM + TSCALE*T0[None,None,:]

        # re-symmetrize the fitted motif according to current symmsubs
        if symmRs is not None:
            updated_template = torch.einsum('sij,raj->srai', symmRs[symmsub], updated_template_asu)
            s,r,a,i = updated_template.shape
            updated_template_symm = updated_template.reshape(s*r,a,3).to(dtype=frames_next.dtype, device=frames_next.device)
            updated_template_symm = updated_template_symm.detach()
        
        # replace the (slightly perturbed) motif with the fitted rigid motif 
        frames_next[motif_mask] = updated_template_symm[:,:3,:]
        # create new xyz_template that contains the current fitted motif
        xyz_template_clone[motif_mask] = updated_template_symm.to(dtype=xyz_template.dtype, device=xyz_template.device)

        return frames_next, xyz_template_clone #updated_template[: updated_template.shape[0] // len(symmsub)]

# def select_true_regions(tensor):
#     true_regions = []
#     start_idx = None

#     for i, value in enumerate(tensor):
#         if value:
#             if start_idx is None:
#                 start_idx = i
#         else:
#             if start_idx is not None:
#                 true_regions.append((start_idx, i))
#                 start_idx = None

#     if start_idx is not None:
#         true_regions.append((start_idx, len(tensor)))

#     return true_regions

def get_grouped_regions(tensor, repeat_length):
    true_regions = []
    start_idx = None
    prev_group = None

    for i, value in enumerate(tensor):
        if value:
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                end_idx = i
                group = (start_idx // repeat_length, end_idx // repeat_length)
                if group != prev_group:
                    true_regions.append([])
                true_regions[-1].append((start_idx, end_idx))
                start_idx = None
                prev_group = group

    if start_idx is not None:
        end_idx = len(tensor)
        group = (start_idx // repeat_length, end_idx // repeat_length)
        if group != prev_group:
            true_regions.append([])
        true_regions[-1].append((start_idx, end_idx))

    return true_regions

def select_true_regions(tensor, repeat_length):
    """
    returns list of masks 
    """
    assert len(tensor) % repeat_length == 0
    nmasks = len(tensor) // repeat_length

    masks = []
    for i in range(nmasks):
        mask = torch.clone(tensor)
        # set any not in this chunk to False 
        cur_start = i * repeat_length
        cur_end = ((i+1) * repeat_length )
        mask[:cur_start]   = False
        mask[(cur_end):] = False
        masks.append(mask)
    
    return masks


# def select_true_regions2(tensor, repeat_length):
#     true_regions = []
#     start_idx = None
#     prev_group = None

#     for i, value in enumerate(tensor):
#         if value:
#             if start_idx is None:
#                 start_idx = i
#         else:
#             if start_idx is not None:
#                 end_idx = i
#                 group = (start_idx // repeat_length, end_idx // repeat_length)
#                 if group != prev_group:
#                     true_regions.append([False] * len(tensor))
#                 true_regions[-1][start_idx:end_idx] = [True] * (end_idx - start_idx)
#                 start_idx = None
#                 prev_group = group

#     if start_idx is not None:
#         end_idx = len(tensor)
#         group = (start_idx // repeat_length, end_idx // repeat_length)
#         if group != prev_group:
#             true_regions.append([False] * len(tensor))
#         true_regions[-1][start_idx:end_idx] = [True] * (end_idx - start_idx)

#     return true_regions




def fit_rigid_motif_repeat(frames_next, 
                           is_motif, 
                           xyz_template, 
                           repeat_length,
                           enforce_repeat_fit=False, 
                           TSCALE=1.0,
                           fit_optim_steps=12):
    """
    Fits motifs rigidly into updated frames

    Parameters:
    -----------

    frames_next (torch.tensor): updated frames, slightly denoised 

    is_motif (torch.tensor): boolena mask of which residues are motif 

    xyz_template (torch.tensor): current 'templated' coordinates of the motif 

    enforce_repeat (bool): whether to enforce repeat symmetry between pairs of repeats

    TSCALE (float): scale of the translation vector fitted 
    """
    xyz_template_clone = xyz_template.clone()
    

    # Save pdb before fitting 
    # util.writepdb(f'frames_before_fit.pdb', frames_next, torch.ones(len(is_motif)).long())
    # util.writepdb(f'template_before_fit.pdb', xyz_template[:,:3], torch.ones(len(is_motif)).long())

    # a coordinate-based error
    def dist_error_comp(R0,T0,frames_next_motif,xyz_template,TSCALE):
            template_COM = xyz_template.mean(dim=0)
            template_corr = torch.einsum('ij,rj->ri', R0, xyz_template-template_COM) + template_COM + TSCALE*T0[None,None,:]
            loss  = torch.abs(frames_next_motif-template_corr).mean()
            return loss


    def Q2R(Q):
        Qs = torch.cat((torch.ones((1),device=Q.device),Q),dim=-1)
        Qs = normQ(Qs)
        return Qs2Rs(Qs[None,:]).squeeze(0)
    

    if not enforce_repeat_fit: # allow symmetry breaking between pairs of repeats 
        # find out where in the motif mask the individual motifs are 
        # i.e., find where the individual contiguous regions of True are
        
        motif_regions = get_grouped_regions(is_motif, repeat_length) # list of tuples (start_idx, end_idx)
        motif_masks = select_true_regions(is_motif, repeat_length) # list of masks (True/False)
        ic(motif_regions)


        def closure():
                lbfgs.zero_grad()
                loss = dist_error_comp(Q2R(Q0), T0, motif_tgt, motif_tmplt, TSCALE)
                loss.backward()
                return loss

        for cur_is_motif in motif_masks:

            # motif_tmplt  = xyz_template[start:end,1,:] # CA of this motif 
            # motif_tmplt_bb = xyz_template[start:end,:3,:] # N, CA, C of this motif
            # motif_tgt = frames_next[start:end,1,:] # CA of the updated (imperfect) motif

            # now use boolean mask 
            motif_tmplt  = xyz_template_clone[cur_is_motif,1,:] # CA of this motif
            motif_tmplt_bb = xyz_template_clone[cur_is_motif,:3,:] # N, CA, C of this motif
            motif_tgt = frames_next[cur_is_motif,1,:] # CA of the updated (imperfect) motif

            with torch.enable_grad():
                T0 = torch.zeros(3,device=xyz_template.device).requires_grad_(True)
                Q0 = torch.zeros(3,device=xyz_template.device).requires_grad_(True)
            
            lbfgs = torch.optim.LBFGS([T0,Q0],
                    history_size=10,
                    max_iter=4,
                    line_search_fn="strong_wolfe")
            

            # fit the motif (ASU) to the updated coordinates
            motif_tgt = motif_tgt.detach()
            motif_tmplt = motif_tmplt.detach()

            best_loss = 1e6 
            same_loss_count = 0
            # print('Fitting motif ', start, end)
            for e in range(fit_optim_steps):
                loss = lbfgs.step(closure)
                # stop if loss is not improving
                if loss < best_loss:
                    best_loss = loss
                    same_loss_count = 0
                else:
                    same_loss_count += 1
                    if same_loss_count > 3:
                        break 

            # replace the slightly perturbed motif with the fitted rigid motif
            com = motif_tmplt.mean(dim=0)
            updated_motif = torch.einsum('ij,raj->rai', Q2R(Q0), motif_tmplt_bb-com) + com + TSCALE*T0[None,:]
            updated_motif = updated_motif.detach()
            
            # print('Replacing motif in frames next')
            # frames_next[start:end] = updated_motif
            # xyz_template_clone[start:end,:3] = updated_motif.to(dtype=xyz_template.dtype, device=xyz_template.device)
            # using boolean mask
            frames_next[cur_is_motif] = updated_motif.to(dtype=frames_next.dtype, device=frames_next.device)
            xyz_template_clone[cur_is_motif,:3] = updated_motif.to(dtype=xyz_template.dtype, device=xyz_template.device)


            
    
    else:
        raise NotImplementedError
    
    # Save pdb after fitting
    # ic(frames_next.shape)
    # ic(xyz_template_clone.shape)
    # util.writepdb(f'frames_after_fit.pdb', frames_next, torch.ones(len(is_motif)).long())
    # util.writepdb(f'template_after_fit.pdb', xyz_template_clone[:,:3], torch.ones(len(is_motif)).long())
    # sys.exit('Debugging')        
    return frames_next, xyz_template_clone


def preprocess(seq, xyz_t, t, T, ppi_design, binderlen, target_res, device):
    """
    Function to prepare inputs to diffusion model
        
        seq (torch.tensor, required): (L) integer sequence 

        msa_masked (torch.tensor, required): (1,1,L,48)

        msa_full  (torch,.tensor, required): (1,1, L,25)
        
        xyz_t (torch,tensor): (L,14,3) template crds (diffused) 
        
        t1d (torch.tensor, required): (1,L,22) this is the t1d before tacking on the chi angles. Last plane is 1/t (conf hacked as timestep)
    """
    set_trace()
    L = seq.shape[-1]
    ### msa_masked ###
    ##################
    msa_masked = torch.zeros((1,1,L,48))
    msa_masked[:,:,:,:22] = nn.one_hot(seq, num_classes=22)[None, None]
    msa_masked[:,:,:,22:44] = nn.one_hot(seq, num_classes=22)[None, None]

    ### msa_full ###
    ################
    msa_full = torch.zeros((1,1,L,25))
    msa_full[:,:,:,:22] = nn.one_hot(seq, num_classes=22)[None, None]

    ### t1d ###
    ########### 
    # NOTE: Not adjusting t1d last dim (confidence) from sequence mask
    t1d = torch.zeros((1,1,L,21))
    t1d[:,:,:,:21] = nn.one_hot(torch.where(seq == 21, 20, seq), num_classes=21)[None,None]
    
    """
    DJ -  next line (commented out) was an attempt to see if model more readily 
          moved the motif if they were all set to same confidence 
          in order to alleveate chainbreaks at motif. Maybe sorta worked? definitely not full fix
    """
    # conf = conf = torch.where(seq == 21, 1-t/T, 1-t/T)[None,None,...,None]
    conf = torch.where(seq == 21, 1-t/T, 1.)[None,None,...,None]
    t1d = torch.cat((t1d, conf), dim=-1)

    # NRB: Adding in dimension for target hotspot residues
    target_residue_feat = torch.zeros_like(t1d[...,0])[...,None]
    if ppi_design and not target_res is None:
        absolute_idx = [resi+binderlen for resi in target_res]
        target_residue_feat[...,absolute_idx,:] = 1

    t1d = torch.cat((t1d, target_residue_feat), dim=-1)

    ### xyz_t ###
    #############
    xyz_t[torch.where(seq == 21, True, False),3:,:] = float('nan')
    xyz_t=xyz_t[None, None]
    xyz_t = torch.cat((xyz_t, torch.full((1,1,L,13,3), float('nan'))), dim=3)
    
    ### t2d ###
    ###########
    t2d = xyz_to_t2d(xyz_t)
  
    ### idx ###
    ###########
    idx = torch.arange(L)[None]
    if ppi_design:
        idx[:,binderlen:] += 200

    ### alpha_t ###
    ###############
    seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,L)
    # alpha, _, alpha_mask, _ = get_torsions(xyz_t.reshape(-1,L,27,3), seq_tmp, TOR_INDICES, TOR_CAN_FLIP, REF_ANGLES)
    alpha, _, alpha_mask, _ = XYZConverter.get_torsions(xyz_t.reshape(-1,L,27,3), seq_tmp, TOR_INDICES, TOR_CAN_FLIP, REF_ANGLES)
    alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))
    alpha[torch.isnan(alpha)] = 0.0
    alpha = alpha.reshape(1,-1,L,10,2)
    alpha_mask = alpha_mask.reshape(1,-1,L,10,1)
    alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 30)
    
    #put tensors on device
    msa_masked = msa_masked.to(device)
    msa_full = msa_full.to(device)
    seq = seq.to(device)
    xyz_t = xyz_t.to(device)
    idx = idx.to(device)
    t1d = t1d.to(device)
    t2d = t2d.to(device)
    alpha_t = alpha_t.to(device)
    return msa_masked, msa_full, seq[None], torch.squeeze(xyz_t, dim=0), idx, t1d, t2d, xyz_t, alpha_t


def parse_pdb(filename, **kwargs):
    '''extract xyz coords for all heavy atoms'''
    lines = open(filename,'r').readlines()
    return parse_pdb_lines(lines, **kwargs)
    # return parse_pdb_na(lines, **kwargs)

# def parse_pdb(filename, **kwargs):
#     '''extract xyz coords for all heavy atoms'''
#     lines = open(filename,'r').readlines()
#     return parse_pdb_lines(lines, **kwargs)

# def parse_pdb_na(lines, parse_hetatom=False, ignore_het_h=True):
#     """
#     adapted from code sent by Ryan
#     """
#     res = []
#     for l in lines:
#         split = l.split()
#         if len(split) < 5:
#             continue
#         if split[0] == 'ATOM' and split[2] == "CA":
#             res.append((split[5], split[3], split[4]))
#         elif split[0] == 'ATOM' and split[2] == "C1'":
#             res.append((split[5], ' ' + split[3], split[4]))


#     # res = [(l[22:26],l[17:20],l[21]) for l in lines if \
#     #         l[:4]=="ATOM" and l[12:16].strip() in ["CA","C1'"] and l[16] in [" ","A"]]
#     # print('res',res)
#     # idx_s = [(int(r[0]),r[2]) for r in res]
#     idx_s = [(r[2], int(r[0])) for r in res]
#     seq = [rf2aa.chemical.aa2num[r[1]] if r[1] in rf2aa.chemical.aa2num.keys() else 20 for r in res]
#     chains = []
#     for r in res:
#         if r[2] not in chains:
#             chains.append(r[2])
#     L_s = [sum([1 for r in res if r[2] == x]) for x in chains]
#     #print('L_s',L_s)
#     xyz = np.full((len(idx_s), rf2aa.chemical.NTOTAL, 3), np.nan, dtype=np.float32)
#     for l in lines:
#         if len(l) < 5 or l[:4] != "ATOM":
#             continue
#         split = l.split()
#         resNo, atom, aa, chid = int(split[5]), split[2], split[3], split[4]
#         if aa in ['DA', 'DC', 'DG', 'DT', 'DX']:
#             aa = ' ' + aa
#         # resNo, atom, aa, chid = int(l[22:26]), l[12:16], l[17:20], l[21]
#         #print(resNo, atom, aa)
#         if (chid,resNo) not in idx_s:
#             continue
#         idx = idx_s.index((chid,resNo))
#         if aa not in rf2aa.chemical.aa2num.keys():
#             continue
#         for i_atm, tgtatm in enumerate(rf2aa.chemical.aa2long[rf2aa.chemical.aa2num[aa]]):
#             if tgtatm and tgtatm.strip() == atom:
#                 #print(i_atm, tgtatm)
#                 # xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
#                 xyz[idx,i_atm,:] = [float(split[6]), float(split[7]), float(split[8])]
#                 break
    
#     mask = np.logical_not(np.isnan(xyz[...,0]))
#     seq = np.array(seq)
#     #xyz[np.isnan(xyz[...,0])] = 0.0
#     # pdbs = []
    
#     # HACK should find a better place for this
#     residue_mask = ~np.isnan(xyz)[:, :3].any(axis=(1, 2))
#     xyz = xyz[residue_mask]
#     # idx = idx[residue_mask]
#     seq = seq[residue_mask]
#     mask = mask[residue_mask]
#     idx_s = [idxpdb for idxpdb, resmask in zip(idx_s, residue_mask) if resmask]
#     out = {
#         'xyz': xyz,
#         'mask': mask,
#         'idx': np.array([i[1] for i in idx_s]),
#         'seq': seq,
#         'pdb_idx': idx_s
#     }

#     return out

def parse_pdb_lines(lines, parse_hetatom=False, ignore_het_h=True):
    # indices of residues observed in the structure
    

    res = []
    pdb_idx = []
    prev_idx_string = 'XXXX'
    for l in lines:
        split = l.split()
        if len(split) < 5:
            continue

        # if l[:4]=="ATOM":
        #     atm_string = l[12:16]
        #     res_string = l[17:20]
        #     chn_string = l[21:22]
        #     idx_string = l[22:26]

        if split[0]=="ATOM":
            atm_string, res_string, chn_string, idx_string = split[2:6]

            # Only add this stuff if we haven't added anything for this resi yet:
            if not idx_string==prev_idx_string:

                # Break this into tower of conditionals so we can exit once we get a valid atom:
                # have to do it this way for a ranking in atom priority, sadly.
                # update this prev_idx_string so we can compare to it in the next iter
                if atm_string.strip()=="CA":
                    res.append((idx_string, res_string))
                    pdb_idx.append(( chn_string.strip(), int(idx_string.strip()) ))
                    prev_idx_string = idx_string

                elif atm_string.strip()=="P":
                    res.append((idx_string, ' ' +res_string))
                    pdb_idx.append(( chn_string.strip(), int(idx_string.strip()) ))
                    prev_idx_string = idx_string

                elif atm_string.strip()=="O5'":
                    res.append((idx_string, ' ' +res_string))
                    pdb_idx.append(( chn_string.strip(), int(idx_string.strip()) ))
                    prev_idx_string = idx_string

                elif atm_string.strip()=="C1'":
                    res.append((idx_string, ' ' +res_string))
                    pdb_idx.append(( chn_string.strip(), int(idx_string.strip()) ))
                    prev_idx_string = idx_string

                # elif atm_string.strip()=="C5'":
                #     print("Really not an ideal situation if we have to use C5 prime...")
                #     res.append((idx_string, res_string))
                #     pdb_idx.append(( chn_string.strip(), int(idx_string.strip()) ))
                #     prev_idx_string = idx_string

    idx_s = [(i[0], i[1]) for i in pdb_idx]
    seq = [util.aa2num[r[1]] if r[1] in rf2aa.chemical.aa2num.keys() else 20 for r in res]
    chains = []
    
    for c in pdb_idx:
        if c[0] not in chains:
            chains.append(c[0])

    L_s = [sum([1 for c in pdb_idx if c[0] == x]) for x in chains]

    xyz = np.full((len(idx_s), rf2aa.chemical.NTOTAL, 3), np.nan, dtype=np.float32)
    for l in lines:
        if len(l) < 5 or l[:4] != "ATOM":
            continue
        split = l.split()
        resNo, atom, aa, chid = int(split[5]), split[2], split[3], split[4]
        # If it is a nucleic acid using the RoseTTAfold naming conventions:
        if aa in ['DA', 'DC', 'DG', 'DT', 'DX', 'RA', 'RC', 'RG', 'RU', 'RX',]:
            aa = ' ' + aa
        # If it is a nucleic acid using the RoseTTAfold naming conventions:
        elif aa in ['A','C','G','U']:
            aa = ' R' + aa


        # resNo, atom, aa, chid = int(l[22:26]), l[12:16], l[17:20], l[21]
        #print(resNo, atom, aa)
        if (chid,resNo) not in idx_s:
            continue
        idx = idx_s.index((chid,resNo))
        if aa not in rf2aa.chemical.aa2num.keys():
            continue

        for i_atm, tgtatm in enumerate(rf2aa.chemical.aa2long[rf2aa.chemical.aa2num[aa]]):
            if tgtatm and tgtatm.strip() == atom:
                #print(i_atm, tgtatm)
                # xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                xyz[idx,i_atm,:] = [float(split[6]), float(split[7]), float(split[8])]
                break

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[...,0]))
    xyz[np.isnan(xyz[...,0])] = 0.0

    # remove duplicated (chain, resi)
    new_idx = []
    i_unique = []
    for i,idx in enumerate(pdb_idx):
        if idx not in new_idx:
            new_idx.append(idx)
            i_unique.append(i)

    pdb_idx = new_idx
    xyz = xyz[i_unique]
    mask = mask[i_unique]

    seq = np.array(seq)[i_unique]

    # remove missing shit:
    residue_mask = ~np.isnan(xyz)[:, :3].any(axis=(1, 2))
    xyz = xyz[residue_mask]
    # idx = idx[residue_mask]
    seq = seq[residue_mask]
    mask = mask[residue_mask]
    idx_s = [idxpdb for idxpdb, resmask in zip(idx_s, residue_mask) if resmask]

    out = {
        'xyz':xyz, # cartesian coordinates, [Lx14]
        'mask':mask, # mask showing which atoms are present in the PDB file, [Lx14]
        'idx':np.array([i[1] for i in idx_s]), # residue numbers in the PDB file, [L]
        # 'idx':np.array([i[1] for i in pdb_idx]), # residue numbers in the PDB file, [L]
        'seq':np.array(seq), # amino acid sequence, [L]
        'pdb_idx': pdb_idx,  # list of (chain letter, residue number) in the pdb file, [L]
    }

    # heteroatoms (ligands, etc)
    if parse_hetatom:
        xyz_het, info_het = [], []
        for l in lines:
            if l[:6]=='HETATM' and not (ignore_het_h and l[77]=='H'):
                info_het.append(dict(
                    idx=int(l[7:11]),
                    atom_id=l[12:16],
                    atom_type=l[77],
                    name=l[16:20]
                ))
                xyz_het.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])

        out['xyz_het'] = np.array(xyz_het)
        out['info_het'] = info_het
        
    return out


# def parse_pdb_lines(lines, parse_hetatom=False, ignore_het_h=True):
#     # indices of residues observed in the structure
    

#     res = []
#     pdb_idx = []
#     prev_idx_string = 'XXXX'
#     for l in lines:
#         if l[:4]=="ATOM":
#             atm_string = l[12:16]
#             res_string = l[17:20]
#             chn_string = l[21:22]
#             idx_string = l[22:26]

#             # Only add this stuff if we haven't added anything for this resi yet:
#             if not idx_string==prev_idx_string:

#                 # Break this into tower of conditionals so we can exit once we get a valid atom:
#                 # have to do it this way for a ranking in atom priority, sadly.
#                 # update this prev_idx_string so we can compare to it in the next iter
#                 if atm_string.strip()=="CA":
#                     res.append((idx_string, res_string))
#                     pdb_idx.append(( chn_string.strip(), int(idx_string.strip()) ))
#                     prev_idx_string = idx_string

#                 elif atm_string.strip()=="P":
#                     res.append((idx_string, res_string))
#                     pdb_idx.append(( chn_string.strip(), int(idx_string.strip()) ))
#                     prev_idx_string = idx_string

#                 elif atm_string.strip()=="O5'":
#                     res.append((idx_string, res_string))
#                     pdb_idx.append(( chn_string.strip(), int(idx_string.strip()) ))
#                     prev_idx_string = idx_string

#                 elif atm_string.strip()=="C5'":
#                     print("Really not an ideal situation if we have to use C5 prime...")
#                     res.append((idx_string, res_string))
#                     pdb_idx.append(( chn_string.strip(), int(idx_string.strip()) ))
#                     prev_idx_string = idx_string

#     seq = [util.aa2num[r[1]] if r[1] in util.aa2num.keys() else 20 for r in res]

#     # 4 BB + up to 10 SC atoms
#     xyz = np.full((len(res), 14, 3), np.nan, dtype=np.float32)
#     for l in lines:
#         if l[:4] != "ATOM":
#             continue
#         chain, resNo, atom, aa = l[21:22], int(l[22:26]), ' '+l[12:16].strip().ljust(3), l[17:20]
#         idx = pdb_idx.index((chain,resNo))
#         # for i_atm, tgtatm in enumerate(util.aa2long[util.aa2num[aa]]):
#         for i_atm, tgtatm in enumerate(util.aa2long[util.aa2num[aa]][:14]): # Nate's proposed change
#             if tgtatm is not None and tgtatm.strip() == atom.strip(): # ignore whitespace
#                 xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
#                 break

#     # save atom mask
#     mask = np.logical_not(np.isnan(xyz[...,0]))
#     xyz[np.isnan(xyz[...,0])] = 0.0

#     # remove duplicated (chain, resi)
#     new_idx = []
#     i_unique = []
#     for i,idx in enumerate(pdb_idx):
#         if idx not in new_idx:
#             new_idx.append(idx)
#             i_unique.append(i)

#     pdb_idx = new_idx
#     xyz = xyz[i_unique]
#     mask = mask[i_unique]

#     seq = np.array(seq)[i_unique]

#     out = {
#         'xyz':xyz, # cartesian coordinates, [Lx14]
#         'mask':mask, # mask showing which atoms are present in the PDB file, [Lx14]
#         'idx':np.array([i[1] for i in pdb_idx]), # residue numbers in the PDB file, [L]
#         'seq':np.array(seq), # amino acid sequence, [L]
#         'pdb_idx': pdb_idx,  # list of (chain letter, residue number) in the pdb file, [L]
#     }

#     # heteroatoms (ligands, etc)
#     if parse_hetatom:
#         xyz_het, info_het = [], []
#         for l in lines:
#             if l[:6]=='HETATM' and not (ignore_het_h and l[77]=='H'):
#                 info_het.append(dict(
#                     idx=int(l[7:11]),
#                     atom_id=l[12:16],
#                     atom_type=l[77],
#                     name=l[16:20]
#                 ))
#                 xyz_het.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])

#         out['xyz_het'] = np.array(xyz_het)
#         out['info_het'] = info_het

#     return out


# def parse_pdb_lines(lines, parse_hetatom=False, ignore_het_h=True):
#     # indices of residues observed in the structure

#     res = [(l[22:26],l[17:20]) for l in lines if l[:4]=="ATOM" and l[12:16].strip() in ["CA","P"]]
#     seq = [util.aa2num[r[1]] if r[1] in util.aa2num.keys() else 20 for r in res]
#     pdb_idx = [( l[21:22].strip(), int(l[22:26].strip()) ) for l in lines if l[:4]=="ATOM" and l[12:16].strip() in ["CA","P"]]  # chain letter, res num

#     # 4 BB + up to 10 SC atoms
#     xyz = np.full((len(res), 14, 3), np.nan, dtype=np.float32)
#     for l in lines:
#         if l[:4] != "ATOM":
#             continue
#         chain, resNo, atom, aa = l[21:22], int(l[22:26]), ' '+l[12:16].strip().ljust(3), l[17:20]
#         idx = pdb_idx.index((chain,resNo))
#         # for i_atm, tgtatm in enumerate(util.aa2long[util.aa2num[aa]]):
#         for i_atm, tgtatm in enumerate(util.aa2long[util.aa2num[aa]][:14]): # Nate's proposed change
#             if tgtatm is not None and tgtatm.strip() == atom.strip(): # ignore whitespace
#                 xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
#                 break

#     # save atom mask
#     mask = np.logical_not(np.isnan(xyz[...,0]))
#     xyz[np.isnan(xyz[...,0])] = 0.0

#     # remove duplicated (chain, resi)
#     new_idx = []
#     i_unique = []
#     for i,idx in enumerate(pdb_idx):
#         if idx not in new_idx:
#             new_idx.append(idx)
#             i_unique.append(i)

#     pdb_idx = new_idx
#     xyz = xyz[i_unique]
#     mask = mask[i_unique]

#     seq = np.array(seq)[i_unique]

#     out = {
#         'xyz':xyz, # cartesian coordinates, [Lx14]
#         'mask':mask, # mask showing which atoms are present in the PDB file, [Lx14]
#         'idx':np.array([i[1] for i in pdb_idx]), # residue numbers in the PDB file, [L]
#         'seq':np.array(seq), # amino acid sequence, [L]
#         'pdb_idx': pdb_idx,  # list of (chain letter, residue number) in the pdb file, [L]
#     }

#     # heteroatoms (ligands, etc)
#     if parse_hetatom:
#         xyz_het, info_het = [], []
#         for l in lines:
#             if l[:6]=='HETATM' and not (ignore_het_h and l[77]=='H'):
#                 info_het.append(dict(
#                     idx=int(l[7:11]),
#                     atom_id=l[12:16],
#                     atom_type=l[77],
#                     name=l[16:20]
#                 ))
#                 xyz_het.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])

#         out['xyz_het'] = np.array(xyz_het)
#         out['info_het'] = info_het

#     return out

# def parse_pdb_lines(lines, parse_hetatom=False, ignore_het_h=True):
#     # indices of residues observed in the structure
#     # res = [(l[22:26],l[17:20]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]
#     # res = [(l[21:22].strip(), l[22:26],l[17:20],l[60:66].strip()) for l in lines if l[:4]=="ATOM" and l[12:16].strip() in ["CA","P"]] # (chain letter, res num, aa, bfactor / plddt)
#     res = [(l[21:22].strip(), l[22:26],l[17:20],l[60:66].strip()) for l in lines if l[:4]=="ATOM" and l[12:16].strip() in ["CA","C5'"]] # (chain letter, res num, aa, bfactor / plddt)

#     seq = [util.aa2num[r[1]] if r[1] in util.aa2num.keys() else 20 for r in res]
#     # pdb_idx = [( l[21:22].strip(), int(l[22:26].strip()) ) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]  # chain letter, res num
#     # pdb_idx = [( l[21:22].strip(), int(l[22:26].strip()) ) for l in lines if l[:4]=="ATOM" and l[12:16].strip() in ["CA","P"]]  # chain letter, res num
#     pdb_idx = [( l[21:22].strip(), int(l[22:26].strip()) ) for l in lines if l[:4]=="ATOM" and l[12:16].strip() in ["CA","C5'"]]  # chain letter, res num

#     # 4 BB + up to 10 SC atoms
#     xyz = np.full((len(res), 14, 3), np.nan, dtype=np.float32)
#     for l in lines:
#         if l[:4] != "ATOM":
#             continue
#         chain, resNo, atom, aa = l[21:22], int(l[22:26]), ' '+l[12:16].strip().ljust(3), l[17:20]
#         # try:
#         idx = pdb_idx.index((chain,resNo))
#         for i_atm, tgtatm in enumerate(util.aa2long[util.aa2num[aa]][:14]): # Nate's proposed change
#         # for i_atm, tgtatm in enumerate(util.aa2long[util.aa2num[aa]]):
#             if tgtatm is not None and tgtatm.strip() == atom.strip(): # ignore whitespace
#                 xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
#                 break


#     # save atom mask
#     mask = np.logical_not(np.isnan(xyz[...,0]))
#     xyz[np.isnan(xyz[...,0])] = 0.0

#     # remove duplicated (chain, resi)
#     new_idx = []
#     i_unique = []
#     for i,idx in enumerate(pdb_idx):
#         if idx not in new_idx:
#             new_idx.append(idx)
#             i_unique.append(i)

#     pdb_idx = new_idx
#     xyz = xyz[i_unique]
#     mask = mask[i_unique]

#     seq = np.array(seq)[i_unique]

#     out = {
#         'xyz':xyz, # cartesian coordinates, [Lx14]
#         'mask':mask, # mask showing which atoms are present in the PDB file, [Lx14]
#         'idx':np.array([i[1] for i in pdb_idx]), # residue numbers in the PDB file, [L]
#         'seq':np.array(seq), # amino acid sequence, [L]
#         'pdb_idx': pdb_idx,  # list of (chain letter, residue number) in the pdb file, [L]
#     }

#     # heteroatoms (ligands, etc)
#     if parse_hetatom:
#         xyz_het, info_het = [], []
#         for l in lines:
#             if l[:6]=='HETATM' and not (ignore_het_h and l[77]=='H'):
#                 info_het.append(dict(
#                     idx=int(l[7:11]),
#                     atom_id=l[12:16],
#                     atom_type=l[77],
#                     name=l[16:20]
#                 ))
#                 xyz_het.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])

#         out['xyz_het'] = np.array(xyz_het)
#         out['info_het'] = info_het
#     return out


def parse_a3m(filename):

    msa = []
    ins = []

    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    #print(filename)

    if filename.split('.')[-1] == 'gz':
        fp = gzip.open(filename, 'rt')
    else:
        fp = open(filename, 'r')

    # read file line by line
    for line in fp:

        # skip labels
        if line[0] == '>':
            continue

        # remove right whitespaces
        line = line.rstrip()

        if len(line) == 0:
            continue

        # remove lowercase letters and append to MSA
        msa.append(line.translate(table))

        # sequence length
        L = len(msa[-1])

        # 0 - match or gap; 1 - insertion
        a = np.array([0 if c.isupper() or c=='-' else 1 for c in line])
        i = np.zeros((L))

        if np.sum(a) > 0:
            # positions of insertions
            pos = np.where(a==1)[0]

            # shift by occurrence
            a = pos - np.arange(pos.shape[0])

            # position of insertions in cleaned sequence
            # and their length
            pos,num = np.unique(a, return_counts=True)

            # append to the matrix of insetions
            i[pos] = num

        ins.append(i)
        if len(msa) == 10000:
            break

    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in msa], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # treat all unknown characters as gaps
    msa[msa > 20] = 20

    ins = np.array(ins, dtype=np.uint8)

    return msa,ins

def symm2nchain(S):
    """Get number of chains in a symmetry group."""
    if S == 'O':
        return 24 
    elif S == 'I':
        return 60
    elif S.startswith('D'):
        return 2*int(S[1:])
    elif S.startswith('T'):
        return 12 
    elif S.startswith('C'):
        return int(S[1:])

def process_target(pdb_path, parse_hetatom=False, center=True, inf_conf=None, d_t1d=32):
    """
    Generally parse target pdb file and return a dictionary of features.

    Handles case where we want to template symmetric proteins into supersymms (e.g., c4 into O, C3 into I)
    """

    # Read target pdb and extract features.
    target_struct = parse_pdb(pdb_path, parse_hetatom=parse_hetatom)

    # Zero-center positions
    ca_center = target_struct['xyz'][:, :1, :].mean(axis=0, keepdims=True)
    if not center:
        ca_center = 0

    xyz       = torch.from_numpy(target_struct['xyz'] - ca_center)
    seq_orig  = torch.from_numpy(target_struct['seq'])
    atom_mask = torch.from_numpy(target_struct['mask'])
    seq_len   = len(xyz)
    
    # is_protein = torch.logical_and((0  <= seq),(seq <= 21)).squeeze()
    # is_nucleic = torch.logical_and((22 <= seq),(seq <= 31)).squeeze()
    # xyz_prot[is_protein,NHEAVYPROT:] = 0
    # xyz_prot[is_nucleic,NHEAVYNUC:] = 0
    # mask_prot[is_protein,NHEAVYPROT:] = False
    # mask_prot[is_nucleic,NHEAVYNUC:] = False

    # Make 27 atom representation
    xyz_27 = torch.full((seq_len, 27, 3), np.nan).float()
    # ipdb.set_trace()
    xyz_27[:, :NHEAVY, :] = xyz[:, :NHEAVY, :]
    # xyz_27[:, :14, :] = xyz[:, :14, :]
    # ipdb.set_trace()
    mask_27 = torch.full((seq_len, 27), False)
    mask_27[:, :NHEAVY] = atom_mask[:, :NHEAVY]
    # mask_27[:, :14] = atom_mask

    out = {
           'xyz_27': xyz_27,
            'mask_27': mask_27,
            'seq': seq_orig,
            'pdb_idx': target_struct['pdb_idx']
            } 
    if parse_hetatom:
        out['xyz_het'] = target_struct['xyz_het']
        out['info_het'] = target_struct['info_het']

    # symmetric template parsing - extract subsymmetry info 
    if inf_conf.subsymm_template is not None:
        # Need to rely on the template pdb for number of chains et. al. 
        tmp_parsed = parse_pdb(inf_conf.subsymm_template, parse_hetatom=False)
        seq_len = len(tmp_parsed['xyz'])

        nchains = len( set(i[0] for i in tmp_parsed['pdb_idx']) )
        Lasu = seq_len // nchains 
        assert seq_len % nchains == 0, "Sequence length must be divisible by number of chains in template."
        Ls = [Lasu] * nchains
        
        # dummy sym metadata 
        symmids, symmRs, symmeta, symmoffset = symmetry.get_pointsym_meta(inf_conf.internal_sym)

        # parse and reshape the symmetric template pdb 
        xyz_t, mask_t, t1d_t, seq_t = parsers.read_multichain_template_pdb(Ls, inf_conf.subsymm_template)
        xyz_t  = xyz_t.reshape(1,len(Ls),-1,27,3).squeeze(dim=0) 
        mask_t = mask_t.reshape(1,len(Ls),-1,27).squeeze(dim=0)
        
        if not d_t1d==22:
            print(f'WARNING: t1d has dim {d_t1d} instead of 22. Hopefully you are using a model trained with NA tokens. ')
        t1d_t  = t1d_t.reshape(1,len(Ls),-1,22).squeeze(dim=0)

        seq_t = torch.argmax(seq_t, dim=-1).reshape(1,len(Ls),-1).squeeze()


        if nchains > 1:
            # one vs all for kabsch --> return axes of symmetry 
            xyz_int = torch.cat( (xyz_t[0:1].repeat(nchains-1,1,1,1), xyz_t[1:]), dim=1)
            mask_int = torch.cat( (mask_t[0:1].repeat(nchains-1,1,1), mask_t[1:]), dim=1)
            

            symmgp, subsymm, symmaxes = symmetry.get_symmetry(xyz_int, mask_int)
            # assert this, means len symmaxes will be 1 - [(nfold, axis, point, i)]
            assert 'c' in symmgp.lower(), "Template must be C-symmetric for now."
            (nfold, _, point, _) = symmaxes[0]
            print('Detected template symmetry group: ',symmgp)


            # map that subsymmetry against the sym of simulation
            mask_t = mask_t[0:1]
            xyz_t = xyz_t[0:1,:Lasu]
            all_possible = inf_conf.subsymm_template_all_possible
            xyz_t, mask_t_2d_subsymm, axis = symmetry.find_subsymmetry(xyz_t, symmgp, symmaxes, symmRs, all_subsymms=all_possible)
            # add intra-chain templating.. ones along diag 
            if all_possible:
                mask_t_2d_subsymm = mask_t_2d_subsymm + torch.eye(mask_t_2d_subsymm.shape[1]).bool()[None]


            angle = 360.0 / nfold
            angle = angle * np.pi / 180.0 
            # make stack of rotations that reconstruct the template from ASU 
            rs = [] 
            for i in range(nfold):
                theta = angle * i
                r = symmetry.matrix_from_rotation(theta, axis)
                rs.append(r)
            rs = torch.stack(rs, dim=0)

            # full length template coordinates and sequence 
            s = rs.shape[0]
            b,l,a,i = xyz_t.shape
            rxyz = torch.einsum('sji,blai->bslaj',rs.transpose(-1,-2), xyz_t).squeeze() # (s,l,a,i) 
            rxyz = rxyz.reshape(l*s,a,i) # (l*s,a,i)
            
            # keeping these for use as template 
            out['mask_2d_subsymm']  = mask_t_2d_subsymm
            out['axis']             = axis
            out['subsymm_xyz']      = rxyz
            out['subsymm_nchains']  = nchains
            out['subsymm_lasu']     = Lasu
            out['subsymm_seq']      = seq_t.reshape(-1) 
            out['subsymm_symbol']   = symmgp
            out['subsymm_axis']     = axis

        else: # C1 symmetric template
            out['mask_2d_subsymm'] = None
            out['axis'] = None 
            out['subsymm_xyz'] = xyz_t.squeeze()
            out['subsymm_seq'] = seq_t.reshape(-1)
            out['subsymm_symbol'] = 'C1'
            out['subsymm_axis'] = None


        # test reconstruction
        # outfolder = '/home/davidcj/projects/rf_diffusion_allatom/rf_diffusion/' 
        # outname = os.path.join(outfolder, 'test_symm_template_reconstruction.pdb')
        # ic(out['template_symm_seq'].shape)
        # ic(out['template_symm_xyz'].shape)
        # ic(util.__file__)
        # util.writepdb(outname, 
        #               out['template_symm_xyz'].squeeze()[:,:3,:],
        #               out['template_symm_seq'].squeeze())


        # sys.exit('debugging exit')
    return out
    

def recycle_schedule(T, rec_sched=None, num_designs=1):
    """  
    Function to convert input recycle schedule into a list of recycles.
    Input:
        - T: Max T
        - rec_sched: timestep:num_recycles|timestep:num_recycles
            e.g. T=200, rec_sched = 50:2/25:4/2:10. At timestep 50, start having 2 recycles, then 4 at t=25, 2=10
    """
    if rec_sched is not None:
        schedule = np.ones(T, dtype='int')
        if "/" in rec_sched:
            parts = rec_sched.split("/")
        else:
            parts = [rec_sched]
        indices=[int(i.split(":")[0]) for i in parts]
        assert all(indices[i] > indices[i+1] for i in range(len(indices) - 1)), "Recycle schedule indices must be in decreasing order"
        for part in parts:
            idx, num = part.split(":")
            schedule[:int(idx)] = int(num)
    else:
        schedule = np.ones(T, dtype='int') * int(num_designs)
    return schedule
 
class BlockAdjacency():
    """
    Class for handling PPI design inference with ss/block_adj inputs.
    Basic idea is to provide a list of scaffolds, and to output ss and adjacency
    matrices based off of these, while sampling additional lengths.
    Inputs:
        - scaffold_list: list of scaffolds (e.g. ['2kl8','1cif']). Can also be a .txt file.
        - scaffold dir: directory where scaffold ss and adj are precalculated
        - sampled_insertion: how many additional residues do you want to add to each loop segment? Randomly sampled 0-this number
        - sampled_N: randomly sample up to this number of additional residues at N-term
        - sampled_C: randomly sample up to this number of additional residues at C-term
        - ss_mask: how many residues do you want to mask at either end of a ss (H or E) block. Fixed value
        - num_designs: how many designs are you wanting to generate? Currently only used for bookkeeping
        - systematic: do you want to systematically work through the list of scaffolds, or randomly sample (default)
        - num_designs_per_input: Not really implemented yet. Maybe not necessary
    Outputs:
        - L: new length of chain to be diffused
        - ss: all loops and insertions, and ends of ss blocks (up to ss_mask) set to mask token (3). Onehot encoded. (L,4)
        - adj: block adjacency with equivalent masking as ss (L,L)     
    """

    def __init__(self, conf, num_designs):
        """
        Parameters:
          inputs:
             conf.scaffold_list as conf
             conf.inference.num_designs for sanity checking
        """

        # either list or path to .txt file with list of scaffolds
        if type(conf.scaffold_list) == list:
            self.scaffold_list = scaffold_list
        elif conf.scaffold_list[-4:] == '.txt':
            #txt file with list of ids
            list_from_file = []
            with open(conf.scaffold_list,'r') as f:
                for line in f:
                    list_from_file.append(line.strip())
            self.scaffold_list = list_from_file
        else:
            raise NotImplementedError

        # path to directory with scaffolds, ss files and block_adjacency files
        self.scaffold_dir = conf.scaffold_dir

        # maximum sampled insertion in each loop segment
        self.sampled_insertion = conf.sampled_insertion

        # maximum sampled insertion at N- and C-terminus
        if '-' in str(conf.sampled_N):
            self.sampled_N = [int(str(conf.sampled_N).split("_")[0]), int(str(conf.sampled_N).split("-")[1])]
        else:
            self.sampled_N = [0, int(conf.sampled_N)]
        if '-' in str(conf.sampled_C):
            self.sampled_C = [int(str(conf.sampled_C).split("_")[0]), int(str(conf.sampled_C).split("-")[1])]
        else:
            self.sampled_C = [0, int(conf.sampled_C)]

        # number of residues to mask ss identity of in H/E regions (from junction)
        # e.g. if ss_mask = 2, L,L,L,H,H,H,H,H,H,H,L,L,E,E,E,E,E,E,L,L,L,L,L,L would become\
        # M,M,M,M,M,H,H,H,M,M,M,M,M,M,E,E,M,M,M,M,M,M,M,M where M is mask
        self.ss_mask = conf.ss_mask

        # whether or not to work systematically through the list
        self.systematic = conf.systematic

        self.num_designs = num_designs

        if len(self.scaffold_list) > self.num_designs:
            print("WARNING: Scaffold set is bigger than num_designs, so not every scaffold type will be sampled")


        # for tracking number of designs
        self.num_completed = 0
        if self.systematic:
            self.item_n = 0

    def get_ss_adj(self, item):
        """
        Given at item, get the ss tensor and block adjacency matrix for that item
        """
        ss = torch.load(os.path.join(self.scaffold_dir, f'{item.split(".")[0]}_ss.pt'))
        adj = torch.load(os.path.join(self.scaffold_dir, f'{item.split(".")[0]}_adj.pt'))

        return ss, adj

    def mask_to_segments(self, mask):
        """
        Takes a mask of True (loop) and False (non-loop), and outputs list of tuples (loop or not, length of element)
        """
        segments = []
        begin=-1
        end=-1
        for i in range(mask.shape[0]):
            # Starting edge case
            if i == 0:
                begin = 0
                continue

            if not mask[i] == mask[i-1]:
                end=i
                if mask[i-1].item() is True:
                    segments.append(('loop', end-begin))
                else:
                    segments.append(('ss', end-begin))
                begin = i

        # Ending edge case: last segment is length one
        if not end == mask.shape[0]:
            if mask[i].item() is True:
                segments.append(('loop', mask.shape[0]-begin))
            else:
                segments.append(('ss', mask.shape[0]-begin))
        return segments

    def expand_mask(self, mask, segments):
        """
        Function to generate a new mask with dilated loops and N and C terminal additions
        """
        N_add = random.randint(self.sampled_N[0], self.sampled_N[1])
        C_add = random.randint(self.sampled_C[0], self.sampled_C[1])

        output = N_add * [False]
        for ss, length in segments:
            if ss == 'ss':
                output.extend(length*[True])
            else:
                # randomly sample insertion length
                ins = random.randint(0, self.sampled_insertion)
                output.extend((length + ins)*[False])
        output.extend(C_add*[False])
        assert torch.sum(torch.tensor(output)) == torch.sum(~mask)
        return torch.tensor(output)

    def expand_ss(self, ss, adj, mask, expanded_mask):
        """
        Given an expanded mask, populate a new ss and adj based on this
        """
        ss_out = torch.ones(expanded_mask.shape[0])*3 #set to mask token
        adj_out = torch.full((expanded_mask.shape[0], expanded_mask.shape[0]), 0.)

        ss_out[expanded_mask] = ss[~mask]
        expanded_mask_2d = torch.full(adj_out.shape, True)
        #mask out loops/insertions, which is ~expanded_mask
        expanded_mask_2d[~expanded_mask, :] = False
        expanded_mask_2d[:,~expanded_mask] = False

        mask_2d = torch.full(adj.shape, True)
        # mask out loops. This mask is True=loop
        mask_2d[mask, :] = False
        mask_2d[:,mask] = False
        adj_out[expanded_mask_2d] = adj[mask_2d]
        adj_out = adj_out.reshape((expanded_mask.shape[0], expanded_mask.shape[0]))

        return ss_out, adj_out


    def mask_ss_adj(self, ss, adj, expanded_mask):
        """
        Given an expanded ss and adj, mask some number of residues at either end of non-loop ss
        """
        original_mask = torch.clone(expanded_mask)
        if self.ss_mask > 0:
            for i in range(1, self.ss_mask+1):
                expanded_mask[i:] *= original_mask[:-i]
                expanded_mask[:-i] *= original_mask[i:]


        ss[~expanded_mask] = 3
        adj[~expanded_mask,:] = 0
        adj[:,~expanded_mask] = 0

        return ss, adj

    def get_scaffold(self):
        """
        Wrapper method for pulling an item from the list, and preparing ss and block adj features
        """
        if self.systematic:
            # reset if num designs > num_scaffolds
            if self.item_n >= len(self.scaffold_list):
                self.item_n = 0
            item = self.scaffold_list[self.item_n]
            self.item_n += 1
        else:
            item = random.choice(self.scaffold_list)
        print("Scaffold constrained based on file: ", item)
        # load files
        ss, adj = self.get_ss_adj(item)
        adj_orig=torch.clone(adj)
        # separate into segments (loop or not)
        mask = torch.where(ss == 2, 1, 0).bool()
        segments = self.mask_to_segments(mask)

        # insert into loops to generate new mask
        expanded_mask = self.expand_mask(mask, segments)

        # expand ss and adj
        ss, adj = self.expand_ss(ss, adj, mask, expanded_mask)

        # finally, mask some proportion of the ss at either end of the non-loop ss blocks
        ss, adj = self.mask_ss_adj(ss, adj, expanded_mask)

        # and then update num_completed
        self.num_completed += 1

        return ss.shape[0], torch.nn.functional.one_hot(ss.long(), num_classes=4), adj

class Target():
    """
    Class to handle targets (fixed chains).
    Inputs:
        - path to pdb file
        - hotspot residues, in the form B10,B12,B60 etc
        - whether or not to crop, and with which method
    Outputs:
        - Dictionary of xyz coordinates, indices, pdb_indices, pdb mask
    """

    def __init__(self, conf: DictConfig):

        self.pdb = parse_pdb(conf.target_path)
        if conf.hotspots:
            hotspots = list(conf.hotspots)
            self.hotspots = [(i[0], int(i[1:])) for i in hotspots]
        else:
            self.hotspots = []
        # sanity check
        if conf.radial_crop is not None and conf.contig_crop is not None:
            raise ValueError("Cannot have both radial and contig cropping")

        # add hotspots
        self.add_hotspots()

        if conf.radial_crop:
#             self.pdb = self.radial_crop(radial_crop)
            raise NotImplementedError("Haven't implemented radial cropping yet")

        elif conf.contig_crop:
            self.pdb = self.contig_crop(conf.contig_crop)
    
    def parse_contig(self, contig_crop):
        """
        Takes contig input and parses
        """
        contig_list = []
        for contig in contig_crop.split(" "):
            subcon=[]
            for crop in contig.split(","):
                if crop[0].isalpha():
                    subcon.extend([(crop[0], p) for p in np.arange(int(crop.split("-")[0][1:]), int(crop.split("-")[1])+1)])
            contig_list.append(subcon)

        return contig_list

    def contig_crop(self, contig_crop, residue_offset=200):
        """
        Method to take a contig string referring to the receptor and output a pdb dictionary with just this crop
        NB there are two ways to provide inputs:
            - 1) e.g. B1-30,0 B50-60,0. This will add a residue offset between each chunk
            - 2) e.g. B1-30,B50-60,B80-100. This will keep the original indexing of the pdb file. 
        Can handle the target being on multiple chains
        """

        # add residue offset between chains if multiple chains in receptor file
        for idx, val in enumerate(self.pdb['pdb_idx']):
            if idx != 0 and val != self.pdb['pdb_idx'][idx-1]:
                self.pdb['idx'][idx:] += (residue_offset + idx)


        # convert contig to mask
        contig_list = self.parse_contig(contig_crop)

        # add residue offset to different parts of contig_list
        for contig in contig_list[1:]:
            start = int(contig[0][1])
            self.pdb['idx'][start:] += residue_offset

        contig_list = np.array(contig_list).flatten()

        mask = np.array([True if i in contig_list else False for i in self.pdb['pdb_idx']])

        # sanity check
        assert np.sum(self.pdb['hotspots']) == np.sum(self.pdb['hotspots'][mask]), "Supplied hotspot residues are missing from the target contig!"

        for key, val in self.pdb:
            self.pdb[key] = val[mask]

    def centre_pdb(self):
        self.pdb['xyz'] = self.pdb['xyz'] - self.pdb['xyz'][:,:1,:].mean(axis=0)

    def add_hotspots(self):
        hotspots = np.array([1. if i in self.hotspots else 0. for i in self.pdb['pdb_idx']])
        self.pdb['hotspots'] = hotspots

    def radial_crop(self, radial_crop):
        #TODO
        pass

    def get_target(self):
        return self.pdb

def assemble_config_from_chk(conf: DictConfig):
    """
    Function for loading model config from checkpoint directly.
    
    Takes:
        - config file
    
    Actions:
        - Loads model checkpoint and looks for "Config"
        - Replaces all -model and -diffuser items
        - Throws a warning if there are items in -model and -diffuser that aren't in the checkpoint
    
    """
    
    # TODO change this so we don't have to load the model twice
    pass


    
def find_breaks(ix, thresh=1):
    # finds positions in ix where the jump is greater than thresh
    breaks = np.where(np.diff(ix) > thresh)[0]
    return np.array(breaks)+1

def pseudo_chainbreak(pdb_idx, break_idx): 
    """
    Breaks the chain at desired index 
    """
    out_idx = torch.clone(pdb_idx)
    prev_break_idx = 0
    Ls = []
    for curr_break_idx in (str(break_idx).split("-")):
        curr_break_idx = int(curr_break_idx)
        #curr_break_idx = curr_break_idx - 1
        out_idx[:,curr_break_idx:] += 200 # 1-indexed
        Ls.append(curr_break_idx - prev_break_idx)
        prev_break_idx = curr_break_idx
    Ls.append(out_idx.shape[1] - prev_break_idx)

    out_chids = []
    for i, L in enumerate(Ls):
        out_chids += [string.ascii_uppercase[i]]*L
    return out_idx, out_chids


def get_breaks(a, cut=1):
    # finds indices where jumps in a occur
    assert len(a.shape) == 1 # must be 1D array

    if torch.is_tensor(a):
        diff = torch.abs( torch.diff(a) )
        breaks = torch.where(diff > cut)[0]

    else:
        diff = np.abs( np.diff(a) )
        breaks = np.where(diff > cut)[0]

    return breaks

def find_true_chunks_indices(tensor):
    # chat gpt algorithm 
    true_indices = torch.nonzero(tensor).flatten().tolist()
    chunks = []
    
    if not true_indices:
        return chunks
    
    start = true_indices[0]
    prev = true_indices[0]
    
    for idx in true_indices[1:]:
        if idx != prev + 1:
            chunks.append((start, prev))
            start = idx
        prev = idx
    
    chunks.append((start, prev))
    return chunks




def find_template_ranges(input_tensor, return_inds=False):
    # mostly afav, some gpt guidance

    input_list = input_tensor.tolist()

    if return_inds:
        regions = []
        start = 0
        for i in range(1, len(input_tensor)):
            if input_tensor[i] != input_tensor[i-1] + 1:
                regions.append((start, i-1))
                start = i
        regions.append((start, len(input_tensor)-1))
    
    else:
        regions = []
        start = 0
        for i in range(1, len(input_tensor)):
            if input_tensor[i] != input_tensor[i-1] + 1:
                regions.append((int(input_tensor[start]), int(input_tensor[i-1])))
                start = i
        
        regions.append((int(input_tensor[start]), int(input_tensor[len(input_tensor)-1])))

    return regions


def merge_regions(regions1, regions2):
    # mostly afav, some gpt guidance
    regions_full = []
    for r1_start, r1_end in regions1:
        for r2_start, r2_end in regions2:
            if r1_start >= r2_start and r1_end <= r2_end:
                regions_full.append((r1_start, r1_end))
            elif r2_start >= r1_start and r2_end <= r1_end:
                regions_full.append((r2_start, r2_end))
            else:
                regions_full.append((r1_start, r1_end))
                regions_full.append((r2_start, r2_end))

    regions_full = sorted(set(regions_full), key=lambda tup: tup[0])

    return regions_full


def get_contig_chunks(contig_map):

    
    # contig_chunk_ranges = []
    all_chunk_ranges = []
    is_motif_chunk = []
    last_idx = 0
    for contig_i in contig_map.sampled_mask:
        for subcontig_ij in contig_i.split(','):
            if subcontig_ij[0].isalpha(): # if it is a template region:
                chain_ij = subcontig_ij[0]
                start_resi_ij, stop_resi_ij = [int(idx) for idx in subcontig_ij[1:].split('-')]
                
                len_ij = stop_resi_ij - start_resi_ij
                new_idx = last_idx + len_ij

                # contig_chunk_ranges.append((last_idx , new_idx))
                is_motif_chunk.append(1)
                all_chunk_ranges.append((last_idx , new_idx))

                last_idx = new_idx + 1

            else: # if it is a diffused region:
                len_ij = int(subcontig_ij.split('-')[0])
                new_idx = last_idx + len_ij

                is_motif_chunk.append(0)
                all_chunk_ranges.append((last_idx, new_idx - 1))

                last_idx = new_idx



    contig_chunk_ranges = [con_tup for con_tup,is_mot in zip(all_chunk_ranges,is_motif_chunk) if is_mot]

    return contig_chunk_ranges


def get_repeat_t2d_mask(L, con_hal_idx0, contig_map, ij_is_visible, nrepeat, supplied_full_contig=True):
    # """
    # Given contig map and motif chunks that can see each other, create t2d mask
    # defining which motif chunks can see each other. 

    # Parameters:
    # -----------
    # L (int): total length of protein being modelled
    
    # con_ref_idx0 (torch.tensor): tensor containing zero-indexed indices of where motif chunks are 
    #                              going to be placed in the output protein.

    # ij_is_visible (list): List of tuples, each tuple defines a set of motif chunks that can see each other.

    # nrepeat (int): Number of repeat units in repeat protein being modelled 
    # """
    # assert all([type(x) == tuple for x in ij_is_visible]), 'ij_is_visible must be list of tuples'
    # assert L%nrepeat == 0
    # Lasu = L // nrepeat

    # # # (1) Define matrix where each row/col is a motif chunk, entries are 1 if motif chunks can see each other
    # # #     and 0 otherwise.

    # # # AF : break regions into the start stop indices based on both template breaks and chain breaks
    # # # this just gets the breaks by mask regions between motifs
    # # mask_breaks = get_breaks(con_hal_idx0)
    # # templ_range_inds = find_template_ranges(con_hal_idx0, return_inds=False)

    # # # add the breaks from the chain jumps
    # # if full_complex_idx0 is not None:
    # #     chain_breaks = get_breaks(full_complex_idx0)
    # #     chain_range_inds = find_template_ranges(full_complex_idx0, return_inds=True)
    # #     # merge these into a list of sub-chunk tuples for template region locations
    # #     chunk_range_inds = merge_regions(templ_range_inds, chain_range_inds)
    # # else:
    # #     chunk_range_inds = templ_range_inds


    # chunk_range_inds = get_contig_chunks(contig_map)
    # # now we have the complete con_hal_idx0 including templates that are separated by chain breaks!
    # true_con_hal_idx0 = torch.tensor([ind for start,end in chunk_range_inds for ind in range(start,end+1)])

    # nchunk = len(chunk_range_inds)
    # nchunk_total = nchunk * nrepeat

    # # initially empty
    # chunk_ij_visible = torch.eye(nchunk_total)
    # # fill in user-defined visibility
    # for S in ij_is_visible:
    #     for i in S:
    #         for j in S: 
    #             if i == j:
    #                 continue # already visible bc eye 
    #             chunk_ij_visible[i,j] = 1
    #             chunk_ij_visible[j,i] = 1

    # # chunk_range_inds_hal, mask_1d_hal = iu.get_hal_contig_ranges(contig_map.contigs, contig_map.inpaint_hal)
    # # (2) Fill in LxL matrix with coarse mask info
    # con_hal_idx0_full = torch.cat([true_con_hal_idx0 + i*Lasu for i in range(nrepeat)])
    # chunk_range_inds_full = [(R[0] + i*Lasu, R[1] + i*Lasu) for i in range(nrepeat) for R in chunk_range_inds]
    

    # mask2d = torch.zeros(L, L)
    # set_trace()
    # # make 1D array designating which chunks are motif
    # is_motif = torch.zeros(L)
    # is_motif[con_hal_idx0_full] = 1 
    # # fill in 2d mask
    # for i in range(len(chunk_range_inds_full)):
    #     for j in range(len(chunk_range_inds_full)):

    #         visible = chunk_ij_visible[i,j] 

    #         if visible: 
    #             start_i, end_i = chunk_range_inds_full[i]
    #             start_j, end_j = chunk_range_inds_full[j]
    #             mask2d[start_i:end_i+1, start_j:end_j+1] = 1
    #             mask2d[start_j:end_j+1, start_i:end_i+1] = 1

    # return mask2d, is_motif
    """
    Given contig map and motif chunks that can see each other, create t2d mask
    defining which motif chunks can see each other. 

    Parameters:
    -----------
    L (int): total length of protein being modelled
    
    con_ref_idx0 (torch.tensor): tensor containing zero-indexed indices of where motif chunks are 
                                 going to be placed in the output protein.

    ij_is_visible (list): List of tuples, each tuple defines a set of motif chunks that can see each other.

    nrepeat (int): Number of repeat units in repeat protein being modelled 
    """
    assert all([type(x) == tuple for x in ij_is_visible]), 'ij_is_visible must be list of tuples'
    assert L%nrepeat == 0
    Lasu = L // nrepeat

    # (1) Define matrix where each row/col is a motif chunk, entries are 1 if motif chunks can see each other
    #     and 0 otherwise.
    breaks = get_breaks(con_hal_idx0)
    nchunk = len(breaks) + 1
    nchunk_total = nchunk * nrepeat

    
    # initially empty
    chunk_ij_visible = torch.eye(nchunk_total)
    # fill in user-defined visibility
    for S in ij_is_visible:
        for i in S:
            for j in S: 
                if i == j:
                    continue # already visible bc eye 
                chunk_ij_visible[i,j] = 1
                chunk_ij_visible[j,i] = 1


    # (2) Fill in LxL matrix with coarse mask info
    L_contigs = len(con_hal_idx0)
    if not supplied_full_contig:
        con_hal_idx0_full = torch.cat([con_hal_idx0 + i*Lasu for i in range(nrepeat)])
    else: 
        con_hal_idx0_full = con_hal_idx0


    mask2d = torch.zeros(L, L)

    # make 1D array designating which chunks are motif
    is_motif = torch.zeros(L)
    is_motif[con_hal_idx0_full] = 1 
    breaks2 = find_true_chunks_indices(is_motif)

    # fill in 2d mask
    for i in range(len(breaks2)):
        for j in range(len(breaks2)):

            visible = chunk_ij_visible[i,j] 

            if visible: 
                start_i, end_i = breaks2[i]
                start_j, end_j = breaks2[j]
                mask2d[start_i:end_i+1, start_j:end_j+1] = 1
                mask2d[start_j:end_j+1, start_i:end_i+1] = 1


    return mask2d, is_motif


def parse_ij_get_repeat_mask(ij_visible, L, n_repeat, con_hal_idx0, supplied_full_contig, full_complex_idx0, contig_map):
    """
    Helper function for getting repeat protein mask 2d info
    """

    abet = 'abcdefghijklmnopqrstuvwxyz'
    abet = [a for a in abet]
    abet2num = {a:i for i,a in enumerate(abet)}

    # ij_visible = self._conf.inference.ij_visible # which chunks can see each other 
    assert ij_visible is not None
    ij_visible = ij_visible.split('-') # e.g., [abc,de,df,...]
    ij_visible_int = [tuple([abet2num[a] for a in s]) for s in ij_visible]

    assert L%n_repeat == 0, 'L must be a multiple of n_repeat'
    Lasu = L//n_repeat 

    ## check that the user-specified ij_visible is valid
    unique_letters      = set([a for a in ''.join(ij_visible)] )
    max_letter          = max([abet2num[a] for a in unique_letters]) # e.g., 5 for abcde
    contig_motif_breaks = get_breaks(con_hal_idx0, cut=1)
    nbreaks             = len(contig_motif_breaks)
    n_motif_contig      = (nbreaks+1)*n_repeat # total number of motif chunks 
    # cannot have more user specified motif chunks than exist in contigs 
    assert max_letter <= n_motif_contig, 'user specified number of motif chunks > number calculated from contigs using {} repeats'.format(n_repeat)


    # create a mask of which chunks are visible to each other compatible with contigs/con_hal_idx0
                  # get_repeat_t2d_mask(L, con_hal_idx0, contig_map, ij_is_visible, nrepeat, supplied_full_contig)
    
    mask_t2d, _ = get_repeat_t2d_mask(L, con_hal_idx0, contig_map, ij_visible_int, n_repeat, supplied_full_contig=supplied_full_contig)

    return mask_t2d



# def get_pdb_contig_ranges(contig_conf, pdb_idx_spec):

#     start_stop_tuples = []
#     for i, contig_i in enumerate(contig_conf):
#         for j, subcontig_ij in enumerate(contig_i.split(',')):
#             if subcontig_ij[0].isalpha():
#                 start_str, stop_str = subcontig_ij[1:].split('-')
#                 start_key_ij = (subcontig_ij[0], int(start_str))
#                 stop_key_ij = (subcontig_ij[0], int(stop_str))
#                 start_stop_tuples.append((pdb_idx_spec.index(start_key_ij),pdb_idx_spec.index(stop_key_ij)))
    

    
#     mask_1d_pdb = torch.full((len(pdb_idx_spec),), 0)
#     for start, stop in start_stop_tuples:
#         mask_1d_pdb[start:stop+1] = 1

#     mask_1d_pdb = mask_1d_pdb.bool()

#     return start_stop_tuples, mask_1d_pdb

# def get_hal_contig_ranges(contig_conf, pdb_idx_spec):

#     start_stop_tuples = []

    
#     contig_list = [subcontig_str for contig_str in contig_conf for subcontig_str in contig_str.split(' ') ]
#     # for i, contig_i in enumerate(contig_conf): # chain num is i
#     for i, contig_i in enumerate(contig_list): # chain num is i
#         running_length = 0
#         for j, subcontig_ij in enumerate(contig_i.split(',')):
#             if subcontig_ij[0].isalpha():
#                 start_str, stop_str = subcontig_ij[1:].split('-')
#                 len_diff_ij = int(stop_str)-int(start_str)
#                 # assert num2abet[i].upper()=='A', "Only single chain diffusion currently supported. Need to find way to specify multi chain free diffusion."
#                 start_key_ij = (num2abet[i].upper(), 1+running_length)
#                 stop_key_ij = (num2abet[i].upper(), 1+running_length+len_diff_ij)
                
#                 # print(start_key_ij)
#                 # print(stop_key_ij)
#                 # ipdb.set_trace()
#                 start_stop_tuples.append((pdb_idx_spec.index(start_key_ij),pdb_idx_spec.index(stop_key_ij)))
#                 running_length += len_diff_ij+1
#             else: # then we at the hal range thingie
#                 if not '-' in subcontig_ij:
#                     len_diff_ij = int(subcontig_ij)
#                     running_length += len_diff_ij
#                 else:
#                     print('INPAINT-RANGE NOT CURRENTLY SUPPORTED. GO BACK AND PROVIDE A FIXED LENGTH CONTIG.')


#     mask_1d_hal = torch.full((len(pdb_idx_spec),), 0)
#     for start, stop in start_stop_tuples:
#         mask_1d_hal[start:stop+1] = 1

#     mask_1d_hal = mask_1d_hal.bool()

#     return start_stop_tuples, mask_1d_hal



def sstr_to_matrix(ss_string, only_basepairs=True):

    def find_any_bracket_pairs(s, open_symbol, close_symbol):
        stack = []
        pairs = []

        for i, char in enumerate(s):
            if char == open_symbol:
                stack.append(i)
            elif char == close_symbol:
                if stack:
                    pairs.append((stack.pop(), i))
                else:
                    raise ValueError(f"No matching open parenthesis for closing parenthesis at index {i}")

        if stack:
            raise ValueError(f"No matching closing parenthesis for open parenthesis at index {stack[0]}")

        return pairs

    def find_loop_bases(s):
        loop_bases = []
        for i, char in enumerate(s):
            if char == '.':
                loop_bases.append(i)
                
        return loop_bases
    open_symbols  = ['(','[','{','<','5','i','f','b']
    close_symbols = [')',']','}','>','3','j','t','e']

    all_pair_dict = {}
    for pair_ind, (open_symbol, close_symbol) in enumerate(zip(open_symbols, close_symbols)):

        num_opens = len([ _ for _ in ss_string if _ == open_symbol])
        num_close = len([ _ for _ in ss_string if _ == close_symbol])
        assert num_opens==num_close , "number of base pairs must be an even number... must be an error somewhere..."
        
        all_pair_dict[pair_ind] = find_any_bracket_pairs(ss_string, open_symbol, close_symbol)
        

    # pairs
    L = len(ss_string)
    ss_adj_mat = np.zeros((L,L))
        
    for pair_ind in all_pair_dict.keys():
        paired_base_list = all_pair_dict[pair_ind]
        for i,j in paired_base_list:
            ss_adj_mat[i,j] = 1
            ss_adj_mat[j,i] = 1

    
    return ss_adj_mat


def get_index_map_dict(contigs):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    index_map_dict = {}
    

    # contig_idx_list = [ np.arange(int(chn_len)) for chn_len in contigs[0].split(' ')]
    
    contig_len_list = []
    for i,contig in enumerate(contigs[0].split(' ')):
        contig_len_list.append([])
        for subcontig in contig.split(','):
            if subcontig[0].isalpha():

                start, stop = subcontig[1:].split('-')
                contig_len_list[i].append(1+int(stop)-int(start))
            else:
                contig_len_list[i].append(int(subcontig))

    
    # contig_idx_list = []
    contig_idx_list = [ np.arange(sum(contig_lengths_i)) for contig_lengths_i in contig_len_list]

    counter = 0
    for i in range(len(contig_idx_list)):
        chn_letter = alphabet[i]

        index_map_dict[chn_letter] = {}
        for chn_i_resi_j in contig_idx_list[i]:
            index_map_dict[chn_letter][chn_i_resi_j + 1] = counter
            counter += 1

    return index_map_dict, counter



def ss_pairs_to_matrix(pair_input, index_map_dict, ss_adj_mat, ss_pair_ori_list=None):

    
    if not ss_pair_ori_list: # without user spec, assume antiparallel for each strand contact.
        ss_pair_ori_list = ['A' for _ in range(len(pair_input))]

    for ss_pair_ori, ss_pair_string in zip(ss_pair_ori_list,pair_input):

        region_i, region_j = ss_pair_string.split(',')

        chn_i = region_i[0]
        bounds_i = [index_map_dict[chn_i][int(_)] for _ in region_i[1:].split('-')]
        start_i, stop_i = min(bounds_i), max(bounds_i)

        chn_j = region_j[0]
        bounds_j = [index_map_dict[chn_j][int(_)] for _ in region_j[1:].split('-')]
        start_j, stop_j = min(bounds_j), max(bounds_j)

        range_i = np.arange(start_i, stop_i+1) # Always orient first stretch in fwd direction

        if not ss_pair_ori=='A': # If pair is not antiparallel, keep them both in fwd direction (parallel)
            range_j = np.arange(start_j, stop_j+1)
        else: # Otherwise default to antiparallel
            range_j = np.arange(start_j, stop_j+1)[ : :-1]
        
        # Ensure that the backgdrop of the paired region is ss-mask token:
        ss_adj_mat[start_i:stop_i, start_j:stop_j] = 2
        ss_adj_mat[start_j:stop_j, start_i:stop_i] = 2
        # ^^^ IS THIS THE RIGHT THING TO DO???

        # Set the pairs as the ss-pair token
        for pair_ind_i, pair_ind_j in zip(range_i, range_j):
            ss_adj_mat[pair_ind_i, pair_ind_j] = 1
            ss_adj_mat[pair_ind_j, pair_ind_i] = 1


    return ss_adj_mat




def ss_pairs_to_matrix_v2(sampler):

    pair_matrix = torch.zeros((sampler.length_init, sampler.length_init)).bool()

    # User can specify the strand orientaton of the pairs:
    if sampler._conf.scaffoldguided.target_ss_pair_ori is not None:
        ss_pair_ori_list = sampler._conf.scaffoldguided.target_ss_pair_ori
    else:# without user spec, assume antiparallel for each strand contact.
        ss_pair_ori_list = ['A' for _ in range(len(sampler._conf.scaffoldguided.target_ss_pairs))]

    for ss_pair_ori, ss_pair_string in zip(ss_pair_ori_list, sampler._conf.scaffoldguided.target_ss_pairs):

        region_i, region_j = ss_pair_string.split(',')

        chn_i = region_i[0]
        bounds_i = [sampler.index_map_dict[chn_i][int(_)] for _ in region_i[1:].split('-')]
        start_i, stop_i = min(bounds_i), max(bounds_i)

        chn_j = region_j[0]
        bounds_j = [sampler.index_map_dict[chn_j][int(_)] for _ in region_j[1:].split('-')]
        start_j, stop_j = min(bounds_j), max(bounds_j)

        range_i = np.arange(start_i, stop_i+1) # Always orient first stretch in fwd direction

        if not ss_pair_ori=='A': # If pair is not antiparallel, keep them both in fwd direction (parallel)
            range_j = np.arange(start_j, stop_j+1)
        else: # Otherwise default to antiparallel
            range_j = np.arange(start_j, stop_j+1)[ : :-1]
        

        # Set the pairs as the ss-pair token
        for pair_ind_i, pair_ind_j in zip(range_i, range_j):
            pair_matrix[pair_ind_i, pair_ind_j] = True
            pair_matrix[pair_ind_j, pair_ind_i] = True


    return pair_matrix


def force_loops(force_loops_list, index_map_dict, ss_adj_mat):

    # set_trace()
    for loop_range_string in force_loops_list:
        # region_i, region_j = loop_range_string.split(',')
        chn_i = loop_range_string[0]
        bounds_i = [index_map_dict[chn_i][int(_)] for _ in loop_range_string[1:].split('-')]
        start_i, stop_i = min(bounds_i), max(bounds_i)
        range_i = np.arange(start_i, stop_i+1)

        for resi in range_i:
            ss_adj_mat[resi,:] = 0
            ss_adj_mat[:,resi] = 0

    return ss_adj_mat

def force_loops_v2(sampler):
    # force_loops_list, sampler._conf.scaffoldguided.force_loops_list
    # index_map_dict,   sampler.index_map_dict
    # ss_adj_mat,       sampler.target_ss_matrix

    loop_vec = torch.zeros((sampler.length_init)).bool()
    for loop_range_string in sampler._conf.scaffoldguided.force_loops_list:
        # region_i, region_j = loop_range_string.split(',')
        chn_i = loop_range_string[0]
        bounds_i = [sampler.index_map_dict[chn_i][int(_)] for _ in loop_range_string[1:].split('-')]
        start_i, stop_i = min(bounds_i), max(bounds_i)
        range_i = np.arange(start_i, stop_i+1)

        for resi in range_i:
            loop_vec[resi] = True

    return loop_vec

def force_multi_contacts(force_multi_contact_list, index_map_dict, ss_adj_mat):

    for contact_set_string in force_multi_contact_list:
        contact_set = contact_set_string.split(',')
        contact_pair_list = [combo for combo in combinations(contact_set,2)]
        for resi_i, resi_j in contact_pair_list:
            idx_i = index_map_dict[resi_i[0]][int(resi_i[1:])]
            idx_j = index_map_dict[resi_j[0]][int(resi_j[1:])]

            ss_adj_mat[idx_i,idx_j] = 1
            ss_adj_mat[idx_j,idx_i] = 1


    return ss_adj_mat

def force_multi_contacts_v2(sampler):

    # force_multi_contact_list,  sampler._conf.scaffoldguided.force_multi_contacts
    # index_map_dict,            sampler.index_map_dict
    # ss_adj_mat                 sampler.target_ss_matrix

    pair_matrix = torch.zeros((sampler.length_init, sampler.length_init)).bool()

    for contact_set_string in sampler._conf.scaffoldguided.force_multi_contacts:
        contact_set = contact_set_string.split(',')
        contact_pair_list = [combo for combo in combinations(contact_set,2)]
        for resi_i, resi_j in contact_pair_list:
            idx_i = sampler.index_map_dict[resi_i[0]][int(resi_i[1:])]
            idx_j = sampler.index_map_dict[resi_j[0]][int(resi_j[1:])]

            pair_matrix[idx_i,idx_j] = True
            pair_matrix[idx_j,idx_i] = True
            # sampler.target_ss_matrix[idx_i,idx_j] = 1
            # sampler.target_ss_matrix[idx_j,idx_i] = 1


    return pair_matrix
      
def ss_matrix_to_t2d_feats(ss_matrix):

    ss_matrix = torch.from_numpy(ss_matrix).long()
    ss_templ_onehot = torch.nn.functional.one_hot(ss_matrix, num_classes=3)
    ss_templ_onehot = ss_templ_onehot.reshape(1, 1, *ss_templ_onehot.shape).repeat(1,3,1,1,1)

    return ss_templ_onehot
    

def get_target_ss_matrix(sampler):


    pair_matrix = torch.zeros((sampler.length_init, sampler.length_init)).bool()
    loop_vector = torch.zeros(sampler.length_init).bool()
    mask_vector = torch.zeros(sampler.length_init).bool()

    # first modification: check for ss-string specification:
    if (sampler._conf.scaffoldguided.target_ss_string_list is not None) or (sampler._conf.scaffoldguided.target_ss_string is not None):

        pair_matrix_list = []
        loop_region_list = []
        ss_loc_list = []

        if sampler._conf.scaffoldguided.target_ss_string_list is not None:
            for string_spec_i in sampler._conf.scaffoldguided.target_ss_string_list:
                ss_loc_i, ss_string_i = string_spec_i.split(':')

                bp_partners_2d = torch.from_numpy(sstr_to_matrix(ss_string_i, only_basepairs=True)).bool()
                loop_regions = torch.tensor([s_i=='.' for s_i in ss_string_i])
                ss_target_loc = tuple(sampler.index_map_dict[ss_loc_i[0]][int(_)] for _ in ss_loc_i[1:].split('-') )

                pair_matrix_list.append(bp_partners_2d)
                loop_region_list.append(loop_regions)
                ss_loc_list.append(ss_target_loc)

        
        elif sampler._conf.scaffoldguided.target_ss_string is not None:
            # Or just replace it with full ss string
            bp_partners_2d = torch.from_numpy(sstr_to_matrix(sampler._conf.scaffoldguided.target_ss_string, only_basepairs=True)).bool()
            loop_regions = torch.tensor([s_i=='.' for s_i in sampler._conf.scaffoldguided.target_ss_string])
            ss_target_loc = (0,sampler.length_init)
            
            pair_matrix_list.append(bp_partners_2d)
            loop_region_list.append(loop_regions)
            ss_loc_list.append(ss_target_loc)

        # Now loop through all the different sub-ss-specs we have made and add them to the pair_matrix and loop_vector
        for bp_partners_i, loop_regions_i, (from_i,to_i) in zip(pair_matrix_list, loop_region_list, ss_loc_list):
            
            full_loop_regions_i = torch.zeros(sampler.length_init).bool()
            full_loop_regions_i[from_i:to_i+1] = loop_regions_i
            pair_matrix[from_i:to_i+1,from_i:to_i+1][bp_partners_i] = True
            loop_vector[full_loop_regions_i] = True

            # sampler.target_ss_matrix[full_loop_regions_i,:] = 0
            # sampler.target_ss_matrix[:,full_loop_regions_i] = 0
            # sampler.target_ss_matrix[from_i:to_i+1,from_i:to_i+1][bp_partners_i] = 1
            
    # second: check for basepair range specifications (best way to specify):
    if sampler._conf.scaffoldguided.target_ss_pairs is not None:
        # Here we add paired regions to the target ss matrix
        # Get this pair spec:
        target_ss_pair_mat = ss_pairs_to_matrix_v2(sampler)
        pair_matrix[target_ss_pair_mat] = True

    # Now we can add force loops, or triple+ contacts
    # Force loops:
    if sampler._conf.scaffoldguided.force_loops_list is not None:

        force_loop_vec = force_loops_v2(sampler)
        loop_vector[force_loop_vec] = True

    # Force the multi-contact:
    if sampler._conf.scaffoldguided.force_multi_contacts is not None:
        multi_contact_mat = force_multi_contacts_v2(sampler)
        pair_matrix[multi_contact_mat] = True

    pair_vector = pair_matrix.any(dim=0)
    loop_vector[pair_vector] = False # Set loops to False if there are *ANY* base pairs involving that position


    # IMPORTANT!!!!!  IMPORTANT!!!!!  IMPORTANT!!!!!  IMPORTANT!!!!!  IMPORTANT!!!!!  IMPORTANT!!!!!
    # IMPORTANT!!!!!  IMPORTANT!!!!!  IMPORTANT!!!!!  IMPORTANT!!!!!  IMPORTANT!!!!!  IMPORTANT!!!!!
    #   During training, I apply striped masks across matrix as last step. 
    #   Maybe that means I should apply stripe mask as last step here too?
    #   IN THEORY IT SHOULD GIVE SAME EXACT MATRIX EITHER WAY, 
    #   but it shouldnt be the "2" stripe from the pair vector, because that is not a fully masked region.
    #   it actually needs to be the logical nor of the pair_vector and the loop_vector.
    # IMPORTANT!!!!!  IMPORTANT!!!!!  IMPORTANT!!!!!  IMPORTANT!!!!!  IMPORTANT!!!!!  IMPORTANT!!!!!
    # IMPORTANT!!!!!  IMPORTANT!!!!!  IMPORTANT!!!!!  IMPORTANT!!!!!  IMPORTANT!!!!!  IMPORTANT!!!!!



    # By default fall back to masking everything:
    # We initialize the ss_matrix as fully masked, then we modify it as we go
    # Legacy way of setting up matrix because of how I used to train: 
    # (1) initialize everything as "2" (mask)
    # (2) set stripes of "0" at all loop and pair locations
    # (3) pairs i,j elements are set to "1"

    # target_ss_matrix = (2*torch.ones((sampler.length_init,sampler.length_init))).long() # (1)
    # target_ss_matrix[loop_vector,:] = 0 # (2i)
    # target_ss_matrix[:,loop_vector] = 0 # (2j)
    # target_ss_matrix[pair_matrix] = 1   # (3)

    # HOW I USED TO TRAIN: SET BACKDROP OF PAIRED REGIONS TO ZERO:
    if sampler._conf.scaffoldguided.target_ss_matrix_loop_backdrop:
        print('USING LOOP BACKDROP NA SS T2D')
        target_ss_matrix = (2*torch.ones((sampler.length_init,sampler.length_init))).long() # (1)
        target_ss_matrix[loop_vector,:] = 0 # (2i)
        target_ss_matrix[:,loop_vector] = 0 # (2j)
        target_ss_matrix[pair_vector,:] = 0 # (2i)
        target_ss_matrix[:,pair_vector] = 0 # (2j)
        target_ss_matrix[pair_matrix] = 1   # (3)
    else:
        print('USING MASK BACKDROP NA SS T2D')
        # target_ss_matrix = (2*torch.ones((sampler.length_init,sampler.length_init))).long() # (1)
        # target_ss_matrix[loop_vector,:] = 0 # (2i)
        # target_ss_matrix[:,loop_vector] = 0 # (2j)
        # target_ss_matrix[pair_matrix] = 1   # (3)
        target_ss_matrix = (2*torch.ones((sampler.length_init,sampler.length_init))).long() # (1)
        target_ss_matrix[pair_vector,:] = 2 # (2i)
        target_ss_matrix[:,pair_vector] = 2 # (2j)
        target_ss_matrix[loop_vector,:] = 0 # (3i)
        target_ss_matrix[:,loop_vector] = 0 # (3j)
        target_ss_matrix[pair_matrix] = 1   # (3)

    

    # Future way of setting up matrix based on how I should train in the future: 
    # (1) initialize everything as "2" (mask)
    # (2) set stripes of "2" at all pair locations
    # (3) set stripes of "0" at all loop locations
    # (4) pairs i,j elements are set to "1"

    # target_ss_matrix = (2*torch.ones((sampler.length_init,sampler.length_init))).long() # (1)
    # target_ss_matrix[pair_vector,:] = 2 # (2i)
    # target_ss_matrix[:,pair_vector] = 2 # (2j)
    # target_ss_matrix[loop_vector,:] = 0 # (3i)
    # target_ss_matrix[:,loop_vector] = 0 # (3j)
    # target_ss_matrix[pair_matrix] = 1   # (3)

    # set_trace()


    # Save the matrix image if we want:
    # if sampler._conf.scaffoldguided.save_ss_matrix_png and (t==sampler._conf.diffuser.T):
    if sampler._conf.scaffoldguided.save_ss_matrix_png:
        print('SAVING SS MATRIX PIC!')
        output_pic_filapath = sampler._conf.inference.output_prefix+'.png'
        output_dirpath = os.path.dirname(output_pic_filapath)

        if not (os.path.exists(output_dirpath) and os.path.isdir(output_dirpath)):
            os.mkdir(output_dirpath)

        fig, ax = plt.subplots(1,1,figsize=(5,5), dpi=300)
        ax.imshow(np.array(target_ss_matrix))
        plt.tight_layout()
        plt.savefig(output_pic_filapath, bbox_inches='tight', dpi='figure')
        plt.close()



    return target_ss_matrix


    


get_bp_partner = {
                'dna': {
                    ' DA': ' DT' ,
                    ' DC': ' DG' ,
                    ' DG': ' DC' ,
                    ' DT': ' DA' ,
                    ' DX': ' DX' ,
                    },
                'rna': {
                    ' RA': ' RU' ,
                    ' RC': ' RG' ,
                    ' RG': ' RC' ,
                    ' RU': ' RA' ,
                    ' RX': ' RX' ,
                    },
                }


letters_to_RF_codes = {
    'protein':{
            'A': 'ALA',
            'R': 'ARG',
            'N': 'ASN',
            'D': 'ASP',
            'C': 'CYS',
            'Q': 'GLN',
            'E': 'GLU',
            'G': 'GLY',
            'H': 'HIS',
            'I': 'ILE',
            'L': 'LEU',
            'K': 'LYS',
            'M': 'MET',
            'F': 'PHE',
            'P': 'PRO',
            'S': 'SER',
            'T': 'THR',
            'W': 'TRP',
            'Y': 'TYR',
            'V': 'VAL',
            '?': 'UNK',
        },
    'dna':{
            'A': ' DA',
            'C': ' DC',
            'G': ' DG',
            'T': ' DT',
            'X': ' DX',
        },
    'rna':{
            'A': ' RA',
            'C': ' RC',
            'G': ' RG',
            'U': ' RU',
            'X': ' RX',
        },
}


def get_sequence_spec(set_sequence_spec, ss_mat, index_map_dict, contig_map, fill_canonical=False):
    # set_sequence_spec=None,
    # fill_canonical=False,
    
    L = ss_mat.shape[0]

    bp_mat = (ss_mat == 1)

    # seq_spec_vec = torch.nan * torch.full((L,), 0, dtype=torch.long) 
    seq_spec_list = []

    for set_sequence_spec_i in set_sequence_spec:
        seq_spec_i = set_sequence_spec_i.upper()
        contig_spec_i, sequence_i = seq_spec_i.split(':')
        chain_i = contig_spec_i[0]
        resis_i = contig_spec_i[1:]

        poly_type_i = contig_map['polymer_chains'][abet2num[chain_i.lower()]]

        # convert to RF codes:


        if '-' in resis_i:
            start_resi_i, stop_resi_i = resis_i.split('-')
            resi_range_i = [resi_ij for resi_ij in range(int(start_resi_i), int(stop_resi_i)+1)]
        else:
            resi_range_i = [int(resis_i)]


        assert len(sequence_i)==len(resi_range_i), "the provided sequence length must match the specified resi range length"

        rf_seq_i = [letters_to_RF_codes[poly_type_i][aa_letter_ij] for aa_letter_ij in sequence_i]

        for resdex_ij, aa_ij in zip(resi_range_i, rf_seq_i):

            resi_ind_ij = index_map_dict[chain_i][resdex_ij]

            aa_num_ij = aa2num[aa_ij]

            # seq_spec_vec[resi_ind_ij] = aa_num_ij
            seq_spec_list.append((resi_ind_ij, aa_num_ij))

            if fill_canonical:

                partner_ind_ij = bp_mat[resi_ind_ij,:].nonzero()[0][0]
                partner_aa_num_ij = aa2num[ get_bp_partner[poly_type_i][aa_ij] ]
                # seq_spec_vec[partner_ind_ij] = partner_aa_num_ij
                seq_spec_list.append((partner_ind_ij, partner_aa_num_ij))


    return seq_spec_list



