# script for diffusion protocols 
import torch 
import numpy as np

from scipy.spatial.transform import Rotation as scipy_R
from scipy.spatial.transform import Slerp 

from util import rigid_from_3_points, get_torsions

from util import torsion_indices as TOR_INDICES 
from util import torsion_can_flip as TOR_CAN_FLIP
from util import reference_angles as REF_ANGLES

from util_module import ComputeAllAtomCoords

from diff_util import th_min_angle, th_interpolate_angles, get_aa_schedule 

from chemical import INIT_CRDS 

import time 

from icecream import ic  

torch.set_printoptions(sci_mode=False)

def cosine_interp(T, eta_max, eta_min):
    """
    Cosine interpolation of some value between its max <eta_max> and its min <eta_min>

    from https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    
    Parameters:
        T (int, required): total number of steps 
        eta_max (float, required): Max value of some parameter eta 
        eta_min (float, required): Min value of some parameter eta 
    """
    
    t = torch.arange(T)
    out = eta_min + 0.5*(eta_max - eta_min)*(1+torch.cos((t/T)*np.pi))
    return out 


def get_beta_schedule(T, b0, bT, schedule_type, schedule_params={}):
    """
    Given a noise schedule type, create the beta schedule 
    """
    assert schedule_type in ['linear', 'geometric', 'cosine']

    # linear noise schedule 
    if schedule_type == 'linear':
        schedule = torch.linspace(b0, bT, T) 

    # geometric noise schedule 
    elif schedule_type == 'geometric': 
        raise NotImplementedError('geometric schedule not ready yet')
    
    # cosine noise schedule 
    else:
        schedule = cosine_interp(T, bT, b0) 

    return schedule 


class EuclideanDiffuser():
    # class for diffusing points 

    def __init__(self,
                 T, 
                 b_0, 
                 b_T, 
                 schedule_type='cosine',
                 schedule_kwargs={},
                 ):
        
        self.T = T 
        
        # make noise/beta schedule 
        self.beta_schedule = get_beta_schedule(T, b_T, b_0, schedule_type, **schedule_kwargs)
        self.alpha_schedule = 1-self.beta_schedule 

    
    # NOTE: this one seems fishy - doesn't match apply_kernel
    #def apply_kernel_closed(self, x0, t, var_scale=1, mask=None):
    #    """
    #    Applies a noising kernel to the points in x 

    #    Parameters:
    #        x0 (torch.tensor, required): (N,3,3) set of backbone coordinates from ORIGINAL backbone 

    #        t (int, required): Which timestep

    #        noise_scale (float, required): scale for noise 
    #    """
    #    t_idx = t-1 # bring from 1-indexed to 0-indexed

    #    assert len(x0.shape) == 3
    #    L,_,_ = x0.shape 

    #    # c-alpha crds 
    #    ca_xyz = x0[:,1,:]


    #    b_t = self.beta_schedule[t_idx]    
    #    a_t = self.alpha_schedule[t_idx]


    #    # get the noise at timestep t
    #    a_bar = torch.prod(self.alpha_schedule[:t_idx], dim=0)

    #    mean  = torch.sqrt(a_bar)*ca_xyz 
    #    var   = torch.ones(L,3)*(1-a_bar)*var_scale


    #    sampled_crds = torch.normal(mean, var)
    #    delta = sampled_crds - ca_xyz

    #    if mask != None:
    #        delta[mask,...] = 0

    #    out_crds = x0 + delta[:,None,:]

    #    return out_crds 


    def diffuse_translations(self, xyz, diffusion_mask=None, var_scale=1):
        return self.apply_kernel_recursive(xyz, diffusion_mask, var_scale)


    def apply_kernel(self, x, t, diffusion_mask=None, var_scale=1):
        """
        Applies a noising kernel to the points in x 

        Parameters:
            x (torch.tensor, required): (N,3,3) set of backbone coordinates 

            t (int, required): Which timestep

            noise_scale (float, required): scale for noise 
        """
        t_idx = t-1 # bring from 1-indexed to 0-indexed

        assert len(x.shape) == 3
        L,_,_ = x.shape 

        # c-alpha crds 
        ca_xyz = x[:,1,:]


        b_t = self.beta_schedule[t_idx]    


        # get the noise at timestep t
        mean  = torch.sqrt(1-b_t)*ca_xyz
        var   = torch.ones(L,3)*(b_t)*var_scale

        sampled_crds = torch.normal(mean, var) 
        delta = sampled_crds - ca_xyz  

        if not diffusion_mask is None:
            delta[diffusion_mask,...] = 0

        out_crds = x + delta[:,None,:]

        return out_crds, delta


    def apply_kernel_recursive(self, xyz, diffusion_mask=None, var_scale=1):
        """
        Repeatedly apply self.apply_kernel T times and return all crds 
        """
        bb_stack = []
        T_stack  = []

        cur_xyz  = torch.clone(xyz)  

        for t in range(1,self.T+1):     
            cur_xyz, cur_T = self.apply_kernel(cur_xyz, 
                                        t, 
                                        var_scale=var_scale, 
                                        diffusion_mask=diffusion_mask)
            bb_stack.append(cur_xyz)
            T_stack.append(cur_T)
        

        return torch.stack(bb_stack).transpose(0,1), torch.stack(T_stack).transpose(0,1)

        
#TODO:  This class uses scipy+numpy for the slerping/matrix generation 
#       Probably could be much faster if everything was in torch
class SLERP():
    """
    Class for taking in a set of backbone crds and performing slerp
    on all of them 
    """

    def __init__(self, T):

        self.T = T 

    def diffuse_frames(self, xyz, diffusion_mask=None):
        return self.slerp(xyz, diffusion_mask)
    
    def slerp(self, xyz, diffusion_mask=None):
        """
        Perform spherical linear interpolation from the True coordinate frame for each 
        residue to a randomly sampled coordinate frame 

        Parameters:
            xyz (np.array or torch.tensor, required): (L,3,3) set of backbone coordinates 

            mask (np.array or torch.tensor, required): (L,1) set of bools. True/1 is NOT diffused, False/0 IS diffused
        Returns:
            np.array : N/CA/C coordinates for each residue in the SLERP 
                        (T,L,3,3), where T is num timesteps
            
        """
        # diffusion_mask = None 

        if torch.is_tensor(xyz):
            xyz = xyz.numpy()

        t = np.arange(self.T)
        alpha = t/self.T
        
        R_rand = scipy_R.random(len(xyz))
        
        N  = torch.from_numpy(  xyz[None,:,0,:]  )
        Ca = torch.from_numpy(  xyz[None,:,1,:]  )
        C  = torch.from_numpy(  xyz[None,:,2,:]  )
        
        # scipy rotation object for true coordinates
        R_true, Ca = rigid_from_3_points(N,Ca,C)
        R_true = scipy_R.from_matrix(R_true.squeeze())
        
        # bad - could certainly vectorize somehow 
        all_interps = []
        for i in range(len(xyz)):

            r_true = R_true[i].as_matrix()
            r_rand = R_rand[i].as_matrix()

            # handle potential nans in BB frames / crds 
            if not np.isnan(r_true).any():
            
                if not diffusion_mask[i]:
                    key_rots = scipy_R.from_matrix(np.stack([r_true, r_rand], axis=0))
                else:
                    key_rots = scipy_R.from_matrix(np.stack([r_true, r_true], axis=0))

            else:
                key_rots = scipy_R.from_matrix(np.stack([np.eye(3), np.eye(3)], axis=0))
        
            key_times = [0,1]
        
            interpolator = Slerp(key_times, key_rots)
            interp_time = alpha
            

            interp_rot  = interpolator(interp_time)

            all_interps.append(interp_rot.as_matrix())
        
        all_interps = np.stack(all_interps, axis=0)
        
        # Now apply all the interpolated rotation matrices to the original rotation matrices and get the frames at each timestep
        slerped_frames = np.einsum('lrij,laj->lrai', all_interps, R_true.as_matrix())
        
        # apply the slerped frames to the coordinates
        
        slerped_crds   = np.einsum('lrij,laj->lrai', slerped_frames, xyz[:,:3,:] - Ca.squeeze()[:,None,...].numpy()) + Ca.squeeze()[:,None,None,...].numpy()
        
        # (T,L,3,3) set of backbone coordinates and frames 
        return slerped_crds, slerped_frames


class INTERP():
    """
    Class for diffusing chi angles by randomly 
    sampling in [-pi,pi] and interpolating on the unit circle from 
    random to native
    """

    def __init__(self, T):
        self.T = T

    def diffuse_torsions(self, xyz14, seq, mask14, diffusion_mask=None, n_steps=100):
        return self.interp(xyz14, seq, mask14, diffusion_mask, n_steps)
    

    def interp(self, xyz14, seq, mask14, diffusion_mask, n_steps):
        """
        Creates a linear interpolation between the true chi angle of a sidechain 
        and a randomly chosen chi angle between -pi<theta<pi

        Parameters:
            xyz14 (torch.tensor, required): (L,14,3) set of heavyatom coordinates 

            seq (torch.tensor, required): integer sequence 

            mask (torch.tensor, required): Lx14 atom mask

            decode_times (torch.tensor, optional): For each residue position, how many 
                                diffusion steps will it get. 

        Returns:
            torch.tensor - (L,T,4,2) tensor of the sin/cos of each interpolated set of chis 
        """

        RAD = True # in radians 
        L = len(xyz14)

        # calculate random decoding order - linear w.r.t. time 
        decode_times, decode_order, idx2steps, aa_masks = get_aa_schedule(self.T, L, n_steps)
        idx2steps = torch.from_numpy(idx2steps)

        xyz = torch.full((L,27,3),np.nan).float()
        xyz[:,:14,:] = xyz14

        mask = torch.full((L, 27), False)
        mask[:,:14] = mask14 

        # replace missing with blackholes & conovert NaN to zeros to avoid any NaN problems during loss calculation
        init = INIT_CRDS.reshape(1, 27, 3).repeat(len(xyz), 1, 1)
        xyz = torch.where(mask[...,None], xyz, init).contiguous()
        xyz = torch.nan_to_num(xyz)


        # cos first, sin second 
        torsions_sincos, torsions_alt_sincos, tors_mask, tors_planar = get_torsions(xyz[None], seq[None], TOR_INDICES, TOR_CAN_FLIP, REF_ANGLES)

        # convert sin/cos to degrees
        torsions_angle     = torch.atan2(torsions_sincos[...,1], 
                                         torsions_sincos[...,0]).squeeze()

        torsions_angle_alt = torch.atan2(torsions_alt_sincos[...,1],
                                         torsions_alt_sincos[...,0]).squeeze()

        # Now sample random torsions from -pi < theta < pi
        r1 = -np.pi 
        r2 =  np.pi
        random_torsions = ((r1 - r2) * torch.rand(L,10,1) + r2).squeeze()
        
        # Grab the torsions which have the minimum difference to randomly sampled one 
        a = th_min_angle(torsions_angle,     random_torsions, radians=RAD)
        b = th_min_angle(torsions_angle_alt, random_torsions, radians=RAD)
        condition = a < b
        torsions_mindiff = torch.where(condition, torsions_angle, torsions_angle_alt)

        # Don't allow movement where diffusion_mask is True 
        if not diffusion_mask is None:
            random_torsions[diffusion_mask,...] = torsions_mindiff[diffusion_mask,...]


        # Now interpolate between the mindiff torsions and the random ones 
        c,d = torsions_mindiff.flatten(), random_torsions.flatten()
        e = idx2steps.repeat(10,1).T.flatten()
        angle_interpolations = th_interpolate_angles(c,d, self.T, n_diffuse=e).reshape(L,10,self.T)
        angle_interpolations = angle_interpolations.transpose(1,2)

        return angle_interpolations, aa_masks


class Diffuser():
    # wrapper for yielding diffused coordinates/frames/rotamers  


    def __init__(self, 
                 T, 
                 b_0=0.001,
                 b_T=0.1,
                 schedule_type='cosine',
                 schedule_kwargs={},
                 so3_type='slerp',
                 chi_type='interp',
                 var_scale=1, 
                 crd_scale=1/15,
                 aa_decode_steps=100):
        """
        
        Parameters:
            
        """
        #print('**********16')
        self.T = T  
        self.b_0 = b_0
        self.b_T = b_T
        self.crd_scale = crd_scale
        self.var_scale = var_scale 
        self.aa_decode_steps=aa_decode_steps
        # self.get_allatom = ComputeAllAtomCoords()

        # get backbone frame diffuser 
        if so3_type == 'slerp':
            self.so3_diffuser =  SLERP(self.T)
        else:
            raise NotImplementedError()

        # get backbone translation diffuser
        self.eucl_diffuser = EuclideanDiffuser(self.T, b_0, b_T, schedule_type=schedule_type, **schedule_kwargs)

        # get chi angle diffuser 
        self.torsion_diffuser = INTERP(self.T)

        print('Successful diffuser __init__')
    
    def diffuse_pose(self, xyz, seq, atom_mask, diffusion_mask=None, t_list=None):
        """
        Given full atom xyz, sequence and atom mask, diffuse the protein 
        translations, rotations, and chi angles

        Parameters:
            
            xyz (L,14/27,3) set of coordinates 

            seq (L,) integer sequence 

            atom_mask: mask describing presence/absence of an atom in pdb 

            diffusion_mask (torch.tensor, optional): Tensor of bools, True means NOT diffused at this residue, False means diffused

            t_list (list, optional): If present, only return the diffused coordinates at timesteps t within the list 


        """
        if diffusion_mask is None:
            diffusion_mask = torch.zeros(len(xyz.squeeze())).to(dtype=bool)

        get_allatom = ComputeAllAtomCoords().to(device=xyz.device)
        L = len(xyz)

        # bring to origin and scale 
        # check if any BB atoms are nan before centering 
        nan_mask = ~torch.isnan(xyz.squeeze()[:,:3]).any(dim=-1).any(dim=-1)

        xyz = xyz - xyz[nan_mask][:,1,:].mean(dim=0)
        xyz = xyz * self.crd_scale

        
        # 1 get translations 
        tick = time.time()
        diffused_T, deltas = self.eucl_diffuser.diffuse_translations(xyz[:,:3,:].clone(), diffusion_mask=diffusion_mask)
        #print('Time to diffuse coordinates: ',time.time()-tick)
        diffused_T /= self.crd_scale
        deltas     /= self.crd_scale


        # 2 get  frames
        tick = time.time()
        diffused_frame_crds, diffused_frames = self.so3_diffuser.diffuse_frames(xyz[:,:3,:].clone(), diffusion_mask=diffusion_mask.numpy())
        diffused_frame_crds /= self.crd_scale 
        #print('Time to diffuse frames: ',time.time()-tick)


        # 3 diffuse chi angles/planar angles and sequence information 
        tick = time.time()
        diffused_torsions,aa_masks = self.torsion_diffuser.diffuse_torsions(xyz[:,:14].clone(), 
                                                                            seq, 
                                                                            atom_mask[:,:14].clone(),
                                                                            diffusion_mask=diffusion_mask, 
                                                                            n_steps=self.aa_decode_steps)
        #print('Time to diffuse torsions: ',time.time()-tick)


        ##### Now combine all the diffused quantities to make full atom diffused poses 
        tick = time.time()
        cum_delta = deltas.cumsum(dim=1)
        # The coordinates of the translated AND rotated frames
        diffused_BB = (torch.from_numpy(diffused_frame_crds) + cum_delta[:,:,None,:]).transpose(0,1)


        diffused_torsions_trig = torch.stack([torch.cos(diffused_torsions), 
                                              torch.sin(diffused_torsions)], dim=-1)

        # Full atom diffusions at all timepoints 
        fa_stack = []
        if t_list is None:
            for t,alphas_t in enumerate(diffused_torsions_trig.transpose(0,1)):
                xyz_bb_t = diffused_BB[t,:,:3]

                _,fullatom_t = get_allatom(seq[None], xyz_bb_t[None], alphas_t[None])
                fa_stack.append(fullatom_t)

        else:
            for t in t_list:
                xyz_bb_t  = diffused_BB[t,:,:3]
                alphas_t = diffused_torsions_trig.transpose(0,1)[t]

                _,fullatom_t = get_allatom(seq[None], xyz_bb_t[None], alphas_t[None])
                fa_stack.append(fullatom_t.squeeze())

        fa_stack = torch.stack(fa_stack, dim=0)


        return diffused_T, deltas, diffused_frame_crds, diffused_frames, diffused_torsions, fa_stack, aa_masks
