#!/home/dimaio/.conda/envs/SE3nv/bin/python
import torch 
import numpy as np
from scipy.spatial.transform import Rotation as scipy_R
from scipy.spatial.transform import Slerp

# diffusion imports 
from diffusion import get_beta_schedule, Diffuser
from util import rigid_from_3_points, writepdb_multi, writepdb, get_torsions
from diff_util import get_aa_schedule, th_interpolate_angles, th_min_angle
from util import torsion_indices as TOR_INDICES
from util import torsion_can_flip as TOR_CAN_FLIP
from util import reference_angles as REF_ANGLES
from util_module import ComputeAllAtomCoords

from inference_utils import preprocess
import util
from util_module import ComputeAllAtomCoords 

from icecream import ic 
import time

def rmsd(V,W, eps=0):
        # First sum down atoms, then sum down xyz
        N = V.shape[-2]
        return np.sqrt(np.sum((V-W)*(V-W), axis=(-2,-1)) / N + eps)

def parse_pdb(filename, **kwargs):
    '''extract xyz coords for all heavy atoms'''
    lines = open(filename,'r').readlines()
    return parse_pdb_lines(lines, **kwargs)

def parse_pdb_lines(lines, parse_hetatom=False, ignore_het_h=True):
    # indices of residues observed in the structure
    res = [(l[22:26],l[17:20]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]
    seq = [util.aa2num[r[1]] if r[1] in util.aa2num.keys() else 20 for r in res]
    pdb_idx = [( l[21:22].strip(), int(l[22:26].strip()) ) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]  # chain letter, res num

    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(res), 14, 3), np.nan, dtype=np.float32)
    for l in lines:
        if l[:4] != "ATOM":
            continue
        chain, resNo, atom, aa = l[21:22], int(l[22:26]), ' '+l[12:16].strip().ljust(3), l[17:20]
        idx = pdb_idx.index((chain,resNo))
        for i_atm, tgtatm in enumerate(util.aa2long[util.aa2num[aa]]):
            if tgtatm is not None and tgtatm.strip() == atom.strip(): # ignore whitespace
                xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
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

    out = {'xyz':xyz, # cartesian coordinates, [Lx14]
            'mask':mask, # mask showing which atoms are present in the PDB file, [Lx14]
            'idx':np.array([i[1] for i in pdb_idx]), # residue numbers in the PDB file, [L]
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
        self.T = T
        self.L = L 
        self.b_0 = b_0
        self.b_T = b_T
        self.schedule_type = schedule_type
        self.so3_type = so3_type
        self.chi_type = chi_type
        self.crd_scale = crd_scale
        self.var_scale = var_scale
        self.aa_decode_steps=aa_decode_steps
        self.schedule, self.alphabar_schedule = get_beta_schedule(self.T, self.b_0, self.b_T, self.schedule_type, schedule_params={}, inference=True)

        # amino acid decoding schedule 
        out = get_aa_schedule(T,L,nsteps=aa_decode_steps)
        self.aa_decode_times, self.decode_order, self.idx2steps, self.aa_mask_stack = out




    def get_next_torsions(self, xt, px0, seq_t, seq_logit_px0, t, diffusion_mask=None):
        """
        Samples the next chi angles and amino acid identities for the pose 

        xt (torch.tensor, required): L,14,3 or L,27,3 set of fullatom crds 

        px0 (torch.tensor, required): Predicted set of full atom crds 

        seq_t (torch.tensor, required): Current sequence in the pose 

        pseq_0 (torch.tensor, required): predicted sequence logits for the x0 prediction

        t (int, required): timestep 
        """
        L = len(xt)
        # argmaxed sequence logits for x0 prediction
        seq_px0 = torch.argmax(seq_logit_px0, dim=-1) 

        ### initialize full atom reps to calculate torsions 
        RAD = True # use radians for angle stuff 

        # build 27-atom representations
        if not xt.shape[1] == 27:
            xt_full = torch.full((L,27,3),np.nan).float()
            xt_full[:,:14,:] = xt[:,:14]


        if not px0.shape[1] == 27:
            px0_full = torch.full((L,27,3),np.nan).float()
            px0_full[:,:14,:] = px0[:,:14]



        # there is no situation where we should have any NaN BB crds here  
        mask = torch.full((L, 27), False)
        mask[:,:14] = True 

        ### Calcualte torsions and interpolate between them 

        # Xt torsions
        torsions_sincos_xt, torsions_alt_sincos_xt, tors_mask_xt, tors_planar_xt = get_torsions(xt_full[None], 
                                                                                                seq_t[None], 
                                                                                                TOR_INDICES, 
                                                                                                TOR_CAN_FLIP, 
                                                                                                REF_ANGLES)
        torsions_angle_xt     = torch.atan2(torsions_sincos_xt[...,1],
                                            torsions_sincos_xt[...,0]).squeeze()
        torsions_angle_xt_alt = torch.atan2(torsions_alt_sincos_xt[...,1],
                                            torsions_alt_sincos_xt[...,0]).squeeze()



        # pX0 torsions - uses same seq_t to calculate 
        torsions_sincos_px0, torsions_alt_sincos_px0, tors_mask_px0, tors_planar_px0 = get_torsions(px0_full[None], 
                                                                                                    seq_t[None], 
                                                                                                    TOR_INDICES, 
                                                                                                    TOR_CAN_FLIP, 
                                                                                                    REF_ANGLES)
        torsions_angle_px0     = torch.atan2(torsions_sincos_px0[...,1],
                                             torsions_sincos_px0[...,0]).squeeze()

        # Take the first one as ground truth --> don't need alt for px0 
        # torsions_angle_alt_px0 = torch.atan2(torsions_alt_sincos_px0[...,1],
        #                                      torsions_alt_sincos_px0[...,0]).squeeze()


        # calculate the minimum angle between both options in xt and the angles in px0
        a = th_min_angle(torsions_angle_xt,     torsions_angle_px0, radians=RAD)
        b = th_min_angle(torsions_angle_xt_alt, torsions_angle_px0, radians=RAD)
        condition = torch.abs(a) < torch.abs(b)
        #condition = a < b
        torsions_xt_mindiff = torch.where(condition, torsions_angle_xt, torsions_angle_xt_alt)
        mindiff = torch.where(condition, a, b)

        # these are in radians now 
        start,end = torsions_xt_mindiff.flatten(), torsions_angle_px0.flatten()
        
        # everybody who is being diffused is going 1/t of the way to px0
        # so interpolate over t steps and use the first interpolation point 
        diff_steps = torch.full_like(start, t)
        angle_interpolations = th_interpolate_angles(start, end, t, n_diffuse=diff_steps, mindiff=mindiff)
        angle_interpolations = angle_interpolations.reshape(L,10,t)

        # don't allow movement where diffusion mask is True 
        if diffusion_mask is not None:
            angle_interpolations[diffusion_mask,:,:] = end.reshape(L,10,-1)[diffusion_mask,:,:1].repeat(1,1,t)
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

        next_torsions_trig = torch.stack([torch.cos(next_torsions),
                                          torch.sin(next_torsions)], dim=-1)

        ## Reveal some amino acids in the sequence according to schedule and predictions
        next_seq = torch.clone(seq_t)
        if t <= len(self.decode_order): 
            decode_positions = self.decode_order[t-1]
            replacement      = seq_px0[decode_positions]
            next_seq[decode_positions] = replacement
        return next_torsions_trig, next_seq





    def align_to_xt_motif(self, px0, xT, diffusion_mask, eps=1e-6):
        """
        Need to align px0 to motif in xT. This is to permit the swapping of residue positions in the px0 motif for the true coordinates.
        First, get rotation matrix from px0 to xT for the motif residues.
        Second, rotate px0 (whole structure) by that rotation matrix
        Third, centre at origin
        """
        assert xT.shape[1] == px0.shape[1], f'xT has shape {xT.shape} and px0 has shape {px0.shape}'

        L,n_atom,_ = xT.shape # A is number of atoms
        atom_mask = ~torch.isnan(px0)
        #convert to numpy arrays
        px0 = px0.cpu().detach().numpy()
        xT = xT.cpu().detach().numpy()
        diffusion_mask = diffusion_mask.cpu().detach().numpy()

        #1 centre motifs at origin and get rotation matrix
        px0_motif = px0[diffusion_mask,:3].reshape(-1,3)
        xT_motif = xT[diffusion_mask,:3].reshape(-1,3)
        px0_motif_mean = np.copy(px0_motif.mean(0)) #need later
        xT_motif_mean = np.copy(xT_motif.mean(0))

        # center at origin
        px0_motif = px0_motif-px0_motif_mean
        xT_motif = xT_motif-xT_motif_mean


        # Computation of the covariance matrix
        #C = px0_motif.T @ xT_motif
        
        # Compute optimal rotation matrix using SVD
        #V, S, W = np.linalg.svd(C)

        # get sign to ensure right-handedness
        #d = np.ones([3,3])
        #d[:,-1] = np.sign(np.linalg.det(V)*np.linalg.det(W))

        # Rotation matrix U
        #U = (d*V) @ W
        # computation of the covariance matrix

        A = px0_motif
        B = xT_motif 

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

        #2 rotate whole px0 by rotation matrix
        atom_mask = atom_mask.cpu()
        px0[~atom_mask] = 0 #convert nans to 0
        px0 = px0.reshape(-1,3) - px0_motif_mean
        px0_ = px0 @ R

        #3 put in same global position as xT
        px0_ = px0_ + xT_motif_mean
        px0_ = px0_.reshape([L,n_atom,3])
        px0_[~atom_mask] = float('nan')
        return torch.Tensor(px0_)
   
    def get_mu_xt_x0(self, xt, px0, t, eps=1e-6):
        """
        Given xt, predicted x0 and the timestep t, give mu of x(t-1)
        Assumes t is 0 indexed
        """
        #sigma is predefined from beta. Often referred to as beta tilde t
        sigma = ((1-self.alphabar_schedule[t-1])/(1-self.alphabar_schedule[t]))*self.schedule[t]

        xt_ca = xt[:,1,:]
        px0_ca = px0[:,1,:]

        a = ((torch.sqrt(self.alphabar_schedule[t-1] + eps)*self.schedule[t])/(1-self.alphabar_schedule[t]))*px0_ca
        b = ((torch.sqrt(1-self.schedule[t] + eps)*(1-self.alphabar_schedule[t-1]))/(1-self.alphabar_schedule[t]))*xt_ca

        mu = a + b

        return mu, sigma

    def get_next_ca(self, xt, px0, xT, t, diffusion_mask):
        """
        Given full atom x0 prediction (xyz coordinates), diffuse to x(t-1)
        
        Parameters:
            
            xt (L, 14/27, 3) set of coordinates
            
            px0 (L, 14/27, 3) set of coordinates
    
            xT (L, 14/27, 3) set of coordinates given to the model at xT. Contains true motif coordinates from input structure

            t: time step. Note this is zero-index current time step, so are generating t-1    

            logits_aa (L x 20 ) amino acid probabilities at each position

            seq_schedule (L): Tensor of bools, True is unmasked, False is masked. For this specific t

            diffusion_mask (torch.tensor, required): Tensor of bools, True means NOT diffused at this residue, False means diffused 

        """
        get_allatom = ComputeAllAtomCoords().to(device=xt.device)
        L = len(xt)

        # bring to origin after global alignment (when don't have a motif) or replace input motif and bring to origin, and then scale 
        px0 = px0 * self.crd_scale
        xt = xt * self.crd_scale

        # get mu(xt, x0)
        mu, sigma = self.get_mu_xt_x0(xt, px0, t)

        sampled_crds = torch.normal(mu, sigma)
        delta = sampled_crds - xt[:,1,:] #check sign of this is correct

        if not diffusion_mask is None:
            delta[diffusion_mask,...] = 0

        out_crds = xt + delta[:, None, :]


        return out_crds/self.crd_scale, delta/self.crd_scale

    def get_next_frames(self, xt, px0, t, diffusion_mask):
        """
        SLERP xt frames towards px0, by factor 1/t
        Rather than generating random rotations (as occurs during forward process), calculate rotation between xt and px0
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

        
        all_interps = []
        for i in range(len(xt)):
            r_0 = R_0[i].as_matrix()
            r_t = R_t[i].as_matrix()
            
            # interpolate FRAMES between one and next 
            if not diffusion_mask[i]:
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

            all_interps.append(interp_rot)


        all_interps = np.stack(all_interps, axis=0)
        # Now apply all the interpolated rotation matrices to the original rotation matrices and get the frames at each timestep
        
        slerped_frames = np.einsum('lrij,ljk->lrik', all_interps, R_t.as_matrix())

        # apply the interpolated rotation matrices to the coordinates
        slerped_crds   = np.einsum('lrij,laj->lrai', all_interps, xt[:,:3,:] - Ca_t.squeeze()[:,None,...].numpy()) + Ca_t.squeeze()[:,None,None,...].numpy()

        # (L,3,3) set of backbone coordinates with slight rotation 
        return slerped_crds.squeeze(1)


    def get_next_pose(self, xt, px0, xT, t, diffusion_mask, seq_t, pseq0, fix_motif=True):
        """
        Wrapper function to take px0, xt and t, and to produce xt-1
        First, aligns px0 to xt
        Then gets coordinates, frames and torsion angles

        Parameters:
            
            xt (torch.tensor, required): Current coordinates at timestep t 

            px0 (torch.tensor, required): Prediction of x0 

            xT (torch.tensor, required): Initial noised set of coordinates 

            t (int, required): timestep t

            diffusion_mask (torch.tensor, required): Mask

            seq_t (torch.tensor, required): Sequence at the current timestep 
        """
        get_allatom = ComputeAllAtomCoords().to(device=xt.device)
        L,n_atom = xt.shape[:2]
        assert (xt.shape[1]  == 14) or (xt.shape[1]  == 27)
        assert (px0.shape[1] == 14) or (px0.shape[1] == 27)# need full atom rep for torsion calculations   

        #align to motif
        #px0 = self.align_to_xt_motif(px0, xt, diffusion_mask)
        px0=px0.to(xt.device)
        # Now done with diffusion mask. if fix motif is False, just set diffusion mask to be all True, and all coordinates can diffuse
        if not fix_motif:
            diffusion_mask[:] = False
        
        # get the next set of CA coordinates 
        _, ca_deltas = self.get_next_ca(xt, px0, xT, t, diffusion_mask)
        
        # get the next set of backbone frames (coordinates)
        frames_next = self.get_next_frames(xt, px0, t, diffusion_mask)
        
        # add the delta to the new frames 
        frames_next = torch.from_numpy(frames_next) + ca_deltas[:,None,:]  # translate 
        
        # get the next set of amino acid identities and chi angles 
        torsions_next, seq_next = self.get_next_torsions(xt, px0, seq_t, pseq0, t, diffusion_mask)       

        # build full atom representation with the new torsions but the current seq 
        _, fullatom_next =  get_allatom(seq_t[None], frames_next[None], torsions_next[None])

        return fullatom_next.squeeze()[:,:14,:], seq_next, torsions_next


def main():
    import arguments 
    from RoseTTAFoldModel import RoseTTAFoldModule

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # make model 
    args, model_param, loader_param, loss_param, diffusion_params = arguments.get_args()
    ckpt_path = '/mnt/home/davidcj/projects/BFF/rf_diffusion/tmp_models/BFF_12.pt'
    model = RoseTTAFoldModule(**model_param).to(device)
    model = model.eval()    
    print('Loading checkpoint from disk...')
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt['model_state_dict']
    print('Putting checkpoint into model...')
    model.load_state_dict(state, strict=True)
    print('Done making model')

    
    # parse pdb and prep inputs for diffusion
    #parsed = parse_pdb('/mnt/home/davidcj/projects/expert-potato/expert-potato/1qys.pdb')
    # parsed = parse_pdb('/mnt/home/jwatson3/projects/BFF_diff/BFF/rf_diffusion/test_sh.pdb')
    parsed = parse_pdb('/home/davidcj/projects/BFF/rf_diffusion/tim_barrel.pdb')
    xyz    = parsed['xyz']
    xyz    = torch.from_numpy( (xyz - xyz[:,:1,:].mean(axis=0)[None,...]) )
    # xyz = torch.from_numpy(xyz)
    L = len(xyz)
    
    get_allatom = ComputeAllAtomCoords().to(device)
    conversion = 'ARNDCQEGHILKMFPSTWYVX-'

    seq = torch.from_numpy( parsed['seq'] )
    atom_mask = torch.from_numpy( parsed['mask'] )

    # writepdb('./test_native.pdb', xyz[:,:3,:], seq, bfacts=torch.ones_like(seq))

    
    # Make diffusion mask just first 20 residues 
    diffusion_mask = torch.zeros(len(seq.squeeze())).to(dtype=bool)
    diffusion_mask[45:90] = True
    # Make the denoiser 
    T = 200
    b_0 = 0.001
    b_T = 0.1
    
    denoise_kwargs = {'T':T,
                      'b_0':b_0,
                      'b_T':b_T,
                      'schedule_type':'cosine',
                      'schedule_kwargs':{},
                      'so3_type':'slerp',
                      'chi_type':'interp',
                      'var_scale':1,
                      'crd_scale':1/15,
                      'aa_decode_steps':20}

    diffuser = Diffuser(**denoise_kwargs)
    denoise_kwargs.update({'L':L})
    denoiser = Denoise(**denoise_kwargs)
    
    # get diffused xT  
    out = diffuser.diffuse_pose(torch.clone(xyz), seq, atom_mask, diffusion_mask)
    diffused_T, deltas, diffused_frame_crds, diffused_frames, diffused_torsions, fa_stack, aa_masks = out 
     
    xT = fa_stack[-1].squeeze()[:,:14,:]
    xt = torch.clone(xT)
    
    
    # set the predicted sequence as the true sequence 
    pseq_0 = torch.nn.functional.one_hot(seq, num_classes=22)
    # set the starting sequence as all masked besides the motif 
    seq_t = torch.clone(seq)
    seq_t[~diffusion_mask] = 21

    denoised_xyz_stack = []
    px0_xyz_stack = []
    seq_stack = []
    chi1_stack = []
    #n_cycle = 2
    n_cycle = np.ones(T)
    # n_cycle[:50] = 2
    # n_cycle[:25] = 3
    #msa_prev = None
    #pair_prev = None
    #state_prev = None
    for t in range(T-1,0,-1):
        if t % 10 == 0:
            print(f'{t} timesteps to go')
        msa_masked, msa_full, seq, xt_in, idx_pdb, t1d, t2d, xyz_t, alpha_t = preprocess(seq_t, xt, t, device)
        msa_prev = None
        pair_prev = None
        state_prev = None
        with torch.no_grad():
            for i in range(int(n_cycle[t])): 
                msa_prev, pair_prev, px0, state_prev, alpha, logits = model(msa_masked,
                                    msa_full,
                                    seq, xt_in,
                                    idx_pdb,
                                    t1d=t1d, t2d=t2d,
                                    xyz_t=xyz_t, alpha_t=alpha_t,
                                    msa_prev = msa_prev,
                                    pair_prev = pair_prev,
                                    state_prev = state_prev,
                                    return_infer=True)
        _,px0 = get_allatom(seq, px0, alpha)
        px0 = px0.squeeze()[:,:14]
        px0_xyz_stack.append(px0) 
        pseq_0 = torch.nn.functional.one_hot(torch.argmax(logits.squeeze(), dim=-1), num_classes=22).to(xt.device)
        print(f'current sequence: {"".join([conversion[int(i)] for i in torch.argmax(pseq_0, dim=-1).tolist()])}')
        print(f'number of recycles = {int(n_cycle[t])}')
        if t != 0:
            xt, seq_t, tors_t = denoiser.get_next_pose( xt              = xt, 
                                            px0            =  px0, 
                                            xT             = xT, 
                                            t              = t, 
                                            diffusion_mask = diffusion_mask, 
                                            seq_t          = seq_t,
                                            pseq0          = pseq_0) 
        
        denoised_xyz_stack.append(xt)
        seq_stack.append(seq_t)
        chi1_stack.append(tors_t[:,:])
    
    denoised_xyz_stack = torch.stack(denoised_xyz_stack)
    denoised_xyz_stack = torch.flip(denoised_xyz_stack, [0,])
    px0_xyz_stack = torch.stack(px0_xyz_stack)
    px0_xyz_stack = torch.flip(px0_xyz_stack, [0,])
    chi1_stack = torch.stack(chi1_stack)
    #with open('chi_stack.pt', 'wb') as fp:
    #    torch.save(chi1_stack, fp)
    
    timestr = time.strftime("%H%M%S")
    out=f'./test_denoising_{timestr}.pdb'
    writepdb_multi(out, denoised_xyz_stack, torch.ones_like(seq.squeeze()), seq.squeeze(), use_hydrogens=False, backbone_only=False)
    
    out=f'./test_px0_{timestr}.pdb'
    writepdb_multi(out, px0_xyz_stack, torch.ones_like(seq.squeeze()), seq.squeeze(), use_hydrogens=False, backbone_only=False)

    out = f'./test_denoise_last_step_{timestr}.pdb'
    writepdb(out, xt, seq_t)


    #out = './rewrite_1qys.pdb'
    #writepdb(out, xyz, seq)


if __name__ == '__main__':
    main()
