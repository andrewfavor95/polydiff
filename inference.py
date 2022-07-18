#!/home/dimaio/.conda/envs/SE3nv/bin/python
import torch 
import numpy as np
from scipy.spatial.transform import Rotation as scipy_R
from scipy.spatial.transform import Slerp

# diffusion imports 
from diffusion import get_beta_schedule, Diffuser
from util import rigid_from_3_points, writepdb_multi, writepdb
import util
import util
from util_module import ComputeAllAtomCoords 

from icecream import ic 


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
        self.b_0 = b_0
        self.b_T = b_T
        self.schedule_type = schedule_type
        self.so3_type = so3_type
        self.chi_type = chi_type
        self.crd_scale = crd_scale
        self.var_scale = var_scale
        self.aa_decode_steps=aa_decode_steps
        self.schedule, self.alphabar_schedule = get_beta_schedule(self.T, self.b_0, self.b_T, self.schedule_type, schedule_params={}, inference=True)

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
        px0 = px0.numpy()
        xT = xT.numpy()
        diffusion_mask = diffusion_mask.numpy()

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

    def get_next_xyz(self, xt, px0, xT, t, diffusion_mask, seq, fix_motif=True):
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
        """
        #align to motif:
        px0 = self.align_to_xt_motif(px0, xt, diffusion_mask)

        # Now done with diffusion mask. if fix motif is False, just set diffusion mask to be all True, and all coordinates can diffuse
        if not fix_motif:
            diffusion_mask[:] = True
    
        _, ca_deltas = self.get_next_ca(xt, px0, xT, t, diffusion_mask)

        frames_next = self.get_next_frames(xt, px0, t, diffusion_mask) # the coordinates of the new N/CA/C slerped once 
        
        frames_next = torch.from_numpy(frames_next) + ca_deltas[:,None,:]  # translate 
        
        # TODO: Build full atom rep with predicted torsions instead of just BB

        return frames_next 


def main():
    import arguments 
    from RoseTTAFoldModel import RoseTTAFoldModule

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # make model 
    #args, model_param, loader_param, loss_param, diffusion_params = arguments.get_args()
    #ckpt_path = '/home/davidcj/projects/BFF/rf_diffusion/checkpoints/small_se3_pred_x0/BFF_9.pt'
    #model = RoseTTAFoldModule(**model_param).to(device)
    
    print('Loading checkpoint from disk...')
    #ckpt  = torch.load(ckpt_path, map_location=device)
    #state = ckpt['model_state_dict']
    print('Putting checkpoint into model...')
    #model.load_state_dict(state, strict=True)
    print('Done making model')

    
    # parse pdb and prep inputs for diffusion
    parsed = parse_pdb('/mnt/home/davidcj/projects/expert-potato/expert-potato/1qys.pdb')
    xyz    = parsed['xyz']
    #xyz    = torch.from_numpy( (xyz - xyz[:,:1,:].mean(axis=0)[None,...]) )
    xyz = torch.from_numpy(xyz)



    seq = torch.from_numpy( parsed['seq'] )
    atom_mask = torch.from_numpy( parsed['mask'] )

    writepdb('./test_native.pdb', xyz[:,:3,:], seq, bfacts=torch.ones_like(seq))

    
    # Make diffusion mask just first 20 residues 
    diffusion_mask = torch.zeros(len(seq.squeeze())).to(dtype=bool)
    diffusion_mask[:20] = True
    
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
                      'aa_decode_steps':100}

    diffuser = Diffuser(**denoise_kwargs)
    denoiser = Denoise(**denoise_kwargs)
    
    # get diffused xT  
    out = diffuser.diffuse_pose(torch.clone(xyz), seq, atom_mask, diffusion_mask)
    diffused_T, deltas, diffused_frame_crds, diffused_frames, diffused_torsions, fa_stack, aa_masks = out 
     
    xT = fa_stack[-1].squeeze()[:,:3,:]
    xt = torch.clone(xT)

    denoised_xyz_stack = []
    for t in range(T-1,0,-1):
        
        xt = denoiser.get_next_xyz(xt=xt, px0=torch.clone( xyz[:,:3,:] ), xT=xT, t=t, diffusion_mask=diffusion_mask, seq=seq) 
        denoised_xyz_stack.append(xt)

    
    denoised_xyz_stack = torch.stack(denoised_xyz_stack)
    
    out='./test_denoising.pdb'
    writepdb_multi(out, denoised_xyz_stack, torch.ones_like(seq), seq, use_hydrogens=False, backbone_only=True)
    
    #out = './test_single_denoising.pdb'
    #writepdb(out, xt, seq)


    #out = './rewrite_1qys.pdb'
    #writepdb(out, xyz, seq)


if __name__ == '__main__':
    main()
