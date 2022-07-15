from diffusion import get_beta_schedule
from util import rigid_from_3_points
       
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
        L,A,_ = xT.shape # A is number of atoms
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
        px0_motif = px0_motif-px0_motif_mean
        xT_motif = xT_motif-xT_motif_mean

        # Computation of the covariance matrix
        C = px0_motif.T @ xT_motif
        
        # Compute optimal rotation matrix using SVD
        V, S, W = np.linalg.svd(C)

        # get sign to ensure right-handedness
        d = np.ones([3,3])
        d[:,-1] = np.sign(np.linalg.det(V)*np.linalg.det(W))

        # Rotation matrix U
        U = (d*V) @ W

        #2 rotate whole px0 by rotation matrix
        px0[~atom_mask] = 0 #convert nans to 0
        px0 = px0.reshape(-1,3) - px0_motif_mean
        px0_ = px0 @ U


#         #3 centre at origin and reshape
#         px0_ = px0_ - px0_[:,:3].reshape(-1,3).mean(0)
#         px0_ = px0_.reshape(L,A,3)
#         px0_[~atom_mask] = float('nan')
        #3 put in same global position as xT
        px0_ = px0_ + xT_motif_mean
        px0_ = px0_.reshape(L,A,3)
        px0_[~atom_mask] = float('nan')
        return torch.Tensor(px0_)
    def get_mu_xt_x0(self, xt, px0, t):
        """
        Given xt, predicted x0 and the timestep t, give mu of x(t-1)
        Assumes t is 0 indexed
        """
        #sigma is predefined from beta. Often referred to as beta tilde t
        sigma = ((1-self.alphabar_schedule[t-1])/(1-self.alphabar_schedule[t]))*self.schedule[t]

        xt_ca = xt[:,1,:]
        px0_ca = px0[:,1,:]

        a = ((torch.sqrt(self.alphabar_schedule[t-1])*self.schedule[t])/(1-self.alphabar_schedule[t]))*px0_ca
        b = ((torch.sqrt(1-self.schedule[t])*(1-self.alphabar_schedule[t-1]))/(1-self.alphabar_schedule[t]))*xt_ca

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

        if torch.sum(diffusion_mask) > 0:
            px0 = self.align_to_xt_motif(px0, xT, diffusion_mask)
        else:
            #align to xt and centre at origin
            px0 = selfalign_to_xt_motif(px0, xT)
        px0 = px0 * self.crd_scale
        xt = xt * self.crd_scale
        # get mu(xt, x0)
        mu, sigma = self.get_mu_xt_x0(xt, px0, t)
        sampled_crds = torch.normal(mu, sigma)
        delta = sampled_crds - xt[:,1,:] #check sign of this is correct

        if not diffusion_mask is None:
            delta[diffusion_mask,...] = 0

        out_crds = xt - delta[:, None, :]


        return out_crds/self.crd_scale

    def get_next_frames(self, xt, px0, t, diffusion_mask):
        """
        SLERP xt frames towards px0, by factor 1/t
        Rather than generating random rotations (as occurs during forward process), calculate rotation between xt and px0
        """

        N_0  = px0[None,:,0,:]
        Ca_0 = px0[None,:,1,:]
        C_0  = px0[None,:,2,:]

        N_t = xt[None, :, 0, :]
        Ca_t = xt[None, :, 1, :]
        C_t = xt[None, :, 2, :]

        R_0, Ca_0 = rigid_from_3_points(N_0, Ca_0, C_0)
        R_t, Ca_t = rigid_from_3_points(N_t, Ca_t, C_t)

        R_0 = scipy_R.from_matrix(R_0.squeeze())
        R_t = scipy_R.from_matrix(R_t.squeeze())
        
        all_interps = []
        for i in range(len(xt)):
            r_0 = R_0[i].as_matrix()
            r_t = R_t[i].as_matrix()
            
            if not diffusion_mask[i]:
                key_rots = scipy_R.from_matrix(np.stack([r_t, r_0], axis=0))
            else:
                key_rots = scipy_R.from_matrix(np.stack([r_t, r_t], axis=0))
            key_times = [0,1]
    
            interpolator = Slerp(key_times, key_rots)
            alpha = [1/t]
            interp_rot  = interpolator(alpha)
            all_interps.append(interp_rot.as_matrix())
        all_interps = np.stack(all_interps, axis=0)
        print(all_interps.shape)
        # Now apply all the interpolated rotation matrices to the original rotation matrices and get the frames at each timestep
        slerped_frames = np.einsum('lrij,laj->lrai', all_interps, R_t.as_matrix())

        # apply the slerped frames to the coordinates

        slerped_crds   = np.einsum('lrij,laj->lrai', slerped_frames, xt[:,:3,:] - Ca_0.squeeze()[:,None,...].numpy()) + Ca_0.squeeze()[:,None,None,...].numpy()
        print(slerped_crds.shape)
        # (T,L,3,3) set of backbone coordinates and frames 
        return slerped_crds

    def get_next_xyz(self, xt, px0, xT, t, diffusion_mask, fix_motif=True):
        """
        Wrapper function to take px0, xt and t, and to produce xt-1
        First, aligns px0 to xt
        Then gets coordinates, frames and torsion angles
        """
    
        #align to motif:
        px0 = self.align_to_xt_motif(px0, xt, diffusion_mask)
        # Now done with diffusion mask. if fix motif is False, just set diffusion mask to be all True, and all coordinates can diffuse
        if fix_motif:
            diffusion_mask[:] = True
    
        ca_next = self.get_next_ca(xt, px0, xT, t, diffusion_mask)
        frames_next = self.get_next_frames(xt*self.crd_scale, px0*self.crd_scale, t, diffusion_mask)
        frames_next /= self.crd_scale
        
        #TODO torsion angle diffusion and put all together

        return xnext          
        
        
