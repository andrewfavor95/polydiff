import torch

from util import generate_Cbeta

class Potential:
    '''
        Interface class that defines the functions a potential must implement
    '''

    def compute(self, seq, xyz):
        '''
            Given the current sequence and structure of the model prediction, return the current
            potential as a PyTorch tensor with a single entry

            Args:
                seq (torch.tensor, size: [L,?]:    The current sequence of the sample.
                                                     TODO: determine whether this is one hot or an 
                                                     integer representation
                xyz (torch.tensor, size: [L,27,3]: The current coordinates of the sample
            
            Returns:
                potential (torch.tensor, size: [1]): A potential whose value will be MAXIMIZED
                                                     by taking a step along it's gradient
        '''
        raise NotImplementedError('Potential compute function was not overwritten')

class monomer_ROG(Potential):
    '''
        Radius of Gyration potential for encouraging monomer compactness

        Written by DJ and refactored into a class by NRB
    '''

    def __init__(self, weight=1, min_dist=15):

        self.weight   = weight
        self.min_dist = min_dist

    def compute(self, seq, xyz):
        Ca = xyz[:,1] # [L,3]

        centroid = torch.mean(Ca, dim=0, keepdim=True) # [1,3]

        dgram = torch.cdist(Ca[None,...].contiguous(), centroid[None,...].contiguous(), p=2) # [1,L,1,3]

        dgram = torch.maximum(self.min_dist * torch.ones_like(dgram.squeeze(0)), dgram.squeeze(0)) # [L,1,3]

        rad_of_gyration = torch.sqrt( torch.sum(torch.square(dgram)) / Ca.shape[0] ) # [1]

        return -1 * self.weight * rad_of_gyration

class binder_ROG(Potential):
    '''
        Radius of Gyration potential for encouraging binder compactness

        Author: NRB
    '''

    def __init__(self, binderlen, weight=1, min_dist=15):

        self.binderlen = binderlen
        self.min_dist  = min_dist
        self.weight    = weight

    def compute(self, seq, xyz):
        
        # Only look at binder residues
        Ca = xyz[:self.binderlen,1] # [Lb,3]

        centroid = torch.mean(Ca, dim=0, keepdim=True) # [1,3]

        # cdist needs a batch dimension - NRB
        dgram = torch.cdist(Ca[None,...].contiguous(), centroid[None,...].contiguous(), p=2) # [1,Lb,1,3]

        dgram = torch.maximum(self.min_dist * torch.ones_like(dgram.squeeze(0)), dgram.squeeze(0)) # [Lb,1,3]

        rad_of_gyration = torch.sqrt( torch.sum(torch.square(dgram)) / Ca.shape[0] ) # [1]

        return -1 * self.weight * rad_of_gyration
        
class binder_distance_ReLU(Potential):
    '''
        Given the current coordinates of the diffusion trajectory, calculate a potential that is the distance between each residue
        and the closest target residue.

        This potential is meant to encourage the binder to interact with a certain subset of residues on the target that 
        define the binding site.

        Author: NRB
    '''

    def __init__(self, binderlen, hotspot_res, weight=1, min_dist=15, use_Cb=False):

        self.binderlen   = binderlen
        self.hotspot_res = [res + binderlen for res in hotspot_res]
        self.weight      = weight
        self.min_dist    = min_dist
        self.use_Cb      = use_Cb

    def compute(self, seq, xyz):
        binder = xyz[:self.binderlen,:,:] # (Lb,27,3)
        target = xyz[self.hotspot_res,:,:] # (N,27,3)

        if self.use_Cb:
            N  = binder[:,0]
            Ca = binder[:,1]
            C  = binder[:,2]

            Cb = generate_Cbeta(N,Ca,C) # (Lb,3)

            N_t  = target[:,0]
            Ca_t = target[:,1]
            C_t  = target[:,2]

            Cb_t = generate_Cbeta(N_t,Ca_t,C_t) # (N,3)

            dgram = torch.cdist(Cb[None,...], Cb_t[None,...], p=2) # (1,Lb,N)

        else:
            # Use Ca dist for potential

            Ca = binder[:,1] # (Lb,3)

            Ca_t = target[:,1] # (N,3)

            dgram = torch.cdist(Ca[None,...], Ca_t[None,...], p=2) # (1,Lb,N)

        closest_dist = torch.min(dgram.squeeze(0), dim=1)[0] # (Lb)

        # Cap the distance at a minimum value
        min_distance = self.min_dist * torch.ones_like(closest_dist) # (Lb)
        potential    = torch.maximum(min_distance, closest_dist) # (Lb)

        # torch.Tensor.backward() requires the potential to be a single value
        potential    = torch.sum(potential, dim=-1)
        
        return -1 * self.weight * potential

class binder_any_ReLU(Potential):
    '''
        Given the current coordinates of the diffusion trajectory, calculate a potential that is the minimum distance between
        ANY residue and the closest target residue.

        In contrast to binder_distance_ReLU this potential will only penalize a pose if all of the binder residues are outside
        of a certain distance from the target residues.

        Author: NRB
    '''

    def __init__(self, binderlen, hotspot_res, weight=1, min_dist=15, use_Cb=False):

        self.binderlen   = binderlen
        self.hotspot_res = [res + binderlen for res in hotspot_res]
        self.weight      = weight
        self.min_dist    = min_dist
        self.use_Cb      = use_Cb

    def compute(self, seq, xyz):
        binder = xyz[:self.binderlen,:,:] # (Lb,27,3)
        target = xyz[self.hotspot_res,:,:] # (N,27,3)

        if use_Cb:
            N  = binder[:,0]
            Ca = binder[:,1]
            C  = binder[:,2]

            Cb = generate_Cbeta(N,Ca,C) # (Lb,3)

            N_t  = target[:,0]
            Ca_t = target[:,1]
            C_t  = target[:,2]

            Cb_t = generate_Cbeta(N_t,Ca_t,C_t) # (N,3)

            dgram = torch.cdist(Cb[None,...], Cb_t[None,...], p=2) # (1,Lb,N)

        else:
            # Use Ca dist for potential

            Ca = binder[:,1] # (Lb,3)

            Ca_t = target[:,1] # (N,3)

            dgram = torch.cdist(Ca[None,...], Ca_t[None,...], p=2) # (1,Lb,N)


        closest_dist = torch.min(dgram.squeeze(0)) # (1)

        potential    = torch.maximum(min_dist, closest_dist) # (1)

        return -1 * self.weight * potential

# Dictionary of types of potentials indexed by name of potential. Used by PotentialManager.
# If you implement a new potential you must add it to this dictionary for it to be used by
# the PotentialManager
implemented_potentials = { 'monomer_ROG':          monomer_ROG,
                           'binder_ROG':           binder_ROG,
                           'binder_distance_ReLU': binder_distance_ReLU,
                           'binder_any_ReLU':      binder_any_ReLU }

require_binderlen      = { 'binder_ROG',
                           'binder_distance_ReLU',
                           'binder_any_ReLU' }

require_hotspot_res    = { 'binder_distance_ReLU',
                           'binder_any_ReLU' }

