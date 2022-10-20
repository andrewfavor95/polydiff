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


class dimer_ROG(Potential):
    '''
        Radius of Gyration potential for encouraging compactness of both monomers when designing dimers

        Author: PV
    '''

    def __init__(self, binderlen, weight=1, min_dist=15):

        self.binderlen = binderlen
        self.min_dist  = min_dist
        self.weight    = weight

    def compute(self, seq, xyz):

        # Only look at monomer 1 residues
        Ca_m1 = xyz[:self.binderlen,1] # [Lb,3]
        
        # Only look at monomer 2 residues
        Ca_m2 = xyz[self.binderlen:,1] # [Lb,3]

        centroid_m1 = torch.mean(Ca_m1, dim=0, keepdim=True) # [1,3]
        centroid_m2 = torch.mean(Ca_m1, dim=0, keepdim=True) # [1,3]

        # cdist needs a batch dimension - NRB
        #This calculates RoG for Monomer 1
        dgram_m1 = torch.cdist(Ca_m1[None,...].contiguous(), centroid_m1[None,...].contiguous(), p=2) # [1,Lb,1,3]
        dgram_m1 = torch.maximum(self.min_dist * torch.ones_like(dgram_m1.squeeze(0)), dgram_m1.squeeze(0)) # [Lb,1,3]
        rad_of_gyration_m1 = torch.sqrt( torch.sum(torch.square(dgram_m1)) / Ca_m1.shape[0] ) # [1]

        # cdist needs a batch dimension - NRB
        #This calculates RoG for Monomer 2
        dgram_m2 = torch.cdist(Ca_m2[None,...].contiguous(), centroid_m2[None,...].contiguous(), p=2) # [1,Lb,1,3]
        dgram_m2 = torch.maximum(self.min_dist * torch.ones_like(dgram_m2.squeeze(0)), dgram_m2.squeeze(0)) # [Lb,1,3]
        rad_of_gyration_m2 = torch.sqrt( torch.sum(torch.square(dgram_m2)) / Ca_m2.shape[0] ) # [1]

        #Potential value is the average of both radii of gyration (is avg. the best way to do this?)
        return -1 * self.weight * (rad_of_gyration_m1 + rad_of_gyration_m2)/2

class binder_ncontacts(Potential):
    '''
        Differentiable way to maximise number of contacts within a protein
        
        Motivation is given here: https://www.plumed.org/doc-v2.7/user-doc/html/_c_o_o_r_d_i_n_a_t_i_o_n.html

        Author: PV
    '''

    def __init__(self, binderlen, weight=1, r_0=8, d_0=4):

        self.binderlen = binderlen
        self.r_0       = r_0
        self.weight    = weight
        self.d_0       = d_0

    def compute(self, seq, xyz):

        # Only look at binder Ca residues
        Ca = xyz[:self.binderlen,1] # [Lb,3]
        
        #cdist needs a batch dimension - NRB
        dgram = torch.cdist(Ca[None,...].contiguous(), Ca[None,...].contiguous(), p=2) # [1,Lb,Lb]
        divide_by_r_0 = (dgram - self.d_0) / self.r_0
        numerator = torch.pow(divide_by_r_0,6)
        denominator = torch.pow(divide_by_r_0,12)
        binder_ncontacts = (1 - numerator) / (1 - denominator)
        
        print("BINDER CONTACTS:", binder_ncontacts.sum())
        #Potential value is the average of both radii of gyration (is avg. the best way to do this?)
        return self.weight * binder_ncontacts.sum()

    
class dimer_ncontacts(Potential):

    '''
        Differentiable way to maximise number of contacts for two individual monomers in a dimer
        
        Motivation is given here: https://www.plumed.org/doc-v2.7/user-doc/html/_c_o_o_r_d_i_n_a_t_i_o_n.html

        Author: PV
    '''


    def __init__(self, binderlen, weight=1, r_0=8, d_0=4):

        self.binderlen = binderlen
        self.r_0       = r_0
        self.weight    = weight
        self.d_0       = d_0

    def compute(self, seq, xyz):

        # Only look at binder Ca residues
        Ca = xyz[:self.binderlen,1] # [Lb,3]
        #cdist needs a batch dimension - NRB
        dgram = torch.cdist(Ca[None,...].contiguous(), Ca[None,...].contiguous(), p=2) # [1,Lb,Lb]
        divide_by_r_0 = (dgram - self.d_0) / self.r_0
        numerator = torch.pow(divide_by_r_0,6)
        denominator = torch.pow(divide_by_r_0,12)
        binder_ncontacts = (1 - numerator) / (1 - denominator)
        #Potential is the sum of values in the tensor
        binder_ncontacts = binder_ncontacts.sum()

        # Only look at target Ca residues
        Ca = xyz[self.binderlen:,1] # [Lb,3]
        dgram = torch.cdist(Ca[None,...].contiguous(), Ca[None,...].contiguous(), p=2) # [1,Lb,Lb]
        divide_by_r_0 = (dgram - self.d_0) / self.r_0
        numerator = torch.pow(divide_by_r_0,6)
        denominator = torch.pow(divide_by_r_0,12)
        target_ncontacts = (1 - numerator) / (1 - denominator)
        #Potential is the sum of values in the tensor
        target_ncontacts = target_ncontacts.sum()
        
        print("DIMER NCONTACTS:", (binder_ncontacts+target_ncontacts)/2)
        #Returns average of n contacts withiin monomer 1 and monomer 2
        return self.weight * (binder_ncontacts+target_ncontacts)/2

class interface_ncontacts(Potential):

    '''
        Differentiable way to maximise number of contacts between binder and target
        
        Motivation is given here: https://www.plumed.org/doc-v2.7/user-doc/html/_c_o_o_r_d_i_n_a_t_i_o_n.html

        Author: PV
    '''


    def __init__(self, binderlen, weight=1, r_0=8, d_0=6):

        self.binderlen = binderlen
        self.r_0       = r_0
        self.weight    = weight
        self.d_0       = d_0

    def compute(self, seq, xyz):

        # Extract binder Ca residues
        Ca_b = xyz[:self.binderlen,1] # [Lb,3]

        # Extract target Ca residues
        Ca_t = xyz[self.binderlen:,1] # [Lt,3]

        #cdist needs a batch dimension - NRB
        dgram = torch.cdist(Ca_b[None,...].contiguous(), Ca_t[None,...].contiguous(), p=2) # [1,Lb,Lt]
        divide_by_r_0 = (dgram - self.d_0) / self.r_0
        numerator = torch.pow(divide_by_r_0,6)
        denominator = torch.pow(divide_by_r_0,12)
        interface_ncontacts = (1 - numerator) / (1 - denominator)
        #Potential is the sum of values in the tensor
        interface_ncontacts = interface_ncontacts.sum()

        print("INTERFACE CONTACTS:", interface_ncontacts.sum())

        return self.weight * interface_ncontacts


class monomer_contacts(Potential):
    '''
        Differentiable way to maximise number of contacts within a protein

        Motivation is given here: https://www.plumed.org/doc-v2.7/user-doc/html/_c_o_o_r_d_i_n_a_t_i_o_n.html
        Author: PV
    '''

    def __init__(self, weight=1, r_0=8, d_0=2, eps=1e-6):

        self.r_0       = r_0
        self.weight    = weight
        self.d_0       = d_0
        self.eps       = eps

    def compute(self, seq, xyz):

        Ca = xyz[:,1] # [L,3]

        #cdist needs a batch dimension - NRB
        dgram = torch.cdist(Ca[None,...].contiguous(), Ca[None,...].contiguous(), p=2) # [1,Lb,Lb]
        divide_by_r_0 = (dgram - self.d_0) / self.r_0
        numerator = torch.pow(divide_by_r_0,6)
        denominator = torch.pow(divide_by_r_0,12)

        ncontacts = (1 - numerator) / ((1 - denominator))


        #Potential value is the average of both radii of gyration (is avg. the best way to do this?)
        return self.weight * ncontacts.sum()


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
                           'binder_any_ReLU':      binder_any_ReLU,
                           'dimer_ROG':            dimer_ROG,
                           'binder_ncontacts':     binder_ncontacts,
                           'dimer_ncontacts':      dimer_ncontacts,
                           'interface_ncontacts':  interface_ncontacts,
                           'monomer_contacts':     monomer_contacts}

require_binderlen      = { 'binder_ROG',
                           'binder_distance_ReLU',
                           'binder_any_ReLU',
                           'dimer_ROG',
                           'binder_ncontacts',
                           'dimer_ncontacts',
                           'interface_ncontacts'}

require_hotspot_res    = { 'binder_distance_ReLU',
                           'binder_any_ReLU' }

