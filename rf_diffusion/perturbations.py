import torch

import rotation_conversions

def se3_perturb(xyz):
    R = rotation_conversions._axis_angle_rotation('X', torch.tensor(1.5))
    T = torch.tensor([1,2,3])
    return torch.einsum('lan,mn->lam', xyz, R) + T
