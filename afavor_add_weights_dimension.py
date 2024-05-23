#!/software/containers/SE3nv.sif



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
from pdb import set_trace
from torch.utils import data
import math 
import argparse
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RF2-allatom'))




# def update_dims(weights,key):

#     # Adding 2D embedding features
#     # d_t1d*2+d_t2d
#     if True:
#         pt1_add_dim     = new_t2d - orig_t2d
#         pt2_add_dim     = new_t1d - orig_t1d
#         pt3_add_dim     = new_t1d - orig_t1d 
        
#         pt1_emb_zeros   = torch.zeros(64, pt1_add_dim)
#         pt2_emb_zeros   = torch.zeros(64, pt2_add_dim)
#         pt3_emb_zeros   = torch.zeros(64, pt3_add_dim)

#         '''
#         The way that the t2d input to embedding is created is not straightforward
#         It looks like this:

#             # Prepare 2D template features
#             left = t1d.unsqueeze(3).expand(-1,-1,-1,L,-1)
#             right = t1d.unsqueeze(2).expand(-1,-1,L,-1,-1)

#             templ = torch.cat((t2d, left, right), -1) # (B, T, L, L, 109)
#             templ = self.emb(templ) # Template templures (B, T, L, L, d_templ)
#         '''
#         set_trace()
#         new_emb_weights = torch.cat( (weights['templ_emb.emb.weight'][:,:orig_t2d], pt1_emb_zeros), dim=-1 )
#         new_emb_weights = torch.cat( (new_emb_weights, weights['templ_emb.emb.weight'][:,orig_t2d:orig_t2d+orig_t1d], pt2_emb_zeros), dim=-1 )
#         new_emb_weights = torch.cat( (new_emb_weights, weights['templ_emb.emb.weight'][:,orig_t2d+orig_t1d:], pt3_emb_zeros), dim=-1 )

#         weights['templ_emb.emb.weight']     = new_emb_weights

#         print(f'Shape of t2d: {new_emb_weights.shape}')
        
#     # Adding 1D embedding features
#     # d_t1d+d_tor
#     if True:
#         t1d_weights_dim = new_t1d + orig_dtor
#         t1d_add_dim     = t1d_weights_dim - orig_t1d - orig_dtor
        
#         t1d_zeros       = torch.zeros(64, t1d_add_dim)
#         new_t1d_weights = torch.cat( (weights['templ_emb.emb_t1d.weight'][:,:orig_t1d], t1d_zeros), dim=-1 )
#         new_t1d_weights = torch.cat( (new_t1d_weights, weights['templ_emb.emb_t1d.weight'][:,orig_t1d:]), dim=-1 )
#         weights['templ_emb.emb_t1d.weight'] = new_t1d_weights
#         print(f'Shape of t1d emb: {new_t1d_weights.shape}')


#         t1d_zeros       = torch.zeros(64, t1d_add_dim)
#         new_t1d_weights = torch.cat( (weights['templ_emb.templ_stack.proj_t1d.weight'][:,:orig_t1d], t1d_zeros), dim=-1 )
#         new_t1d_weights = torch.cat( (new_t1d_weights, weights['templ_emb.templ_stack.proj_t1d.weight'][:,orig_t1d:]), dim=-1 )
#         weights['templ_emb.templ_stack.proj_t1d.weight'] = new_t1d_weights
#         print(f'Shape of t1d proj: {new_t1d_weights.shape}')




#     return weights


def update_dims(weights,key):

    # Adding 2D embedding features
    # d_t1d*2+d_t2d
    if True:
        pt1_add_dim     = new_t2d - orig_t2d
        pt2_add_dim     = new_t1d - orig_t1d
        pt3_add_dim     = new_t1d - orig_t1d 
        
        pt1_emb_zeros   = torch.zeros(64, pt1_add_dim)
        pt2_emb_zeros   = torch.zeros(64, pt2_add_dim)
        pt3_emb_zeros   = torch.zeros(64, pt3_add_dim)

        '''
        The way that the t2d input to embedding is created is not straightforward
        It looks like this:

            # Prepare 2D template features
            left = t1d.unsqueeze(3).expand(-1,-1,-1,L,-1)
            right = t1d.unsqueeze(2).expand(-1,-1,L,-1,-1)

            templ = torch.cat((t2d, left, right), -1) # (B, T, L, L, 109)
            templ = self.emb(templ) # Template templures (B, T, L, L, d_templ)
        '''
        # set_trace()
        new_emb_weights = torch.cat( (weights['model.templ_emb.emb.weight'][:,:orig_t2d], pt1_emb_zeros), dim=-1 )
        new_emb_weights = torch.cat( (new_emb_weights, weights['model.templ_emb.emb.weight'][:,orig_t2d:orig_t2d+orig_t1d], pt2_emb_zeros), dim=-1 )
        new_emb_weights = torch.cat( (new_emb_weights, weights['model.templ_emb.emb.weight'][:,orig_t2d+orig_t1d:], pt3_emb_zeros), dim=-1 )

        weights['model.templ_emb.emb.weight']     = new_emb_weights

        print(f'Shape of t2d: {new_emb_weights.shape}')
        
    # Adding 1D embedding features
    # d_t1d+d_tor
    if True:
        t1d_weights_dim = new_t1d + orig_dtor
        t1d_add_dim     = t1d_weights_dim - orig_t1d - orig_dtor
        
        t1d_zeros       = torch.zeros(64, t1d_add_dim)
        new_t1d_weights = torch.cat( (weights['model.templ_emb.emb_t1d.weight'][:,:orig_t1d], t1d_zeros), dim=-1 )
        new_t1d_weights = torch.cat( (new_t1d_weights, weights['model.templ_emb.emb_t1d.weight'][:,orig_t1d:]), dim=-1 )
        weights['model.templ_emb.emb_t1d.weight'] = new_t1d_weights
        print(f'Shape of t1d emb: {new_t1d_weights.shape}')


        t1d_zeros       = torch.zeros(64, t1d_add_dim)
        new_t1d_weights = torch.cat( (weights['model.templ_emb.templ_stack.proj_t1d.weight'][:,:orig_t1d], t1d_zeros), dim=-1 )
        new_t1d_weights = torch.cat( (new_t1d_weights, weights['model.templ_emb.templ_stack.proj_t1d.weight'][:,orig_t1d:]), dim=-1 )
        weights['model.templ_emb.templ_stack.proj_t1d.weight'] = new_t1d_weights
        print(f'Shape of t1d proj: {new_t1d_weights.shape}')




    return weights


parser = argparse.ArgumentParser()

parser.add_argument('--input_ckpt', '-i', type=str, required=True, help='input model checkpoint path.')
parser.add_argument('--output_ckpt', '-o', type=str, required=True, help='output model checkpoint path.')
parser.add_argument('--delta_t1d_dim', '-dt1d', type=int, default=0, required=True, help='change in dimension of t1d for output checkpoint.')
parser.add_argument('--delta_t2d_dim', '-dt2d', type=int, default=0, required=True, help='change in dimension of t2d for output checkpoint.')
parser.add_argument('--orig_t1d_dim', '-ot1d', type=int, default=81, required=False, help='original dimension of t1d for input checkpoint.')
parser.add_argument('--orig_t2d_dim', '-ot2d', type=int, default=69, required=False, help='original dimension of t2d for input checkpoint.')
# parser.add_argument('--delta_tor_dim', '-odtor', type=int, default=0, required=True, help='original dimension of the torsion angles.')
# parser.add_argument('--orig_tor_dim', '-odtor', type=int, default=30, required=False, help='original dimension of the torsion angles.')
args = parser.parse_args()


# orig_t1d  = 81
# orig_t2d  = 69
orig_t1d  = args.orig_t1d_dim
orig_t2d  = args.orig_t2d_dim
orig_dtor = 30

# Adding seq diffused feature 
new_t1d   = orig_t1d + args.delta_t1d_dim
new_t2d   = orig_t2d + args.delta_t2d_dim


ckpt = torch.load(args.input_ckpt, map_location=torch.device('cpu'))

for k in ['final_state_dict', 'model_state_dict']:
    weights_in = ckpt[k]
    ckpt[k] = update_dims(weights_in, k)


torch.save(ckpt, args.output_ckpt)

