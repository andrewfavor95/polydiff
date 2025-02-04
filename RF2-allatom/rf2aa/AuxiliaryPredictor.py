import torch
import torch.nn as nn

from rf2aa.chemical import NAATOKENS
from rf2aa.Attention_module import BiasedAxialAttention
import torch.nn.functional as F

class DistanceNetwork(nn.Module):
    def __init__(self, n_feat, p_drop=0.1):
        super(DistanceNetwork, self).__init__()
        #
        self.proj_symm = nn.Linear(n_feat, 61+37) # must match bin counts defined in kinematics.py
        self.proj_asymm = nn.Linear(n_feat, 37+19)
    
        self.reset_parameter()
    
    def reset_parameter(self):
        # initialize linear layer for final logit prediction
        nn.init.zeros_(self.proj_symm.weight)
        nn.init.zeros_(self.proj_asymm.weight)
        nn.init.zeros_(self.proj_symm.bias)
        nn.init.zeros_(self.proj_asymm.bias)

    def forward(self, x):
        # input: pair info (B, L, L, C)

        # predict theta, phi (non-symmetric)
        logits_asymm = self.proj_asymm(x)
        logits_theta = logits_asymm[:,:,:,:37].permute(0,3,1,2)
        logits_phi = logits_asymm[:,:,:,37:].permute(0,3,1,2)

        # predict dist, omega
        logits_symm = self.proj_symm(x)
        logits_symm = logits_symm + logits_symm.permute(0,2,1,3)
        logits_dist = logits_symm[:,:,:,:61].permute(0,3,1,2)
        logits_omega = logits_symm[:,:,:,61:].permute(0,3,1,2)

        return logits_dist, logits_omega, logits_theta, logits_phi

class MaskedTokenNetwork(nn.Module):
    def __init__(self, n_feat, p_drop=0.1):
        super(MaskedTokenNetwork, self).__init__()

        #fd note this predicts probability for the mask token (which is never in ground truth)
        #   it should be ok though(?)
        self.proj = nn.Linear(n_feat, NAATOKENS)
        
        self.reset_parameter()
    
    def reset_parameter(self):
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, N, L = x.shape[:3]
        logits = self.proj(x).permute(0,3,1,2).reshape(B, -1, N*L)

        return logits

class LDDTNetwork(nn.Module):
    def __init__(self, n_feat, n_bin_lddt=50):
        super(LDDTNetwork, self).__init__()
        self.proj = nn.Linear(n_feat, n_bin_lddt)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        logits = self.proj(x) # (B, L, 50)

        return logits.permute(0,2,1)

class PAENetwork(nn.Module):
    def __init__(self, n_feat, n_bin_pae=64):
        super(PAENetwork, self).__init__()
        self.proj = nn.Linear(n_feat, n_bin_pae)
        self.reset_parameter()
    def reset_parameter(self):
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        logits = self.proj(x) # (B, L, L, 64)

        return logits.permute(0,3,1,2)

# class BinderNetwork(nn.Module):
#     def __init__(self, n_feat):
#         super(BinderNetwork, self).__init__()
#         self.downsample = torch.nn.Linear(n_feat, 1)
#         self.reset_parameter()
# 
#     def reset_parameter(self):
#         nn.init.zeros_(self.downsample.weight)
#         nn.init.zeros_(self.downsample.bias)
# 
#     def forward(self, pair, state, same_chain):
#         L = pair.shape[1]
#         left = state.unsqueeze(2).expand(-1,-1,L,-1)
#         right = state.unsqueeze(1).expand(-1,L,-1,-1)
#         logits = self.downsample( torch.cat((pair, left, right), dim=-1) ) # (B, L, L, 1)
#         logits_inter = torch.mean( logits[same_chain==0], dim=0 ).nan_to_num() # all zeros if single chain
#         prob = torch.sigmoid( logits_inter )
#         return prob

class BinderNetwork(nn.Module):
    def __init__(self, d_pair=128, d_state=32, d_rbf=64, p_drop=0.15):
        super(BinderNetwork, self).__init__()

        self.rbf2attn = nn.Linear(d_rbf, 1)
        self.downsample = torch.nn.Linear(d_pair+2*d_state, 1)
        self.dropout = torch.nn.Dropout(p_drop)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.zeros_(self.downsample.weight)
        nn.init.zeros_(self.downsample.bias)
        nn.init.zeros_(self.rbf2attn.weight)
        nn.init.zeros_(self.rbf2attn.bias)

    def forward(self, pair, rbf_feat, state, seq, idx, bond_feats, dist_matrix, same_chain):
        B, L = pair.shape[:2]

        # 1. get attention map
        # pair: (B, L, L, d_pair)
        attn = self.rbf2attn(rbf_feat)

        # 2. get logits
        left = state.unsqueeze(2).expand(-1,-1,L,-1)
        right = state.unsqueeze(1).expand(-1,L,-1,-1)
        logits = self.downsample( torch.cat((pair, left, right), dim=-1) )

        # 3. dot product
        if (torch.sum(same_chain==0)==0):
            logits = logits.flatten()
            attn = attn.flatten()
        else:
            logits = logits[same_chain==0]
            attn = attn[same_chain==0]

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn) # randomly zero out 15% of pairwise positions

        logits_inter = torch.mean( logits * attn, dim=0 ).nan_to_num() # all zeros if single chain

        prob = torch.sigmoid( logits_inter )

        return prob
