import torch
from icecream import ic

def calc_displacement(pred, true):
    """
    Calculates the displacement between predicted and true CA 

    pred - (I,B,L,3, 3)
    true - (  B,L,27,3)
    """
    B = pred.shape[1]


    assert B == 1
    pred = pred.squeeze(1)
    true = true.squeeze(0)

    pred_ca = pred[:,:,1,...] # (I,L,3)
    true_ca = true[:,1,...]   # (L,3)

    return pred_ca - true_ca[None,...]



def displacement(logit_s, label_s,
                  logit_aa_s, label_aa_s, mask_aa_s, logit_exp,
                  pred, pred_tors, true, mask_crds, mask_BB, mask_2d, same_chain,
                  pred_lddt, idx, dataset, chosen_task, diffusion_mask, t, unclamp=False, negative=False,
                  w_dist=1.0, w_aa=1.0, w_str=1.0, w_all=0.5, w_exp=1.0,
                  w_lddt=1.0, w_blen=1.0, w_bang=1.0, w_lj=0.0, w_hb=0.0,
                  lj_lin=0.75, use_H=False, w_disp=0.0, eps=1e-6, **kwargs):
    d_clamp = None if unclamp else 10.0
    disp = calc_displacement(pred, true)
    dist = torch.norm(disp, dim=-1)

    I, L = dist.shape[0:2]
    if diffusion_mask is None:
        diffusion_mask = torch.full((L,), False)

    o = {}
    for region, mask in (
            ('given', diffusion_mask),
            ('masked', ~diffusion_mask),
            ('total', torch.full_like(diffusion_mask, True))):
        o[f'displacement_{region}'] = torch.mean(dist[:,mask])
        fraction_clamped = 0.0
        if d_clamp is not None:
            # set squared distance clamp to d_clamp**2
            d_clamp=torch.tensor(d_clamp)[None].to(dist.device)
            fraction_clamped = torch.mean((dist>d_clamp).float()).item()
        o[f'displacement_fraction_clamped_{region}'] = fraction_clamped
        for i in range(I):
            o[f'displacement_{region}_i{i}'] = torch.mean(dist[i,mask])
    
    return o
 

