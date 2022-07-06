#!/home/dimaio/.conda/envs/SE3nv/bin/python
import diffusion
import parsers
import torch 
import numpy as np 

from util import writepdb_multi
from util_module import ComputeAllAtomCoords

from icecream import ic 

parsed = parsers.parse_pdb('/mnt/home/davidcj/projects/expert-potato/expert-potato/1qys.pdb')
xyz    = parsed['xyz']
xyz    = torch.from_numpy( (xyz - xyz[:,:1,:].mean(axis=0)[None,...]) )

seq = torch.from_numpy( parsed['seq'] )
atom_mask = torch.from_numpy( parsed['mask'] )

diffusion_mask = torch.zeros(len(seq.squeeze())).to(dtype=bool)
diffusion_mask[:20] = True

T = 200
b_0 = 0.001
b_T = 0.1

kwargs = {'T'  : T,
          'b_0': b_0,
          'b_T': b_T,
          'schedule_type':'cosine',
          'schedule_kwargs':{},
          'so3_type':'slerp',
          'chi_type':'interp',
          'var_scale':1.,
          'crd_scale':1/15,
          'aa_decode_steps':100}


diffuser = diffusion.Diffuser(**kwargs)

diffused_T,\
deltas,\
diffused_frame_crds,\
diffused_frames,\
diffused_torsions,\
diffused_FA_crds = diffuser.diffuse_pose(xyz, seq, atom_mask, diffusion_mask=diffusion_mask)


# print('Writing translation pdb')
# outpath1 = './translation_only.pdb'
# seq = torch.from_numpy(seq)
# writepdb_multi(outpath1, diffused_T.transpose(0,1), torch.ones_like(seq), seq, backbone_only=True)



# print('Writing slerp pdb')
# outpath1 = './slerp_only.pdb'

# writepdb_multi(outpath1, torch.from_numpy(diffused_frame_crds).transpose(0,1), torch.ones_like(seq), seq, backbone_only=True)




# print('Writing combo slerp / translation pdb')
# cum_delta = deltas.cumsum(dim=1)
# ic(torch.is_tensor(diffused_frame_crds))
# ic(torch.is_tensor(cum_delta))
# translated_slerp = torch.from_numpy(diffused_frame_crds) + cum_delta[:,:,None,:]

# ic(cum_delta[0,10])
# ic(diffused_T[0,10])

# outpath1 = './slerp_and_translate.pdb'
# writepdb_multi(outpath1, translated_slerp.transpose(0,1), torch.ones_like(seq), seq, backbone_only=True)



## Create full atom crds from chi-only diffusion 
# diffused_torsions_sincos = torch.stack( [torch.cos(diffused_torsions), torch.sin(diffused_torsions)], dim=-1 )
# get_allatom = ComputeAllAtomCoords()
# fullatom_stack = []
# for alphas in diffused_torsions_sincos.transpose(0,1):

#     _,full_atoms = get_allatom(seq[None], xyz[None, :,:3], alphas[None])

#     fullatom_stack.append(full_atoms.squeeze())


# print('Writing chi angle interpolation only')
# outpath1 = './chi_interp.pdb'
# writepdb_multi(outpath1, fullatom_stack, torch.ones_like(seq), seq, backbone_only=False)


# Create full atom coords from combined diffusion
print('Writing combined diffusion pdb...')
outpath1 = './diffuse_all.pdb'
# ic(diffused_FA_crds.shape)
writepdb_multi(outpath1, diffused_FA_crds.squeeze(), torch.ones_like(seq), seq, backbone_only=False)