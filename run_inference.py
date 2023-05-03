#!/software/conda/envs/SE3nv/bin/python
"""
Inference script.

To run with base.yaml as the config,

> python run_inference.py

To specify a different config,

> python run_inference.py --config-name symmetry

where symmetry can be the filename of any other config (without .yaml extension)
See https://hydra.cc/docs/advanced/hydra-command-line-flags/ for more options.

"""
import os
import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
aa_se3_path = os.path.join(script_dir, 'RF2-allatom/rf2aa/SE3Transformer')
sys.path.insert(0, aa_se3_path)
sys.path.append(os.path.join(script_dir, 'RF2-allatom'))

import re
import os, time, pickle
import dataclasses
import torch 
from omegaconf import DictConfig, OmegaConf
import hydra
import logging
from util import writepdb_multi, writepdb
from inference import utils as iu
from icecream import ic
from hydra.core.hydra_config import HydraConfig
import numpy as np
import random
import glob
import inference.model_runners
import rf2aa.tensor_util
import rf2aa.util
import aa_model
import guide_posts as gp
import copy
import atomize
# ic.configureOutput(includeContext=True)

def make_deterministic(seed=0):
    torch.use_deterministic_algorithms(True)
    seed_all(seed)

def seed_all(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_seeds():
    return {
        'torch': torch.get_rng_state(),
        'np': np.random.get_state(),
        'python': random.getstate(),
    }

@hydra.main(version_base=None, config_path='config/inference', config_name='base')
def main(conf: HydraConfig) -> None:
    sampler = get_sampler(conf)
    sample(sampler)

def get_sampler(conf):
    if conf.inference.deterministic:
        seed_all()

    # Loop over number of designs to sample.
    design_startnum = conf.inference.design_startnum
    if conf.inference.design_startnum == -1:
        existing = glob.glob(conf.inference.output_prefix + '*.pdb')
        indices = [-1]
        for e in existing:
            m = re.match('.*_(\d+)\.pdb$', e)
            if not m:
                continue
            m = m.groups()[0]
            indices.append(int(m))
        design_startnum = max(indices) + 1   

    conf.inference.design_startnum = design_startnum
    # Initialize sampler and target/contig.
    sampler = inference.model_runners.sampler_selector(conf)
    return sampler

def expand_config(conf):
    confs = {}
    if conf.inference.guidepost_xyz_as_design:
        sub_conf = copy.deepcopy(conf)
        ic(conf.inference.guidepost_xyz_as_design_bb)
        for val in conf.inference.guidepost_xyz_as_design_bb:
            ic(val)
            sub_conf.inference.guidepost_xyz_as_design_bb = val
            suffix = f'atomized-bb-{val}'
            confs[suffix] = copy.deepcopy(sub_conf)
    else:
        confs = {'': conf}
    return confs


def sample(sampler):

    log = logging.getLogger(__name__)
    des_i_start = sampler._conf.inference.design_startnum
    des_i_end = sampler._conf.inference.design_startnum + sampler.inf_conf.num_designs
    for i_des in range(sampler._conf.inference.design_startnum, sampler._conf.inference.design_startnum + sampler.inf_conf.num_designs):
        if sampler._conf.inference.deterministic:
            seed_all(i_des)

        start_time = time.time()
        out_prefix = f'{sampler.inf_conf.output_prefix}_{i_des}'
        sampler.output_prefix = out_prefix
        log.info(f'Making design {out_prefix}')
        if sampler.inf_conf.cautious and os.path.exists(out_prefix+'.pdb'):
            log.info(f'(cautious mode) Skipping this design because {out_prefix}.pdb already exists.')
            continue
        ic(f'making design {i_des} of {des_i_start}:{des_i_end}')
        sampler_out = sample_one(sampler)
        log.info(f'Finished design in {(time.time()-start_time)/60:.2f} minutes')
        original_conf = copy.deepcopy(sampler._conf)
        confs = expand_config(sampler._conf)
        for suffix, conf in confs.items():
            sampler._conf = conf
            out_prefix_suffixed = out_prefix
            if suffix:
                out_prefix_suffixed += f'-{suffix}'
            print(f'{out_prefix_suffixed=}, {conf.inference.guidepost_xyz_as_design_bb=}')
            # TODO: See what is being altered here, so we don't have to copy sampler_out
            save_outputs(sampler, out_prefix_suffixed, *(copy.deepcopy(o) for o in sampler_out))
            sampler._conf = original_conf

def sample_one(sampler, simple_logging=False):
    # For intermediate output logging
    indep = sampler.sample_init()

    denoised_xyz_stack = []
    px0_xyz_stack = []
    seq_stack = []

    rfo = None

    # Loop over number of reverse diffusion time steps.
    for t in range(int(sampler.t_step_input), sampler.inf_conf.final_step-1, -1):
        if simple_logging:
            e = '.'
            if t%10 == 0:
                e = t
            print(f'{e}', end='')
        px0, x_t, seq_t, tors_t, plddt, rfo = sampler.sample_step(
            t, indep, rfo)
        # assert_that(indep.xyz.shape).is_equal_to(x_t.shape)
        rf2aa.tensor_util.assert_same_shape(indep.xyz, x_t)
        indep.xyz = x_t
            
        aa_model.assert_has_coords(indep.xyz, indep)
        # missing_backbone = torch.isnan(indep.xyz).any(dim=-1)[...,:3].any(dim=-1)
        # prot_missing_bb = missing_backbone[~indep.is_sm]
        # sm_missing_ca = torch.isnan(indep.xyz).any(dim=-1)[...,1]
        # try:
        #     assert not prot_missing_bb.any(), f'{t}:prot_missing_bb {prot_missing_bb}'
        #     assert not sm_missing_ca.any(), f'{t}:sm_missing_ca {sm_missing_ca}'
        # except Exception as e:
        #     print(e)
        #     import ipdb
        #     ipdb.set_trace()

        px0_xyz_stack.append(px0)
        denoised_xyz_stack.append(x_t)
        seq_stack.append(seq_t)
    
    # deatomize features, if applicable
    if (sampler.model_adaptor.atomizer is not None):
        indep, px0_xyz_stack, denoised_xyz_stack, seq_stack = \
            deatomize_sampler_outputs(sampler, indep, px0_xyz_stack, denoised_xyz_stack, seq_stack)
           
    # Flip order for better visualization in pymol
    denoised_xyz_stack = torch.stack(denoised_xyz_stack)
    denoised_xyz_stack = torch.flip(denoised_xyz_stack, [0,])
    px0_xyz_stack = torch.stack(px0_xyz_stack)
    px0_xyz_stack = torch.flip(px0_xyz_stack, [0,])

    return indep, denoised_xyz_stack, px0_xyz_stack, seq_stack

def deatomize_sampler_outputs(sampler, indep, px0_xyz_stack, denoised_xyz_stack, seq_stack):
    """Converts atomized residues back to residue-as-residue representation in
    the outputs of a single design trajectory.

    NOTE: `indep` will have `idx`, `bond_features`, `same_chain` updated to
    de-atomized versions, but other features will remain unchanged (and
    therefore become inconsistent).
    """
    indep.xyz = aa_model.pad_dim(indep.xyz, 1, rf2aa.chemical.NTOTAL, torch.nan)
    indep = atomize.deatomize(sampler.model_adaptor.atomizer, indep)
    indep.seq = torch.where(~indep.is_sm * indep.seq >= rf2aa.chemical.UNKINDEX, 0, indep.seq)
    px0_xyz_stack_new = []
    denoised_xyz_stack_new = []
    seq_stack_new = []
    for i in range(len(px0_xyz_stack)):
        px0_xyz = aa_model.pad_dim(px0_xyz_stack[i], 1, rf2aa.chemical.NTOTAL, torch.nan)
        denoised_xyz = aa_model.pad_dim(denoised_xyz_stack[i], 1, rf2aa.chemical.NTOTAL, torch.nan)

        seq_, xyz_, idx_, bond_feats_, same_chain_ = \
            sampler.model_adaptor.atomizer.get_deatomized_features(seq_stack[i], px0_xyz)
        px0_xyz_stack_new.append(xyz_)

        seq_, xyz_, idx_, bond_feats_, same_chain_ = \
            sampler.model_adaptor.atomizer.get_deatomized_features(seq_stack[i], denoised_xyz)
        denoised_xyz_stack_new.append(xyz_)
        seq_cat = torch.argmax(seq_, dim=-1)
        alanine_one_hot = torch.nn.functional.one_hot(torch.tensor([0]), rf2aa.chemical.NAATOKENS)
        cond=~indep.is_sm[...,None] * (seq_ >= rf2aa.chemical.UNKINDEX)
        seq_ = torch.where(cond, alanine_one_hot, seq_)
        seq_stack_new.append(seq_)

    return indep, px0_xyz_stack_new, denoised_xyz_stack_new, seq_stack_new


def save_outputs(sampler, out_prefix, indep, denoised_xyz_stack, px0_xyz_stack, seq_stack):
    log = logging.getLogger(__name__)

    final_seq = seq_stack[-1]

    if sampler._conf.seq_diffuser.seqdiff is not None:
        # When doing sequence diffusion the model does not make predictions beyond category 19
        final_seq = final_seq[:,:20] # [L,20]

    # All samplers now use a one-hot seq so they all need this step
    final_seq[~indep.is_sm, 22:] = 0
    final_seq = torch.argmax(final_seq, dim=-1)

    # replace mask and unknown tokens in the final seq with alanine
    final_seq = torch.where((final_seq == 20) | (final_seq==21), 0, final_seq)
    seq_design = final_seq.clone()
    xyz_design = denoised_xyz_stack[0].clone()
    gp_contig_mappings = {}
    # If using guideposts, infer their placement from the final pX0 prediction.
    if sampler._conf.inference.contig_as_guidepost:
        gp_to_contig_idx0 = sampler.contig_map.gp_to_ptn_idx0  # map from gp_idx0 to the ptn_idx0 in the contig string.
        is_gp = torch.zeros_like(indep.seq, dtype=bool)
        is_gp[list(gp_to_contig_idx0.keys())] = True

        # Infer which diffused residues ended up on top of the guide post residues
        diffused_xyz = denoised_xyz_stack[0, ~is_gp * ~indep.is_sm]
        gp_alone_xyz = denoised_xyz_stack[0, is_gp]
        idx_by_gp_sequential_idx = torch.nonzero(is_gp)[:,0].numpy()
        gp_alone_to_diffused_idx0 = gp.greedy_guide_post_correspondence(diffused_xyz, gp_alone_xyz)
        match_idx_by_gp_idx = {}
        for k, v in gp_alone_to_diffused_idx0.items():
            match_idx_by_gp_idx[idx_by_gp_sequential_idx[k]] = v

        gp_contig_mappings = gp.get_infered_mappings(
            gp_to_contig_idx0,
            match_idx_by_gp_idx,
            sampler.contig_map.get_mappings()
        )

        if sampler._conf.inference.guidepost_xyz_as_design:
            gp_idx, match_idx = zip(*match_idx_by_gp_idx.items())
            gp_idx = np.array(gp_idx)
            match_idx = np.array(match_idx)
            seq_design[match_idx] = seq_design[gp_idx]
            if sampler._conf.inference.guidepost_xyz_as_design_bb:
                xyz_design[match_idx] = xyz_design[gp_idx]
            else:
                xyz_design[match_idx, 4:] = xyz_design[gp_idx, 4:]
        xyz_design = xyz_design[~is_gp]
        seq_design = seq_design[~is_gp]

    # Save outputs 
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    # determine lengths of protein and ligand for correct chain labeling in output pdb
    chain_Ls = rf2aa.util.Ls_from_same_chain_2d(indep.same_chain)

    # pX0 last step
    out = f'{out_prefix}.pdb'
    aa_model.write_traj(out, xyz_design[None,...], seq_design, indep.bond_feats, chain_Ls=chain_Ls, lig_name=sampler._conf.inference.ligand, idx_pdb=indep.idx)
    des_path = os.path.abspath(out)

    # trajectory pdbs
    traj_prefix = os.path.dirname(out_prefix)+'/traj/'+os.path.basename(out_prefix)
    os.makedirs(os.path.dirname(traj_prefix), exist_ok=True)

    out = f'{traj_prefix}_Xt-1_traj.pdb'
    aa_model.write_traj(out, denoised_xyz_stack, final_seq, indep.bond_feats, chain_Ls=chain_Ls, lig_name=sampler._conf.inference.ligand, idx_pdb=indep.idx)
    xt_traj_path = os.path.abspath(out)

    out=f'{traj_prefix}_pX0_traj.pdb'
    aa_model.write_traj(out, px0_xyz_stack, final_seq, indep.bond_feats, chain_Ls=chain_Ls, lig_name=sampler._conf.inference.ligand, idx_pdb=indep.idx)
    x0_traj_path = os.path.abspath(out)

    # run metadata
    sampler._conf.inference.input_pdb = os.path.abspath(sampler._conf.inference.input_pdb)
    trb = dict(
        config = OmegaConf.to_container(sampler._conf, resolve=True),
        device = torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else 'CPU',
        px0_xyz_stack = px0_xyz_stack.detach().cpu().numpy(),
        indep={k:v.detach().cpu().numpy() for k,v in dataclasses.asdict(indep).items()},
    )
    if hasattr(sampler, 'contig_map'):
        for key, value in sampler.contig_map.get_mappings().items():
            trb[key] = value

    if sampler.model_adaptor.atomizer:
        motif_deatomized = atomize.convert_atomized_mask(sampler.model_adaptor.atomizer, ~sampler.is_diffused)
        trb['motif'] = motif_deatomized
    if sampler._conf.inference.contig_as_guidepost:
        # Store the literal location of the guide post residues
        for k in ['con_hal_pdb_idx', 'con_hal_idx0', 'sampled_mask']:
            trb[k+'_literal'] = copy.deepcopy(trb[k])

        # Saved infered guidepost locations. This is probably what downstream applications want.
        trb.update(gp_contig_mappings)        
    
    for out_path in des_path, xt_traj_path, x0_traj_path:
        aa_model.rename_ligand_atoms(sampler._conf.inference.input_pdb, out_path)

    with open(f'{out_prefix}.trb','wb') as f_out:
        pickle.dump(trb, f_out)

    log.info(f'design : {des_path}')
    log.info(f'Xt traj: {xt_traj_path}')
    log.info(f'X0 traj: {x0_traj_path}')


if __name__ == '__main__':
    main()
