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
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RF2-allatom'))

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
import aa_model
# ic.configureOutput(includeContext=True)

def make_deterministic(seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


@hydra.main(version_base=None, config_path='config/inference', config_name='base')
def main(conf: HydraConfig) -> None:
    sampler = get_sampler(conf)
    sample(sampler)

def get_sampler(conf):
    if conf.inference.deterministic:
        make_deterministic()
     
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

def sample(sampler):

    log = logging.getLogger(__name__)
    for i_des in range(sampler._conf.inference.design_startnum, sampler._conf.inference.design_startnum + sampler.inf_conf.num_designs):
        if sampler._conf.inference.deterministic:
            make_deterministic(i_des)

        start_time = time.time()
        out_prefix = f'{sampler.inf_conf.output_prefix}_{i_des}'
        sampler.output_prefix = out_prefix
        log.info(f'Making design {out_prefix}')
        if sampler.inf_conf.cautious and os.path.exists(out_prefix+'.pdb'):
            log.info(f'(cautious mode) Skipping this design because {out_prefix}.pdb already exists.')
            continue
        sampler_out = sample_one(sampler)
        log.info(f'Finished design in {(time.time()-start_time)/60:.2f} minutes')
        save_outputs(sampler, out_prefix, *sampler_out)
 
        # TEMPORARY HACK (jue): Rename ligand and ligand atoms
        if sampler._conf.inference.ligand:
            rename_ligand_atoms(sampler._conf.inference.input_pdb, out_prefix+'.pdb')

def rename_ligand_atoms(ref_fn, out_fn):
    """Copies names of ligand residue and ligand heavy atoms from input pdb
    into output (design) pdb."""

    def is_H(atomname):
        """Returns true if a string starts with 'H' followed by non-letters (numbers)"""
        letters = ''.join([c for c in atomname.strip() if c.isalpha()])
        return letters=='H'

    # get input ligand lines
    with open(ref_fn) as f:
        input_lig_lines = [line.strip() for line in f.readlines()
                           if line.startswith('HETATM') and not is_H(line[13:17])]

    # get output pdb file lines
    with open(out_fn) as f:
        lines = [line.strip() for line in f.readlines()]

    # replace output ligand atom and residue names with those from input ligand
    lines2 = []
    i_input = 0
    for line in lines:
        if line.startswith('HETATM'):
            # col 13-16: atom name; col 17-20: ligand name
            lines2.append(line[:13] + input_lig_lines[i_input][13:21] + line[21:]) 
            i_input += 1
        else:
            lines2.append(line)

    # write new output pdb file
    with open(out_fn,'w') as f:
        for line in lines2:
            print(line, file=f)


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
        
        # Flip order for better visualization in pymol
        denoised_xyz_stack = torch.stack(denoised_xyz_stack)
        denoised_xyz_stack = torch.flip(denoised_xyz_stack, [0,])
        px0_xyz_stack = torch.stack(px0_xyz_stack)
        px0_xyz_stack = torch.flip(px0_xyz_stack, [0,])

        return indep, denoised_xyz_stack, px0_xyz_stack, seq_stack

def save_outputs(sampler, out_prefix, indep, denoised_xyz_stack, px0_xyz_stack, seq_stack):
    log = logging.getLogger(__name__)
    # Save outputs 
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    final_seq = seq_stack[-1]

    if sampler._conf.seq_diffuser.seqdiff is not None:
        # When doing sequence diffusion the model does not make predictions beyond category 19
        final_seq = final_seq[:,:20] # [L,20]

    # All samplers now use a one-hot seq so they all need this step
    final_seq = torch.argmax(final_seq, dim=-1)

    bfacts = torch.ones_like(final_seq.squeeze())

    # replace mask and unknown tokens in the final seq with alanine
    final_seq = torch.where((final_seq == 20) | (final_seq==21), 0, final_seq)
    
    # pX0 last step
    out = f'{out_prefix}.pdb'
    aa_model.write_traj(out, denoised_xyz_stack[0:1], final_seq, indep.bond_feats)
    des_path = os.path.abspath(out)

    # trajectory pdbs
    traj_prefix = os.path.dirname(out_prefix)+'/traj/'+os.path.basename(out_prefix)
    os.makedirs(os.path.dirname(traj_prefix), exist_ok=True)

    out = f'{traj_prefix}_Xt-1_traj.pdb'
    aa_model.write_traj(out, denoised_xyz_stack, final_seq, indep.bond_feats)
    xt_traj_path = os.path.abspath(out)

    out=f'{traj_prefix}_pX0_traj.pdb'
    aa_model.write_traj(out, px0_xyz_stack, final_seq, indep.bond_feats)
    x0_traj_path = os.path.abspath(out)

    # run metadata
    sampler._conf.inference.input_pdb = os.path.abspath(sampler._conf.inference.input_pdb)
    trb = dict(
        config = OmegaConf.to_container(sampler._conf, resolve=True),
        device = torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else 'CPU',
        px0_xyz_stack = px0_xyz_stack,
        indep=dataclasses.asdict(indep),
    )
    if hasattr(sampler, 'contig_map'):
        for key, value in sampler.contig_map.get_mappings().items():
            trb[key] = value
    with open(f'{out_prefix}.trb','wb') as f_out:
        pickle.dump(trb, f_out)

    log.info(f'design : {des_path}')
    log.info(f'Xt traj: {xt_traj_path}')
    log.info(f'X0 traj: {x0_traj_path}')


if __name__ == '__main__':
    main()
