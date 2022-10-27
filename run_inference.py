#! /home/dimaio/.conda/envs/SE3nv/bin/python
"""
Inference script.

To run with base.yaml as the config,

> python run_inference.py

To specify a different config,

> python run_inference.py --config-name symmetry

where symmetry can be the filename of any other config (without .yaml extension)
See https://hydra.cc/docs/advanced/hydra-command-line-flags/ for more options.

"""

import os, time, pickle
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

def make_deterministic(seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


@hydra.main(version_base=None, config_path='config/inference', config_name='base')
def main(conf: HydraConfig) -> None:
    log = logging.getLogger(__name__)
    if conf.inference.deterministic:
        make_deterministic()
    
    # Initialize sampler and target/contig.
    sampler = iu.sampler_selector(conf)
    
    # Loop over number of designs to sample.
    for i_des in range(sampler.inf_conf.design_startnum, sampler.inf_conf.design_startnum + sampler.inf_conf.num_designs):
        if conf.inference.deterministic:
            make_deterministic(i_des)

        start_time = time.time()
        out_prefix = f'{sampler.inf_conf.output_prefix}_{i_des}'
        log.info(f'Making design {out_prefix}')
        if sampler.inf_conf.cautious and os.path.exists(out_prefix+'.pdb'):
            log.info(f'(cautious mode) Skipping this design because {out_prefix}.pdb already exists.')
            continue

        x_init, seq_init = sampler.sample_init()

        denoised_xyz_stack = []
        px0_xyz_stack = []
        seq_stack = []
        chi1_stack = []
        plddt_stack = []

        x_t = torch.clone(x_init)
        seq_t = torch.clone(seq_init)

        # Loop over number of reverse diffusion time steps.
        for t in range(int(sampler.t_step_input), sampler.inf_conf.final_step-1, -1):
            px0, x_t, seq_t, tors_t, plddt = sampler.sample_step(
                t=t, seq_t=seq_t, x_t=x_t, seq_init=seq_init, final_step=sampler.inf_conf.final_step)
            px0_xyz_stack.append(px0)
            denoised_xyz_stack.append(x_t)
            seq_stack.append(seq_t)
            chi1_stack.append(tors_t[:,:])
            plddt_stack.append(plddt[0]) # remove singleton leading dimension
        
        # Flip order for better visualization in pymol
        denoised_xyz_stack = torch.stack(denoised_xyz_stack)
        denoised_xyz_stack = torch.flip(denoised_xyz_stack, [0,])
        px0_xyz_stack = torch.stack(px0_xyz_stack)
        px0_xyz_stack = torch.flip(px0_xyz_stack, [0,])

        # For logging -- don't flip
        plddt_stack = torch.stack(plddt_stack)

        # Save outputs 
        os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
        final_seq = seq_stack[-1]

        if not conf.seq_diffuser.seqdiff is None:
            # When doing sequence diffusion the model does not make predictions beyond category 19
            final_seq = final_seq[:,:20] # [L,20]

        # All samplers now use a one-hot seq so they all need this step
        final_seq = torch.argmax(final_seq, dim=-1)

        bfacts = torch.ones_like(final_seq.squeeze())

        # pX0 last step
        out = f'{out_prefix}.pdb'

        # replace mask and unknown tokens in the final seq with alanine
        final_seq = torch.where(final_seq == 20, 0, final_seq)
        final_seq = torch.where(final_seq == 21, 0, final_seq)

        writepdb(out, denoised_xyz_stack[0], final_seq, sampler.ppi_conf.binderlen, chain_idx=sampler.chain_idx)

        # run metadata
        trb = dict(
            config = OmegaConf.to_container(sampler._conf, resolve=True),
            plddt = plddt_stack.cpu().numpy(),
            device = torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else 'CPU',
            time = time.time() - start_time
        )
        if hasattr(sampler, 'contig_map'):
            for key, value in sampler.contig_map.get_mappings().items():
                trb[key] = value
        with open(f'{out_prefix}.trb','wb') as f_out:
            pickle.dump(trb, f_out)

        # trajectory pdbs
        traj_prefix = os.path.dirname(out_prefix)+'/traj/'+os.path.basename(out_prefix)
        os.makedirs(os.path.dirname(traj_prefix), exist_ok=True)

        out = f'{traj_prefix}_Xt-1_traj.pdb'
        writepdb_multi(out, denoised_xyz_stack, bfacts, 
            final_seq.squeeze(), use_hydrogens=False, backbone_only=False, chain_ids=sampler.chain_idx)

        out=f'{traj_prefix}_pX0_traj.pdb'
        writepdb_multi(out, px0_xyz_stack, bfacts, 
            final_seq.squeeze(), use_hydrogens=False, backbone_only=False, chain_ids=sampler.chain_idx)

        log.info(f'Finished design in {(time.time()-start_time)/60:.2f} minutes')

if __name__ == '__main__':
    main()
