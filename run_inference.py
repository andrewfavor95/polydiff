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
import copy 
from util import writepdb_multi, writepdb
from inference import utils as iu
from inference.motif_manager import create_motif_manager
from icecream import ic
from hydra.core.hydra_config import HydraConfig
import numpy as np
import random
import glob
import inference.model_runners
import rf2aa.tensor_util
import rf2aa.util
from rf2aa.RoseTTAFoldModel import reset_model_attrs
import aa_model
import util
from icecream import ic 
import json 
import ipdb
# ic.configureOutput(includeContext=True)

def make_deterministic(seed=0):
    torch.use_deterministic_algorithms(True)
    seed_all(seed)

def seed_all(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def replace_dict_values(config_dict, replacement_dict):
    for key, value in replacement_dict.items():
        # ic(key, value)
        if key in config_dict:
            if isinstance(value, dict) and isinstance(config_dict[key], dict):
                # print('recursing into', key)
                replace_dict_values(config_dict[key], value)
            else:
                # print('replacing', key, 'with', value)
                config_dict[key] = value

    return config_dict


def get_overrides(args, prefix=''):
    overrides = []
    for key, value in args.items():
        if type(value) == dict:
            overrides += get_overrides(value, prefix + key + '.')
        else:
            overrides.append(f'{prefix}{key}={value}')
    return overrides

if torch.cuda.is_available():
    global_device = torch.device('cuda')
else:
    global_device = torch.device('cpu')


@hydra.main(version_base=None, config_path='config/inference', config_name='base')
def main(conf: HydraConfig) -> None:

    orig_conf = copy.deepcopy(conf)
    # pydebug.set_trace()
    # rfmotif = create_motif_manager(conf, device=global_device)

    # unpack potential json args into conf 
    if conf.inference.json_args: 
        print('Detected json args, unpacking...')
        # change conf into dict 
        dict_conf = OmegaConf.to_container(conf, resolve=True)
        
        # load the json into a dict 
        with open(conf.inference.json_args) as f:
            json_args = json.load(f)
            assert isinstance(json_args, list), 'json args must be a list of dicts'
            assert all(isinstance(x, dict) for x in json_args), 'json args must be a list of dicts'
        
    else: 
        json_args = None
            
        
    if json_args == None:
        sampler = get_sampler(conf)
        sample(sampler)
    
    else: 
        # loop over json args 
        preloaded_ckpts = {}
        prebuilt_models = {}

        for json_arg in json_args:
            tmp_dict_conf = copy.deepcopy(dict_conf)
            tmp_dict_conf = replace_dict_values(tmp_dict_conf, json_arg) # replace args for this run
            tmp_conf = OmegaConf.create(tmp_dict_conf)

            # mini_conf = OmegaConf.create(json_arg) # fake, just to get overrides and add those to tmp_conf overrides 
            mini_overrides = get_overrides(json_arg)
            tmp_conf.inference.overrides = mini_overrides # HACK: to get overrides into assemble_config_from_chk

            # DJ -  repeat/symm specific hack
            #       There are some arguments in conf that make their way into RF object attributes      
            #       Luckily, they should all be under the 'model' key in conf, so iterate over those and reset 
            #       those params in preloaded ckpts 
            
            # check for symmetrize_repeats in overrides 
            if any('symmetrize_repeats' in s for s in mini_overrides):
                reset_model_attrs(prebuilt_models, json_arg)  

            # get sampler and then sample 
            sampler = get_sampler(tmp_conf, preloaded_ckpts, prebuilt_models) 

            preloaded_ckpts = sampler.preloaded_ckpts   # __init__ loads ckpts and updates this dict
            prebuilt_models = sampler.prebuilt_models   # likewise here 

            sample(sampler)

def get_sampler(conf, preloaded_ckpts={}, prebuilt_models={}):
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
    sampler = inference.model_runners.sampler_selector(conf, preloaded_ckpts, prebuilt_models)
    return sampler

def sample(sampler):

    log = logging.getLogger(__name__)
    des_i_start = sampler._conf.inference.design_startnum
    des_i_end = sampler._conf.inference.design_startnum + sampler.inf_conf.num_designs
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
        ic(f'making design {i_des} of {des_i_start}:{des_i_end}')

        sampler_out = sample_one(sampler,sampler._conf, i_des)

        log.info(f'Finished design in {(time.time()-start_time)/60:.2f} minutes')
        save_outputs(sampler, out_prefix, *sampler_out)
 
        # TEMPORARY HACK (jue): Rename ligand and ligand atoms
        if sampler._conf.inference.ligand:
            rename_ligand_atoms(sampler._conf.inference.input_pdb, out_prefix+'.pdb')

def rename_ligand_atoms(ref_fn, out_fn):
    """Copies names of ligand residue and ligand heavy atoms from input pdb
    into output (design) pdb."""

    def is_H(atomname):
        """Returns true if the first non-numeric character of `atomname` is 'H'
        and any subsequent letter is uppercase."""
        letters = ''.join([c for c in atomname.strip() if c.isalpha()])
        return letters.startswith('H') and letters.isupper() # True for "HB", False for "Hg"

    # get input ligand lines
    with open(ref_fn) as f:
        input_lig_lines = [line.strip() for line in f.readlines()
                           if line.startswith('HETATM') and not is_H(line[12:16])]

    # get output pdb file lines
    with open(out_fn) as f:
        lines = [line.strip() for line in f.readlines()]

    # replace output ligand atom and residue names with those from input ligand
    lines2 = []
    i_input = 0
    for line in lines:
        if line.startswith('HETATM'):
            # col 12-15: atom name; col 17-19: ligand name
            lines2.append(line[:12] + input_lig_lines[i_input][12:20] + line[20:]) 
            i_input += 1
        else:
            lines2.append(line)

    # write new output pdb file
    with open(out_fn,'w') as f:
        for line in lines2:
            print(line, file=f)


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

    with open(ref_fn) as f:
        input_lig_lines = [line.strip() for line in f.readlines()
                           if line.startswith('HETATM') and not is_H(line[13:17])]

    with open(out_fn) as f:
        lines = [line.strip() for line in f.readlines()]

    lines2 = []
    i_input = 0
    for line in lines:
        if line.startswith('HETATM'):
            lines2.append(line[:13] + input_lig_lines[i_input][13:21] + line[21:])
            i_input += 1
        else:
            lines2.append(line)

    with open(out_fn,'w') as f:
        for line in lines2:
            print(line, file=f)

def sample_one(sampler,inf_conf, i_des, simple_logging=False):
        # For intermediate output logging
        indep = sampler.sample_init()

        x_init = indep.xyz
        seq_init = indep.seq
        print('For now only assuming asymmetric')
        symmsub = None

        denoised_xyz_stack = []
        px0_xyz_stack = []
        seq_stack = []
        chi1_stack = []
        # plddt_stack = []
        # motif_rmsd_stack = []

        x_t = torch.clone(x_init)
        seq_t = torch.clone(seq_init)

        rfo = None

        # Loop over number of reverse diffusion time steps.
        xyz_template_t = None
        # success = False
        for t in range(int(sampler.t_step_input), sampler.inf_conf.final_step-1, -1):
            if simple_logging:
                e = '.'
                if t%10 == 0:
                    e = t
                print(f'{e}', end='')

            # if sampler.rfmotif: 
            # if sampler.rfmotif and (t <= sampler.t_motif_fit): 
            if sampler.rfmotif and (t in sampler.motif_fit_tsteps): 
                sampler.rfmotif.start_new_design_step(i_des, t, sampler.t_step_input, x_t, symmsub)

            px0, x_t, seq_t, tors_t, plddt, rfo = sampler.sample_step(t, indep, rfo)

            assert xyz_template_t is None or not sampler.rfmotif, 'cant use rfmotif and regular template'

            # assert_that(indep.xyz.shape).is_equal_to(x_t.shape)
            # ipdb.set_trace()
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
            # ipdb.set_trace()
            
            px0_xyz_stack.append(px0)
            denoised_xyz_stack.append(x_t)
            seq_stack.append(seq_t)
            chi1_stack.append(tors_t[:,:])




            # plddt_stack.append(plddt[0]) # remove singleton leading dimension
            # motif_rmsd_stack.append(motif_rmsd) 
        # else:
        #     success = True

        # ipdb.set_trace()
        # if not success:
        #     print(f'failed to produce design {i_des} {t}, continuing...')
        #     continue

        # if doing new symmetry, dump full complex:
        if sampler._conf.inference.internal_sym:
            symmRs = sampler.symmRs.cpu()   # get symmetry operations for whole complex
            O = symmRs.shape[0]             # get total num of subunits 
            # find number of subunits that were modeled
            Nsub = sampler.cur_symmsub.shape[0]
            Lasu = px0.shape[0] // Nsub

            # grab ASU and propogate full complex 
            xyz_particle = torch.full( (O*Lasu,23,3), float('nan'), device=px0.device )
            seq_particle = torch.zeros( (O*Lasu),dtype=seq_t.dtype, device=seq_t.device )
            ipdb.set_trace()
            # put first asu in 
            xyz_particle[:Lasu,:14]   = px0[:Lasu]

            ic(seq_particle.shape)
            ic(seq_t.shape)
            seq_particle[:Lasu] = torch.argmax( seq_t[:Lasu] )

            for i in range(1,O):
                xyz_particle[(i*Lasu):((i+1)*Lasu),:14] = torch.einsum('ij,raj->rai', symmRs[i], px0[:Lasu])
                seq_particle[(i*Lasu):((i+1)*Lasu)] = torch.argmax( seq_t[:Lasu] )
        else:
            xyz_particle = None 
            seq_particle = None
            Lasu = None 

        
        # Flip order for better visualization in pymol
        denoised_xyz_stack = torch.stack(denoised_xyz_stack)
        denoised_xyz_stack = torch.flip(denoised_xyz_stack, [0,])
        px0_xyz_stack = torch.stack(px0_xyz_stack)
        px0_xyz_stack = torch.flip(px0_xyz_stack, [0,])
        return indep, denoised_xyz_stack, px0_xyz_stack, seq_stack, xyz_particle, seq_particle, Lasu

def save_outputs(sampler, out_prefix, indep, denoised_xyz_stack, px0_xyz_stack, seq_stack, xyz_particle, seq_particle, Lasu):
    log = logging.getLogger(__name__)
    # Save outputs 
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    final_seq = seq_stack[-1]

    if sampler._conf.seq_diffuser.seqdiff is not None:
        assert 0, "BEFORE USING THIS SEQ DIFFUSON OPTION, MODIFY TO INCLUDE NA BASE TOKENS!"
        # When doing sequence diffusion the model does not make predictions beyond category 19
        final_seq = final_seq[:,:20] # [L,20]

    # All samplers now use a one-hot seq so they all need this step
    final_seq = torch.argmax(final_seq, dim=-1)

    bfacts = torch.ones_like(final_seq.squeeze())

    # replace mask and unknown tokens in the final seq: 
    # for protein chains, replace with alanine
    pro_condition = ((final_seq == 20) | (final_seq==21)) & indep.is_protein
    final_seq = torch.where(pro_condition, 0, final_seq)
    # for DNA chains, replace with deoxyribo-adenine
    dna_condition = ((final_seq == 26) | (final_seq==21)) & indep.is_dna
    final_seq = torch.where(dna_condition, 22, final_seq)
    # for RNA chains, replace with ribo-adenine
    rna_condition = ((final_seq == 31) | (final_seq==21)) & indep.is_rna
    final_seq = torch.where(rna_condition, 27, final_seq)
    # final_seq = torch.where((final_seq == 20) | (final_seq==21), 0, final_seq)
    
    

    # determine lengths of protein and ligand for correct chain labeling in output pdb
    sm_mask = rf2aa.util.is_atom(final_seq)
    # chain_Ls = [len(sm_mask)-sm_mask.sum(), sm_mask.sum()] # assumes 1 protein followed by 1 ligand
    chain_Ls = rf2aa.util.Ls_from_same_chain_2d(indep.same_chain)
    
    # pX0 last step
    out = f'{out_prefix}.pdb'
    # aa_model.write_traj(out, denoised_xyz_stack[0:1], final_seq, indep.bond_feats, chain_Ls=chain_Ls)
    aa_model.write_traj(out, denoised_xyz_stack[0:1], final_seq, indep.bond_feats, chain_Ls=chain_Ls, natoms=sampler.num_atoms_saved )

    
    des_path = os.path.abspath(out)

    # symmetric oligomer PDB dump (New point symmetry protocol)
    if xyz_particle is not None:
        assert seq_particle is not None, 'Why is xyz_particle not None but seq_particle is None?'
        chain_Ls_symm = [Lasu]*(xyz_particle.shape[0]//Lasu)
        out = f'{out_prefix}_symm.pdb'

        if len(chain_Ls_symm) > 26:
            print('Truncating full particle for PDB writing')
            xyz_particle = xyz_particle[:Lasu*26]
            seq_particle = seq_particle[:Lasu*26]
            chain_Ls_symm = chain_Ls_symm[:26]
            print(len(chain_Ls_symm))

        with open(out, 'w') as f:
            rf2aa.util.writepdb_file(f, xyz_particle.cpu(), seq_particle.long(), chain_Ls=chain_Ls_symm)

    # trajectory pdbs
    traj_prefix = os.path.dirname(out_prefix)+'/traj/'+os.path.basename(out_prefix)
    os.makedirs(os.path.dirname(traj_prefix), exist_ok=True)

    out = f'{traj_prefix}_Xt-1_traj.pdb'
    aa_model.write_traj(out, denoised_xyz_stack, final_seq, indep.bond_feats, chain_Ls=chain_Ls)
    xt_traj_path = os.path.abspath(out)

    out=f'{traj_prefix}_pX0_traj.pdb'
    aa_model.write_traj(out, px0_xyz_stack, final_seq, indep.bond_feats, chain_Ls=chain_Ls)
    x0_traj_path = os.path.abspath(out)

    # run metadata
    sampler._conf.inference.input_pdb = os.path.abspath(sampler._conf.inference.input_pdb)
    indep_out ={}
    for k,v in dataclasses.asdict(indep).items():
        if torch.is_tensor(v):
            indep_out[k] = v.detach().cpu().numpy()
        else:
            indep_out[k] = v

    trb = dict(
        config = OmegaConf.to_container(sampler._conf, resolve=True),
        device = torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else 'CPU',
        px0_xyz_stack = px0_xyz_stack.detach().cpu().numpy(),
        indep=indep_out,
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
