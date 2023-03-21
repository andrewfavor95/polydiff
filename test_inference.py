import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RF2-allatom'))

import unittest
from unittest import mock
import subprocess
from pathlib import Path
from inspect import signature
from io import StringIO

import hydra
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from icecream import ic
import torch
import numpy as np

import test_utils
import run_inference
from functools import partial
from rf2aa import tensor_util
import rf2aa.chemical
from rf2aa.RoseTTAFoldModel import RoseTTAFoldModule
import rf2aa.loss
import inference.utils

REWRITE = False
def infer(overrides):
    conf = construct_conf(overrides)
    run_inference.main(conf)
    p = Path(conf.inference.output_prefix + '_0.pdb')
    return p, conf

def construct_conf(overrides):
    initialize(version_base=None, config_path="config/inference", job_name="test_app")
    conf = compose(config_name='aa_small.yaml', overrides=overrides, return_hydra_config=True)
    # This is necessary so that when the model_runner is picking up the overrides, it finds them set on HydraConfig.
    HydraConfig.instance().set_config(conf)
    conf = compose(config_name='aa_small.yaml', overrides=overrides)
    return conf

def get_trb(conf):
    path = conf.inference.output_prefix + '_0.trb'
    return np.load(path,allow_pickle=True)

class TestRegression(unittest.TestCase):

    def tearDown(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    
    # Example regression test.
    def test_t2(self):
        run_inference.make_deterministic()
        pdb, _ = infer([
            'diffuser.T=2',
            'inference.num_designs=1',
            'inference.output_prefix=tmp/test_0',
            'inference.design_startnum=0',
        ])
        pdb_contents = inference.utils.parse_pdb(pdb)
        cmp = partial(tensor_util.cmp, atol=5e-2, rtol=0)
        test_utils.assert_matches_golden(self, 'T2', pdb_contents, rewrite=REWRITE, custom_comparator=cmp)
    
    def test_partial_sidechain(self):
        run_inference.make_deterministic()
        pdb, _ = infer([
            'diffuser.T=2',
            'inference.num_designs=1',
            'inference.output_prefix=tmp/test_3',
            'inference.design_startnum=0',
            "contigmap.contigs=['1,A518-518,1']",
            "+contigmap.contig_atoms=\"{'A518':'CG,OD1,OD2'}\"",
            'contigmap.length=3-3'
        ])
        pdb_contents = inference.utils.parse_pdb(pdb)
        # Currently does not pass
        # cmp = partial(tensor_util.cmp, atol=5e-2, rtol=0)
        # test_utils.assert_matches_golden(self, 'partial_sc', pdb_contents, rewrite=REWRITE, custom_comparator=cmp)


class TestInference(unittest.TestCase):

    def tearDown(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    
    # Test that the motif remains fixed throughout inference.
    def test_motif_remains_fixed(self):
        T = 3
        conf = construct_conf([
            f'diffuser.T={T}',
            'inference.num_designs=1',
            'inference.output_prefix=tmp/test_1',
            'inference.design_startnum=0',
        ])

        func_sig = signature(RoseTTAFoldModule.forward)
        fake_forward = mock.patch.object(RoseTTAFoldModule, "forward", autospec=True)

        def side_effect(self, *args, **kwargs):
            ic("mock forward", type(self), side_effect.call_count)
            side_effect.call_count += 1
            return fake_forward.temp_original(self, *args, **kwargs)
        side_effect.call_count = 0

        with fake_forward as mock_forward:
            mock_forward.side_effect = side_effect
            run_inference.main(conf)

            mapped_calls = []
            for args, kwargs in mock_forward.call_args_list:
                args = (None,) + args[1:]
                argument_binding = func_sig.bind(*args, **kwargs)
                argument_map = argument_binding.arguments
                argument_map = tensor_util.cpu(argument_map)
                mapped_calls.append(argument_map)
        
        is_motif = 1
        def constant(mapped_call):
            c = {}
            c['xyz'] = mapped_call['xyz'][0,is_motif]
            is_sidechain_torsion = torch.ones(3*rf2aa.chemical.NTOTALDOFS).bool()
            is_sidechain_torsion[0:2] = False
            is_sidechain_torsion[3:5] = False
            c['alpha'] = mapped_call['alpha_t'][0,0,is_motif]
            # Remove backbone torsions
            c['alpha'][~is_sidechain_torsion] = torch.nan
            c['sctors'] = mapped_call['sctors'][0, is_motif]
            # Remove backbone torsions
            c['sctors'][0:2] = torch.nan
            return c
        
        constants = []
        for mapped_call in mapped_calls:
            constants.append(constant(mapped_call))
        
        self.assertEqual(len(constants), T)
        cmp = partial(tensor_util.cmp, atol=1e-9, rtol=1e-4)
        for i in range(1, T):
            test_utils.assertEqual(self, cmp, constants[0], constants[i])
    
    def test_motif_fixed_in_output(self):
        output_pdb, conf = infer([
            'diffuser.T=3',
            'inference.num_designs=1',
            'inference.output_prefix=tmp/test_2',
            "contigmap.contigs=['1,A518-519,1']",
            'inference.design_startnum=0',
            'contigmap.length=4-4'
        ])

        input_feats = inference.utils.parse_pdb(conf.inference.input_pdb)
        output_feats = inference.utils.parse_pdb(output_pdb)

        trb = get_trb(conf)
        is_motif = torch.tensor(trb['con_hal_idx0'])
        is_motif_ref = torch.tensor(trb['con_ref_idx0'])
        n_motif = len(is_motif)
        
        input_motif_xyz = input_feats['xyz'][is_motif_ref]
        output_motif_xyz = output_feats['xyz'][is_motif]
        atom_mask = input_feats['mask'][is_motif_ref] # rk. should this be is_motif_ref?
        self.assertEqual(n_motif, 2)

        # Backbone only
        backbone_atom_mask = torch.zeros((n_motif, 14)).bool()
        backbone_atom_mask[:,:3] = True
        backbone_rmsd = rf2aa.loss.calc_crd_rmsd(
                torch.tensor(input_motif_xyz)[None],
                torch.tensor(output_motif_xyz)[None],
                backbone_atom_mask[None])
        # The motif gets rotated and translated, so the accuracy is somewhat limited
        # due to the precision of coordinates in a PDB file.
        self.assertLess(backbone_rmsd, 0.02)

        # All atoms
        rmsd = rf2aa.loss.calc_crd_rmsd(
                torch.tensor(input_motif_xyz)[None],
                torch.tensor(output_motif_xyz)[None],
                torch.tensor(atom_mask)[None])
        self.assertLess(rmsd, 0.02)

    def test_partial_sidechain(self):
        """
        test that network atomizes protein
        """
        output_pdb, conf = infer([
            'diffuser.T=3',
            'inference.num_designs=1',
            'inference.output_prefix=tmp/test_2',
            "contigmap.contigs=['1,A518-518,1']",
            "+contigmap.contig_atoms=\"{'A518':'CG,OD1,OD2'}\"",
            'inference.design_startnum=0',
            'inference.atomized_output=True',
            'contigmap.length=3-3'
        ])

        input_feats = inference.utils.parse_pdb(conf.inference.input_pdb)
        output_feats = inference.utils.parse_pdb(output_pdb)

        trb = get_trb(conf)
        is_motif = torch.tensor(trb['con_hal_idx0'])
        is_motif_ref = torch.tensor(trb['con_ref_idx0'])
        
        input_motif_xyz = input_feats['xyz'][is_motif_ref]
        atom_mask = input_feats['mask'][is_motif_ref]
        output_xyz = trb['indep']['xyz']
        is_sm = trb['indep']['is_sm']
        
        residue_to_atomize = input_motif_xyz[atom_mask]
        post_atomized = output_xyz[is_sm, 1]
        self.assertEqual(residue_to_atomize.shape, post_atomized.shape)
        # hard coded to choose the atoms that are the motif 
        rmsd, _ = rf2aa.util.kabsch(torch.tensor(residue_to_atomize)[-3:], torch.tensor(post_atomized)[-3:])
        self.assertLess(rmsd, 0.02)


if __name__ == '__main__':
        unittest.main()
