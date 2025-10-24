import itertools
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RF2-allatom'))

import unittest
from icecream import ic
import torch
import numpy as np

import inference.utils
import aa_model
import contigs
import bond_geometry
import perturbations
import atomize

# def is_se3_invariant(loss, true, pred):

class TestLoss(unittest.TestCase):

    def test_atom_bond_loss(self):
        test_pdb = 'benchmark/input/gaa.pdb'
        
        target_feats = inference.utils.process_target(test_pdb)
        contig_map =  contigs.ContigMap(target_feats,
                                       contigs=['2,A518-518'],
                                       contig_atoms="{'A518':'CA,C,N,O,CB,CG,OD1,OD2'}",
                                       length='3-3',
                                       )
        indep = aa_model.make_indep(test_pdb)
        adaptor = aa_model.Model({})
        indep_contig,is_diffused,_ = adaptor.insert_contig(indep, contig_map)

        true = indep_contig.xyz

        perturbed = perturbations.se3_perturb(true)

        expected_losses = list(f'{a}:{b}' for a,b in itertools.combinations_with_replacement(
            ['diffused_residue', 'motif_atom'], 2))
        bond_losses = bond_geometry.calc_atom_bond_loss(indep_contig, perturbed, is_diffused)
        for k in expected_losses:
            self.assertLess(bond_losses.pop(k), 1e-6, msg=k)
        for k, v in bond_losses.items():
            self.assertTrue(torch.isnan(v), msg=k)
        
        perturbed = true.clone()
        T = torch.tensor([1,1,1])
        perturbed[-1,1,:] += T
        bond_losses = bond_geometry.calc_atom_bond_loss(indep_contig, perturbed, is_diffused)
        should_change = 'motif_atom:motif_atom'
        bond_loss = bond_losses.pop(should_change)
        self.assertGreater(bond_loss, 0.1)
        for k in expected_losses:
            if k == should_change:
                continue
            self.assertLess(bond_losses.pop(k), 1e-6, msg=k)
        for k, v in bond_losses.items():
            self.assertTrue(torch.isnan(v), msg=k)

    def test_rigid_loss(self):
        test_pdb = 'benchmark/input/gaa.pdb'
        
        target_feats = inference.utils.process_target(test_pdb)
        contig_map =  contigs.ContigMap(target_feats,
                                       contigs=['2,A518-518'],
                                       contig_atoms="{'A518':'CG,OD1,OD2'}",
                                       length='3-3',
                                       )
        indep = aa_model.make_indep(test_pdb)
        adaptor = aa_model.Model({})
        indep_contig,is_diffused,_ = adaptor.insert_contig(indep, contig_map)

        true = indep_contig.xyz
        perturbed = perturbations.se3_perturb(true)
        T = torch.tensor([1,1,1])
        cg_atomized_idx = atomize.atomized_indices_atoms(adaptor.atomizer, {2: ['CG']})
        perturbed[cg_atomized_idx,1,:] += T

        rigid_losses = bond_geometry.calc_rigid_loss(indep_contig, perturbed, is_diffused)
        self.assertLess(rigid_losses['diffused_atom'], 1e-6)
        self.assertGreater(rigid_losses['diffused_atom:motif_atom_determined'], 1)



if __name__ == '__main__':
        unittest.main()

