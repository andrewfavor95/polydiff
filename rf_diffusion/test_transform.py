import copy
import os
import sys
import unittest
import json

import assertpy
import torch
from icecream import ic

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RF2-allatom'))
from aa_model import Model, make_indep
import inference.utils
import contigs
import atomize
from rf2aa import tensor_util
import guide_posts as gp
import aa_model
import test_utils
from argparse import Namespace
ic.configureOutput(includeContext=True)

class TestTransform(unittest.TestCase):
    
    def test_atomized_placement_agnostic(self):
        testcases = [
            (
                {'contigs': ['A508-511']},
                None,
                {2: ['CB', 'CG1', 'CG2']},
                [],
                ['ASP', 'GLN', 'VAL', 'PRO', 'N', 'C', 'C', 'O', 'C', 'C', 'C']
            ),
            (
                {'contigs': ['A508-511']},
                'LG1',
                {2: ['CB', 'CG1', 'CG2']},
                [1],
                ['ASP', 'GLN', 'VAL', 'PRO', 'C', 'O', 'C', 'O', 'C', 'O', 'C', 'O', 'C', 'O', 'C', 'O', 'C', 'O', 'C', 'O', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'GLN', 'N', 'C', 'C', 'O', 'C', 'C', 'C']
            ),
        ]

        for contig_kwargs, ligand, is_atom_str_shown, res_str_shown_idx, want_seq in testcases:
            test_pdb = 'benchmark/input/gaa.pdb'
            target_feats = inference.utils.process_target(test_pdb)
            contig_map =  contigs.ContigMap(target_feats,
                                        **contig_kwargs
                                        )
            indep = make_indep(test_pdb, ligand)
            adaptor = Model({})
            indep, _, _ = adaptor.insert_contig(indep, contig_map) # ['ASP', 'GLN', 'VAL', 'PRO']
            indep.xyz = atomize.set_nonexistant_atoms_to_nan(indep.xyz, indep.seq)
            indep_init = copy.deepcopy(indep)
            is_res_str_shown = torch.zeros(indep.length()).bool()
            is_res_str_shown[res_str_shown_idx] = True
            n_res_shown = is_res_str_shown.sum()
            indep, is_diffused, is_masked_seq, atomizer, _ = aa_model.transform_indep(indep, is_res_str_shown, is_atom_str_shown, True)
            is_diffused_deatomized = atomize.convert_atomized_mask(atomizer, is_diffused)

            n_ligand = indep_init.is_sm.sum()
            n_motif = sum(len(v) for v in is_atom_str_shown.values()) + n_res_shown + n_ligand
            n_valine = 7
            assertpy.assert_that(indep.length()).is_equal_to(n_valine + indep_init.length() + n_res_shown)
            assertpy.assert_that((~is_diffused).sum()).is_equal_to(n_motif)
            assertpy.assert_that((~is_masked_seq).sum()).is_equal_to(n_valine + n_res_shown + n_ligand)
            assertpy.assert_that(indep.is_sm.sum()).is_equal_to(n_valine + n_ligand)
            tensor_util.assert_equal(indep.bond_feats, indep.bond_feats.T)
            n_gp_atomized = n_valine + n_res_shown
            is_gp = torch.zeros(indep.length()).bool()
            is_gp[-n_gp_atomized:] = True
            self_bonds = indep.bond_feats[is_gp][:, is_gp]
            is_gp_receptor = ~is_gp.clone() * ~indep.is_sm
            other_bonds = indep.bond_feats[is_gp][:, is_gp_receptor]
            assert torch.all(other_bonds == 7), other_bonds
            CB = 4
            CG1 = 5
            assertpy.assert_that(indep.human_readable_seq()).is_equal_to(want_seq)
            atomized_res_bonds = indep.bond_feats[indep.is_sm * is_gp][:, indep.is_sm * is_gp]
            assertpy.assert_that(atomized_res_bonds[CB, CG1].item()).is_equal_to(1)
            
            # Deatomize
            indep_deatomized = atomize.deatomize(atomizer, indep)
            n_gp = len(is_atom_str_shown) + n_res_shown
            want_same_chain = torch.ones(indep_deatomized.length()).bool()
            want_same_chain[-n_gp:] = False
            want_same_chain[indep_deatomized.is_sm] = False
            assertpy.assert_that(indep_deatomized.same_chain[0].tolist()).is_equal_to(want_same_chain.tolist())
            is_gp = torch.zeros(indep_deatomized.length()).bool()
            is_gp[-n_gp:] = True
            aa_model.pop_mask(indep_deatomized, ~is_gp)

            diff = test_utils.cmp_pretty(indep_deatomized, indep_init)
            if diff:
                print(diff)
                self.fail(f'{contig_kwargs=} {diff=}')


if __name__ == '__main__':
        unittest.main()
