import copy
import os
import sys
import unittest
import json

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

class TestGuidepost(unittest.TestCase):

    def test_guidepost_appends_only(self):
        testcases = [
            (
                {'contigs': ['A510-515']},
                None,
            ),
            (
                {'contigs': ['A510-515']},
                'LG1',
            ),
        ]

        for contig_kwargs, ligand in testcases:
            test_pdb = 'benchmark/input/gaa.pdb'
            target_feats = inference.utils.process_target(test_pdb)
            contig_map =  contigs.ContigMap(target_feats,
                                        **contig_kwargs
                                        )
            indep = make_indep(test_pdb, ligand)
            adaptor = Model({})
            indep, _, _ = adaptor.insert_contig(indep, contig_map)
            ic(indep.chirals.shape)
            indep.xyz = atomize.set_nonexistant_atoms_to_nan(indep.xyz, indep.seq)
            indep_init = copy.deepcopy(indep)

            is_ptn = torch.zeros(indep.length()).bool()
            is_ptn[[1,3]] = True
            indep_gp, _, gp_to_ptn_idx0 = gp.make_guideposts(indep, is_ptn, placement='anywhere')
            is_gp = torch.zeros(indep_gp.length()).bool()
            for k in gp_to_ptn_idx0:
                is_gp[k] = True
            
            indep_ungp = copy.deepcopy(indep_gp)
            aa_model.pop_mask(indep_ungp, ~is_gp)
            ic(indep_ungp.length())
            

            diff = test_utils.cmp_pretty(indep_ungp, indep_init)
            if diff:
                print(diff)
                self.fail(f'{contig_kwargs=} {diff=}')

if __name__ == '__main__':
        unittest.main()
