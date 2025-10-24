
import logging
LOGGER = logging.getLogger(__name__)
import copy
import os
import sys
import unittest
import json

import torch
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RF2-allatom'))
from aa_model import Model, make_indep
import inference.utils
import contigs
import atomize
from rf2aa import tensor_util

class TestAtomization(unittest.TestCase):

    def test_atomization_table(self):
        testcases = [
            (
                {'contigs': ['A517-518']},
                [0]
            ),
            (
                {'contigs': ['A518-527']},
                [0,2,4],
            ),
            # Should fail as you cannot atomize C-terminal residues
            # (
            #     {'contigs': ['A517-518']},
            #     [1]
            # ),
        ]

        for contig_kwargs, atomize_indices in testcases:
            test_pdb = 'benchmark/input/gaa.pdb'
            target_feats = inference.utils.process_target(test_pdb)
            contig_map =  contigs.ContigMap(target_feats,
                                        **contig_kwargs
                                        )
            indep = make_indep(test_pdb, 'LG1')
            adaptor = Model({})
            indep, _, _ = adaptor.insert_contig(indep, contig_map)
            indep.xyz = atomize.set_nonexistant_atoms_to_nan(indep.xyz, indep.seq)
            indep_init = copy.deepcopy(indep)

            L = indep.seq.shape[0]
            is_residue_atomized = torch.zeros(L).bool()
            is_residue_atomized[atomize_indices] = True

            # Atomize
            indep_atomized, atomizer = atomize.atomize(indep, is_residue_atomized)

            # Deatomize
            indep_deatomized = atomize.deatomize(atomizer, indep_atomized)
            
            # Compare original indep with deatomize(atomize(indep))
            diff = cmp_pretty(indep_deatomized, indep_init)
            if diff:
                self.fail(f'{contig_kwargs=} {atomize_indices=} {diff=}')

def cmp_pretty(got, want, **kwargs):
    diff = tensor_util.cmp(got, want, **kwargs)
    if not diff:
        return
    try:
        jsoned = diff.to_json()
        loaded = json.loads(jsoned)
        return json.dumps(loaded, indent=4)
    except Exception as e:
        LOGGER.debug('failed to pretty print output', e)
        return json.dumps(diff.pop('tensors unequal', ''), indent=4) + '\n' + str(diff) 

if __name__ == '__main__':
        unittest.main()