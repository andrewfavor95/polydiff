import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RF2-allatom'))
from rf2aa.RoseTTAFoldModel import RoseTTAFoldModule
import torch
import shlex
import unittest
import subprocess
from pathlib import Path
from unittest import mock
from icecream import ic
import run_inference
from deepdiff import DeepDiff
from rf2aa import tensor_util

from inspect import signature

import test_utils
import train_multi_deep

REWRITE = False
class CallException(Exception):
    pass

class TestRegression(unittest.TestCase):
    
    # Example regression test.
    def test_regression(self):
        # This test must be run on a CPU.
        assert torch.cuda.device_count() == 0
        run_inference.make_deterministic()
        # Uncomment to assert test is checking the appropriate things.
        # run_inference.make_deterministic(1)
        path = 'goldens'
        from arguments import get_args
        argString = "-p_drop 0.15 -accum 2 -crop 100 -w_disp 0.5 -w_frame_dist 1.0 -w_aa 0 -w_blen 0.0 -w_bang 0.0 -w_lj 0.0 -w_hb 0.0 -w_str 0.0 -maxlat 256 -maxseq 1024 -num_epochs 2 -lr 0.0005 -seed 42 -seqid 150.0 -mintplt 1 -use_H -max_length 100 -max_complex_chain 250 -task_names diff,seq2str -task_p 1.0,0.0 -diff_T 200 -aa_decode_steps 0 -wandb_prefix debug_sm_conditional -diff_so3_type igso3 -diff_chi_type interp -use_tschedule -maxcycle 1 -diff_b0 0.01 -diff_bT 0.07 -diff_schedule_type linear -prob_self_cond 0.5 -str_self_cond -dataset pdb_aa,sm_complex -dataset_prob 0.0,1.0 -sidechain_input False -motif_sidechain_input True -ckpt_load_path /home/ahern/projects/rf_diffusion/train_session2023-01-09_1673291857.7027779/models/BFF_4.pt -d_t1d 22 -new_self_cond -diff_crd_scale 0.25 -metric displacement -metric contigs -diff_mask_probs get_triple_contact:1.0 -w_motif_disp 10     -data_pkl test_dataset_100.pkl -data_pkl_aa test_allatom_dataset_100.pkl     -n_extra_block 4     -n_main_block 32     -n_ref_block 4     -n_finetune_block 0     -ref_num_layers 2     -d_pair 192     -n_head_pair 6     -freeze_track_motif     -interactive     -n_write_pdb 1 -zero_weights     -debug"
        split_args = shlex.split(argString)
        all_args = get_args(split_args)

        func_sig = signature(RoseTTAFoldModule.forward)
        with mock.patch.object(RoseTTAFoldModule, "forward") as submethod_mocked:
            submethod_mocked.side_effect = CallException('Function called!')
            train = train_multi_deep.make_trainer(*all_args)
            try:
                train.run_model_training(torch.cuda.device_count())
            except CallException:
                print("Called!")

            args, kwargs = submethod_mocked.call_args
            args = (None,) + args
            argument_binding = func_sig.bind(*args, **kwargs)
            argument_map = argument_binding.arguments
            argument_map = tensor_util.cpu(argument_map)
            test_utils.assert_matches_golden(self, 'model_input_0', argument_map, rewrite=REWRITE, custom_comparator=tensor_util.cmp)

if __name__ == '__main__':
        unittest.main()
