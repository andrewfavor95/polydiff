import os
import sys
import dataclasses
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RF2-allatom'))
from rf2aa.RoseTTAFoldModel import RoseTTAFoldModule as RoseTTAFoldModuleReal
from rf2aa.RoseTTAFoldModel import RoseTTAFoldModule
import torch
from torch import tensor
import shlex
import unittest
import ast
import subprocess
from pathlib import Path
from unittest import mock
from icecream import ic
import run_inference
from deepdiff import DeepDiff
from rf2aa import tensor_util
import torch.distributed
from inspect import signature
from unittest.mock import MagicMock

import test_utils
import train_multi_deep
import aa_model

from functools import partial

REWRITE = False
class CallException(Exception):
    pass

item = """{
    'chosen_dataset': 'sm_complex',                                                                                                          
    'mask_gen_seed': 65682867,                                                                                                               
    'sel_item': {   'ASSEMBLY': 1,                                                                                                           
                    'CHAINID': '5neg_A',                                                                                                     
                    'CLUSTER': 22021,
                    'COVALENT': [],
                    'DEPOSITION': '2017-03-10',          
                    'HASH': '030830',        
                    'LEN_EXIST': 109,                                                                                                        
                    'LIGAND': [('C', '201', '8VK')],        
                    'LIGATOMS': 52,            
                    'LIGATOMS_RESOLVED': 52,
                    'LIGXF': [('C', 2)],
                    'PARTNERS': [   (   'A',
                                        0,
                                        264, 
                                        2.5206124782562256,
                                        'polypeptide(L)'),
                                    (   [('E', '203', 'NO3')],
                                        [('E', 4)],
                                        0,
                                        8.707673072814941,
                                        'nonpoly'),
                                    (   'B',
                                        1,
                                        0,
                                        10.479164123535156,
                                        'polypeptide(L)'),
                                    (   [('G', '205', 'NO3')],
                                        [('G', 6)],
                                        0,
                                        16.69165802001953,                                                                                   
                                        'nonpoly'),
                                    (   [('D', '202', 'NO3')],
                                        [('D', 3)],
                                        0, 
                                        17.07767105102539,
                                        'nonpoly'),                                                                                          
                                    (   [('F', '204', 'NO3')],
                                        [('F', 5)],
                                        0,
                                        20.71928596496582,
                                        'nonpoly'),
                                    (   [('I', '202', 'NO3')],
                                        [('I', 8)],
                                        0,
                                        27.311317443847656,
                                        'nonpoly')],
                    'RESOLUTION': 1.29,        
                    'SEQUENCE': 'GSMSEQSICQARAAVMVYDDANKKWVPAGGSTGFSRVHIYHHTGNNTFRVVGRKIQDHQVVINCAIPKGLKYNQATQTFHQWRDARQVYGLNFGSKEDANVFASAMMHALEVL'},
    'task': 'diff'
}"""

# aa_pickle = "/projects/ml/RF2_allatom/dataset_diffusion_20230201_taxid.pkl"
mol_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data/pkl')
aa_pickle = "test_fake_allatom_dataset.pkl"
arg_string = f"-p_drop 0.15 -accum 2 -crop 161 -w_disp 0.5 -w_frame_dist 1.0 -w_aa 0 -w_blen 0.0 -w_bang 0.0 -w_lj 0.0 -w_hb 0.0 -w_str 0.0 -maxlat 256 -maxseq 1024 -num_epochs 2 -lr 0.0005 -seed 42 -seqid 150.0 -mintplt 1 -use_H -max_length 100 -max_complex_chain 250 -task_names diff,seq2str -task_p 1.0,0.0 -diff_T 200 -aa_decode_steps 0 -wandb_prefix debug_sm_conditional -diff_so3_type igso3 -diff_chi_type interp -use_tschedule -maxcycle 1 -diff_b0 0.01 -diff_bT 0.07 -diff_schedule_type linear -prob_self_cond 0.00001 -str_self_cond -dataset pdb_aa,sm_complex -dataset_prob 0.0,1.0 -sidechain_input False -motif_sidechain_input True -ckpt_load_path /home/ahern/projects/rf_diffusion/train_session2023-01-09_1673291857.7027779/models/BFF_4.pt -d_t1d 22 -new_self_cond -diff_crd_scale 0.25 -metric displacement -metric contigs -diff_mask_probs get_triple_contact:1.0 -w_motif_disp 10     -data_pkl test_dataset_100.pkl -data_pkl_aa {aa_pickle}     -n_extra_block 4     -n_main_block 32     -n_ref_block 4     -n_finetune_block 0     -ref_num_layers 2     -d_pair 192     -n_head_pair 6     -freeze_track_motif     -interactive     -n_write_pdb 1 -zero_weights     -debug -spoof_item \"{item}\" -p_uncond 0 -mol_dir {mol_dir}"

class TestDistributed(unittest.TestCase):

    def test_distributed_teardown(self):
        """
        This test passes, but destroy_process_group does not work in practice, so we set the MASTER_PORT explicitly.
        """
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12317'
        ic('start 1')
        torch.distributed.init_process_group(backend="gloo", world_size=1, rank=0)
        ic('end 1')
        torch.distributed.destroy_process_group()
        ic('start 2')
        torch.distributed.init_process_group(backend="gloo", world_size=1, rank=0)
        ic('end 2')
        torch.distributed.destroy_process_group()

class ModelInputs(unittest.TestCase):
    def test_no_self_cond(self):
        os.environ['MASTER_PORT'] = '12318'
        run_regression(self, arg_string, 'model_input_no_self_cond')

    def test_self_cond_tp1(self):
        os.environ['MASTER_PORT'] = '12320'
        self_cond_arg_string = arg_string[:]
        old_self_cond = '-prob_self_cond 0.00001'
        new_self_cond = '-prob_self_cond 1.0'
        # Since we are doing this the lazy way with string replacement rather
        # than using a data structure for the arguments, assert that we're actually
        # replacing something.
        assert old_self_cond in self_cond_arg_string
        self_cond_arg_string = self_cond_arg_string.replace(old_self_cond, new_self_cond)
        run_regression(self, self_cond_arg_string, 'model_input_self_cond_tp1')

    def test_self_cond_t(self):
        os.environ['MASTER_PORT'] = '12321'
        self_cond_arg_string = arg_string[:]
        old_self_cond = '-prob_self_cond 0.00001'
        new_self_cond = '-prob_self_cond 1.0'
        # Since we are doing this the lazy way with string replacement rather
        # than using a data structure for the arguments, assert that we're actually
        # replacing something.
        assert old_self_cond in self_cond_arg_string
        self_cond_arg_string = self_cond_arg_string.replace(old_self_cond, new_self_cond)
        run_regression(self, self_cond_arg_string, 'model_input_self_cond_t', call_number=2)

        
    def test_atomize_regression(self):
        os.environ['MASTER_PORT'] = '12321'
        self_cond_arg_string = arg_string[:]
        old_task = '-diff_mask_probs get_triple_contact:1.0'
        new_task = '-diff_mask_probs get_triple_contact_atomize:1.0'
        # Since we are doing this the lazy way with string replacement rather
        # than using a data structure for the arguments, assert that we're actually
        # replacing something.
        assert old_task in self_cond_arg_string
        self_cond_arg_string = self_cond_arg_string.replace(old_task, new_task)
        run_regression(self, self_cond_arg_string, 'model_input_atomize', call_number=3)

class DataloaderToCalcLoss(unittest.TestCase):
    def test_no_self_cond_loss(self):
        os.environ['MASTER_PORT'] = '12319'
        run_regression(self, arg_string, 'loss_no_self_cond', call_number=2, assert_loss=True)


# Example regression test, with one faked return value from the model.
def run_regression(self, arg_string, golden_name, call_number=1, assert_loss=False):
    # This test must be run on a CPU.
    assert torch.cuda.device_count() == 0
    run_inference.make_deterministic()
    # Uncomment to assert test is checking the appropriate things.
    # run_inference.make_deterministic(1)
    from arguments import get_args

    split_args = shlex.split(arg_string)
    all_args = get_args(split_args)

    func_sig = signature(RoseTTAFoldModule.forward)
    train = train_multi_deep.make_trainer(*all_args)
    loss_func_sig = signature(train.calc_loss)
    with mock.patch.object(train, "init_model") as submethod_mocked:
        with mock.patch.object(train, "calc_loss") as calc_loss_mocked:

            def side_effect(*args, **kwargs):
                side_effect.call_count += 1
                if side_effect.call_count < call_number:
                    ic(kwargs.keys())
                    run_inference.make_deterministic()
                    if 'xyz' in kwargs:
                        xyz = kwargs['xyz']
                    else:
                        xyz = args[4] # [1, L, 36 3]
                    L = xyz.shape[1]
                    px0_xyz = xyz[None,:,:,:3].repeat(40, 1, 1, 1, 1)
                    px0_xyz = torch.normal(0, 1, px0_xyz.shape) + px0_xyz
                    logits = (
                        torch.normal(0, 1, (1, 61, L, L)),
                        torch.normal(0, 1, (1, 61, L, L)),
                        torch.normal(0, 1, (1, 37, L, L)),
                        torch.normal(0, 1, (1, 19, L, L)),
                    )
                    logits_aa = torch.normal(0, 1, (1, 80, L))
                    logits_pae = torch.normal(0, 1, (1, 64, L, L))
                    logits_pde = torch.normal(0, 1, (1, 64, L, L))
                    alpha_s = torch.normal(0, 1, (40, 1, L, 20, 2))
                    xyz_allatom = torch.normal(0, 1, (1, L, 36, 3))
                    lddt = torch.normal(0, 1, (1, 50, L))
                    return logits, logits_aa, logits_pae, logits_pde, px0_xyz, alpha_s, xyz_allatom, lddt, None, None, None
                else:
                    raise CallException('called')
            side_effect.call_count = 0
            mymock_method = mock.MagicMock()
            mymock_method.side_effect = side_effect
            calc_loss_mocked.side_effect = CallException('called calc_loss')

            submethod_mocked.return_value = mymock_method, mock.MagicMock(), mock.MagicMock(), mock.MagicMock(), 0
            train.group_name = golden_name
            try:
                train.run_model_training(torch.cuda.device_count())
            except CallException:
                print("Called!")
            torch.distributed.destroy_process_group()

            mocked_method = mymock_method
            if assert_loss:
                mocked_method = calc_loss_mocked
                func_sig = loss_func_sig

            args, kwargs = mocked_method.call_args
            if not assert_loss:
                args = (None,) + args
            argument_binding = func_sig.bind(*args, **kwargs)
            argument_map = argument_binding.arguments
            argument_map = tensor_util.cpu(argument_map)
            cmp = partial(tensor_util.cmp, atol=1e-3, rtol=0)
            test_utils.assert_matches_golden(self, golden_name, argument_map, rewrite=REWRITE, custom_comparator=cmp)

class Loss(unittest.TestCase):
    def test_loss_grad(self):
        os.environ['MASTER_PORT'] = '12400'
        self.run_regression_loss_grad(arg_string, 'loss_grad_no_self_cond', call_number=2, assert_loss=True)

    # Example regression test, with one faked return value from the model.
    def run_regression_loss_grad(self, arg_string, golden_name, call_number=1, assert_loss=False):
        # This test must be run on a CPU.
        assert torch.cuda.device_count() == 0
        run_inference.make_deterministic()
        from arguments import get_args

        split_args = shlex.split(arg_string)
        all_args = get_args(split_args)

        func_sig = signature(RoseTTAFoldModule.forward)
        train = train_multi_deep.make_trainer(*all_args)
        loss_func_sig = signature(train.calc_loss)
        with mock.patch.object(train, "init_model") as submethod_mocked:
            def side_effect(*args, **kwargs):
                side_effect.call_count += 1
                if side_effect.call_count < call_number:
                    ic(kwargs.keys())
                    if 'xyz' in kwargs:
                        xyz = kwargs['xyz']
                    else:
                        xyz = args[4] # [1, L, 36 3]
                    L = xyz.shape[1]
                    px0_xyz = xyz[None,:,:,:3].repeat(40, 1, 1, 1, 1)
                    px0_xyz = torch.normal(0, 1, px0_xyz.shape) + px0_xyz
                    logits = (
                        torch.normal(0, 1, (1, 61, L, L)),
                        torch.normal(0, 1, (1, 61, L, L)),
                        torch.normal(0, 1, (1, 37, L, L)),
                        torch.normal(0, 1, (1, 19, L, L)),
                    )
                    logits_aa = torch.normal(0, 1, (1, 80, L))
                    logits_pae = torch.normal(0, 1, (1, 64, L, L))
                    logits_pde = torch.normal(0, 1, (1, 64, L, L))
                    alpha_s = torch.normal(0, 1, (40, 1, L, 20, 2))
                    xyz_allatom = torch.normal(0, 1, (1, L, 36, 3))
                    lddt = torch.normal(0, 1, (1, 50, L))
                    ic(xyz_allatom.requires_grad)
                    xyz_allatom.requires_grad = True
                    side_effect.rfo = aa_model.RFO(logits, logits_aa, logits_pae, logits_pde, px0_xyz, alpha_s, xyz_allatom, lddt, None, None, None)
                    side_effect.rfo = tensor_util.to_ordered_dict(side_effect.rfo)
                    tensor_util.require_grad(side_effect.rfo)
                    return side_effect.rfo.values()
                else:
                    raise CallException('called')
            side_effect.call_count = 0
            mymock_method = mock.MagicMock()
            mymock_method.side_effect = side_effect

            scaler_mock = mock.MagicMock()
            scaler_mock.scale.side_effect = CallException('called scaler')
            submethod_mocked.return_value = mymock_method, mock.MagicMock(), mock.MagicMock(), scaler_mock, 0
            train.group_name = golden_name
            try:
                train.run_model_training(torch.cuda.device_count())
            except CallException as e:
                print("CalledException", e)
            torch.distributed.destroy_process_group()

            ic(scaler_mock.scale.call_args)
            ic(len(scaler_mock.scale.call_args))
            ic(type(scaler_mock.scale.call_args))

            (loss,), _ = scaler_mock.scale.call_args
            ic(loss)
            rfo = side_effect.rfo
            run_inference.seed_all()
            loss.backward()
            grads = tensor_util.get_grad(rfo)
            print(f'grad shapes: {tensor_util.info(grads)}')
            print(f'grad (min, max): {tensor_util.minmax(grads)}')
            print(f'loss: {loss}')

            cmp = partial(tensor_util.cmp, atol=1e-9, rtol=1e-2)
            test_utils.assert_matches_golden(self, golden_name, grads, rewrite=REWRITE, custom_comparator=cmp)

if __name__ == '__main__':
        unittest.main()
