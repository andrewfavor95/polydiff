#!/usr/bin/env python
#
# Takes a folder of pdb & trb files, generates MPNN features (fixing AAs at
# contig positions), makes list of MPNN jobs on batches of those designs,
# and optionally submits slurm array job and outputs job ID
# 
from collections import defaultdict
import sys, os, argparse, itertools, json, glob
import numpy as np
import copy

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir,'util'))
import slurm_tools

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir',type=str,help='Folder of designs to score')
    parser.add_argument('--chunk',type=int,default=-1,help='How many designs to process in each job')
    parser.add_argument('-p', type=str, default='gpu',help='-p argument for slurm (partition)')
    parser.add_argument('-J', type=str, help='name of slurm job')
    parser.add_argument('--gres', type=str, default='gpu:a4000:1',help='--gres argument for slurm, e.g. gpu:rtx2080:1')
    parser.add_argument('--no_submit', dest='submit', action="store_false", default=True, help='Do not submit slurm array job, only generate job list.')
    parser.add_argument('--cautious', action="store_true", default=False, help='Skip design if output file exists')
    parser.add_argument('--no_logs', dest='keep_logs', action="store_false", default=True, help='Don\'t keep slurm logs.')
    parser.add_argument('--num_seq_per_target', default=8,type=int, help='How many mpnn sequences per design? Default = 8')
    parser.add_argument('--in_proc', dest='in_proc', action="store_true", default=False, help='Do not submit slurm array job, run on current node.')
    parser.add_argument('--use_ligand', action="store_true", default=False, help='Use ligandMPNN.')
    args, unknown = parser.parse_known_args()
    if len(unknown)>0:
        print(f'WARNING: Unknown arguments {unknown}')
    return args

def main():
    args = get_args()
    filenames = glob.glob(args.datadir+'/*.pdb')
    
    if not args.use_ligand:
        run_mpnn(args, filenames)
        return

    filenames_by_ligand_presence = defaultdict(list)
    for fn in filenames:
        trb_path = os.path.splitext(fn)[0] + '.trb'
        trb = np.load(trb_path,allow_pickle=True)
        has_ligand = bool(trb['config']['inference']['ligand'])
        filenames_by_ligand_presence[has_ligand].append(fn)
    
    for use_ligand, filenames in filenames_by_ligand_presence.items():
        args_for_mpnn_flavor = copy.deepcopy(args)
        args_for_mpnn_flavor.use_ligand = use_ligand
        run_mpnn(args_for_mpnn_flavor, filenames)

def run_mpnn(args, filenames):

    mpnn_flavor = 'mpnn'
    if args.use_ligand:
        mpnn_flavor = 'ligmpnn'
    mpnn_folder = args.datadir+f'/{mpnn_flavor}/'
    os.makedirs(mpnn_folder, exist_ok=True)

    
    # skip designs that have already been done
    if args.cautious:
        filenames = [fn for fn in filenames 
            if not os.path.exists(mpnn_folder+'/seqs/'+os.path.basename(fn).replace('.pdb','.fa'))]

    if args.chunk == -1:
        args.chunk = len(filenames)

    # run parser script
    job_fn = args.datadir + f'/jobs.{mpnn_flavor}.parse.list'
    job_list_file = open(job_fn, 'w') if args.submit else sys.stdout
    if args.use_ligand:
        parse_script = f'{script_dir}/util/parse_multiple_chains_ligand.py'
    else:
        parse_script = f'{script_dir}/util/parse_multiple_chains.py'

    for i in range(0, len(filenames), args.chunk):
        with open(mpnn_folder+f'parse_multiple_chains.list.{i}','w') as outf:
            for fn in filenames[i:i+args.chunk]:
                print(fn,file=outf)
        print(f'python {parse_script} --input_files {mpnn_folder}/parse_multiple_chains.list.{i} '\
              f'--datadir {args.datadir} '\
              f'--output_parsed {mpnn_folder}/pdbs_{i}.jsonl '\
              f'--output_fixed_pos {mpnn_folder}/pdbs_position_fixed_{i}.jsonl', file=job_list_file)
    if args.submit: job_list_file.close()

    # submit to slurm
    if args.submit:
        pre = 'ligmpnn_pre' if args.use_ligand else 'mpnn_pre'
        slurm_job, proc = slurm_tools.array_submit(job_fn, p='cpu', gres=None, J=pre, log=args.keep_logs, in_proc=args.in_proc)
        print(f'Submitted array job {slurm_job} with {int(np.ceil(len(filenames)/args.chunk))} jobs to preprocess {len(filenames)} designs for MPNN')

        prev_job = slurm_job
    else:
        prev_job = None

    job_fn = args.datadir + '/jobs.mpnn.list'
    job_list_file = open(job_fn, 'w') if args.submit else sys.stdout
    if args.use_ligand:
        mpnn_script = '/net/databases/mpnn/github_repo/ligandMPNN/protein_mpnn_run.py'
        model_name = 'v_32_020'
    else:
        mpnn_script = '/net/databases/mpnn/github_repo/protein_mpnn_run.py'
        model_name = 'v_48_020'

    for i in range(0, len(filenames), args.chunk):
        print(f'source activate SE3-nvidia; python {mpnn_script} '\
              f'--model_name "{model_name}" '\
              f'--jsonl_path {mpnn_folder}pdbs_{i}.jsonl '\
              f'--fixed_positions_jsonl {mpnn_folder}pdbs_position_fixed_{i}.jsonl '\
              f'--out_folder {mpnn_folder} '\
              f'--num_seq_per_target  {args.num_seq_per_target} '\
              f'--sampling_temp="0.1" '\
              f'--batch_size {8 if args.num_seq_per_target > 8 else args.num_seq_per_target} '\
              f'--omit_AAs XC',
              file=job_list_file)
    if args.submit: job_list_file.close()

    # submit job
    if args.submit:
        if args.J is not None:
            job_name = args.J
        else:
            pre = 'ligmpnn_' if args.use_ligand else 'mpnn_'
            job_name = pre + os.path.basename(args.datadir.strip('/'))
        slurm_job, proc = slurm_tools.array_submit(job_fn, p = args.p, gres=args.gres, log=args.keep_logs, J=job_name, wait_for=[prev_job], in_proc=args.in_proc)
        print(f'Submitted array job {slurm_job} with {int(np.ceil(len(filenames)/args.chunk))} jobs to MPNN {len(filenames)} designs')

if __name__ == "__main__":
    main()
