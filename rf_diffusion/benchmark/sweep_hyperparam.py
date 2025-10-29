#!/usr/bin/env python
#
# Generates and slurm array jobs for hyperparameter sweeps on design
# scripts, optionally submits array job and outputs slurm job ID
#

import logging
LOGGER = logging.getLogger(__name__)
import sys, os, argparse, itertools, json, shutil, re
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.realpath(__file__))+'/'
sys.path.append(script_dir+'util/')
import slurm_tools
import assertpy

def split_string_with_parentheses(string, delimiter=None):
    '''
    Splits a string using on delimiter (or whitespace if delimiter is None),
    ignoring delimiters in between pairs of parentheses.
    '''
    if delimiter is None:
        is_delimiter = lambda x: x.isspace()
    else:
        is_delimiter = lambda x: x==delimiter
    result = []
    current_word = ''
    paren_count = 0
    
    for char in string:
        if char == '(':
            paren_count += 1
        elif char == ')':
            paren_count -= 1
        
        if is_delimiter(char) and paren_count == 0:
            if current_word:
                result.append(current_word)
            current_word = ''
        else:
            current_word += char
    
    if current_word:
        result.append(current_word)
    
    return result

def get_arg_combos(arg_str):
    '''
    Params:
        arg_str: key=value string like `
            c=5
            (a=1 b=2)|(a=3 b=4)
        `
    
    Returns:
        List of dictionaries like:
        `[
            {'c':'5','a':'1', 'b':'2'},
            {'c':'5','a':'3', 'b':'4'},
        ]`
    '''
    all_arg_dicts = []
    for arg in split_string_with_parentheses(arg_str):
        if arg.startswith('('):
            arg_dicts = []
            for c in split_string_with_parentheses(arg, '|'):
                assertpy.assert_that(c).starts_with('(')
                assertpy.assert_that(c).ends_with(')')
                arg_dicts.extend(get_arg_combos(c[1:-1]))
        else:
            # base case
            k, vs = arg.split('=')
            arg_dicts = []
            for v in vs.split('|'):
                arg_dicts.append({k:v})
        all_arg_dicts.append(arg_dicts)
            
    arg_dicts = [dict()]
    for sub_arg_dicts in all_arg_dicts:
        next_arg_dicts = []
        for d1 in arg_dicts:
            for d2 in sub_arg_dicts:
                next_arg_dicts.append(dict(d1, **d2))
        arg_dicts = next_arg_dicts

    return arg_dicts

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--command',type=str,help='design script to run')
    parser.add_argument('--args',type=str,nargs='+',required=True,help='string with all arguments to pass to the command, '\
                        'with pipe (|)-delimited value options for each')
    parser.add_argument('--benchmarks', type=str, nargs='+',help='Space-separated list of benchmark names, as defined in "benchmarks.json"')
    parser.add_argument('--num_per_condition', type=int, default=1,help='Number of designs to make for each condition')
    parser.add_argument('--num_per_job', type=int, default=1,help='Split runs for each condition into this many designs per job')
    parser.add_argument('-p', type=str, default='gpu',help='-p argument for slurm (partition)')
    parser.add_argument('-t', type=str, help='-t argument for slurm')
    parser.add_argument('-J', type=str, help='name of slurm job')
    parser.add_argument('--gres', type=str, default='gpu:rtx2080:1',help='--gres argument for slurm, e.g. gpu:rtx2080:1')
    parser.add_argument('--no_submit', dest='submit', action="store_false", default=True, help='Do not submit slurm array job, only generate job list.')
    parser.add_argument('--in_proc', dest='in_proc', action="store_true", default=False, help='Do not submit slurm array job, only generate job list.')
    parser.add_argument('--no_logs', dest='keep_logs', action="store_false", default=True, help='Don\'t keep slurm logs.')
    parser.add_argument('--out', type=str, default='out/out',help='Path prefix for output files')
    parser.add_argument('--benchmark_json', type=str, default='benchmarks.json', help='Path to non-standard custom json file of benchmarks')
    parser.add_argument('--use_ligand', default=False, action='store_true', help='Use LigandMPNN instead of regular MPNN.')
    parser.add_argument('--pilot', dest='pilot', action="store_true", default=False)
    parser.add_argument('--pilot_single', dest='pilot_single', action="store_true", default=False)

    args, unknown = parser.parse_known_args()
    if len(unknown)>0:
        print(f'WARNING: Unknown arguments {unknown}')

    if args.num_per_job > args.num_per_condition:
        sys.exit('ERROR: --num_per_job cannot be greater than --num_per_condition '\
                 '(different conditions can\'t be in the same job.)')
    
    if args.pilot:
        args.num_per_condition = 1
        args.num_per_job = 1

    args_vals = [] # argument names and lists of values for passing to design script

    # default design script
    if args.command is None:
        args.command = os.path.abspath(script_dir+'../run_inference.py')

    # parse pre-defined benchmarks
    print('This is benchmarks json')
    print(args.benchmark_json)
    if not args.benchmark_json.startswith('/'):
        args.benchmark_json =script_dir+args.benchmark_json
    with open(args.benchmark_json) as f: 
        benchmarks = json.load(f)
    input_path = script_dir+'input/' # prepend path to input pdbs in current repo
    benchmark_list = []
    if args.benchmarks is not None:
        if args.benchmarks[0]=='all':
            to_run = benchmarks
        else:
            to_run = args.benchmarks
        for bm in to_run:
            benchmark_list.append([
                f'inference.output_prefix={bm}',
                benchmarks[bm].replace('inference.input_pdb=','inference.input_pdb='+input_path)
            ])

    # parse names of arguments and their value options to be passed into the design script
    arg_str = ''.join(args.args)
    if '--config-name' in arg_str.split():
        raise Exception('config names must be passed like: --config-name=name_here')

    if len(benchmark_list) > 0:
        benchmark_arg_groups = []
        for benchmark in benchmark_list: # [output path, input pdb, contig spec]
            benchmark_arg_groups.append(f"({' '.join(benchmark)})")
        arg_str += ' ' + '|'.join(benchmark_arg_groups)
    arg_dicts = get_arg_combos(arg_str)

    df = pd.DataFrame.from_dict(arg_dicts, dtype=str)

    # make output folder
    os.makedirs(os.path.dirname(args.out), exist_ok=True) 
    os.makedirs(os.path.dirname(args.out)+'/input', exist_ok=True)

    def get_input_copy_path(input_pdb):
        return os.path.join(os.path.dirname(args.out), 'input', os.path.basename(input_pdb))
    for input_pdb in df['inference.input_pdb'].unique():
        shutil.copyfile(input_pdb, get_input_copy_path(input_pdb))
    
    df['inference.input_pdb'] = df['inference.input_pdb'].apply(get_input_copy_path)

    out_dir, basename = os.path.split(args.out)
    def get_output_path(row):
        output_path_components = []
        if basename != '':
            output_path_components.append(basename)
        existing_prefix = row.get('inference.output_prefix', '')
        if existing_prefix and not pd.isna(existing_prefix):
            output_path_components.append(os.path.basename(existing_prefix))

        output_path_components.append(f'cond{row.name}')
        return os.path.join(out_dir, '_'.join(output_path_components))
    df['inference.output_prefix'] = df.apply(get_output_path, axis=1)

    # output commands with all combos of argument values
    job_fn = os.path.dirname(args.out) + '/jobs.list'
    job_list_file = open(job_fn, 'w') if args.submit else sys.stdout
    for _, arg_row in df.iterrows():
        arg_dict = arg_row.dropna().to_dict()
        combo = []
        for k,v in arg_dict.items():
            combo.append(f'{k}={v}')
        extra_args = ' '.join(combo)

        for istart in np.arange(0, args.num_per_condition, args.num_per_job):
            log_fn = f'{arg_row["inference.output_prefix"]}_{istart}.log'
            print(f'source activate SE3-nvidia; python {args.command} {extra_args} '\
                  f'inference.num_designs={args.num_per_job} inference.design_startnum={istart} >> {log_fn}', file=job_list_file)

    if args.submit or args.in_proc:
        job_list_file.close()
    # submit job
    if args.submit:
        job_fn = prune_jobs_list(job_fn)
        if args.pilot:
            job_fn = pilot_jobs_list(job_fn, args.pilot_single)

        if args.J is not None:
            job_name = args.J
        else:
            job_name = 'sweep_hyp_'+os.path.basename(os.path.dirname(args.out))
        if args.p == 'cpu':
            args.gres = ""
        slurm_job, proc = slurm_tools.array_submit(job_fn, p = args.p, gres=args.gres, log=args.keep_logs, J=job_name, t=args.t, in_proc=args.in_proc)
        print(f'Submitted array job {slurm_job} with {len(df)*args.num_per_condition/args.num_per_job} jobs to make {len(df)*args.num_per_condition} designs for {len(df)} conditions')

def pilot_jobs_list(jobs_path, single=False):
    pilot_path = os.path.join(os.path.split(jobs_path)[0], 'jobs.list.pilot')
    with open(jobs_path, 'r') as fh:
        jobs = fh.readlines()
    job_by_input_pdb = {}
    for job in jobs:
        input_pdb = re.match('.*inference\.input_pdb=(\S+).*', job).groups()[0]
        if input_pdb not in job_by_input_pdb:
            job_by_input_pdb[input_pdb] = job
    jobs = list(job_by_input_pdb.values())
    with open(pilot_path, 'w') as fh:
        if single:
            jobs = jobs[0:1]
        fh.writelines(jobs)
    LOGGER.debug(f'running {len(jobs)} pilot jobs for PDBS: {list(job_by_input_pdb.keys())}')
    return pilot_path
        
def prune_jobs_list(jobs_path):
    pruned_path = os.path.join(os.path.split(jobs_path)[0], 'jobs.list.pruned')
    pruned = []
    with open(jobs_path, 'r') as fh:
        jobs = fh.readlines()
    for job in jobs:
        want_outs = expected_outputs(job)
        if not all(os.path.exists(want) for want in want_outs):
            pruned.append(job)
    with open(pruned_path, 'w') as fh:
        fh.writelines(pruned)
    return pruned_path

import re
def expected_outputs(job):
    output_prefix = re.match('.*inference\.output_prefix=(\S+).*', job).groups()[0]
    design_startnum = re.match('.*inference\.design_startnum=(\S+).*', job).groups()[0]
    num_designs = re.match('.*inference\.num_designs=(\S+).*', job).groups()[0]

    design_startnum = int(design_startnum)
    num_designs = int(num_designs)

    des_i_start = design_startnum
    des_i_end = design_startnum + num_designs
    return [f'{output_prefix}_{i}.pdb' for i in range(des_i_start, des_i_end)]

if __name__ == "__main__":
    main()