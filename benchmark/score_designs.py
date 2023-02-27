#!/usr/bin/env python
#
# Takes a folder of pdb & trb files, generates list of AF2 prediction & scoring
# jobs on batches of those designs, and optionally submits slurm array job and
# outputs job ID
# 

import sys, os, argparse, itertools, json, glob
import numpy as np
from icecream import ic

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, 'util'))
import slurm_tools

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir',type=str,help='Folder of designs to score')
    parser.add_argument('--chunk',type=int,default=-1,help='How many designs to score in each job')
    parser.add_argument('--tmp_pre',type=str,default='score.list', help='Name prefix of temporary files with lists of designs to score')
    parser.add_argument('-p', type=str, default='gpu',help='-p argument for slurm (partition)')
    parser.add_argument('-J', type=str, help='name of slurm job')
    parser.add_argument('--gres', type=str, default='gpu:rtx2080:1',help='--gres argument for slurm, e.g. gpu:rtx2080:1')
    parser.add_argument('--no_submit', dest='submit', action="store_false", default=True, help='Do not submit slurm array job, only generate job list.')
    parser.add_argument('--in_proc', dest='in_proc', action="store_true", default=False, help='Do not submit slurm array job, only generate job list.')
    parser.add_argument('--no_logs', dest='keep_logs', action="store_false", default=True, help='Don\'t keep slurm logs.')
    parser.add_argument('--pipeline', '-P', action='store_true', default=False, help='Pipeline mode: submit the next script to slurm with a dependency on jobs from this script.')
    parser.add_argument('-r', '--run', default='af2', help='Comma-separated (no whitespace) list of scoring scripts to run (e.g. "af2,pyrosetta"). Can contain any of {"af2", pyrosetta", "chemnet", "rosettalig"}')
    parser.add_argument('--trb_dir', type=str, help='Folder containing .trb files (if not same as pdb folder)')
    parser.add_argument('--filenames', type=str, help='A path to a list of PDBs to score, rather than scoring everything in datadir')

    args, unknown = parser.parse_known_args()
    if len(unknown)>0:
        print(f'WARNING: Unknown arguments {unknown}')

    if args.filenames:
        filenames = [l.strip() for l in open(args.filenames).readlines()]
    else:
        filenames = sorted(glob.glob(args.datadir+'/*.pdb'))
    if len(filenames)==0: sys.exit('No pdbs to score. Exiting.')

    args.run = args.run.split(',')

    if args.chunk == -1:
        args.chunk = len(filenames)

    # AF2 predictions
    if 'af2' in args.run:
        job_fn = args.datadir + '/jobs.score.af2.list'
        job_list_file = open(job_fn, 'w') if args.submit else sys.stdout
        for i in np.arange(0,len(filenames),args.chunk):
            tmp_fn = f'{args.datadir}/{args.tmp_pre}.{i}'
            with open(tmp_fn,'w') as outf:
                for j in np.arange(i,min(i+args.chunk, len(filenames))):
                    print(filenames[j], file=outf)
            print(f'source activate /software/conda/envs/PPI_design; python {script_dir}/util/af2_metrics.py --use_ptm '\
                  f'--outcsv {args.datadir}/af2_metrics.csv.{i} '\
                  f'--trb_dir {args.trb_dir} '\
                  f'{tmp_fn}', file=job_list_file)

        # submit job
        if args.submit: 
            job_list_file.close()
            if args.J is not None:
                job_name = args.J 
            else:
                job_name = 'af2_'+os.path.basename(args.datadir.strip('/'))
            af2_job, proc = slurm_tools.array_submit(job_fn, p = args.p, gres=None if args.p=='cpu' else args.gres, log=args.keep_logs, J=job_name, in_proc=args.in_proc)
            print(f'Submitted array job {af2_job} with {int(np.ceil(len(filenames)/args.chunk))} jobs to AF2-predict {len(filenames)} designs')

    # Rosetta metrics
    if 'pyrosetta' in args.run:
        # pyrosetta metrics (rog, SS)
        job_fn = args.datadir + '/jobs.score.pyr.list'
        job_list_file = open(job_fn, 'w') if args.submit else sys.stdout
        for i in np.arange(0,len(filenames),args.chunk):
            tmp_fn = f'{args.datadir}/{args.tmp_pre}.pyr.{i}'
            with open(tmp_fn,'w') as outf:
                for j in np.arange(i,min(i+args.chunk, len(filenames))):
                    print(filenames[j], file=outf)
            print(f'{script_dir}/util/pyrosetta_metrics.py '\
                  f'--outcsv {args.datadir}/pyrosetta_metrics.csv.{i} '\
                  f'{tmp_fn}', file=job_list_file)

        # submit job
        if args.submit: 
            job_list_file.close()
            if args.J is not None:
                job_name = args.J 
            else:
                job_name = 'pyr_'+os.path.basename(args.datadir.strip('/'))
            pyr_job, proc = slurm_tools.array_submit(job_fn, p = 'cpu', gres=None, log=args.keep_logs, J=job_name, in_proc=args.in_proc)
            print(f'Submitted array job {pyr_job} with {int(np.ceil(len(filenames)/args.chunk))} jobs to get PyRosetta metrics for {len(filenames)} designs')

    # Ligand metrics (chemnet)
    if 'chemnet' in args.run:
        job_fn = args.datadir + '/jobs.score.chemnet.list'
        job_list_file = open(job_fn, 'w') if args.submit else sys.stdout
        chemnet_script = '/home/jue/git/chemnet/arch.22-08-19/predict/DALigandDock_v02.py'                
        for i in range(0, len(filenames), args.chunk):
            tmp_fn = f'{args.datadir}/{args.tmp_pre}.chemnet.{i}'
            with open(tmp_fn,'w') as outf:
                for j in np.arange(i,min(i+args.chunk, len(filenames))):
                    print(filenames[j], file=outf)
            print(f'source activate dlchem; python {chemnet_script} '\
                  f'-n 10 --ifile {tmp_fn} '\
                  f'--odir {args.datadir}/chemnet/ '\
                  f'--ocsv {args.datadir}/chemnet_scores.csv.{i} ',
                  file=job_list_file)

        # submit job
        if args.submit:
            job_list_file.close()
            if args.J is not None:
                job_name = args.J
            else:
                pre = 'chemnet_'
                job_name = pre + os.path.basename(args.datadir.strip('/')) 
            cn_job, proc = slurm_tools.array_submit(job_fn, p = args.p, gres=None if args.p=='cpu' else args.gres, log=args.keep_logs, J=job_name, in_proc=args.in_proc)
            print(f'Submitted array job {cn_job} with {int(np.ceil(len(filenames)/args.chunk))} jobs to ChemNet-predict {len(filenames)} designs')

    # Ligand metrics (rosetta)
    if 'rosettalig' in args.run:
        job_fn = args.datadir + '/jobs.score.rosettalig.list'
        job_list_file = open(job_fn, 'w') if args.submit else sys.stdout
        rosettalig_script = script_dir+'/util/rosetta_ligand_metrics.py'
        for i in range(0, len(filenames), args.chunk):
            tmp_fn = f'{args.datadir}/{args.tmp_pre}.rosettalig.{i}'
            with open(tmp_fn,'w') as outf:
                for j in np.arange(i,min(i+args.chunk, len(filenames))):
                    print(filenames[j], file=outf)
            print(f'source activate SE3-nvidia; python {rosettalig_script} '\
                  f'--list {tmp_fn} '\
                  f'--outdir {args.datadir}/rosettalig/ '\
                  f'--outcsv {args.datadir}/rosettalig_scores.csv.{i} ',
                  file=job_list_file)

        # submit job
        if args.submit:
            job_list_file.close()
            if args.J is not None:
                job_name = args.J
            else:
                pre = 'rosetta_lig_'
                job_name = pre + os.path.basename(args.datadir.strip('/')) 
            cn_job, proc = slurm_tools.array_submit(job_fn, p = 'cpu', gres=None, log=args.keep_logs, J=job_name, in_proc=args.in_proc)
            print(f'Submitted array job {cn_job} with {int(np.ceil(len(filenames)/args.chunk))} jobs to compute Rosetta ligand metrics on {len(filenames)} designs')


if __name__ == "__main__":
    main()
