#!/bin/bash 
cd "$(dirname "$0")"

export PYTHONPATH=/home/ahern/projects/dev_rf_diffusion:$PYTHONPATHa

rm -r test_output/test
mkdir test_output/test

echo "DEBUG 1"
cd test_output

echo "DEBUG 1"
python ../pipeline.py \
        --in_proc \
        --num_per_condition 2 --num_per_job 2 --out test/run \
        --args "--config-name=aa diffuser.T=3 contigmap.length=200-200 inference.input_pdb=/home/ahern/projects/dev_rf_diffusion/benchmark/input/gaa.pdb inference.ligand=UNL contigmap.contigs=[\'4-4,A518-519\'] contigmap.length=6-6" \
        --num_seq_per_target 1 --af2_gres=gpu:a6000:1 -p cpu
    
