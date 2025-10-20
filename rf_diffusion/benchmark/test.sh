#!/bin/bash 
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH=$SCRIPT_DIR/../:$PYTHONPATH
cd "$SCRIPT_DIR"

rm -r test_output/test
mkdir -p test_output/test

echo "DEBUG 1"
cd test_output

echo "DEBUG 1"
python ../pipeline.py \
        --in_proc \
        --use_ligand \
        --num_per_condition 2 --num_per_job 2 --out $SCRIPT_DIR/test_output/test/run \
        --args "--config-name=aa diffuser.T=3 contigmap.length=200-200 inference.input_pdb=$SCRIPT_DIR/input/gaa.pdb inference.ligand=LG1 contigmap.contigs=[\'4-4,A518-519\'] contigmap.length=6-6" \
        --num_seq_per_target 1 --af2_gres=gpu:a6000:1 -p cpu --no_tmalign
    
