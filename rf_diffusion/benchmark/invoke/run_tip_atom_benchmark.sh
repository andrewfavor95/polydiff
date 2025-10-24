#!/bin/bash 

outdir='/net/scratch/'$USER'/tip_atoms/benchmark'
benchmark_json='bench_tip_debug.json'

script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
benchmark_dir="$(dirname "$script_dir")"
repo_dir="$(dirname "$benchmark_dir")"

mkdir $outdir
cd $outdir

export PYTHONPATH=$repo_dir:$PYTHONPATH

python $repo_dir/benchmark/pipeline.py \
        --num_per_condition 20 --num_per_job 20 --out $outdir/out/run \
        --benchmark_json $benchmark_json \
        --benchmarks all \
        --args "
        (--config-name=aa_tip_atoms_positioned inference.ckpt_path=/home/ahern/projects/dev_rf_diffusion/train_session2023-04-16_1681708054.946131/models/BFF_4.pt)|(--config-name=aa_tip_atoms_position_agnostic inference.ckpt_path=/home/ahern/projects/dev_rf_diffusion/train_session2023-04-18_1681815206.1173074/models/BFF_4.pt|/home/ahern/projects/dev_rf_diffusion/train_session2023-04-18_1681815206.1173074/models/BFF_8.pt)
        diffuser.T=30
        " \
        --num_seq_per_target 8 --af2_chunk=300 --af2_gres=gpu --af2_p=gpu-bf --use_ligand --no_tmalign --score_scripts='af2' --gres=gpu -p gpu-bf --pilot --pilot_single --in_proc
