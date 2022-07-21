#!/bin/bash
#SBATCH -p gpu
#SBATCH -J diff_predict_prev_L2disp
#SBATCH --cpus-per-task=4
#SBATCH --mem=196g
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:4
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=4


### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=36843

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

crop=384
max_len=384
max_complex_chain=250

w_disp=1.0
w_blen=0.0
w_bang=0.0
w_lj=0.0
w_hb=0.0
w_str=5.0

source activate /home/jwatson3/.conda/envs/SE3nv
python -u ./train_multi_deep.py -wandb_prefix 'T200_pred_prev_disp1_str5_noclassical' -p_drop 0.15 -accum 16 -crop $crop -w_disp $w_disp -w_blen $w_blen -w_bang $w_bang -w_lj $w_lj -w_hb $w_hb -w_str $w_str -maxlat 256 -maxseq 1024 -num_epochs 200 -lr 0.0005 -n_main_block 32 -seed 450 -seqid 150.0 -mintplt 1 -use_H -max_length $max_len -max_complex_chain $max_complex_chain -task_names diff -task_p 1.0 -interactive -diff_T 200 -predict_previous
