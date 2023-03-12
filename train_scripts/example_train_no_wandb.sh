#!/bin/bash
### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=11010

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

crop=256
max_len=256
max_complex_chain=250
wandb_pref='debug_sm_conditional'
so3_type='igso3'
euclidean_schedule='linear'
b0=0.01 # 1e-2
bT=0.07 # 7e-2 
chi_type='interp'
uncond_diff=0.0
seqdiff='none'

prob_self_cond=0.5
diff_crd_scale=0.25

ckpt_load_path='/home/rohith/rf2a-fd3/models/rf2a_fd3_20221125_398.pt'

python -u train_multi_deep.py -p_drop 0.15 -accum 2 -crop $crop -w_disp 0.5 -w_frame_dist 1.0 -w_aa 0 -w_blen 0.0 -w_bang 0.0 -w_lj 0.0 -w_hb 0.0 -w_str 0.0 -maxlat 256 -maxseq 1024 -num_epochs 2 -lr 0.0005 -seed 42 -seqid 150.0 -mintplt 1 -use_H -max_length $max_len -max_complex_chain $max_complex_chain -task_names diff,seq2str -task_p 1.0,0.0 -diff_T 200 -aa_decode_steps 0 -wandb_prefix $wandb_pref -diff_so3_type $so3_type -diff_chi_type $chi_type -use_tschedule -maxcycle 1 -diff_b0 $b0 -diff_bT $bT -diff_schedule_type $euclidean_schedule -prob_self_cond $prob_self_cond -str_self_cond -dataset pdb_aa,sm_complex -dataset_prob 0.9,0.1 -sidechain_input False -motif_sidechain_input True -ckpt_load_path $ckpt_load_path -d_t1d 22 -new_self_cond -diff_crd_scale $diff_crd_scale -metric displacement -metric contigs -diff_mask_probs get_triple_contact:1.0 -w_motif_disp 10 \
    -data_pkl_aa aa_dataset_$max_len.pkl \
    -n_extra_block 4 \
    -n_main_block 32 \
    -n_ref_block 4 \
    -n_finetune_block 0 \
    -ref_num_layers 2 \
    -d_pair 192 \
    -n_head_pair 6 \
    -freeze_track_motif \
    -interactive \
    -n_write_pdb 1 \
    -log_inputs \
    -no_wandb \
    2>&1 | tee log.txt

