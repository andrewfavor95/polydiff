#!/bin/bash


# comment out some stuff
# ### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
# ### change WORLD_SIZE as gpus/node * num_nodes
# export MASTER_PORT=12651

# ### get the first node name as master address - customized for vgg slurm
# ### e.g. master(gnodee[2-5],gnoded1) == gnodee2
# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)


#This training run has these new variables manipulated


crop=512
max_len=512
max_complex_chain=250
wandb_pref='afav_first_run'
so3_type='slerp'
euclidean_schedule='linear'
b0=0.01 # 1e-2
bT=0.07 # 7e-2 
chi_type='interp'

prob_self_cond=0.5
diff_crd_scale=0.25

#ckpt_load_path='/mnt/home/davidcj/from/rohith/models/rf2a_25c_414_t1d_81_t2d_69.pt'    # expanded t1d and t2d features 
ckpt_load_path='/home/afavor/git/RFD_AF/3template_na/checkpoints/from_rohith/rf2a_25c_414_t1d_81_t2d_69.pt'
# ckpt_load_path='/home/afavor/git/RFD_AF/3template_na/checkpoints/from_davidj/train_session2023-09-06_1694049198.4710052/BFF_12.pt'
# loss weights 
W_DISP=0.5
W_FRAME_DIST=0.0
W_AA=0.0
W_BLEN=0.0
W_BANG=0.0
W_LJ=0.0
W_HB=0.0
W_STR=0.0
W_DIST=0.05

# FAPE associated params 
# W_FAPE 5.0 instead of 10.0 because we now have 2 FAPE terms (motif and non-motif)
W_MOTIF_FAPE=10
W_NONMOTIF_FAPE=0.0
NORM_FAPE=10
CUT_FAPE=10


/software/containers/SE3nv.sif -u train_multi_deep.py -p_drop 0.15 \
    -accum 2 \
    -crop $crop \
    -w_disp $W_DISP \
    -w_frame_dist $W_FRAME_DIST \
    -w_aa $W_AA \
    -w_blen $W_BLEN \
    -w_bang $W_BANG \
    -w_lj $W_LJ \
    -w_hb $W_HB \
    -w_str $W_STR \
    -w_motif_fape $W_MOTIF_FAPE \
    -w_nonmotif_fape $W_NONMOTIF_FAPE \
    -norm_fape $NORM_FAPE \
    -clamp_fape $CUT_FAPE \
    -w_dist $W_DIST \
    -maxlat 256 \
    -maxseq 1024 \
    -num_epochs 200 \
    -lr 0.0005 \
    -seed 42 \
    -seqid 150.0 \
    -mintplt 1 \
    -use_H \
    -max_length $max_len \
    -max_complex_chain $max_complex_chain \
    -task_names diff,seq2str \
    -task_p 1.0,0.0 \
    -diff_T 200 \
    -aa_decode_steps 0 \
    -wandb_prefix $wandb_pref \
    -diff_so3_type $so3_type \
    -diff_chi_type $chi_type \
    -use_tschedule \
    -maxcycle 1 \
    -diff_b0 $b0 \
    -diff_bT $bT \
    -diff_schedule_type $euclidean_schedule \
    -prob_self_cond $prob_self_cond \
    -str_self_cond \
    -dataset pdb_aa,na_compl,tf_distil,sm_complex \
    -dataset_prob 0.5,0.25,0.25,0.0 \
    -sidechain_input False \
    -motif_sidechain_input True \
    -ckpt_load_path $ckpt_load_path \
    -d_t1d 22 \
    -new_self_cond \
    -diff_crd_scale $diff_crd_scale  \
    -diff_mask_probs get_diffusion_mask_chunked:0.4,get_triple_contact_3template:0.6 \
    -w_motif_disp 0 \
    -data_pkl_aa dataset_3template_na_20230926.pkl \
    -metric atom_bonds \
    -n_extra_block 4 \
    -n_main_block 32 \
    -n_ref_block 4 \
    -n_finetune_block 0 \
    -ref_num_layers 2 \
    -d_pair 192 \
    -n_head_pair 6 \
    -freeze_track_motif \
    -n_write_pdb 5 \
    -saves_per_epoch 10 \
    -epoch_size 25600 \
    -motif_only_2d \
    -d_templ_1d 81 \
    -d_templ_2d 69 \
    -eye_frames \
    -p_show_motif_seq 0.5 \
    -no_wandb
