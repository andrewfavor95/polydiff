import torch
import torch.nn as nn
import assertpy
from assertpy import assert_that
from icecream import ic
from rf2aa.Embeddings import MSA_emb, Extra_emb, Bond_emb, Templ_emb, Recycling
from rf2aa.Track_module import IterativeSimulator
from rf2aa.AuxiliaryPredictor import (
    DistanceNetwork,
    MaskedTokenNetwork,
    LDDTNetwork,
    PAENetwork,
    BinderNetwork,
)
from rf2aa.chemical import INIT_CRDS, NAATOKENS, NBTYPES, NTOTAL
from rf2aa.tensor_util import assert_shape, assert_equal
import rf2aa.util
from rf2aa.util_module import rbf
from pdb import set_trace

def get_shape(t):
    if hasattr(t, "shape"):
        return t.shape
    if type(t) is tuple:
        return [get_shape(e) for e in t]
    else:
        return type(t)

def reset_model_attrs(prebuilt_models, overrides):
    """
    Loops through overrides and resets those attributes within RF 
    """
    assert type(overrides['model']) is dict, "overrides['model'] must be a dict"

    for _, model in prebuilt_models.items(): 
        # some things are RF attrs, some things are deeper. 
        for module_name, module in model.named_modules():

            for key,val in overrides['model'].items():

                if hasattr(module, key):
                    
                    # simulator shares main_block name
                    # check for that and skip if so
                    if module_name == 'simulator' and key == 'main_block':
                        continue
                    try: 
                        setattr(module, key, val)
                        print('Set {} to {} in {}'.format(key, val, module_name))
                    except Exception as e:
                        print('Error setting {} to {} in {}: {}'.format(key, val, module_name, e))
                        sys.exit(1)

                else:
                    pass 

class RoseTTAFoldModule(nn.Module):
    def __init__(
        self, 
        symmetrize_repeats=None,       # whether to symmetrize repeats in the pair track 
        repeat_length=None,            # if symmetrizing repeats, what length are they? 
        symmsub_k=None,                # if symmetrizing repeats, which diagonals?
        sym_method=None,               # if symmetrizing repeats, which block symmetrization method? 
        main_block=None,               # if copying template blocks along main diag, which block is main block? (the one w/ motif)
        copy_main_block_template=None, # whether or not to copy main block template along main diag
        n_extra_block=4, 
        n_main_block=8, 
        n_ref_block=4, 
        n_finetune_block=0,
        d_msa=256, 
        d_msa_full=64, 
        d_pair=128, 
        d_templ=64,
        n_head_msa=8, 
        n_head_pair=4, 
        n_head_templ=4,
        d_hidden=32, 
        d_hidden_templ=64, 
        p_drop=0.15,
        SE3_param={}, SE3_ref_param={},
        atom_type_index=None, 
        aamask=None, 
        ljlk_parameters=None, 
        lj_correction_parameters=None, 
        cb_len=None, 
        cb_ang=None, 
        cb_tor=None,
        num_bonds=None, 
        lj_lin=0.6, 
        use_extra_l1=True, 
        use_atom_frames=True,
        # New for diffusion
        freeze_track_motif=False,
        assert_single_sequence_input=False,
        symm_fit_start=-1,
        # half_precision=False,
        clash=4.0,
        lock_symmsubs_at=-1,
        binder_net=True,         # associated with newer version of the network 
        allow_particle_recenter=True, # when updating Rs, whether or not to recenter particle at origin
        d_templ_1d=None,
        d_templ_2d=None,
    ):
        super(RoseTTAFoldModule, self).__init__()
        self.freeze_track_motif = freeze_track_motif
        self.symm_fit_start = symm_fit_start
        self.assert_single_sequence_input = assert_single_sequence_input
        # self.half_precision = half_precision
        self.clash = clash
        self.lock_symmsubs_at = lock_symmsubs_at # if > 0, doesn't allow symmsub swaps after this timestep
        self.binder_net = binder_net
        self.allow_particle_recenter = allow_particle_recenter
        #
        # Input Embeddings
        d_state = SE3_param["l0_out_features"]
        self.latent_emb = MSA_emb(
            d_msa=d_msa, d_pair=d_pair, d_state=d_state, p_drop=p_drop
        )
        self.full_emb = Extra_emb(
            d_msa=d_msa_full, d_init=NAATOKENS - 1 + 4, p_drop=p_drop
        )
        self.bond_emb = Bond_emb(d_pair=d_pair, d_init=NBTYPES)

        self.templ_emb = Templ_emb(d_pair=d_pair, 
                                   d_t1d=d_templ_1d,
                                   d_t2d=d_templ_2d,
                                   d_templ=d_templ, 
                                   d_state=d_state, 
                                   n_head=n_head_templ,
                                   d_hidden=d_hidden_templ, 
                                   p_drop=0.25,
                                   symmetrize_repeats=symmetrize_repeats, # repeat protein stuff 
                                   repeat_length=repeat_length, 
                                   symmsub_k=symmsub_k,
                                   sym_method=sym_method, 
                                   main_block=main_block, 
                                   copy_main_block=copy_main_block_template)

        # Update inputs with outputs from previous round
        self.recycle = Recycling(d_msa=d_msa, d_pair=d_pair, d_state=d_state)
        #
        self.simulator = IterativeSimulator(
            n_extra_block=n_extra_block,
            n_main_block=n_main_block,
            n_ref_block=n_ref_block,
            n_finetune_block=n_finetune_block,
            d_msa=d_msa,
            d_msa_full=d_msa_full,
            d_pair=d_pair,
            d_hidden=d_hidden,
            n_head_msa=n_head_msa,
            n_head_pair=n_head_pair,
            SE3_param=SE3_param,
            SE3_ref_param=SE3_ref_param,
            p_drop=p_drop,
            atom_type_index=atom_type_index,  # change if encoding elements instead of atomtype
            aamask=aamask,
            ljlk_parameters=ljlk_parameters,
            lj_correction_parameters=lj_correction_parameters,
            num_bonds=num_bonds,
            cb_len=cb_len,
            cb_ang=cb_ang,
            cb_tor=cb_tor,
            lj_lin=lj_lin,
            use_extra_l1=use_extra_l1,
            symmetrize_repeats=symmetrize_repeats,
            repeat_length=repeat_length,
            symmsub_k=symmsub_k,
            sym_method=sym_method,
            main_block=main_block
        )

        ##
        self.c6d_pred = DistanceNetwork(d_pair, p_drop=p_drop)
        self.aa_pred = MaskedTokenNetwork(d_msa, p_drop=p_drop)
        self.lddt_pred = LDDTNetwork(d_state)
        if (
            use_extra_l1
        ):  # extra l1 features introduced at the same time as pAE/pDE heads
            self.pae_pred = PAENetwork(d_pair)
            self.pde_pred = PAENetwork(
                d_pair
            )  # distance error, but use same architecture as aligned error

        # binder predictions are made on top of the pair features, just like
        # PAE predictions are. It's not clear if this is the best place to insert
        # this prediction head.
        # self.binder_network = BinderNetwork(d_pair, d_state)

        self.use_extra_l1 = use_extra_l1

        # self.bind_pred = BinderNetwork() #fd - expose n_hidden as variable?
        self.bind_pred = BinderNetwork(
            d_pair=d_pair,
            d_state=d_state,
            d_rbf=64
        )

        self.use_atom_frames = use_atom_frames
        self.verbose_checks = False

    def forward(
        self,
        msa_latent,
        msa_full,
        seq,
        seq_unmasked,
        xyz,
        sctors,
        idx,
        bond_feats,
        dist_matrix,
        chirals, 
        t=None,
        atom_frames=None, t1d=None, t2d=None, xyz_t=None, alpha_t=None, mask_t=None, same_chain=None,
        msa_prev=None, pair_prev=None, state_prev=None, mask_recycle=None, is_motif=None,
        return_raw=False,
        use_checkpoint=False,
        return_infer=False, #fd ?
        p2p_crop=-1, topk_crop=-1,   # striping
        symmids=None, symmsub=None, symmRs=None, symmeta=None,  # symmetry
        ss_adj_conditioned=False,
        poly_class_conditioned=False,
        poly_hotspot_conditioned=False,
    ):
        # if self.half_precision:
        #     msa_latent = msa_latent.half()
        #     msa_full = msa_full.half()
        #     seq = seq.half()
        #     seq_unmasked = seq_unmasked.half()

        # ic(get_shape(msa_latent))
        # ic(get_shape(msa_full))
        # ic(get_shape(seq))
        # ic(get_shape(seq_unmasked))
        # ic(get_shape(xyz))
        # ic(get_shape(sctors))
        # ic(get_shape(idx))
        # ic(get_shape(bond_feats))
        # ic(get_shape(chirals))
        # ic(get_shape(atom_frames))
        # ic(get_shape(t1d))
        # ic(get_shape(t2d))
        # ic(get_shape(xyz_t))
        # ic(get_shape(alpha_t))
        # ic(get_shape(mask_t))
        # ic(get_shape(same_chain))
        # ic(get_shape(msa_prev))
        # ic(get_shape(pair_prev))
        # ic(get_shape(state_prev))
        # ic(get_shape(mask_recycle))
        # ic()
        # ic()
        B, N, L = msa_latent.shape[:3]
        A = atom_frames.shape[1]
        dtype = msa_latent.dtype

        if self.assert_single_sequence_input:
            assert_shape(msa_latent, (1, 1, L, 164))
            assert_shape(msa_full, (1, 1, L, 83))
            assert_shape(seq, (1, L))
            assert_shape(seq_unmasked, (1, L))
            assert_shape(xyz, (1, L, NTOTAL, 3))
            assert_shape(sctors, (1, L, 20, 2))
            assert_shape(idx, (1, L))
            assert_shape(bond_feats, (1, L, L))
            assert_shape(dist_matrix, (1, L, L))
            # assert_shape(chirals,     (1, 0))
            # assert_shape(atom_frames, (1, 4, L)) # This is set to 4 for the recycle count, but that can't be right
            assert_shape(atom_frames, (1, A, 3, 2))  # What is 4?


            if self.binder_net:
                if t1d.shape[1] == 2: # 2template model
                    assert_shape(t1d, (1, 2, L, 80))
                    assert_shape(t2d, (1, 2, L, L, 68))
                    assert_shape(xyz_t, (1, 2, L, 3))
                    assert_shape(alpha_t, (1, 2, L, 60))
                    assert_shape(mask_t, (1, 2, L, L))
                else: # 3template model
                    
                    assert_shape(xyz_t, (1, 3, L, 3))
                    assert_shape(alpha_t, (1, 3, L, 60))
                    assert_shape(mask_t, (1, 3, L, L))

                    if ss_adj_conditioned: # using ss conditioning
                        assert_shape(t2d, (1, 3, L, L, 72))
                    else: # no ss adjacency conditioning
                        assert_shape(t2d, (1, 3, L, L, 69))

                    if poly_class_conditioned:

                        if poly_hotspot_conditioned:
                            assert_shape(t1d, (1, 3, L, 90))
                        else:
                            assert_shape(t1d, (1, 3, L, 84))

                    else:
                        if poly_hotspot_conditioned:
                            assert_shape(t1d, (1, 3, L, 87))
                        else:
                            assert_shape(t1d, (1, 3, L, 81))
                        


            else:
                assert_shape(t1d, (1, 1, L, 80))
                assert_shape(t2d, (1, 1, L, L, 68))
                assert_shape(xyz_t, (1, 1, L, 3))
                assert_shape(alpha_t, (1, 1, L, 60))
                assert_shape(mask_t, (1, 1, L, L))


            assert_shape(same_chain, (1, L, L))
            device = msa_latent.device
            assert_that(msa_full.device).is_equal_to(device)
            assert_that(seq.device).is_equal_to(device)
            assert_that(seq_unmasked.device).is_equal_to(device)
            assert_that(xyz.device).is_equal_to(device)
            assert_that(sctors.device).is_equal_to(device)
            assert_that(idx.device).is_equal_to(device)
            assert_that(bond_feats.device).is_equal_to(device)
            assert_that(dist_matrix.device).is_equal_to(device)
            assert_that(atom_frames.device).is_equal_to(device)
            assert_that(t1d.device).is_equal_to(device)
            assert_that(t2d.device).is_equal_to(device)
            assert_that(xyz_t.device).is_equal_to(device)
            assert_that(alpha_t.device).is_equal_to(device)
            assert_that(mask_t.device).is_equal_to(device)
            assert_that(same_chain.device).is_equal_to(device)

        if self.verbose_checks:
            ic(is_motif.shape)
            is_sm = rf2aa.util.is_atom(seq[0])  # (L)
            is_protein_motif = is_motif & ~is_sm
            if is_motif.any():
                motif_protein_i = torch.where(is_motif)[0][0]
            is_motif_sm = is_motif & is_sm
            if is_sm.any():
                motif_sm_i = torch.where(is_motif_sm)[0][0]
            diffused_protein_i = torch.where(~is_sm & ~is_motif)[0][0]

            """
            msa_full: NSEQ,N_INDEL,N_TERMINUS,
            msa_masked: NSEQ,NSEQ,N_INDEL,N_INDEL,N_TERMINUS
            """
            import numpy as np
            from rf2aa.chemical import NAATOKENS

            NINDEL = 1
            NTERMINUS = 2
            NMSAFULL = NAATOKENS + NINDEL + NTERMINUS
            NMSAMASKED = NAATOKENS + NAATOKENS + NINDEL + NINDEL + NTERMINUS
            assert_that(msa_latent.shape[-1]).is_equal_to(NMSAMASKED)
            assert_that(msa_full.shape[-1]).is_equal_to(NMSAFULL)

            msa_full_seq = np.r_[0:NAATOKENS]
            msa_full_indel = np.r_[NAATOKENS : NAATOKENS + NINDEL]
            msa_full_term = np.r_[NAATOKENS + NINDEL : NMSAFULL]

            msa_latent_seq1 = np.r_[0:NAATOKENS]
            msa_latent_seq2 = np.r_[NAATOKENS : 2 * NAATOKENS]
            msa_latent_indel1 = np.r_[2 * NAATOKENS : 2 * NAATOKENS + NINDEL]
            msa_latent_indel2 = np.r_[
                2 * NAATOKENS + NINDEL : 2 * NAATOKENS + NINDEL + NINDEL
            ]
            msa_latent_terminus = np.r_[2 * NAATOKENS + 2 * NINDEL : NMSAMASKED]

            i_name = [(diffused_protein_i, "diffused_protein")]
            if is_sm.any():
                i_name.insert(0, (motif_sm_i, "motif_sm"))
            if is_motif.any():
                i_name.insert(0, (motif_protein_i, "motif_protein"))

            for i, name in i_name:
                ic(f"------------------{name}:{i}----------------")
                msa_full_seq = msa_full[0, 0, i, np.r_[0:NAATOKENS]]
                msa_full_indel = msa_full[
                    0, 0, i, np.r_[NAATOKENS : NAATOKENS + NINDEL]
                ]
                msa_full_term = msa_full[0, 0, i, np.r_[NAATOKENS + NINDEL : NMSAFULL]]

                msa_latent_seq1 = msa_latent[0, 0, i, np.r_[0:NAATOKENS]]
                msa_latent_seq2 = msa_latent[0, 0, i, np.r_[NAATOKENS : 2 * NAATOKENS]]
                msa_latent_indel1 = msa_latent[
                    0, 0, i, np.r_[2 * NAATOKENS : 2 * NAATOKENS + NINDEL]
                ]
                msa_latent_indel2 = msa_latent[
                    0,
                    0,
                    i,
                    np.r_[2 * NAATOKENS + NINDEL : 2 * NAATOKENS + NINDEL + NINDEL],
                ]
                msa_latent_term = msa_latent[
                    0, 0, i, np.r_[2 * NAATOKENS + 2 * NINDEL : NMSAMASKED]
                ]

                assert_equal(msa_full_seq, msa_latent_seq1)
                assert_equal(msa_full_seq, msa_latent_seq2)
                assert_equal(msa_full_indel, msa_latent_indel1)
                assert_equal(msa_full_indel, msa_latent_indel2)
                assert_equal(msa_full_term, msa_latent_term)
                # if 'motif' in name:
                msa_cat = torch.where(msa_full_seq)[0]
                ic(msa_cat, seq[0, i])
                assert_equal(seq[0, i : i + 1], msa_cat)
                assert_equal(seq[0, i], seq_unmasked[0, i])
                ic(
                    name,
                    # torch.where(msa_latent[0,0,i,:80]),
                    # torch.where(msa_full[0,0,i]),
                    seq[0, i],
                    seq_unmasked[0, i],
                    torch.where(t1d[0, 0, i]),
                    xyz[0, i, :4, 0],
                    xyz_t[0, 0, i, 0],
                )

        # Get embeddings
        msa_latent, pair, state = self.latent_emb(
            msa_latent, seq, idx, bond_feats,  dist_matrix, same_chain
        )
        msa_full = self.full_emb(msa_full, seq, idx)
        pair = pair + self.bond_emb(bond_feats)

        msa_latent, pair, state = msa_latent.to(dtype), pair.to(dtype), state.to(dtype)
        msa_full = msa_full.to(dtype)

        #
        # Do recycling
        if msa_prev is None:
            msa_prev = torch.zeros_like(msa_latent[:,0])
        if pair_prev is None:
            pair_prev = torch.zeros_like(pair)
        if state_prev is None:
            state_prev = torch.zeros_like(state)

        msa_recycle, pair_recycle, state_recycle = self.recycle(msa_prev, pair_prev, xyz, state_prev, sctors, mask_recycle)
        msa_recycle, pair_recycle, state_recycle = msa_recycle.to(dtype), pair_recycle.to(dtype), state_recycle.to(dtype)

        msa_latent[:,0] = msa_latent[:,0] + msa_recycle.reshape(B,L,-1)
        pair = pair + pair_recycle
        state = state + state_recycle

        # add template embedding
        pair, state = self.templ_emb(t1d, t2d, alpha_t, xyz_t, mask_t, pair, state, use_checkpoint=use_checkpoint, p2p_crop=p2p_crop)

        # Predict coordinates from given inputs
        is_motif = is_motif if self.freeze_track_motif else torch.zeros_like(seq).bool()[0]
        msa, pair, xyz, alpha_s, xyz_allatom, state, symmsub = self.simulator(
            seq_unmasked, msa_latent, msa_full, pair, xyz[:,:,:3], state, idx, 
            bond_feats, dist_matrix, same_chain, chirals, is_motif, atom_frames, 
            use_checkpoint=use_checkpoint, use_atom_frames=self.use_atom_frames,
            symmids=symmids, symmsub=symmsub, symmRs=symmRs, symmeta=symmeta, 
            p2p_crop=p2p_crop, topk_crop=topk_crop, clash=self.clash, 
            recenter_particle=self.allow_particle_recenter)
        # msa, pair, xyz, alpha_s, xyz_allatom, state, symmsub = self.simulator(
        #     seq_unmasked, msa_latent, msa_full, pair, xyz[:,:,:3], state, idx, 
        #     bond_feats, dist_matrix, same_chain, chirals, is_motif, atom_frames, 
        #     use_checkpoint=use_checkpoint, use_atom_frames=self.use_atom_frames,
        #     symmids=symmids, symmsub=symmsub, symmRs=symmRs, symmeta=symmeta, 
        #     p2p_crop=p2p_crop, topk_crop=topk_crop, fit_symm=fit_symm, clash=self.clash, 
        #     lock_symmsubs=lock_symmsubs, recenter_particle=self.allow_particle_recenter)


        
        if return_raw:
            # get last structure
            xyz_last = xyz_allatom[-1].unsqueeze(0)
            return msa[:,0], pair, xyz_last, state, alpha_s[-1], None

        # predict masked amino acids
        logits_aa = self.aa_pred(msa)

        # predict distogram & orientograms
        logits = self.c6d_pred(pair)

        # Predict LDDT
        lddt = self.lddt_pred(state)

        if self.verbose_checks:
            pseq_0 = logits_aa.permute(0, 2, 1)
            ic(pseq_0.shape)
            pseq_0 = pseq_0[0]
            ic(
                f"motif    sequence: { rf2aa.chemical.seq2chars(torch.argmax(pseq_0[is_motif], dim=-1).tolist())}"
            )
            ic(
                f"diffused sequence: { rf2aa.chemical.seq2chars(torch.argmax(pseq_0[~is_motif], dim=-1).tolist())}"
            )

        # if return_infer:
        #     xyz_last = xyz_allatom[-1].unsqueeze(0)
        #     # return msa[:,0], pair, xyz_last, state, alpha_s[-1], logits_aa.permute(0,2,1), lddt
        #     # return msa[:,0], pair, xyz_last, state, alpha_s[-1], logits_aa.permute(0,2,1), lddt
        #     return msa[:,0], pair, xyz_last, state, alpha_s[-1], logits_aa.permute(0,2,1), lddt

        logits_pae = logits_pde = p_bind = None
        if not return_infer:
            # predict aligned error and distance error
            if self.use_extra_l1:
                logits_pae = self.pae_pred(pair)        
                logits_pde = self.pde_pred(pair + pair.permute(0,2,1,3)) # symmetrize pair features

                #fd  predict bind/no-bind
                # p_bind = self.bind_pred(logits_pae,same_chain)

                #fd  predict bind/no-bind
                cas = xyz[-1,:,:,1].contiguous()
                rbf_feat = rbf(torch.cdist(cas, cas))
                p_bind = self.bind_pred(pair, rbf_feat, state, seq_unmasked, idx, bond_feats, dist_matrix, same_chain)


            else:
                logits_pae = None
                logits_pde = None


        return (
            logits, logits_aa, logits_pae, logits_pde, p_bind, 
            xyz, alpha_s, xyz_allatom, lddt, msa[:,0], pair, state, symmsub
        )

