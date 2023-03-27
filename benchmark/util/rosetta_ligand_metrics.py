#!/usr/bin/env python

"""
Rosetta scoring script for ligand-binding protein designs.

Adapted from /home/linnaan/scripts/scripts/pseudocycle_scripts/design_ligand_full_noHBNet_fromDesign_onlyFR_noFD.py on 2023-2-7
"""
import sys, os, glob, pickle, time, argparse
from argparse import Namespace
import numpy as np
import torch
from collections import OrderedDict
import pandas as pd

from pyrosetta import *
from pyrosetta.rosetta import *
import pyrosetta.distributed.tasks.rosetta_scripts as rosetta_scripts

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,script_dir+'/../')
import pdb_to_params

sys.path.insert(0,script_dir+'/../../')
sys.path.insert(0,script_dir+'/../../RF2-allatom/')
import inference.utils
import rf2aa.parsers
import rf2aa.chemical
from parsers import load_ligand_from_pdb


def parse_args(in_args):
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument( "--pdb", type=str, default="", help='The name of a pdb file to run this metric on.' )
    argparser.add_argument( "--list", type=str, help='List of a pdb file names to run this metric on.' )
    argparser.add_argument( "--outdir", type=str, help='folder to save outputs' )
    argparser.add_argument( "--outcsv", type=str, help='CSV file to output scores in')
    argparser.add_argument( "--suffix", type=str, default="_FR", help='suffix to add, need to include _ sign' )
    argparser.add_argument( "--use_genpot", type=bool, default="False", help='use genpot? default use beta_nov16' )
    argparser.add_argument( "--cache", type=str, default="./tmp/", help='cache directory for psipred SSpred etc' )
    argparser.add_argument( "--constraint_sd", type=float, default=1.0, help='strength of the constraints' )
    args = argparser.parse_args(in_args)

    if args.pdb is None and args.list is None:
        sys.exit('ERROR: one of --pdb or --list is required.')

    return args

def main(args):

    if args.pdb:
        filenames = [args.pdb]
    if args.list:
        with open(args.list) as f:
            filenames = [line.strip() for line in f.readlines()]
    if args.outdir is None:
        args.outdir = os.path.dirname(filenames[0])+'/rosettalig/'
        print('No --outdir specified, using {args.outdir} by default')
    os.makedirs(args.outdir, exist_ok=True)

    cache_dir = args.cache
    os.makedirs(cache_dir, exist_ok=True)

    # extract metadata needed to score each design
    metadata = []
    for fn in filenames:

        name = os.path.basename(fn).replace('.pdb','')
        parent_name = '_'.join(name.split('_')[:-1])
        parent_dir = os.path.dirname(fn)+'/../'

        with open(parent_dir+parent_name+'.trb','rb') as f:
            trb = pickle.load(f)
        lig_name = trb['config']['inference']['ligand']
        in_pdb = trb['config']['inference']['input_pdb']
        params_fn = pdb_to_params.aux_file(parent_dir, in_pdb, lig_name, 'params')

        # identify potential H-bond donors & acceptors
        mol, xyz_sm, mask_sm, msa_sm, bond_feats_sm, atom_names = \
            load_ligand_from_pdb(fn, lig_name=lig_name, remove_H=False)

        atom_type = [''.join([c for c in a if c.isalpha()]) for a in atom_names]

        H_bond_atoms = set()
        for i, aname, atype in zip(range(len(atom_names)), atom_names,atom_type):
            if atype=='H':
                neighbors_nonC = [atom_names[j].strip() for j in torch.where(bond_feats_sm[i])[0]
                                  if atom_type[j]!='C']
                if len(neighbors_nonC)>0:
                    H_bond_atoms.add(aname.strip())
                    H_bond_atoms.update(neighbors_nonC)

        # rosetta numbering for the ligand residue
        parsed_pdb = inference.utils.parse_pdb(fn)
        L_prot = parsed_pdb['xyz'].shape[0]
        lig_res_num = L_prot+1

        metadata.append(dict(
            tag=name,
            params_fn=params_fn,
            H_bond_atoms=H_bond_atoms,
            lig_res_num=lig_res_num
        ))

    # initialize rosetta with all the ligands we need
    all_params_fns = set([dat['params_fn'] for dat in metadata])
    params_str = ' '.join(['-extra_res '+x for x in all_params_fns])

    if args.use_genpot:
        pyrosetta.init(f"-beta -holes:dalphaball /software/rosetta/DAlphaBall.gcc -use_terminal_residues true -mute basic.io.database core.scoring -dunbrack_prob_buried 0.8 -dunbrack_prob_nonburied 0.8 -dunbrack_prob_buried_semi 0.8 -dunbrack_prob_nonburied_semi 0.8 -no_nstruct_label true -nstruct 1 -precompute_ig -out:path:scratch {cache_dir} -out:file:write_pdb_parametric_info True -out:file:scorefile score.sc -run:preserve_header {params_str}") #-indexed_structure_store:fragment_store /databases/vall/ss_grouped_vall_helix_shortLoop.h5 
    else:
        pyrosetta.init(f"-beta_nov16 -corrections:beta_nov16 -holes:dalphaball /software/rosetta/DAlphaBall.gcc -use_terminal_residues true -mute basic.io.database core.scoring -dunbrack_prob_buried 0.8 -dunbrack_prob_nonburied 0.8 -dunbrack_prob_buried_semi 0.8 -dunbrack_prob_nonburied_semi 0.8 -no_nstruct_label true -nstruct 1 -precompute_ig -out:path:scratch {cache_dir} -out:file:write_pdb_parametric_info True -out:file:scorefile score.sc -run:preserve_header {params_str}") #-indexed_structure_store:fragment_store /databases/vall/ss_grouped_vall_helix_shortLoop.h5 

    # score each design
    records = []
    for fn, dat in zip(filenames, metadata):
        t0 = time.time()
            
        pose = pose_from_file(fn)
        designed_pose = design(pose, dat['H_bond_atoms'], dat['lig_res_num'], args.constraint_sd)

        outname = f"{args.outdir}{dat['tag']}{args.suffix}"
        designed_pose.dump_pdb(f'{outname}.pdb')
        print(f'saved relaxed pose to {outname}.pdb')

        record = OrderedDict()
        record['name'] = dat['tag']
        record.update(score_dict_from_pdb(outname+'.pdb'))
        records.append(record)

        seconds = int(time.time() - t0)
        print( f"protocols.jd2.JobDistributor: {dat['tag']} reported success, generated in {seconds} seconds" )

    # output scores
    df = pd.DataFrame.from_records(records)
    if args.outcsv is not None:
        df.to_csv(args.outcsv)


def design(pose, heavy_atms, ligand_res_number, constraint_sd):

    t0 = time.time()
   
    filters,protocols = generate_hb_filters(heavy_atms,'sfxn',ligand_res_number)
    print(f'generated filters')
    
    xml = f"""
    <ROSETTASCRIPTS>  
    #this protocol moves the ligand too much during minimization, transfer this to
        <SCOREFXNS>
            <ScoreFunction name="sfxn_design" weights="beta">
              <Reweight scoretype="res_type_constraint" weight="0.3"/>
              <Reweight scoretype="arg_cation_pi" weight="3"/>
          <Reweight scoretype="approximate_buried_unsat_penalty" weight="5"/>
              <Set approximate_buried_unsat_penalty_burial_atomic_depth="3.5"/>
              <Set approximate_buried_unsat_penalty_hbond_energy_threshold="-0.5"/>
              <Set approximate_buried_unsat_penalty_hbond_bonus_cross_chain="-1"/>
              <Reweight scoretype="atom_pair_constraint" weight="0.3"/>
              <Reweight scoretype="dihedral_constraint" weight="0.1"/>
              <Reweight scoretype="angle_constraint" weight="0.1"/>
              <Reweight scoretype="aa_composition" weight="1.0" />
            </ScoreFunction>
            <ScoreFunction name="sfxn" weights="beta"/>    
            <ScoreFunction name="sfxn_softish" weights="beta">
                <Reweight scoretype="fa_rep" weight="0.15" />
            </ScoreFunction>
        <ScoreFunction name="vdw_sol" weights="empty" >
          <Reweight scoretype="fa_atr" weight="1.0" />
          <Reweight scoretype="fa_rep" weight="0.55" />
          <Reweight scoretype="fa_sol" weight="1.0" />
        </ScoreFunction>
          </SCOREFXNS>
          
          <RESIDUE_SELECTORS>
            <Layer name="init_core_SCN" select_core="True" use_sidechain_neighbors="True" surface_cutoff="1.0" /> 
            <Layer name="init_boundary_SCN" select_boundary="True" use_sidechain_neighbors="True" surface_cutoff="1.0" />
            <Layer name="surface_SCN" select_surface="True" use_sidechain_neighbors="True" surface_cutoff="1.0" />
            <Layer name="coreRes" select_core="true" use_sidechain_neighbors="true" core_cutoff="2.1" surface_cutoff="1.0"/>
            <ResiduePDBInfoHasLabel name="hbnet_res" property="HBNet" />
            Not name="not_hbnet_res" selector="hbnet_res" /> 
            And name="surface_SCN_and_not_hbnet_res" selectors="surface_SCN,not_hbnet_res"/>
            <ResidueName name="select_AVLI" residue_names="ALA,VAL,LEU,ILE" />
            <Not name="not_AVLI" selector="select_AVLI" />
            <ResiduePDBInfoHasLabel name="all_rifres_res" property="RIFRES"/>
            <And name="rifres_res" selectors="all_rifres_res,not_AVLI" />
            <Chain name="chainA" chains="A"/>
            <Chain name="chainB" chains="B"/>
        <Index name="ligand" resnums="{ligand_res_number}"/>
        <Not name="not_ligand" selector="ligand"/>
        <CloseContact name="interface_by_contact" residue_selector="ligand" contact_threshold="8" /> /this will select the ligand as well
        <And name="interface" selectors="interface_by_contact,not_ligand"/>
            <Not name="not_interface" selector="interface"/>
            <Or name="interface_and_ligand" selectors="interface,ligand"/>
            <And name="not_interface_or_ligand" selectors="chainA,not_interface" />
            <ResidueName name="select_polar" residue_names="GLU,ASP,ARG,HIS,GLN,ASN,THR,SER,TYR,TRP" />
            <ResidueName name="select_PG" residue_names="PRO,GLY" />
            #layer design definition
            Layer name="surface" select_core="false" select_boundary="false" select_surface="true" use_sidechain_neighbors="true"/>
            Layer name="boundary" select_core="false" select_boundary="true" select_surface="false" use_sidechain_neighbors="true"/>
            <Layer name="core_by_SC" select_core="true" select_boundary="false" select_surface="false" use_sidechain_neighbors="true" core_cutoff="5.2"/>
            <Layer name="core_by_SASA" select_core="true" select_boundary="false" select_surface="false" use_sidechain_neighbors="false" ball_radius="2" core_cutoff="20"/>
        <Or name="core_by_SC_SASA" selectors="core_by_SC,core_by_SASA"/>
        <And name="core" selectors="core_by_SC_SASA,not_ligand"/>
            <Not name="not_core" selector="core"/>
            <And name="not_core_chA" selectors="not_core,chainA,not_interface_or_ligand"/>
            <SecondaryStructure name="sheet" overlap="0" minH="3" minE="2" include_terminal_loops="false" use_dssp="true" ss="E"/>
            <SecondaryStructure name="entire_loop" overlap="0" minH="3" minE="2" include_terminal_loops="true" use_dssp="true" ss="L"/>
            <SecondaryStructure name="entire_helix" overlap="0" minH="3" minE="2" include_terminal_loops="false" use_dssp="true" ss="H"/>
            <And name="helix_cap" selectors="entire_loop">
              <PrimarySequenceNeighborhood lower="1" upper="0" selector="entire_helix"/>
            </And>
            <And name="helix_start" selectors="entire_helix">
              <PrimarySequenceNeighborhood lower="0" upper="1" selector="helix_cap"/>
            </And>
            <And name="helix" selectors="entire_helix">
              <Not selector="helix_start"/>
            </And>
            <And name="loop" selectors="entire_loop">
              <Not selector="helix_cap"/>
            </And>  
          </RESIDUE_SELECTORS>

          <RESIDUE_LEVEL_TASK_OPERATIONS>
            <PreventRepackingRLT name="PreventRepacking" />
            <RestrictToRepackingRLT name="RestrictToRepacking" />
          </RESIDUE_LEVEL_TASK_OPERATIONS>
          
          <TASKOPERATIONS>
              <SetIGType name="precompute_ig" lin_mem_ig="false" lazy_ig="false" double_lazy_ig="false" precompute_ig="true"/> 
              SeqprofConsensus name="pssm_cutoff" filename="%%pssmFile%%" min_aa_probability="-1" convert_scores_to_probabilities="0" probability_larger_than_current="0" debug="1" ignore_pose_profile_length_mismatch="1"/>
              <RestrictAbsentCanonicalAAS name="noCys" keep_aas="ADEFGHIKLMNPQRSTVWY"/>
              <PruneBuriedUnsats name="prune_buried_unsats" allow_even_trades="false" atomic_depth_cutoff="3.5" minimum_hbond_energy="-0.5" />
              <LimitAromaChi2 name="limitchi2" chi2max="110" chi2min="70" include_trp="True" />
              <ExtraRotamersGeneric name="ex1_ex2aro" ex1="1" ex2aro="1" />
              <IncludeCurrent name="ic"/>

              <DesignRestrictions name="design_task">
    <!--             <Action selector_logic="surface AND helix_start"  aas="DEHKPQR"/>
                <Action selector_logic="surface AND helix"    aas="EHKQR"/>
                <Action selector_logic="surface AND sheet"    aas="EHKNQRST"/>
                <Action selector_logic="surface AND loop"   aas="DEGHKNPQRST"/>
                <Action selector_logic="boundary AND helix_start" aas="ADEHIKLMNPQRSTVWY"/>
                <Action selector_logic="boundary AND helix"   aas="ADEHIKLMNQRSTVWY"/>
                <Action selector_logic="boundary AND sheet"   aas="DEFHIKLMNQRSTVWY"/>
                <Action selector_logic="boundary AND loop"    aas="ADEFGHIKLMNPQRSTVWY"/>
                <Action selector_logic="surface"  residue_level_operations="PreventRepacking" /> -->
                <Action selector_logic="not_core_chA"  residue_level_operations="PreventRepacking" />
                <!-- <Action selector_logic="core AND helix_start"   aas="AFILMPVWY"/>
                <Action selector_logic="core AND helix"     aas="AFILMVWY"/>
                <Action selector_logic="core AND sheet"     aas="FILMVWY"/> -->
                <Action selector_logic="core NOT rifres_res"      aas="AFILMVWYSTQN"/>
            <Action selector_logic="rifres_res"      aas="AFILMVWYSTQNDERKH"/>
            <Action selector_logic="interface NOT core"      aas="AFILMVWYSTQNDERKH"/>
                <!-- <Action selector_logic="helix_cap"      aas="DNST"/> -->
              </DesignRestrictions>

              <OperateOnResidueSubset name="restrict_to_packing_not_interface" selector="not_interface"><RestrictToRepackingRLT/></OperateOnResidueSubset>
              <OperateOnResidueSubset name="restrict_to_interface" selector="not_interface_or_ligand"><PreventRepackingRLT/></OperateOnResidueSubset>
              <!-- <OperateOnResidueSubset name="ld_surface_not_hbnets" selector="surface_SCN_and_not_hbnet_res"><PreventRepackingRLT/></OperateOnResidueSubset> -->
              <OperateOnResidueSubset name="ld_surface" selector="surface_SCN"><PreventRepackingRLT/></OperateOnResidueSubset>
              <OperateOnResidueSubset name="restrict_packing_rifres_res" selector="rifres_res"><RestrictToRepackingRLT/></OperateOnResidueSubset>
              <OperateOnResidueSubset name="restrict_packing_interface" selector="interface"><RestrictToRepackingRLT/></OperateOnResidueSubset>
              <!-- <OperateOnResidueSubset name="fix_hbnet_residues" selector="hbnet_res"><RestrictToRepackingRLT/></OperateOnResidueSubset> -->
              <OperateOnResidueSubset name="restrict_target2repacking" selector="ligand"><PreventRepackingRLT/></OperateOnResidueSubset> #change from RestrictToRepackingRLT to PreventRepackingRLT
                  
              <ProteinProteinInterfaceUpweighter name="upweight_interface" interface_weight="3" />
              <ProteinInterfaceDesign name="pack_long" design_chain1="0" design_chain2="0" jump="1" interface_distance_cutoff="15"/>
          </TASKOPERATIONS>
          
          <MOVERS>
        <AddResidueLabel name="label_core" residue_selector="core" label="core" />
        <AddResidueLabel name="label_interface" residue_selector="interface" label="interface" />
            <AddConstraintsToCurrentConformationMover name="add_bb_cst" use_distance_cst="False" cst_weight="1" bb_only="1" sc_tip_only="0" />
            <ClearConstraintsMover name="rm_bb_cst" />
            <TaskAwareMinMover name="min" scorefxn="sfxn" bb="0" chi="1" task_operations="pack_long" />
            FavorSequenceProfile name="FSP" scaling="none" weight="1" pssm="%%pssmFile%%" scorefxns="sfxn_design"/>
            
            <!-- <PackRotamersMover name="hard_pack" scorefxn="sfxn_design"  task_operations="ex1_ex2aro,ld_surface_not_hbnets,fix_hbnet_residues,ic,limitchi2,pssm_cutoff,noCys,restrict_packing_rifres_res,upweight_interface,restrict_to_interface"/> 
            <TaskAwareMinMover name="softish_min" scorefxn="sfxn_softish" chi="1" bb="1" jump="1" tolerance="0.01" cartesian="false" task_operations="restrict_to_packing_not_interface" />
            <TaskAwareMinMover name="hard_min" scorefxn="sfxn" chi="1" bb="1" jump="1" tolerance="0.01" cartesian="false" task_operations="restrict_to_packing_not_interface" />  -->
                    
            <FastRelax name="fastRelax" scorefxn="sfxn" task_operations="ex1_ex2aro,ic"/>
            <ClearConstraintsMover name="rm_csts" />
            
            <FastDesign name="fastDesign_stage1" scorefxn="sfxn_design" repeats="1" task_operations="precompute_ig,design_task,ex1_ex2aro,ic,limitchi2,noCys,restrict_packing_rifres_res,upweight_interface,restrict_to_packing_not_interface,restrict_target2repacking,restrict_packing_interface" batch="false" ramp_down_constraints="false" cartesian="False" bondangle="false" bondlength="false" min_type="dfpmin_armijo_nonmonotone" relaxscript="MonomerDesign2019"/> /do not design for interace, but allow relax
            
            <FastDesign name="fastDesign_stage2" scorefxn="sfxn_design" repeats="3" task_operations="precompute_ig,design_task,ex1_ex2aro,ic,limitchi2,noCys,upweight_interface,restrict_to_packing_not_interface,restrict_target2repacking,restrict_packing_interface" batch="false" ramp_down_constraints="false" cartesian="False" bondangle="false" bondlength="false" min_type="dfpmin_armijo_nonmonotone" relaxscript="MonomerDesign2019"/> /do not design for interace, but allow relax
            
            MutateResidue name="install_protonated_his" residue_selector="the_hip" new_res="000" preserve_atom_coords="false" mutate_self="false"/>
            MutateResidue name="remove_protonated_his" residue_selector="the_hip" new_res="HIS" preserve_atom_coords="false" mutate_self="false"/>
            DumpPdb name="dump_test" fname="test.pdb" scorefxn="sfxn"/>
            
            <SwitchChainOrder name="chain1onlypre" chain_order="1" />
            <ScoreMover name="scorepose" scorefxn="sfxn" verbose="false" />
            <ParsedProtocol name="chain1only">
                <Add mover="chain1onlypre" />
                <Add mover="scorepose" />
            </ParsedProtocol>
            
            <!-- <ParsedProtocol name="chain1only_no_his_protonation">
                <Add mover="chain1onlypre" />
                Add mover="remove_protonated_his"/>
                <Add mover="scorepose" />
            </ParsedProtocol> -->
            
            <!-- <ParsedProtocol name="short_repack_and_min">
             <Add mover="hard_pack" />
             <Add mover="softish_min" />
             <Add mover="hard_min" />
            </ParsedProtocol> -->
            
            <DeleteRegionMover name="delete_polar" residue_selector="select_polar" rechain="false" />
            <SavePoseMover name="save_pose" restore_pose="0" reference_name="after_design" />
            
          </MOVERS>
          
          <FILTERS>
            <Rmsd name="lig_rmsd_after_final_relax" reference_name="after_design" superimpose_on_all="0" superimpose="1" threshold="5" confidence="0" >
               <span begin_res_num="{ligand_res_number}" end_res_num="{ligand_res_number}"/>
            </Rmsd>

            <MoveBeforeFilter name="move_then_lig_rmsd" mover="fastRelax" filter="lig_rmsd_after_final_relax" confidence="0" />

            <ScoreType name="totalscore" scorefxn="sfxn" threshold="9999" confidence="1"/>
            <ResidueCount name="nres" confidence="1" />
            <CalculatorFilter name="score_per_res" confidence="1" equation="SCORE/NRES" threshold="999">
                <Var name="SCORE" filter_name="totalscore" />
                <Var name="NRES" filter_name="nres" />
            </CalculatorFilter>

            <Geometry name="geometry" omega="165" cart_bonded="20" start="1" end="9999" count_bad_residues="true" confidence="0"/>

            <Ddg name="ddg_norepack"  threshold="0" jump="1" repeats="1" repack="0" relax_mover="min" confidence="0" scorefxn="sfxn"/>  
            <Report name="ddg1" filter="ddg_norepack"/>
            <Report name="ddg2" filter="ddg_norepack"/>


            <ShapeComplementarity name="SC" min_sc="0" min_interface="0" verbose="0" quick="0" jump="1" confidence="0"/>
            HbondsToResidue name="hbonds2lig" scorefxn="sfxn" partners="0" energy_cutoff="-0.5" backbone="0" bb_bb="0" sidechain="1" residue="{2}"/>
            BuriedUnsatHbonds2 name="interf_uhb2" cutoff="200" scorefxn="sfxn" jump_number="1"/>
            <ContactMolecularSurface name="contact_molecular_surface" distance_weight="0.5" target_selector="chainA" binder_selector="chainB" confidence="0" use_rosetta_radii="1"/>
            <Holes name="hole" threshold="20.0" residue_selector="coreRes" exclude_bb_atoms="false" />

            {filters}

            <Sasa name="interface_buried_sasa" confidence="0" />      

            <InterfaceHydrophobicResidueContacts name="hydrophobic_residue_contacts" target_selector="chainB" binder_selector="chainA" scorefxn="sfxn" confidence="0"/>

            <SSPrediction name="pre_mismatch_probability" confidence="0" cmd="/software/psipred4/runpsipred_single" use_probability="1" mismatch_probability="1" use_svm="0" use_scratch_dir="1"/>
            <MoveBeforeFilter name="mismatch_probability" mover="chain1only" filter="pre_mismatch_probability" confidence="0" />

            <SSPrediction name="pre_sspred_overall" cmd="/software/psipred4/runpsipred_single" use_probability="0" use_svm="0" threshold="0.85" confidence="0" use_scratch_dir="1" />
            <MoveBeforeFilter name="sspred_overall" mover="chain1only" filter="pre_sspred_overall" confidence="0" />
            <!-- <MoveBeforeFilter name="clash_check" mover="short_repack_and_min" filter="ddg1" confidence="0"/> -->
            <Ddg name="ddg_hydrophobic_pre"  threshold="-10" jump="1" repeats="1" repack="0" confidence="0" scorefxn="vdw_sol" />
            <MoveBeforeFilter name="ddg_hydrophobic" mover="delete_polar" filter="ddg_hydrophobic_pre" confidence="0"/>

            <ResidueCount name="nMET" count_as_percentage="1" max_residue_count="300" min_residue_count="0" residue_types="MET" confidence="0" />
            <ResidueCount name="nALA" count_as_percentage="1" max_residue_count="300" min_residue_count="0" residue_types="ALA" confidence="0" />
            <ResidueCount name="nARG" count_as_percentage="1" max_residue_count="300" min_residue_count="0" residue_types="ARG" confidence="0" />
            <ResidueCount name="nHIS" count_as_percentage="1" max_residue_count="300" min_residue_count="0" residue_types="HIS" confidence="0" />
            ResidueCount name="ala_loop_count" max_residue_count="300" residue_types="ALA" count_as_percentage="1" residue_selector="loop" confidence="0"/>
            <ResidueCount name="ala_core_count" max_residue_count="300" residue_types="ALA" count_as_percentage="1" residue_selector="core" confidence="0"/>
            ResidueCount name="ala_bdry_count" max_residue_count="300" residue_types="ALA" count_as_percentage="1" residue_selector="boundary" confidence="0"/>
            <ResidueCount name="res_count_all" max_residue_count="9999" confidence="0"/>
            <ScoreType name="hb_lr_bb" scorefxn="sfxn" score_type="hbond_lr_bb" confidence="0" threshold="0"/>
            <ResidueCount name="nres_all"/>
            <CalculatorFilter name="hb_lr_bb_per_res" equation="FAA/RES" threshold="0" confidence="0">
                    <Var name="FAA" filter="hb_lr_bb" />
                    <Var name="RES" filter="nres_all"/>
            </CalculatorFilter>
            <ScoreType name="hb_sr_bb" scorefxn="sfxn" score_type="hbond_sr_bb" confidence="0" threshold="0"/>
            <CalculatorFilter name="hb_sr_bb_per_res" equation="FAA/RES" threshold="0" confidence="0">
                    <Var name="FAA" filter="hb_sr_bb" />
                    <Var name="RES" filter="nres_all"/>
            </CalculatorFilter>
            Worst9mer name="worst9mer" rmsd_lookup_threshold="1.1"  only_helices="false" confidence="0" />
            <Holes name="holes_around_lig" threshold="-0.5" residue_selector="interface" normalize_per_atom="true" exclude_bb_atoms="true" confidence="0"/>
            <CavityVolume name="cavity" confidence="0"/>
            <BuriedUnsatHbonds name="buns_all_heavy_ball" report_all_heavy_atom_unsats="true" scorefxn="sfxn" cutoff="5" residue_surface_cutoff="20.0" ignore_surface_res="true" print_out_info_to_pdb="true" dalphaball_sasa="1" probe_radius="1.1" confidence="0" />
            <BuriedUnsatHbonds name="buns_bb_heavy_ball"  report_bb_heavy_atom_unsats="true"  scorefxn="sfxn" cutoff="5" residue_surface_cutoff="20.0" ignore_surface_res="true" print_out_info_to_pdb="true" dalphaball_sasa="1" probe_radius="1.1" confidence="0" />
            <BuriedUnsatHbonds name="buns_sc_heavy_ball"  report_sc_heavy_atom_unsats="true"  scorefxn="sfxn" cutoff="5" residue_surface_cutoff="20.0" ignore_surface_res="true" print_out_info_to_pdb="true" dalphaball_sasa="1" probe_radius="1.1" confidence="0" /> 
            <BuriedUnsatHbonds name="vbuns_all_heavy" use_reporter_behavior="true" report_all_heavy_atom_unsats="true" scorefxn="sfxn" ignore_surface_res="false" print_out_info_to_pdb="true" atomic_depth_selection="5.5" burial_cutoff="1000" confidence="0" />
            <BuriedUnsatHbonds name="sbuns_all_heavy" use_reporter_behavior="true" report_all_heavy_atom_unsats="true" scorefxn="sfxn" cutoff="4" residue_surface_cutoff="20.0" ignore_surface_res="true" print_out_info_to_pdb="true" dalphaball_sasa="1" probe_radius="1.1" atomic_depth_selection="5.5" atomic_depth_deeper_than="false" confidence="0" />
            <MoveBeforeFilter name="vbuns_all_heavy_no_ligand" mover="chain1only" filter="vbuns_all_heavy" confidence="0" />
            <MoveBeforeFilter name="sbuns_all_heavy_no_ligand" mover="chain1only" filter="sbuns_all_heavy" confidence="0" />
            <DSasa name="dsasa" lower_threshold="0" upper_threshold="1"/> 
            <ShapeComplementarity name="interface_sc" verbose="0" min_sc="0.55" write_int_area="1" write_median_dist="1" jump="1" confidence="0"/>

            <Time name="timed"/>
          </FILTERS>
          
        <SIMPLE_METRICS>
            <SapScoreMetric name="sap" />
            <SapScoreMetric name="sap_A"
                score_selector="chainA"
                sap_calculate_selector="chainA" sasa_selector="chainA" />
            <SecondaryStructureMetric name="dssp_string" />

        </SIMPLE_METRICS>

        <MOVERS>
            /turn on and off for test
            <DumpPdb name="after_fd1" fname="after_fd1_noFD.pdb" tag_time="1" scorefxn="sfxn" />
            <DumpPdb name="after_fd2" fname="after_fd2_noFD.pdb" tag_time="1" scorefxn="sfxn" />
            <DumpPdb name="after_relax" fname="after_relax_noFD.pdb" tag_time="1" scorefxn="sfxn" />
            <DumpPdb name="after_rmsd" fname="after_rmsd_noFD.pdb" tag_time="1" scorefxn="sfxn" />
        </MOVERS>
          
          
        <PROTOCOLS>
          <Add filter_name="timed"/>
          Add mover="FSP"/>
          Add filter="is_target_hbond_maintained"/>
          Add mover="install_protonated_his"/>
          Add mover="short_repack_and_min"/>
          Add filter="is_target_hbond_maintained"/>
          Add filter="ddg1"/>
          <Add mover="label_core"/>
          <Add mover="add_bb_cst"/>

          # turn off for only FR the MPNN model+ligand
          Add mover="fastDesign_stage1"/> 
          Add mover="after_fd1"/>
          Add mover="fastDesign_stage2"/> 
          Add mover="after_fd2"/>
          Add mover="save_pose"/>    
          <Add mover="fastRelax"/>
          Add mover="after_relax"/>
          Add filter="move_then_lig_rmsd"/> 
          Add mover="after_rmsd"/>
          <Add mover="rm_bb_cst"/>
                 
          <Add filter="score_per_res"/>
          <Add filter="geometry"/>

          <Add filter="contact_molecular_surface"/>
          <Add filter="ddg2"/>
          <Add filter="interface_buried_sasa"/>
          <Add filter="SC"/>
          <Add filter="holes_around_lig"/>
          <Add filter="nMET"/>
          <Add filter="nALA"/>
          <Add filter="nARG"/>
          <Add filter="nHIS"/>
          Add filter="ala_bdry_count"/>
          <Add filter="ala_core_count"/>
          Add filter="ala_loop_count"/>
          <Add filter="hb_lr_bb"/>
          <Add filter="hb_lr_bb_per_res"/>
          <Add filter="hb_sr_bb"/>
          <Add filter="hb_sr_bb_per_res"/>
          Add filter="worst9mer"/>
          Add filter="hole"/>
          <Add filter="cavity"/>

          {protocols}
                    
          <Add filter="vbuns_all_heavy"/>
          <Add filter="sbuns_all_heavy"/>
          <Add filter="buns_all_heavy_ball"/>
          <Add filter="buns_bb_heavy_ball"/>
          <Add filter="buns_sc_heavy_ball"/>
          <Add filter="vbuns_all_heavy_no_ligand"/>
          <Add filter="sbuns_all_heavy_no_ligand"/>

          <Add filter="hydrophobic_residue_contacts"/>
          <Add filter="mismatch_probability"/>
          <Add filter="sspred_overall"/>

          <Add filter="dsasa" /> /measure ligand burial rate, 0 is totally
          <Add filter="interface_sc"/>
          <Add metrics="sap_A" labels="sap_A"/>
          <Add metrics="sap" labels="sap_all"/>
          <Add filter_name="timed"/>

        </PROTOCOLS>
        <OUTPUT scorefxn="sfxn" /> /LA: this is the only way to export score to PDB file
    </ROSETTASCRIPTS>
    """
    #========================================================================
    # Build option parser
    #========================================================================
    print('generated xml')
    key_contacts = get_all_close_res(pose, ligand_res_number)
    print(f'freese {key_contacts}')
    Add_key_contacts_to_rifres = f"""//////////Add key contacts////////
    <Index name="key_residues" resnums="{key_contacts}"/>
    <And name="dock_rifres_res" selectors="all_rifres_res,not_AVLI" />
    <Or name="rifres_res" selectors="dock_rifres_res,key_residues" />
        ////////////////
        """
    xml = xml.replace('<And name="rifres_res" selectors="all_rifres_res,not_AVLI" />', Add_key_contacts_to_rifres)

    design_task = rosetta_scripts.SingleoutputRosettaScriptsTask(xml)
    print('get xml')
    design_task.setup()
    print('setup xml')

    cst_list = get_all_atom_close_csts( pose, ligand_res_number, bb_only=False, sd=constraint_sd, no_ligand_cst=False)
    for cst in cst_list:
        pose.add_constraint(cst)
    print('added cst')

    designed_pose = design_task.apply(pose)
    print(f'finished design')

    t1 = time.time()
    print("Design took ", t1-t0)

    return designed_pose

def get_all_close_res(pose, ligand_res_number):
    """
    get cst for ligand
    atm_list - one ligand atms, use for estimate distance
    usage:
    cst_list = get_all_atom_close_csts(pose, bb_only=False, bb_sd=0.5, sc_sd=10.0)

        for cst in cst_list:
            pose.add_constraint(cst)
    """
    print(pose.pdb_info())
    # bbs = ["N", "O", "C", "CA", "CB"]
    ligand = int(ligand_res_number)
    close_res = []
    close_dist_cutoff = 5
    for resi in range(1, pose.size()):
        for at_i in range(1, pose.residue(resi).natoms() + 1):
            for at_j in range(1, pose.residue(ligand).natoms() + 1):
                if pose.residue(resi).xyz("CA").distance_squared(pose.residue(ligand).xyz(pose.residue(ligand).atom_name(at_j).strip())) >= 100:
                    continue
                id_i = pyrosetta.rosetta.core.id.AtomID(at_i, resi)
                id_j = pyrosetta.rosetta.core.id.AtomID(at_j, ligand)

                i_j_dist = pose.residue(resi).xyz(at_i).distance(pose.residue(ligand).xyz(at_j))

                if (i_j_dist < close_dist_cutoff) and (resi not in close_res):
                    close_res.append(resi)
    if ligand in close_res:
        close_res.remove(ligand)

    print(f'{len(close_res)} close_res {close_res} generated')

    return ','.join(str(i) for i in set(close_res))

def get_all_atom_close_csts(pose, ligand_res_number, bb_only=False, sd=1.0, no_ligand_cst=False):
    """
    get cst for ligand
    atm_list - one ligand atms, use for estimate distance
    usage:
    cst_list = get_all_atom_close_csts(pose, bb_only=False, bb_sd=0.5, sc_sd=10.0)

        for cst in cst_list:
            pose.add_constraint(cst)
    """
    # bbs = ["N", "O", "C", "CA", "CB"]
    cst_list = []
    ligand = int(ligand_res_number)
    if no_ligand_cst == False:
        for resi in range(1, pose.size()):
            for at_i in range(1, pose.residue(resi).natoms() + 1):
                if pose.residue(resi).atom_name(at_i).strip() != "CA": #only do CA, could do CB instead but then need logic for GLY
                    continue    
                best_dist = 11
                cst = ""
                for at_j in range(1, pose.residue(ligand).natoms() + 1):
                    if "H" in pose.residue(ligand).atom_name(at_j).strip():
                        continue
                    if pose.residue(resi).xyz("CA").distance_squared(pose.residue(ligand).xyz(pose.residue(ligand).atom_name(at_j).strip())) >= 100:
                        continue
                    id_i = pyrosetta.rosetta.core.id.AtomID(at_i, resi)
                    id_j = pyrosetta.rosetta.core.id.AtomID(at_j, ligand)

                    i_j_dist = pose.residue(resi).xyz(at_i).distance(pose.residue(ligand).xyz(at_j))

                    if i_j_dist < best_dist:
                        best_dist = i_j_dist
                        func = pyrosetta.rosetta.core.scoring.func.HarmonicFunc(best_dist, sd)
                        cst = pyrosetta.rosetta.core.scoring.constraints.AtomPairConstraint(id_i, id_j, func)
                # if cst != "":
                        cst_list.append(cst) 
    print(f'after generating protein-ligand cst {len(cst_list)}')
    for resi in range(1, pose.size()):
        for resj in range(1, pose.size()):
            for at_i in range(1, pose.residue(resi).natoms() + 1):
                if pose.residue(resi).atom_name(at_i).strip() != "CA": #only do CA, could do CB instead but then need logic for GLY
                    continue    
                best_dist = 11
                cst = ""
                for at_j in range(1, pose.residue(resj).natoms() + 1):
                    if pose.residue(resj).atom_name(at_j).strip() != "CA": #only do CA, could do CB instead but then need logic for GLY
                        continue 
                    elif pose.residue(resi).xyz("CA").distance_squared(pose.residue(resj).xyz("CA")) >= 100:
                        continue
                    id_i = pyrosetta.rosetta.core.id.AtomID(at_i, resi)
                    id_j = pyrosetta.rosetta.core.id.AtomID(at_j, resj)

                    i_j_dist = pose.residue(resi).xyz(at_i).distance(pose.residue(resj).xyz(at_j))

                    if i_j_dist < best_dist:
                        best_dist = i_j_dist
                        func = pyrosetta.rosetta.core.scoring.func.HarmonicFunc(best_dist, sd)
                        cst = pyrosetta.rosetta.core.scoring.constraints.AtomPairConstraint(id_i, id_j, func)
                # if cst != "":
                        cst_list.append(cst)

    print(f'Add protein bb cst, in total {len(cst_list)} cst generated')

    return cst_list

def generate_hb_filters(atm_list,scorefxn,ligand_res_number):
    filters,protocols = [],[]
    for atm in atm_list:
        filters.append(f'<SimpleHbondsToAtomFilter name="hb_to_{atm}" n_partners="1" hb_e_cutoff="-0.3" target_atom_name="{atm}" res_num="{ligand_res_number}" scorefxn="{scorefxn}" confidence="0"/>')
        protocols.append(f'<Add filter="hb_to_{atm}"/>')
    return '\n'.join(filters),'\n'.join(protocols)

def score_dict_from_pdb(fn):
    """Loads scores from a PDB file into an OrderedDict"""
    with open(fn) as f:
        lines = [line.strip() for line in f.readlines()]

    record = OrderedDict()
    is_reading = False
    for i,line in enumerate(lines):
        # read score terms
        if line.startswith('#BEGIN_POSE_ENERGIES_TABLE'):
            assert(lines[i+1].startswith('label'))
            assert(lines[i+3].startswith('pose'))
            keys = lines[i+1].split()[1:]
            values = lines[i+3].split()[1:]
            record.update(OrderedDict(zip(keys,values)))

        # read metrics/filters
        if line.startswith('#END_POSE_ENERGIES_TABLE'):
            is_reading = True
            continue
        if is_reading and len(line)>0:
            tokens = line.split()
            record[tokens[0]] = tokens[1]

    for k in record:
        record[k] = float(record[k])

    return record

if __name__ == '__main__':
    args = parse_args(None)
    main(args)

