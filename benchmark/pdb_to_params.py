#!/usr/bin/env python

import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, root_dir)

import subprocess
import openbabel
import fire
import run_inference
import aa_model
import glob
import numpy as np
from icecream import ic
from collections import defaultdict

def aux_file(rundir, pdb, ligand, kind):
    input_dir = os.path.join(rundir, 'input')
    pdb_name = trim_suffix(os.path.basename(pdb), '.pdb')
    return os.path.join(input_dir, kind, f'{pdb_name}_{ligand}.{kind}')

def convert_all(indir):
    pdb_ligand = set()
    pdb_ligand_count = defaultdict(list)
    for trb_path in glob.glob(os.path.join(indir, '*.trb')):
        trb = np.load(trb_path,allow_pickle=True)
        pdb_ligand.add((trb['config']['inference']['input_pdb'], trb['config']['inference']['ligand']))
        pdb_ligand_count[((trb['config']['inference']['input_pdb'], trb['config']['inference']['ligand']))].append(trb_path)
    
    ic(pdb_ligand_count)
    pdb_ligand_count = {k: len(v) for k,v in pdb_ligand_count.items()}
    ic(pdb_ligand_count)

    for pdb, ligand in pdb_ligand:
        if ligand is None:
            continue
        print(f'Making params for pdb: {pdb} ligand: {ligand}')
        mol2_path = aux_file(indir, pdb, ligand, 'mol2')
        ligandmpnn_params_path = aux_file(indir, pdb, ligand, 'params')
        for p in (mol2_path, ligandmpnn_params_path):
            d = os.path.dirname(p)
            os.makedirs(d, exist_ok=True)
        convert(pdb, mol2_path, ligand)
        params_from_mol2(mol2_path, ligandmpnn_params_path, ligand)

def convert(pdb, mol2, ligand):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdb", "mol2")

    mol = openbabel.OBMol()
    with open(pdb, 'r') as fh:
        stream = [l for l in fh if "HETATM" in l or "CONECT" in l]
    stream = aa_model.filter_het(stream, ligand)
    stream = "".join(stream)
    obConversion.ReadString(mol, stream)

    obConversion.WriteFile(mol, mol2)

def trim_suffix(s, suffix):
    if s.endswith(suffix):
        s = s[:-(len(suffix))]
    return s

def params_from_mol2(mol2, params, ligand):
    params = trim_suffix(params, '.params')

    # the -n argument MUST be name of ligand, and this will ONLY output <ligandname>.params to the CWD
    proc = subprocess.run(f'/software/rosetta/main/source/scripts/python/public/molfile_to_params.py --keep-names --clobber  {mol2} -n {ligand}',
        shell=True)

    if proc.returncode != 0:
        raise Exception(f'params_from_mol2({mol2}, {params} failed')

    # move/rename the .params file after making it to be more tidy
    os.rename(ligand+'.params', params+'.params')
    print(f'Moved {ligand}.params to {params}.params')

if __name__ == '__main__':
    fire.Fire(convert_all)
