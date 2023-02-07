#!/usr/bin/env python
#
# Threads MPNN sequences onto original PDBs

import sys, os, argparse, glob, tqdm
import numpy as np
from icecream import ic
import torch

script_dir = os.path.dirname(os.path.realpath(__file__))+'/'
sys.path.insert(0,script_dir+'/../')
sys.path.insert(0,script_dir+'/../RF2-allatom/')
import inference.utils
import rf2aa.parsers
import rf2aa.chemical


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir',type=str,help='Folder of designs')
    parser.add_argument('--seqdir',type=str,help='Folder of MPNN sequences, one .fa file per design')
    parser.add_argument('--outdir',type=str,help='Folder to put threaded MPNN PDBs')
    parser.add_argument('--use_ligand',action='store_true',default=False,
        help='Whether to parse and write ligand in the PDB. Also affects default I/O paths')
    parser.add_argument('--init_sc',action='store_true',default=False,
        help='Initialize sidechain coordinates in output pdb to C-alpha. For running chemnet afterwards.')
    args = parser.parse_args()

    if args.seqdir is None: 
        args.seqdir = args.datadir+('/ligmpnn/seqs/' if args.use_ligand else '/mpnn/seqs/')
    if args.outdir is None: 
        args.outdir = args.datadir+('/ligmpnn/' if args.use_ligand else '/mpnn/')

    return args

def main():
    args = get_args()

    for fn in tqdm.tqdm(glob.glob(args.datadir+'/*.pdb')):
        print('processing',fn)
        name = os.path.basename(fn).replace('.pdb','')

        # parse protein
        parsed_pdb = inference.utils.parse_pdb(fn)
        xyz_prot = torch.tensor(parsed_pdb['xyz'])
        mask_prot = torch.tensor(parsed_pdb['mask'])
        idx_prot = torch.tensor([x[1] for x in parsed_pdb['pdb_idx']])
        
        L_prot, N_atoms_prot = xyz_prot.shape[:2]
        Ls = [L_prot]

        # parse ligand
        if args.use_ligand:
            with open(fn, 'r') as fh:
                stream = [l for l in fh if "HETATM" in l or "CONECT" in l]
            if len(stream)==0:
                sys.exit('ERROR (thread_mpnn.py): --use_ligand set but no HETATM records found '\
                         'in input PDB file.')

            mol, msa_sm, ins_sm, xyz_sm, mask_sm = \
                rf2aa.parsers.parse_mol("".join(stream), filetype="pdb", string=True)
            G = rf2aa.util.get_nxgraph(mol)
            bond_feats_sm = rf2aa.util.get_bond_feats(mol)

            L_sm = xyz_sm.shape[1]
            Ls = [L_prot, L_sm]
            idx_sm = torch.full((L_sm,), idx_prot.max()+10)

            atom_names = []
            lig_names = []
            for line in stream:
                if line.startswith('HETATM'):
                    # https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html
                    atom_names.append(line[12:16])
                    lig_names.append(line[17:20].strip())
        else:
            # blank ligand features for concatenating
            xyz_sm = torch.zeros((1,0,3))
            mask_sm = torch.zeros((1,0))
            msa_sm = torch.tensor([])
            idx_sm = torch.tensor([])
            atom_names = None
            lig_names = ['LG1']
            bond_feats_sm = torch.tensor([])
            
        # combine protein & ligand features
        L = sum(Ls)
        xyz = torch.zeros((L,rf2aa.chemical.NTOTAL,3))
        xyz[:L_prot, :N_atoms_prot] = xyz_prot
        xyz[L_prot:, 1] = xyz_sm[0,:]

        # initialize sidechains to C-alpha coords (for chemnet)
        if args.init_sc:
            xyz[:L_prot, 4:] = xyz_prot[:, 1:2]

        mask = torch.zeros((L, rf2aa.chemical.NTOTAL)).bool()
        mask[:L_prot, :4] = mask_prot[:, :4] # remove sidechain atoms
        mask[L_prot:, 1] = mask_sm[0]

        bond_feats = torch.zeros((L,L))
        bond_feats[:L_prot, :L_prot] = rf2aa.util.get_protein_bond_feats(L_prot)
        bond_feats[L_prot:, L_prot:] = bond_feats_sm

        idx = torch.cat([idx_prot, idx_sm])

        with open(args.seqdir+name+'.fa') as f:
            lines = f.readlines()
            n_designs = int(len(lines)/2-1)
            for i in range(n_designs):
                print('writing file', args.outdir+name+f"_{i}.pdb")
                seq = lines[2*i + 3].strip() # 2nd seq is 1st design
                seq_num = torch.tensor([rf2aa.util.aa2num[rf2aa.util.aa_123[a]] for a in seq])

                if args.init_sc:
                    mask[:L_prot, :14] = rf2aa.util.allatom_mask[seq_num][:,:14]

                pdbstr = rf2aa.util.writepdb(
                    args.outdir+name+f"_{i}.pdb",
                    atoms = xyz,
                    atom_mask = mask,
                    seq = torch.cat([seq_num, msa_sm]).long(),
                    idx_pdb = idx,
                    chain_Ls = Ls,
                    bond_feats = bond_feats[None],
                    atom_names = atom_names,
                    lig_name = lig_names[0]
                )

if __name__ == "__main__":
    main()
