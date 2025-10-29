import logging
LOGGER = logging.getLogger(__name__)
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
sys.path.append('/home/ahern/tools/pdb-tools/')
from dev import analyze
import shutil
import glob
from tqdm import tqdm
import fire
import numpy as np
import Bio.PDB.PDBParser

def get_input_aligned_pdb(row, out_path=None):
    input_pdb = analyze.get_input_pdb(row)
    des_pdb = analyze.get_design_pdb(row)
    input_p = analyze.sak.parse_pdb(input_pdb)
    des_p = analyze.sak.parse_pdb(des_pdb)
    self_idx, other_idx = analyze.get_idx_motif(row, mpnn=False)
    trb = analyze.get_trb(row)
    other_ch = trb['con_ref_pdb_idx'][0][0]
    self_ch = 'A'
    # LOGGER.debug(self_ch, other_ch, other_ch, other_idx)
    des_p = des_p.aligned_to_chain_idxs(input_p, self_ch, self_idx, other_ch, other_idx)
    des_p.chains[self_ch].xyz[self_idx, 3:] = input_p[other_ch].xyz[other_idx, 3:]
    aligned_path = des_p.write_pdb(out_path)
    return aligned_path
    

def get_input_aligned_pdb_with_ligand(row, out_path):
    pdb = get_input_aligned_pdb(row)
    substrate_name = row['inference.ligand']
    input_pdb = analyze.get_input_pdb(row)
    with open(input_pdb) as fh, open(pdb) as aligned:
        o = pdb_selresname.run(fh, substrate_name)
        o = pdb_selhetatm.run(o)
        o = pdb_merge.run([o, aligned])
        o = pdb_sort.run(o, [])
        o = pdb_tidy.run(o)
        
        with open(out_path, 'w') as of:
            for l in o:
                of.write(l)
    
    shutil.copy(get_trb(analyze.get_design_pdb(row)), get_trb(out_path))
        
def get_trb(pdb):
    return pdb[:-4] + '.trb'

def get_res_atm_idx(pdb, res_idx, atom_id):
    parser = Bio.PDB.PDBParser()
    model = parser.get_structure('a', pdb)
    for residue in model.get_residues():
        if residue.get_id()[1] == res_idx:
            for atom in residue.get_atoms():
                if atom.get_id() == atom_id:
                    return atom.serial_number
            raise Exception('atom not found')
    raise Exception('residue not found')


def get_hetatm(pdb, chain, atom_id):
    parser = Bio.PDB.PDBParser()
    model = parser.get_structure('a', pdb)
    for residue in model[0][chain].get_residues():
        for atom in residue:
            if atom.get_id() == atom_id:
                return atom.serial_number
    raise Exception('atom not found')

def main(input_dir, ref_ch, ref_res, ref_prot_atom, ref_het_atom, output_dir=None, prefix='', cautious=False):
    '''
    For each PDB in the input directory, create a PDB with a protein-ligand bond specified by
    ref_ch, ref_res, ref_prot_atom, and ref_het_atom.
    '''
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'bonded')
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    pdbs_to_graft = glob.glob(os.path.join(input_dir, '*.pdb'))
    pdbs_to_graft.sort()
    for pdb in tqdm(pdbs_to_graft):
        out_pdb = os.path.join(output_dir, prefix + os.path.split(pdb)[1])
        LOGGER.debug(cautious)
        if cautious and os.path.exists(out_pdb):
            continue

        # Do it here.
        # row = analyze.make_row_from_traj(pdb[:-4])
        # mpnned = os.path.join(row['rundir'], 'ligandmpnn', 
        root_dir, filename = os.path.split(pdb)
        name, _ = os.path.splitext(filename)
        name = '_'.join(name.split('_')[:-1])
        # trb = os.path.join(os.path.dirname(root_dir), name + '.trb')
        # print(trb)
        # assert os.path.exists(trb)
        # trb = np.load(trb, allow_pickle=True)
        # LOGGER.debug(trb)
        orig_name = os.path.join(os.path.dirname(root_dir), name)
        row = analyze.make_row_from_traj(orig_name)
        # self_idx, other_idx = analyze.get_idx_motif(row, mpnn=False)
        trb = analyze.get_trb(row)
        pdb_des_from_ref = {ref: des for ref, des in zip(trb['con_ref_pdb_idx'], trb['con_hal_pdb_idx'])}
        ref_res = int(ref_res)
        LOGGER.debug(pdb_des_from_ref)
        _, des_res_pdb_idx = pdb_des_from_ref[(ref_ch, ref_res)]
        LOGGER.debug(pdb, des_res_pdb_idx, ref_prot_atom)
        prot_atm_idx = get_res_atm_idx(pdb, des_res_pdb_idx, ref_prot_atom)
        het_atm_idx = get_hetatm(pdb, 'B', ref_het_atom)
        LOGGER.debug(prot_atm_idx, het_atm_idx)
        shutil.copy(pdb, out_pdb)
        with open(out_pdb, 'a') as fh:
            a_i = str(prot_atm_idx)
            a_j = str(het_atm_idx)
            LOGGER.debug('writing')
            fh.write(f"CONECT{a_i:>5}{a_j:>5}\n")
    
    print(f'grafted PDBs from {input_dir} to {output_dir}')

if __name__ == '__main__':
    fire.Fire(main)