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
from pdbtools import *

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

def main(input_dir, output_dir=None, prefix=''):
    '''
    For each PDB in the input directory, create a PDB with the native motif sidechains grafted onto the design.
    '''
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'grafted')
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    pdbs_to_graft = glob.glob(os.path.join(input_dir, '*.pdb'))
    pdbs_to_graft.sort()
    for pdb in tqdm(pdbs_to_graft):
        out_pdb = os.path.join(output_dir, prefix + os.path.split(pdb)[1])
        if os.path.exists(out_pdb):
            continue
        row = analyze.make_row_from_traj(pdb[:-4])
        get_input_aligned_pdb_with_ligand(row, out_pdb)
    
    print(f'grafted PDBs from {input_dir} to {output_dir}')

if __name__ == '__main__':
    fire.Fire(main)