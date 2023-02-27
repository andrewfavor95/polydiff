import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
sys.path.append('/home/ahern/tools/pdb-tools/')
import shutil
import glob
from icecream import ic
from tqdm import tqdm
import fire
from pdbtools import *

def main(input_dir, output_dir=None, prefix='', cautious=True):
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'renumbered_chains')
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    pdbs_to_graft = glob.glob(os.path.join(input_dir, '*.pdb'))
    pdbs_to_graft.sort()
    for pdb in tqdm(pdbs_to_graft):
        out_pdb = os.path.join(output_dir, prefix + os.path.split(pdb)[1])
        if cautious and os.path.exists(out_pdb):
            continue
        with open(pdb) as fh, open(pdb) as fh2:
            prot = pdb_delhetatm.run(fh)
            het = pdb_selhetatm.run(fh2)
            het = pdb_rplchain.run(het, ('A', 'B'))
            o = pdb_merge.run([prot, het])
            # o = pdb_sort.run(o, [])
            o = pdb_tidy.run(o)
        
            with open(out_pdb, 'w') as of:
                for l in o:
                    of.write(l)

if __name__ == '__main__':
    fire.Fire(main)