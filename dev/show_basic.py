import os
import sys

import fire


def main(
    design, # Path to pdb
    rf_diffusion_dir='/home/ahern/projects/dev_rf_diffusion',
    keep=False
    ):

    sys.path.insert(0, rf_diffusion_dir)
    from dev import analyze
    cmd = analyze.cmd

    pdb_prefix = os.path.splitext(design)[0]
    srow=analyze.make_row_from_traj(pdb_prefix)

    if not keep:
        analyze.sak.clear(cmd)
        cmd.do('@~/.pymolrc')
    structures = analyze.show_motif_simple(srow, srow['name'], traj_types=['des', 'X0', 'Xt'], show_af2=False)
    native = structures['native']
    des = structures['trajs'][0]
    cmd.hide('everything', native.name)
    cmd.center(des.name)

    for structure in [native, des]:
        cmd.show('licorice', f'{structure.name} and resn {srow["inference.ligand"]}')
        cmd.show('licorice', f'{structure.motif_sele()}')


if __name__ == '__main__':
    fire.Fire(main)

