import os
import sys
from icecream import ic

import fire


def main(
    design, # Path to pdb
    rf_diffusion_dir='/home/ahern/projects/dev_rf_diffusion',
    keep=False,
    remote_ip=None,
    ):
    '''
    Shows a diffusion trajectory, native motif, and AF2 (if present) for a given design.
    If the design pdb is /dir/pdb_0.pdb, call this function with design=/dir/pdb_0
    '''

    sys.path.insert(0, rf_diffusion_dir)
    from dev import analyze
    cmd = analyze.cmd
    if remote_ip:
        cmd = analyze.set_remote_cmd(remote_ip)
        analyze.cmd = cmd

    pdb_prefix = os.path.splitext(design)[0]
    srow=analyze.make_row_from_traj(pdb_prefix)

    if not keep:
        analyze.sak.clear(cmd)
        cmd.do('@~/.pymolrc')
    
    has_af2 = os.path.exists(analyze.get_af2(srow))
    structures = analyze.show_motif_simple(srow, srow['name'], traj_types=['des', 'X0', 'Xt'], show_af2=has_af2)
    native = structures['native']
    des = structures['trajs'][0]
    des, X0, Xt = structures['trajs']
    # cmd.show_as('cartoon', native.name)
    # ic(structures)
    cmd.color('paper_melon', des.name)
    cmd.color('paper_lightblue', X0.name)
    cmd.color('paper_darkblue', Xt.name)
    cmd.hide('everything', des.name)
    cmd.center(des.name)

    ic(native)
    if not native:
        ic('not native')
        native_pdb = analyze.get_input_pdb(srow)
        native = analyze.Structure(srow['name']+'_native', [])
        cmd.load(native_pdb, native.name)
        cmd.super(native.name, X0.name)
    if has_af2:
        af2 = structures['af2']
        ic(af2)
        cmd.super(af2.name, X0.name)

    for structure in [native, des]:
        if srow["inference.ligand"]:
            cmd.show('licorice', f'{structure.name} and resn {srow["inference.ligand"]}')
        # cmd.show('licorice', f'{structure.motif_sele()}')


if __name__ == '__main__':
    fire.Fire(main)

