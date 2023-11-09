from abc import ABC, abstractmethod
import numpy as np, itertools, torch
from opt_einsum import contract as einsum

from inference.dynamic_parameters import *
import torch
import sys

import willutil as wu
from willutil.sym import SymElem
from willutil.pdb.pdbread import readpdb as wu_readpdb
from willutil.pdb.pdbdump import dumppdb as wu_dumppdb
from willutil.sym.sym import axes as symaxes
from willutil.sym.symfit import aligncx
from willutil.homog.hgeom import hcross, hxform, hrot, hconstruct
from willutil.homog.thgeom import t_rot
from willutil.search.slidedock import slide_dock_oligomer
from willutil.motif.motif_placer import place_motif_dme_fast
import ipdb

def create_motif_manager(conf, device='cuda'):
    rfmotif = MotifManager(conf, device)
    rfmotif.printinfo()
    rfmotif.dumppdbs('test_motifs')
    return rfmotif

class RFMotif(ABC):
    @abstractmethod
    def printinfo(self, prefix):
        pass

    @abstractmethod
    def dumppdb(self, fname, alignto=None, filemode=None, startchain=0, doseqaln=False):
        pass

    @abstractmethod
    def apply_motif_RT(self, xyz, mask, symmsub, symmRs, alpha=1.0, debuglabel=None):
        return newxyz, mask, wu.Bunch(extra_info=None)

    @abstractmethod
    def apply_motif_t2d(self, t2d, mask, xyz, xyz_to_t2d_func, symmsub):
        pass

    def start_new_design(self):
        pass

class MotifManager:
    def __init__(self, hconf, device=None):
        self.hconf = hconf
        self.init_dynamic_parameters()
        self.device = device
        # 

        # Make visible motif indices
        abet = 'abcdefghijklmnopqrstuvwxyz'
        abet = [a for a in abet]
        abet2num = {a:i for i,a in enumerate(abet)} 

        self.ij_visible_int = [[abet2num[a] for a in s] for s in self.hconf.inference.ij_visible.split('-')]
        


        self._set_regions_AF(hconf.contigmap.contigs)
        pointsyms = hconf.rfmotif.pointsyms or None #* len(hconf.rfmotif.pdbs)
        self._sanity_check(self.hconf)
        self.float_seq_pos = self.hconf.rfmotif.float_seq_pos
        self.motifs = list()
        self.i_des = None
        flip = hconf.rfmotif.flip or False #* len(hconf.rfmotif.pdbs)


        pdb = self.hconf.inference.input_pdb

        for i in range(len(self.motifregions)):

            if pointsyms is not None:
                m = PointSymMotif(pdb, self.motifregions[0], pointsyms, hconf.inference.symmetry, flip=flip, device=self.device, index=1, fit_only_junctions=hconf.rfmotif.fit_only_junctions, manager=self)
                print('THIS LINE IS PROBABLY BUGGY!!!! WAS ONLY MADE FOR ASYM SO FAR!!!!')
            else:
                m = AsymMotif(pdb, self.motifregions[i], device=self.device, index=i, fit_only_junctions=hconf.rfmotif.fit_only_junctions, region_fudge_factor=hconf.rfmotif.region_fudge_factor, manager=self)

            self.motifs.append(m)

        # self.motifs.append(m)

    def init_dynamic_parameters(self):
        self.params = DynamicParameters()
        self.params.add_hydraconf(self.hconf)
        self.params.add_param('disable_motifs_this_iter', False)

        # self.params.add_param('do_apply_motifs_t2d', FalseOnIters(diffuse=[0], rfold=None))
        self.params.add_param('do_apply_motifs_t2d', False)

        # self.params.add_param('do_apply_motifs_rfold', FalseOnIters(diffuse=[0,1,2,3,4], rfold=None))
        # self.params.add_param('do_apply_motifs_rfold', False)
        self.params.add_param('do_apply_motifs_rfold', True)        

        self.params.add_param('do_motif_sequence_alignment', TrueOnIters(diffuse=None, rfold=0))

    def start_new_design(self, i_des, x_t, symmsub):
        self.i_des = i_des
        self.nres = len(x_t)
        self.nsub = 1 if symmsub is None else len(symmsub)
        self.nasym = self.nres // self.nsub
        for m in self.motifs: m.start_new_design()

    def start_new_design_step(self, i_des, i_step, n_step, x_t, symmsub):
        if i_step == n_step:
            ic('start_new_design')
            self.start_new_design(i_des, x_t, symmsub)
        self.i_step, self.n_step = i_step, n_step
        num_rfold_steps = 40
        self.params.set_progress(step=(i_des, n_step - i_step, -1), totstep=(self.hconf.inference.num_designs, n_step, num_rfold_steps))

    def new_rfold_iter(self, tag):
        self.params.new_rfold_iter(tag)

    def apply_motifs_t2d(self, t2d, xyz, xyz_to_t2d_func, symmsub):
        if self.params.disable_motifs_this_iter: return t2d
        if not self.params.do_apply_motifs_t2d: return t2d

        xyz = xyz[0, 0, :, :3]
        assert not torch.allclose(xyz, torch.tensor(0.0))
        nsub = 1 if symmsub is None else len(symmsub)
        mask = torch.zeros(len(xyz) // nsub, dtype=bool, device=t2d.device)
        for m in self.motifs:
            t2d, mask, extra = m.apply_motif_t2d(t2d, mask, xyz, xyz_to_t2d_func, symmsub)

            # slice in 2D information for the template
            # template_coords = torch.zeros(self.symmetry.Lasu,27,3).float().to(xyz_t.device)
            # template_coords[self.motif_mask[:,:self.symmetry.Lasu].squeeze()] = self.target_feats['xyz_27'].to(xyz_t.device)
            # template_t2d = xyz_to_t2d(template_coords[None,None,...])
            # template_t2d_mask = self.motif_mask.unsqueeze(-1).expand_as(t2d)
            # t2d[template_t2d_mask] = template_t2d.repeat(1,1,self.symmetry.symmmeta[1][0],self.symmetry.symmmeta[1][0],1)[template_t2d_mask]

        return t2d

    def apply_motifs_RT(self, R_in, T_in, xyzorig_in, symmsub, symmRs, alpha=1.0, debuglabel=None, dock=False):
        if self.params.disable_motifs_this_iter: return R_in, T_in
        if not self.params.do_apply_motifs_rfold: return R_in, T_in

        assert len(self.motifs), 'apply_motifs_RT called with no motifs, maybe this is a mistake...'
        assert len(R_in) == len(T_in) == len(xyzorig_in) == 1

        # np.save('R.npy', R.cpu().numpy())
        # np.save('T.npy', T.cpu().numpy())
        # np.save('xyzorig_in.npy', xyzorig_in.cpu().numpy())
        # np.save('symmsub.npy', symmsub.cpu().numpy())
        # np.save('symmRs.npy', symmRs.cpu().numpy())
        # assert 0

        R, T, xyzorig = R_in[0], T_in[0], xyzorig_in[0]
        xyz = einsum('rij,raj->rai', R, xyzorig) + T[:, None]

        # debuglabel = 'TEST'
        # if debuglabel:
        # wu_dumppdb(f'{debuglabel}_IN.pdb', xyz.reshape(len(symmsub), -1, 3, 3))

        nsub = 1 if symmsub is None else len(symmsub)
        mask = torch.zeros(len(R) // nsub, dtype=bool, device=R.device)
        for m in self.motifs:
            xyz, mask, extra = m.apply_motif_RT(
                xyz=xyz,
                mask=mask,
                symmsub=symmsub,
                symmRs=symmRs,
                alpha=alpha,
                debuglabel=debuglabel,
            )
        mask = mask.tile(nsub)

        nasym = len(R) // nsub
        if dock and 'cxaxis' in extra:
            # ic(extra.cxaxis)
            # wu_dumppdb('test_dock_pre.pdb', xyz.reshape(nsub,-1,3,3))
            docked = self.dock(xyz[:nasym], extra.cxaxis)
            xyz[:nasym] = torch.tensor(docked, device=xyz.device, dtype=xyz.dtype)
            # wu_dumppdb('test_dock_post_asym.pdb', xyz[:nasym])

        if symmsub is not None:
            xyz = torch.einsum('sij,raj->srai', symmRs[symmsub], xyz[:nasym]).reshape(-1, 3, 3)

        # if dock:
        # wu_dumppdb('test_dock_post_resym.pdb', xyz.reshape(nsub,-1,3,3))
        # assert 0

        # oldstubs = stub_rots(xyzorig[mask])
        # newstubs = stub_rots(xyz[mask])
        # T[mask] = xyz[mask, 1]
        # R[mask] = newstubs @ torch.linalg.inv(oldstubs)
        oldstubs = stub_rots(xyzorig)
        newstubs = stub_rots(xyz)
        T = xyz[:, 1]
        R = newstubs @ torch.linalg.inv(oldstubs)

        # Rnew = torch.einsum('sij,rjk,slk->sril', symmRs[symmsub], R[:nasym], symmRs[symmsub]).reshape(-1, 3, 3)
        # Rnew = torch.einsum('sij,rjk->srik', symmRs[symmsub], R[:nasym]).reshape(-1, 3, 3)
        # Tnew = torch.einsum('sij,rj->sri', symmRs[symmsub], T[:nasym]).reshape(-1, 3)
        # ic(torch.sum(~mask))
        # R symmetry is broken !?!?! maybe doesn't matter because frank resym later??

        # assert torch.allclose(T[~mask], Tnew[~mask])
        # assert torch.allclose(R[~mask], Rnew[~mask])

        # out.write(repr(hconstruct(R.cpu(),T.cpu())))

        if self.hconf.rfmotif.dump_debug_pdbs:
            xyzOUT = einsum('nij,naj->nai', R, xyzorig) + T.unsqueeze(-2)
            wu_dumppdb(f'{debuglabel}_OUT.pdb', xyzOUT.reshape(nsub, -1, 3, 3))
            if dock:
                wu_dumppdb(f'{debuglabel}_DOCKED.pdb', xyzOUT.reshape(nsub, -1, 3, 3))
                # assert 0

        R_in[0] = R
        T_in[0] = T
        return R_in, T_in

    def apply_motifs_xyz(self, xyz, symmsub, symmRs, alpha=1.0, debuglabel=None, dock=False):
        """
        Should be equivalent to apply_motifs_xyz, but for all atom because all atom does not use R and T .
        (since ligands dont have same protein atom sets).
        Should take in only xyz, and return only xyz. Strip shit out.

        
        """
        if self.params.disable_motifs_this_iter: return xyz
        if not self.params.do_apply_motifs_rfold: return xyz

        assert len(self.motifs), 'apply_motifs_xyz called with no motifs, maybe this is a mistake...'

        
        # xyz = xyz_in # IS THIS THE RIGHT WAY TO DO THIS SHIT????
        if xyz.shape[0]==1:
            xyz = xyz[0]


        nsub = 1 if symmsub is None else len(symmsub)
        mask = torch.zeros(len(xyz) // nsub, dtype=bool, device=xyz.device)

        for m in self.motifs:
            # ic(xyz.shape)
            xyz, mask, extra = m.apply_motif_RT(
                xyz=xyz,
                mask=mask,
                symmsub=symmsub,
                symmRs=symmRs,
                alpha=alpha,
                debuglabel=debuglabel,
            )
        mask = mask.tile(nsub)
        # ic(xyz.shape)

        nasym = len(xyz) // nsub
        if dock and 'cxaxis' in extra:
            # ic(extra.cxaxis)
            # wu_dumppdb('test_dock_pre.pdb', xyz.reshape(nsub,-1,3,3))
            docked = self.dock(xyz[:nasym], extra.cxaxis)
            xyz[:nasym] = torch.tensor(docked, device=xyz.device, dtype=xyz.dtype)
            ic(xyz.shape)
            # wu_dumppdb('test_dock_post_asym.pdb', xyz[:nasym])

        if symmsub is not None:
            xyz = torch.einsum('sij,raj->srai', symmRs[symmsub], xyz[:nasym]).reshape(-1, 3, 3)
            ic(xyz.shape)

        # if dock:
        # wu_dumppdb('test_dock_post_resym.pdb', xyz.reshape(nsub,-1,3,3))
        # assert 0

        # oldstubs = stub_rots(xyzorig[mask])
        # newstubs = stub_rots(xyz[mask])
        # T[mask] = xyz[mask, 1]
        # R[mask] = newstubs @ torch.linalg.inv(oldstubs)
        # oldstubs = stub_rots(xyzorig)
        # newstubs = stub_rots(xyz)
        # T = xyz[:, 1]
        # R = newstubs @ torch.linalg.inv(oldstubs)

        # Rnew = torch.einsum('sij,rjk,slk->sril', symmRs[symmsub], R[:nasym], symmRs[symmsub]).reshape(-1, 3, 3)
        # Rnew = torch.einsum('sij,rjk->srik', symmRs[symmsub], R[:nasym]).reshape(-1, 3, 3)
        # Tnew = torch.einsum('sij,rj->sri', symmRs[symmsub], T[:nasym]).reshape(-1, 3)
        # ic(torch.sum(~mask))
        # R symmetry is broken !?!?! maybe doesn't matter because frank resym later??

        # assert torch.allclose(T[~mask], Tnew[~mask])
        # assert torch.allclose(R[~mask], Rnew[~mask])

        # out.write(repr(hconstruct(R.cpu(),T.cpu())))

        # if self.hconf.rfmotif.dump_debug_pdbs:
        #     xyzOUT = einsum('nij,naj->nai', R, xyzorig) + T.unsqueeze(-2)
        #     ic(xyzOUT.shape)
        #     wu_dumppdb(f'{debuglabel}_OUT.pdb', xyzOUT.reshape(nsub, -1, 3, 3))
        #     if dock:
        #         wu_dumppdb(f'{debuglabel}_DOCKED.pdb', xyzOUT.reshape(nsub, -1, 3, 3))
        #         # assert 0

        # R_in[0] = R
        # T_in[0] = T
        return xyz[None]

    def dock(self, xyz, cxaxis):
        # wu_dumppdb('test0.pdb', xyz)

        assert len(self.motifs) == 1
        sym, psym = self.motifs[0].wholesym, self.motifs[0].pointsym

        #!!!!!!!!!!!!!!!!!!!!!
        nsym = 'c2'

        dockxyz = slide_dock_oligomer(sym, psym, nsym, xyz, startaxis=cxaxis, step=1.0, clash_radius=2.5, contact_dis=7)

        # wu_dumppdb('test1.pdb', dockxyz)
        # assert 0

        return dockxyz

    def __str__(self):
        s = 'MotifManager(\n'
        for m in self.motifs:
            s += '    ' + str(m) + '\n'
        s += ')'
        return s

    def __bool__(self):
        return bool(self.motifs)

    def _sanity_check(self, conf):
        # assert len(self.motifregions) == len(conf.rfmotif.pdbs) or 1 == len(conf.rfmotif.pdbs)
        # assert len(conf.rfmotif.pointsyms) in (0, len(conf.rfmotif.pdbs))
        pass

    def _set_regions(self, regions):
        self.regions, self.motifregions = list(), list()
        if not regions: return
        assert len(regions) == 1
        startres = 0
        for i, c in enumerate(regions[0].split(',')):
            label = None
            if not str.isnumeric(c[0]):
                lb, ub = [int(_) for _ in c[1:].split('-')]
                label, n, tbeg, tend = c[0], ub - lb + 1, lb - 1, ub
            else:
                label, n, tbeg, tend = None, int(c), None, None
            region = (label, (startres, startres + n), (tbeg, tend))
            self.regions.append(region)
            if label: self.motifregions.append(region)
            startres += n

    def _set_regions_AF(self, regions):


        self.sep_regions, self.sep_motifregions = list(), list()
        if not regions: return
        assert len(regions) == 1

        

        startres = 0


        for i, c in enumerate(regions[0].split(',')):
            label = None
            if not str.isnumeric(c[0]):
                lb, ub = [int(_) for _ in c[1:].split('-')]
                label, n, tbeg, tend = c[0], ub - lb + 1, lb - 1, ub
            else:
                label, n, tbeg, tend = None, int(c), None, None
            region = (label, (startres, startres + n), (tbeg, tend))
            self.sep_regions.append(region)
            if label: self.sep_motifregions.append(region)
            startres += n

        self.regions, self.motifregions = list(), list()

        for i, group_i in enumerate(self.ij_visible_int):
            self.regions.append(list())
            self.motifregions.append(list())
            for j, motif_j in enumerate(group_i):

                block_ind_ij = self.ij_visible_int[i][j]
                self.motifregions[i].append(self.sep_motifregions[block_ind_ij])
                self.regions[i].append(self.sep_regions[ 2*block_ind_ij + 0])
                self.regions[i].append(self.sep_regions[ 2*block_ind_ij + 1])


    def active(self):
        return bool(self.motifs)

    def printinfo(self):
        print('MotifManager motifs:')
        for m in self.motifs:
            m.printinfo(prefix='    ')

    def dumppdbs(self, prefix, alignto=None, filemode=None):
        if filemode != 'a' and prefix.endswith('.pdb'): prefix = prefix[:-4]
        for i, m in enumerate(self.motifs):
            fname = prefix
            if filemode != 'a':
                fname = f'{prefix}_{m.index}_{m.label}.pdb'
            rslt = m.dumppdb(fname, alignto=alignto, filemode=filemode, startchain=i)

class AsymMotif(RFMotif):
    def __init__(
        self,
        pdb,
        regions,
        device,
        index,
        fit_only_junctions,
        region_fudge_factor,
        manager,
    ):
        self.manager = manager
        self.index = index
        self.regions = regions
        self.region_fudge_factor = region_fudge_factor
        self.pdb = wu_readpdb(pdb)
        self.label = f'AsymMotif{index}'

        self.fit_only_junctions = fit_only_junctions

        crds, masks = self.pdb.atomcoords(['n', 'ca', 'c'], splitchains=True)

        assert all([np.all(m) for m in masks])
        self.chaincoords = [torch.tensor(c, device=device) for c in crds]

        # for ichain, crd in enumerate(self.chaincoords):
        #     ic(ichain, crd.shape, self.regions)
        #     ic(*self.regions[ichain][2])
            # 
            # assert 0 == len(self.regions) or len(crd) == len(range(*self.regions[ichain][2]))

        chain_coords_crop = []
        for chain_coords_i, motif_spec_i in zip(self.chaincoords, self.regions):
            from_i, to_i = motif_spec_i[2]
            chain_coords_crop.append(chain_coords_i[from_i:to_i,:,:])

        self.chaincoords = chain_coords_crop
        # self.motifcoords = torch.concatenate(chain_coords_crop)
        
        self.motifcoords = torch.concatenate(self.chaincoords)
        # self.motifcoords = chain_coords_crop
        self.distmat = list()
        for ichain, icrd in enumerate(self.chaincoords):
            self.distmat.append(list())
            for jchain, jcrd in enumerate(self.chaincoords):
                self.distmat[ichain].append(torch.cdist(icrd[:, 1], jcrd[:, 1]))
        # self.make_floating_offsets(50, 2, minsep=10, minbeg=0, minend=1)
        self.offset = None

        self.motif_sizes = [len(m) for m in self.chaincoords]
        self.motif_offsets = np.cumsum([0] + self.motif_sizes[:-1])



    def apply_motif_RT(self, xyz, mask, symmsub, symmRs, debuglabel=None, **kw):
        # if we want to allow the motif to float in sequence space: not touching for now
        # assert 0, "yeeet"
        if self.manager.float_seq_pos:
            if self.manager.params.do_motif_sequence_alignment:
                # floating placement
                placement = self._do_fast_drms_rms_placement(xyz, symmsub)
                if self.offset is None or any(self.offset != placement.offset[0]):
                    print(f'found new offset {self.offset} -> {placement.offset[0]}')
                self.offset = placement.offset[0]
                print(f'do_motif_sequence_alignment, RMS {placement.rms[0]} {self.offset}', flush=True)
            if self.offset is not None:
                # ic(self.offset)
                junct = self.fit_only_junctions
                if junct > 0:
                    # comupte coords of only ends
                    mxyz = list()
                    for s, o in zip(self.motif_sizes, self.motif_offsets):
                       if junct == 0 or 2 * junct >= s:
                          mxyz.append(self.motifcoords[o:o + s, 1])
                       else:
                          mxyz.append(self.motifcoords[o:o + junct, 1])
                          mxyz.append(self.motifcoords[o + s - junct:o + s, 1])
                    mxyz = torch.concatenate(mxyz)
                    regxyz = list()
                    for s, o in zip(self.motif_sizes, self.offset):
                       if junct == 0 or 2 * junct >= s:
                          regxyz.append(xyz[o:o + s, 1])
                       else:
                          regxyz.append(xyz[o:o + junct, 1])
                          regxyz.append(xyz[o + s - junct:o + s, 1])
                    regxyz = torch.concatenate(regxyz)
                    r, _, x = wu.hrmsfit(mxyz.cpu(), regxyz.cpu())
                else:
                    regxyz = torch.cat([xyz[o:o + s, 1] for o, s in zip(self.offset, [len(_) for _ in self.chaincoords])])
                    r, _, x = wu.hrmsfit(self.motifcoords[:, 1].cpu(), regxyz.cpu())
            
        # this is if we have specific fixed contigs for our motif in space
        else:
            self.offset = None # reassigning to make sure None
            # 
            regxyz = xyz.clone()[:self.manager.Lasu].unsqueeze(dim=0)[self.manager.motif_mask[:,:self.manager.Lasu]]
            r, _, x = wu.hrmsfit(self.motifcoords[:, 1].cpu(), regxyz[:,1].cpu())

        if 'placement' in vars():
            
            if not np.allclose(r, placement.rms[0], atol=1e-4):
                print('WE GOT THE PLACEMENT ISSUE!!!!!')
                ic(r, placement.rms[0])
                print('WEEEOOO!!!! WEEEEOOO!!!!!!!!')
            # assert np.allclose(r, placement.rms[0], atol=1e-4)

        x = torch.tensor(x).to(xyz.device)
        if self.offset is not None:
            for ofst, mcrd in zip(self.offset, self.chaincoords):
                xyz[ofst:ofst + len(mcrd)] = wu.th_xform(x, mcrd)
                mask[ofst:ofst + len(mcrd)] = True
        else:
            new_cords = wu.th_xform(x,self.motifcoords)
            xyz[:self.manager.Lasu][self.manager.motif_mask.squeeze()[:self.manager.Lasu]] = new_cords.to(dtype=xyz.dtype)
            mask = self.manager.motif_mask

        return xyz, mask, wu.Bunch()

    def apply_motif_t2d(self, t2d, mask, xyz, xyz_to_t2d_func, symmsub):
        if self.offset is not None: 
            # print('already placed motif', flush=True)
            return t2d, mask, wu.Bunch() # don't bother, prev px0 has motif

        placement = self._do_fast_drms_rms_placement(xyz, symmsub)

        template_t2d = xyz_to_t2d_func(self.motifcoords.reshape(1, 1, *self.motifcoords.shape))
        # ic(template_t2d.shape) # torch.Size([1, 1, 54, 54, 44])

        for o1, mo1, msize1 in zip(placement.offset[0], self.motif_offsets, self.motif_sizes):
            assert not torch.any(mask[o1:o1 + msize1])
            mask[o1:o1 + msize1] = True
            for o2, mo2, msize2 in zip(placement.offset[0], self.motif_offsets, self.motif_sizes):
                t2d[:, :, o1:o1 + msize1, o2:o2 + msize2] = template_t2d[:, :, mo1:mo1 + msize1, mo2:mo2 + msize2]

        print('apply_motifs_t2d', placement.rms[0], placement.offset[0], flush=True)

        return t2d, mask, wu.Bunch()

    def _do_fast_drms_rms_placement(self, xyz, symmsub):

        nsub = 1 if symmsub is None else len(symmsub)

        nasym = len(xyz) // nsub
        args = dict(
            nasym=nasym,
            cbreaks=[],
            nrmsalign=100_000,
            nolapcheck=1_000_000,
            minsep=10,
            minbeg=0,
            minend=0,
            junct=self.fit_only_junctions,
            return_alldme=False,
            motif_occlusion_weight=self.manager.hconf.rfmotif.motif_occlusion_weight,
            motif_occlusion_dist=self.manager.hconf.rfmotif.motif_occlusion_dist,            
        )
        # 
        # assert 
        # if xyz.shape[0]==1:
        #     placement = place_motif_dme_fast(xyz[0], self.chaincoords, **args)
        # else:
        #     placement = place_motif_dme_fast(xyz, self.chaincoords, **args)
        # 
        # placement = place_motif_dme_fast(xyz[0], self.chaincoords, **args)
        # ic(xyz.shape)
        placement = place_motif_dme_fast(xyz, self.chaincoords, **args)
        

        if len(placement.offset) == 0:
            place_motif_dme_fast(xyz, self.chaincoords, debug=True, **args)
            raise ValueError(f'place_motif_dme_fast can\'t place motif')

        if placement.occ is not None:
            print(f'DME {placement.dme[0]} OCC {placement.occ[0]}', flush=True)

        return placement

    def printinfo(self, prefix=''):
        print(f'{prefix}AsymMotif:')
        print(f'{prefix}    regions    {self.regions}', flush=True)
        print(f'{prefix}    resfudge   {self.region_fudge_factor}', flush=True)
        print(f'{prefix}    index      {self.index}', flush=True)
        print(f'{prefix}    device     {self.chaincoords[0].device}', flush=True)
        for i, c in enumerate(self.chaincoords):
            print(f'{prefix}    xyz      {i} {c.shape}', flush=True)
        # print(f'{prefix}    stubs    {self.stubs.shape}', flush=True)

    def dumppdb(self, fname, alignto=None, filemode=None, startchain=0, doseqaln=False):
        if alignto is None:
            for ichain0, crd in enumerate(self.chaincoords):
                wu_dumppdb(f'{fname}_chain_{startchain+ichain0}.pdb', crd, filemode=filemode)
                return wu.Bunch(nchains=len(self.chaincoords))
        elif self.offset is not None:
            assert not doseqaln
            assert alignto.ndim == 3 and alignto.shape[1:] == (3, 3)
            regxyz = torch.cat([alignto[o:o + s, 1] for o, s in zip(self.offset, [len(_) for _ in self.chaincoords])])
            r, _, x = wu.hrmsfit(self.motifcoords[:, 1].cpu(), regxyz.cpu())
            xpdb = self.pdb.xformed(x)  #, startchain=startchain)
            xpdb.df.ch = 'yz'[startchain % 3].encode()
            xpdb.dump_pdb(fname, filemode=filemode)
            return wu.Bunch(nchains=1)

    def start_new_design(self):
        self.offset = None



class PointSymMotif(RFMotif):
    def __init__(
        self,
        pdbfile,
        region,
        pointsym=None,
        wholesym=None,
        index=None,
        device=None,
        flip=False,
        fit_only_junctions=0,
        manager=None,
    ):
        self._region = region
        self.label, self.res, self.motifres = self._region
        self.pointsym, self.wholesym = pointsym, wholesym
        if self.pointsym == 'None': self.pointsym = None
        self.device = device
        self.index = index
        self.fit_only_junctions = fit_only_junctions
        self.manager = manager
        # self.mainres = torch.arange(*self.res)
        # self.motifres = torch.arange(*self.motifres)

        self._set_coords(pdbfile, flip)

    def _set_coords(self, pdbfile, flip=False):
        self.pdbfile = pdbfile

        # ic(self.pointsym)
        # ic(self.wholesym)
        self.xyz = wu_readpdb(self.pdbfile).ncac(splitchains=True)
        self.flip = flip
        self.symaxis = None
        if self.pointsym is not None:
            nfold = int(self.pointsym[1:])
            if len(self.xyz) != nfold:
                raise ValueError(f'nfold mismatch on {pdbfile}, motif pointsym is {self.pointsym}, '
                                 f'but pdb has only {len(self.xyz)} chains')
            self.symaxis = torch.tensor(symaxes(self.wholesym, self.pointsym)[:3], device=self.device, dtype=torch.float32)
            symelem = SymElem(nfold, self.symaxis.cpu().numpy())
            if len(self.xyz) > 1:
                self.xyz, xfit = aligncx(self.xyz, symelem)
            if self.flip:
                flipaxis = hcross([1, 2, 3], self.symaxis.cpu())
                ic(flipaxis.dtype)
                ic(type(self.xyz))
                self.xyz = hxform(hrot(flipaxis, np.pi), self.xyz)

        self.xyz = torch.tensor(self.xyz, device=self.device, dtype=torch.float32)
        # shift away from origin along symmaxis, needed for dumb COM alignment to work
        if self.symaxis is not None: self.xyz += 30.0 * self.symaxis

        # self.stubs = stub_rots(self.xyz)
        # assert torch.allclose(torch.linalg.det(self.stubs), torch.tensor(1.0, dtype=self.stubs.dtype))

        assert self.motifres[0] == 0
        assert self.motifres[1] == len(self.xyz[0])

    def crappy_com_fit(self, curxyz, symmsub, symmRs, term=None, debuglabel=None):
        # crappy fit com
        dev = curxyz.device

        assert len(curxyz) == len(self.xyz[0])
        if term and self.fit_only_junctions:
            if term == 'N': fitres = torch.arange(len(curxyz) - self.fit_only_junctions, len(curxyz), device=dev)
            elif term == 'C': fitres = torch.arange(self.fit_only_junctions, device=dev)
            else: assert 0
        else:
            fitres = torch.arange(len(curxyz), device=dev)
        curcom = curxyz[fitres].mean(axis=(0, 1))
        motif_com_shift = self.xyz[0, fitres].mean(axis=(0, 1)) + self.symaxis * 1000.0

        # ic(curcom)
        # ic(motif_com_shift)

        motif_comsym = einsum('sij,j->si', symmRs, motif_com_shift)
        symcomdist = torch.linalg.norm(curcom - motif_comsym, axis=-1)
        whichsub = torch.argmin(symcomdist)
        motif_crdsub = einsum('ij,raj->rai', symmRs[whichsub], self.xyz[0])
        motif_comsub = motif_crdsub[fitres].mean(axis=(0, 1))
        cxaxis = einsum('ij,j->i', symmRs[whichsub], self.symaxis)
        zero = torch.tensor([0.0, 0.0, 0.0], device=dev)
        curcom_fit = curxyz[fitres].mean(axis=(0, 1))
        dang = dihedral(motif_comsub, zero, cxaxis, curcom_fit)
        rotcom = t_rot(cxaxis, dang)

        motif_fit = einsum('ij,raj->rai', rotcom, motif_crdsub)
        # wu_dumppdb(f'{debuglabel}_motif_rot.pdb', motif_fit)
        motif_fit += proj(cxaxis, curcom_fit - motif_comsub).to(dev)
        # wu_dumppdb(f'{debuglabel}_motif_fit.pdb', motif_fit)

        return motif_fit, cxaxis

    def apply_motif_RT(self, xyz, mask, symmsub, symmRs, alpha=1.0, debuglabel=None):
        motifres = torch.arange(*self.res, device=xyz.device)
        assert not torch.any(mask[motifres])
        mask = mask.clone()
        mask[motifres] = True
        term = None
        if self.res[0] == 0: term = 'N'
        elif self.res[1] == len(R) // len(symmsub): term = 'C'
        xyz = xyz.clone()
        motif_fit, cxaxis = self.crappy_com_fit(xyz[motifres], symmsub, symmRs, term=term, debuglabel=debuglabel)
        xyz[motifres] = motif_fit
        return xyz, mask, wu.Bunch(cxaxis=cxaxis)

    def apply_motif_t2d(self, t2d, mask, xyz, xyz_to_t2d_func, symmsub):
        assert 0

    def __str__(self):
        axs = ','.join([str(float(_)) for _ in self.symaxis])
        return f'PointSymMotif(pdbfile={self.pdbfile}, region={self._region}, symaxis={axs})'

    def printinfo(self, prefix=''):
        print(f'{prefix}PointSymMotif:')
        print(f'{prefix}    label    {self.label}', flush=True)
        print(f'{prefix}    index    {self.index}', flush=True)
        print(f'{prefix}    res      {self.res}', flush=True)
        print(f'{prefix}    motifres {self.motifres}', flush=True)
        print(f'{prefix}    device   {self.device}', flush=True)
        print(f'{prefix}    symmetry {self.pointsym}', flush=True)
        print(f'{prefix}    symaxis  {self.symaxis}', flush=True)
        print(f'{prefix}    xyz      {self.xyz.shape}', flush=True)
        # print(f'{prefix}    stubs    {self.stubs.shape}', flush=True)

    def dumppdb(self, fname, alignto=None, filemode=None, startchain=0, doseqaln=False):
        if alignto is not None:
            print('TODO ALIGN PointSymMotif!')
            return
        wu_dumppdb(fname, self.xyz)


def normalized(x):
    return x / torch.linalg.norm(x, axis=-1)[..., None]

def stub_rots(xyz):
    # stub definition a little strange... easy to see visually on top of bb coordinatesnn,
    N, CA, C = xyz[..., 0, :], xyz[..., 1, :], xyz[..., 2, :]
    a = normalized(CA - N)
    # b = normalized(torch.linalg.cross(a, C - CA))
    b = normalized(torch.cross(a, C - CA))
    c = torch.linalg.cross(a, b)
    stubs = torch.stack([a, b, c], axis=-1)
    assert np.allclose(1, torch.linalg.det(stubs).cpu().detach())

    return stubs

def set_RT_from_coords(newxyz, idx, xyzorig, Rs, Ts):
    oldstubs = stub_rots(xyzorig[idx])
    newstubs = stub_rots(newxyz)
    Ts[idx] = newxyz[:, 1]
    Rs[idx] = newstubs @ torch.linalg.inv(oldstubs)

def dot(a, b):
    return torch.sum(a * b)

def dihedral(p1, p2, p3, p4):
    a = normalized(p2 - p1)
    b = normalized(p3 - p2)
    c = normalized(p4 - p3)
    x = torch.clip(dot(a, b) * dot(b, c) - dot(a, c), -1, 1)
    # m = torch.linalg.cross(b, c)
    m = torch.cross(b, c)
    y = torch.clip(dot(a, m), -1, 1)
    return torch.arctan2(y, x)

def proj(u, v):
    return dot(u, v) / dot(u, u) * u

#def apply_sym_template(Rs, Ts, xyzorig, symmids, symmsub, symmRs, metasymm, tpltcrd, tpltidx):
#
#   assert len(Rs) == len(Ts) == len(xyzorig) == 1
#   Rs, Ts, xyzorig = Rs[0], Ts[0], xyzorig[0]
#   Lasu, nsub = len(Rs) // len(symmsub), len(symmsub)
#   assert torch.all(tpltidx < Lasu)
#   dev = Rs.device
#
#   stubs = stub_rots(xyzorig)
#
#   # N = 10
#   # print(xyzorig.shape)
#   # print(stubs.shape)
#   # print(normalized(xyzorig[N, 0] - xyzorig[N, 1]))
#   # print(stubs[N, 0])
#   # print('--------------', flush=True)
#   # wu.showme(wu.th_construct(stubs)[N:N + 1], name='RT_in')
#   # wu.showme(xyzorig[N:N + 1].reshape(1, 1, 3, 3), name='pose_in')
#   # assert 0
#
#   curxyz = einsum('rij,raj->rai', Rs[tpltidx], xyzorig[tpltidx]) + Ts[tpltidx, None]
#   curcom = curxyz.mean(axis=(0, 1))
#   tpltcom = tpltcrd.mean(axis=(0, 1))
#   tpltcomsym = einsum('sij,j->si', symmRs, tpltcom)
#   whichsub = torch.argmin(torch.linalg.norm(curcom - tpltcomsym, axis=-1))
#   tpltcrdsub = einsum('ij,raj->rai', symmRs[whichsub], tpltcrd)
#   tpltcomsub = tpltcrdsub.mean(axis=(0, 1))
#
#   cxaxis = normalized(torch.tensor([1.0, 1.0, 1.0], device=dev))
#   cxaxis = einsum('ij,j->i', symmRs[whichsub], cxaxis)
#   zero = torch.tensor([0.0, 0.0, 0.0], device=dev)
#
#   dang = dihedral(tpltcomsub, zero, cxaxis, curcom)
#   rotcom = wu.t_rot(cxaxis, dang)
#   templatefitcom = einsum('ij,raj->rai', rotcom, tpltcrdsub)
#   templatefitcom += proj(cxaxis, curcom - tpltcomsub).to(dev)
#
#   wu.showme(curxyz, name='curxyz')
#   wu.showme(templatefitcom, name='templatefitcom')
#
#   wu.showme(tpltcrd, name='template_in')
#   wu.showme(tpltcrdsub, name='template_in_sub')
#   wu.showme(wu.th_construct(Rs @ stubs, Ts), name='RT_in')
#   tmp = einsum('rij,raj->rai', Rs, xyzorig) + Ts.unsqueeze(-2)
#   wu.showme(tmp.reshape(len(symmsub), -1, 3, 3), name='pose_in')
#
#   #
#
#   #
#
#   set_RT_from_coords(templatefitcom, tpltidx, xyzorig, Rs, Ts)
#
#   Rs = torch.einsum('sij,rjk,slk->sril', symmRs[symmsub], Rs[:Lasu], symmRs[symmsub]).reshape(-1, 3, 3)
#   Ts = torch.einsum('sij,rj->sri', symmRs[symmsub], Ts[:Lasu]).reshape(-1, 3)
#
#   # check symmetry of Rs ??!?!?!
#
#   #
#   wu.showme(templatefitcom, name='template_out')
#   wu.showme(wu.th_construct(Rs @ stubs, Ts), name='RT_out')
#   tmp = einsum('rij,raj->rai', Rs, xyzorig) + Ts.unsqueeze(-2)
#   wu.showme(tmp.reshape(len(symmsub), -1, 3, 3), name='pose_out')
#
#   return

# def t_rot(axis, angle, shape=(3, 3), squeeze=True):
#
#    # axis = torch.tensor(axis, dtype=dtype, requires_grad=requires_grad)
#    # angle = angle * np.pi / 180.0 if degrees else angle
#    # angle = torch.tensor(angle, dtype=dtype, requires_grad=requires_grad)
#
#    if axis.ndim == 1: axis = axis[None, ]
#    if angle.ndim == 0: angle = angle[None, ]
#    # if angle.ndim == 0
#    if axis.shape and angle.shape and not is_broadcastable(axis.shape[:-1], angle.shape):
#       raise ValueError(f'axis/angle not compatible: {axis.shape} {angle.shape}')
#    zero = torch.zeros(*angle.shape)
#    axis = th_normalized(axis)
#    a = torch.cos(angle / 2.0)
#    tmp = axis * -torch.sin(angle / 2)[..., None]
#    b, c, d = tmp[..., 0], tmp[..., 1], tmp[..., 2]
#    aa, bb, cc, dd = a * a, b * b, c * c, d * d
#    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
#    if shape == (3, 3):
#       rot = torch.stack([
#          torch.stack([aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)], axis=-1),
#          torch.stack([2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)], axis=-1),
#          torch.stack([2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc], axis=-1),
#       ], axis=-2)
#    elif shape == (4, 4):
#       rot = torch.stack([
#          torch.stack([aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac), zero], axis=-1),
#          torch.stack([2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab), zero], axis=-1),
#          torch.stack([2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc, zero], axis=-1),
#          torch.stack([zero, zero, zero, zero + 1], axis=-1),
#       ], axis=-2)
#    else:
#       raise ValueError(f't_rot shape must be (3,3) or (4,4), not {shape}')
#    # ic('foo')
#    # ic(axis.shape)
#    # ic(angle.shape)
#    # ic(rot.shape)
#    if squeeze and rot.shape == (1, 3, 3): rot = rot.reshape(3, 3)
#    if squeeze and rot.shape == (1, 4, 4): rot = rot.reshape(4, 4)
#    return rot

def Rs2Qs(Rs):
    Qs = torch.zeros((*Rs.shape[:-2], 4), device=Rs.device)

    Qs[..., 0] = 1.0 + Rs[..., 0, 0] + Rs[..., 1, 1] + Rs[..., 2, 2]
    Qs[..., 1] = 1.0 + Rs[..., 0, 0] - Rs[..., 1, 1] - Rs[..., 2, 2]
    Qs[..., 2] = 1.0 - Rs[..., 0, 0] + Rs[..., 1, 1] - Rs[..., 2, 2]
    Qs[..., 3] = 1.0 - Rs[..., 0, 0] - Rs[..., 1, 1] + Rs[..., 2, 2]
    Qs[Qs < 0.0] = 0.0
    Qs = torch.sqrt(Qs) / 2.0
    Qs[..., 1] *= torch.sign(Rs[..., 2, 1] - Rs[..., 1, 2])
    Qs[..., 2] *= torch.sign(Rs[..., 0, 2] - Rs[..., 2, 0])
    Qs[..., 3] *= torch.sign(Rs[..., 1, 0] - Rs[..., 0, 1])

    return Qs

def Qs2Rs(Qs):
    Rs = torch.zeros((*Qs.shape[:-1], 3, 3), device=Qs.device)

    Rs[..., 0, 0] = Qs[..., 0] * Qs[..., 0] + Qs[..., 1] * Qs[..., 1] - Qs[..., 2] * Qs[..., 2] - Qs[..., 3] * Qs[..., 3]
    Rs[..., 0, 1] = 2 * Qs[..., 1] * Qs[..., 2] - 2 * Qs[..., 0] * Qs[..., 3]
    Rs[..., 0, 2] = 2 * Qs[..., 1] * Qs[..., 3] + 2 * Qs[..., 0] * Qs[..., 2]
    Rs[..., 1, 0] = 2 * Qs[..., 1] * Qs[..., 2] + 2 * Qs[..., 0] * Qs[..., 3]
    Rs[..., 1, 1] = Qs[..., 0] * Qs[..., 0] - Qs[..., 1] * Qs[..., 1] + Qs[..., 2] * Qs[..., 2] - Qs[..., 3] * Qs[..., 3]
    Rs[..., 1, 2] = 2 * Qs[..., 2] * Qs[..., 3] - 2 * Qs[..., 0] * Qs[..., 1]
    Rs[..., 2, 0] = 2 * Qs[..., 1] * Qs[..., 3] - 2 * Qs[..., 0] * Qs[..., 2]
    Rs[..., 2, 1] = 2 * Qs[..., 2] * Qs[..., 3] + 2 * Qs[..., 0] * Qs[..., 1]
    Rs[..., 2, 2] = Qs[..., 0] * Qs[..., 0] - Qs[..., 1] * Qs[..., 1] - Qs[..., 2] * Qs[..., 2] + Qs[..., 3] * Qs[..., 3]

    return Rs