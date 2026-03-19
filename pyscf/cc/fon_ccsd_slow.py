#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Restricted CCSD

Ref: Stanton et al., J. Chem. Phys. 94, 4334 (1990)
Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004)
'''

from functools import reduce

import numpy as np

from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.lib import linalg_helper

#einsum = np.einsum
einsum = lib.einsum

# note MO integrals are treated in chemist's notation

def update_amps(cc, t1, t2, eris):
    # Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004) Eqs.(35)-(36)
    nocc, nvir = t1.shape
    nfullocc = cc.nfullocc
    fock = eris.fock

    fov = fock[:nocc,nfullocc:].copy()
    foo = fock[:nocc,:nocc].copy()
    fvv = fock[nfullocc:,nfullocc:].copy()

    Foo = foncc_Foo(t1,t2,eris, cc.mo_occ_of, cc.mo_occ_fv)
    Fvv = foncc_Fvv(t1,t2,eris, cc.mo_occ_of, cc.mo_occ_fv)
    Fov = foncc_Fov(t1,t2,eris, cc.mo_occ_of, cc.mo_occ_fv)

    # Move energy terms to the other side
    Foo -= np.diag(np.diag(foo))
    Fvv -= np.diag(np.diag(fvv))

    # T1 equation
    t1new = np.asarray(fov).conj().copy()
    t1new +=-2*einsum('kc,ka,ic,k,c->ia', fov, t1, t1, cc.mo_occ_of, cc.mo_occ_fv)
    t1new +=   einsum('ac,ic,c->ia', Fvv, t1, cc.mo_occ_fv)
    t1new +=  -einsum('ki,ka,k->ia', Foo, t1, cc.mo_occ_of)
    t1new += 2*einsum('kc,kica,k,c->ia', Fov, t2, cc.mo_occ_of, cc.mo_occ_fv)
    t1new +=  -einsum('kc,ikca,k,c->ia', Fov, t2, cc.mo_occ_of, cc.mo_occ_fv)
    t1new +=   einsum('kc,ic,ka,k,c->ia', Fov, t1, t1, cc.mo_occ_of, cc.mo_occ_fv)
    t1new += 2*einsum('kcai,kc,k,c->ia', eris.ovvo, t1, cc.mo_occ_of, cc.mo_occ_fv)
    t1new +=  -einsum('kiac,kc,k,c->ia', eris.oovv, t1, cc.mo_occ_of, cc.mo_occ_fv)
    eris_ovvv = np.asarray(eris.ovvv)
    t1new += 2*einsum('kdac,ikcd,k,c,d->ia', eris_ovvv, t2, cc.mo_occ_of, cc.mo_occ_fv, cc.mo_occ_fv)
    t1new +=  -einsum('kcad,ikcd,k,c,d->ia', eris_ovvv, t2, cc.mo_occ_of, cc.mo_occ_fv, cc.mo_occ_fv)
    t1new += 2*einsum('kdac,kd,ic,k,c,d->ia', eris_ovvv, t1, t1, cc.mo_occ_of, cc.mo_occ_fv, cc.mo_occ_fv)
    t1new +=  -einsum('kcad,kd,ic,k,c,d->ia', eris_ovvv, t1, t1, cc.mo_occ_of, cc.mo_occ_fv, cc.mo_occ_fv)
    t1new +=-2*einsum('kilc,klac,k,l,c->ia', eris.ooov, t2, cc.mo_occ_of, cc.mo_occ_of, cc.mo_occ_fv)
    t1new +=   einsum('likc,klac,k,l,c->ia', eris.ooov, t2, cc.mo_occ_of, cc.mo_occ_of, cc.mo_occ_fv)
    t1new +=-2*einsum('kilc,lc,ka,k,l,c->ia', eris.ooov, t1, t1, cc.mo_occ_of, cc.mo_occ_of, cc.mo_occ_fv)
    t1new +=   einsum('likc,lc,ka,k,l,c->ia', eris.ooov, t1, t1, cc.mo_occ_of, cc.mo_occ_of, cc.mo_occ_fv)

    # T2 equation
    t2new = np.asarray(eris.ovov).conj().transpose(0,2,1,3).copy()
    if cc.cc2:
        raise NotImplementedError('FON-CC2 update_amps is not implemented yet')
    else:
        Loo = fon_Loo(t1, t2, eris, cc.mo_occ_of, cc.mo_occ_fv)
        Lvv = fon_Lvv(t1, t2, eris, cc.mo_occ_of, cc.mo_occ_fv)
        Loo -= np.diag(np.diag(foo))
        Lvv -= np.diag(np.diag(fvv))
        Woooo = foncc_Woooo(t1, t2, eris, cc.mo_occ_of, cc.mo_occ_fv)
        Wvoov = foncc_Wvoov(t1, t2, eris, cc.mo_occ_of, cc.mo_occ_fv)
        Wvovo = foncc_Wvovo(t1, t2, eris, cc.mo_occ_of, cc.mo_occ_fv)
        Wvvvv = foncc_Wvvvv(t1, t2, eris, cc.mo_occ_of, cc.mo_occ_fv)
        tau = t2 + einsum('ia,jb->ijab', t1, t1)
        t2new += einsum('klij,klab,k,l->ijab', Woooo, tau, cc.mo_occ_of, cc.mo_occ_of)
        t2new += einsum('abcd,ijcd,c,d->ijab', Wvvvv, tau, cc.mo_occ_fv, cc.mo_occ_fv)
        tmp = einsum('ac,ijcb,c->ijab', Lvv, t2, cc.mo_occ_fv)
        t2new += (tmp + tmp.transpose(1,0,3,2))
        tmp = einsum('ki,kjab,k->ijab', Loo, t2, cc.mo_occ_of)
        t2new -= (tmp + tmp.transpose(1,0,3,2))
        tmp  = 2*einsum('akic,kjcb,k,c->ijab', Wvoov, t2, cc.mo_occ_of, cc.mo_occ_fv)
        tmp -=   einsum('akci,kjcb,k,c->ijab', Wvovo, t2, cc.mo_occ_of, cc.mo_occ_fv)
        t2new += (tmp + tmp.transpose(1,0,3,2))
        tmp = einsum('akic,kjbc,k,c->ijab', Wvoov, t2, cc.mo_occ_of, cc.mo_occ_fv)
        t2new -= (tmp + tmp.transpose(1,0,3,2))
        tmp = einsum('bkci,kjac,k,c->ijab', Wvovo, t2, cc.mo_occ_of, cc.mo_occ_fv)
        t2new -= (tmp + tmp.transpose(1,0,3,2))

    tmp2  = einsum('kibc,ka,k->abic', eris.oovv, -t1, cc.mo_occ_of)
    tmp2 += np.asarray(eris.ovvv).conj().transpose(1,3,0,2)
    tmp = einsum('abic,jc,c->ijab', tmp2, t1, cc.mo_occ_fv)
    t2new += (tmp + tmp.transpose(1,0,3,2))
    tmp2  = einsum('kcai,jc,c->akij', eris.ovvo, t1, cc.mo_occ_fv)
    tmp2 += np.asarray(eris.ooov).transpose(3,1,2,0).conj()
    tmp = einsum('akij,kb,k->ijab', tmp2, t1, cc.mo_occ_of)
    t2new -= (tmp + tmp.transpose(1,0,3,2))

    mo_e = eris.fock.diagonal().real.copy()
    if cc.denorm_fac == "renorm":
        mo_e_o = mo_e[:nocc] * cc.mo_occ_of
        mo_e_v = mo_e[nfullocc:] * cc.mo_occ_fv
        eia = mo_e_o[:,None] - mo_e_v[None,:]
    else:
        eia = mo_e[:nocc,None] - mo_e[None,nfullocc:]
    eia[np.abs(eia) < cc.denorm_thres] = np.inf
    eijab = lib.direct_sum('ia,jb->ijab',eia,eia)
    t1new /= eia
    t2new /= eijab

    return t1new, t2new


def energy(cc, t1, t2, eris):
    nocc, nvir = t1.shape
    nfullocc = cc.nfullocc
    fock = eris.fock
    e = 2*np.einsum('ia,ia,i,a->', fock[:nocc,nfullocc:], t1, cc.mo_occ_of, cc.mo_occ_fv, optimize=True)
    tau = einsum('ia,jb->ijab',t1,t1)
    tau += t2
    eris_ovov = np.asarray(eris.ovov)
    e += 2*np.einsum('ijab,iajb,i,a,j,b->', tau, eris_ovov, cc.mo_occ_of, cc.mo_occ_fv, cc.mo_occ_of, cc.mo_occ_fv, optimize=True)
    e +=  -np.einsum('ijab,ibja,i,a,j,b->', tau, eris_ovov, cc.mo_occ_of, cc.mo_occ_fv, cc.mo_occ_of, cc.mo_occ_fv, optimize=True)
    return e.real


def vector_to_amplitudes(vector, nocc, nvir):
    nov = nocc * nvir
    t1 = vector[:nov].copy().reshape((nocc,nvir))
    # filltriu=lib.SYMMETRIC because t2[iajb] == t2[jbia]
    t2 = lib.unpack_tril(vector[nov:], filltriu=lib.SYMMETRIC)
    t2 = t2.reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3)
    return t1, np.asarray(t2, order='C')



def get_idx_metal(mo_occ, threshold=1.0e-6):
    """Get index of occupied/virtual/fractional orbitals of metals.

    Parameters
    ----------
    mo_occ : double 1d array
        occupation number
    threshold : double, optional
        threshold to determine fractionally occupied orbitals, by default 1.0e-6

    Returns
    -------
    idx_occ : list
        list of occupied orbital indexes
    idx_frac : list
        list of fractionally occupied orbital indexes
    idx_vir : list
        list of virtual orbital indexes
    """
    idx_occ = np.where(mo_occ > 2.0 - threshold)[0]
    idx_vir = np.where(mo_occ < threshold)[0]
    
    idx_frac = list(range(len(idx_occ), len(mo_occ) - len(idx_vir)))

    return idx_occ, idx_frac, idx_vir

class FON_CCSD(ccsd.CCSD):
    _keys = {'max_space'}

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, frac_tol=1e-6, denorm_fac="renorm", denorm_thres=1e-3):
        ccsd.CCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.max_space = 20
        assert hasattr(mf, 'sigma'), 'FON_CCSD only works for smearing-HF reference'
        self.frac_tol = frac_tol
        self.denorm_thres = denorm_thres
        idx_occ, idx_frac, idx_vir = get_idx_metal(self.mo_occ, threshold=frac_tol)
        self.nfullocc = len(idx_occ)
        self.nfrac = len(idx_frac)
        self.nfullvir = len(idx_vir)
        assert self.nfullocc + self.nfrac + self.nfullvir == self.nmo, f"Number of orbitals {self.nmo} does not match the sum of full occupied {self.nfullocc}, fractional {self.nfrac}, and full virtual {self.nfullvir} orbitals."
        self.nocc = self.nfrac + self.nfullocc
        self.nvir = self.nfrac + self.nfullvir
        self.mo_occ_of = self.mo_occ[:self.nocc] / 2.0
        self.mo_occ_fv = 1.0 - self.mo_occ[self.nfullocc:] / 2.0
        assert denorm_fac in ["renorm", "con"], 'denorm_fac should be either "renorm" or "con"'
        self.denorm_fac = denorm_fac
        

    def init_amps(self, eris):
        nocc = self.nocc
        nfullocc = self.nfullocc
        mo_e = eris.fock.diagonal().real.copy()
        if self.denorm_fac == "renorm":
            mo_e_o = mo_e[:nocc] * self.mo_occ_of
            mo_e_v = mo_e[nfullocc:] * self.mo_occ_fv
            eia = mo_e_o[:,None] - mo_e_v[None,:]
        else:
            eia = mo_e[:nocc,None] - mo_e[None,nfullocc:]
        eia[np.abs(eia) < self.denorm_thres] = np.inf
        eijab = lib.direct_sum('ia,jb->ijab', eia, eia)
        t1 = eris.fock[:nocc,nfullocc:].conj() / eia
        eris_ovov = np.asarray(eris.ovov)
        t2 = eris_ovov.transpose(0,2,1,3).conj() / eijab
        self.emp2  = 2*np.einsum('ijab,iajb,i,a,j,b->', t2, eris_ovov, self.mo_occ_of, self.mo_occ_fv, self.mo_occ_of, self.mo_occ_fv, optimize=True)
        self.emp2 -=   np.einsum('ijab,ibja,i,a,j,b->', t2, eris_ovov, self.mo_occ_of, self.mo_occ_fv, self.mo_occ_of, self.mo_occ_fv, optimize=True)
        lib.logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        return self.emp2, t1, t2


    def kernel(self, t1=None, t2=None, eris=None, mbpt2=False, cc2=False):
        return self.ccsd(t1, t2, eris, mbpt2, cc2)
    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False, cc2=False):
        '''Ground-state CCSD.

        Kwargs:
            mbpt2 : bool
                Use one-shot MBPT2 approximation to CCSD.
            cc2 : bool
                Use CC2 approximation to CCSD.
        '''
        if mbpt2 and cc2:
            raise RuntimeError('MBPT2 and CC2 are mutually exclusive approximations to the CCSD ground state.')
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        self.e_hf = self.get_e_hf()
        self.eris = eris
        self.dump_flags()
        if mbpt2:
            cctyp = 'MBPT2'
            self.e_corr, self.t1, self.t2 = self.init_amps(eris)
        else:
            if cc2:
                cctyp = 'CC2'
                self.cc2 = True
            else:
                cctyp = 'CCSD'
                self.cc2 = False
            self.converged, self.e_corr, self.t1, self.t2 = \
                    ccsd.kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                                tol=self.conv_tol, tolnormt=self.conv_tol_normt,
                                verbose=self.verbose)
            if self.converged:
                logger.info(self, '%s converged', cctyp)
            else:
                logger.info(self, '%s not converged', cctyp)
        if self.e_hf == 0:
            logger.note(self, 'E_corr = %.16g', self.e_corr)
        else:
            logger.note(self, 'E(%s) = %.16g  E_corr = %.16g',
                        cctyp, self.e_tot, self.e_corr)
        return self.e_corr, self.t1, self.t2

    def ao2mo(self, mo_coeff=None):
        return _ChemistsERIs(self, mo_coeff)
    
    def vector_to_amplitudes(self, vec, nmo=None, nocc=None):
        return vector_to_amplitudes(vec, self.nocc, self.nvir)

    energy = energy
    update_amps = update_amps

class _ChemistsERIs:
    def __init__(self, cc, mo_coeff=None, method='incore',
                 ao2mofn=ao2mo.outcore.general_iofree):
        if mo_coeff is None:
            self.mo_coeff = mo_coeff = ccsd._mo_without_core(cc, cc.mo_coeff)
        else:
            self.mo_coeff = mo_coeff = ccsd._mo_without_core(cc, mo_coeff)
        dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
        fockao = cc._scf.get_hcore() + cc._scf.get_veff(cc.mol, dm)
        self.fock = reduce(np.dot, (mo_coeff.T, fockao, mo_coeff))

        nocc = cc.nocc
        nfullocc = cc.nfullocc
        nmo = cc.nmo
        eri1 = ao2mo.incore.full(cc._scf._eri, mo_coeff)
        eri1 = ao2mo.restore(1, eri1, nmo)
        self.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
        self.ooov = eri1[:nocc,:nocc,:nocc,nfullocc:].copy()
        self.ovoo = eri1[:nocc,nfullocc:,:nocc,:nocc].copy()
        self.ovov = eri1[:nocc,nfullocc:,:nocc,nfullocc:].copy()
        self.oovv = eri1[:nocc,:nocc,nfullocc:,nfullocc:].copy()
        self.ovvo = eri1[:nocc,nfullocc:,nfullocc:,:nocc].copy()
        self.ovvv = eri1[:nocc,nfullocc:,nfullocc:,nfullocc:].copy()
        self.vvvv = eri1[nfullocc:,nfullocc:,nfullocc:,nfullocc:].copy()

    def get_ovvv(self, *slices):
        '''To access a subblock of ovvv tensor'''
        if slices:
            return self.ovvv[slices]
        else:
            return self.ovvv


def foncc_Foo(t1, t2, eris, mo_occ_of, mo_occ_fv):
    nocc, nvir = t1.shape
    nfullocc = eris.fock.shape[0] - nvir
    foo = eris.fock[:nocc,:nocc]
    eris_ovov = np.asarray(eris.ovov)
    Fki  = 2*lib.einsum('kcld,ilcd,c,d,l->ki', eris_ovov, t2, mo_occ_fv, mo_occ_fv, mo_occ_of)
    Fki -=   lib.einsum('kdlc,ilcd,c,d,l->ki', eris_ovov, t2, mo_occ_fv, mo_occ_fv, mo_occ_of)
    Fki += 2*lib.einsum('kcld,ic,ld,c,d,l->ki', eris_ovov, t1, t1, mo_occ_fv, mo_occ_fv, mo_occ_of)
    Fki -=   lib.einsum('kdlc,ic,ld,c,d,l->ki', eris_ovov, t1, t1, mo_occ_fv, mo_occ_fv, mo_occ_of)
    Fki += foo
    return Fki

def foncc_Fvv(t1, t2, eris, mo_occ_of, mo_occ_fv):
    nocc, nvir = t1.shape
    nfullocc = eris.fock.shape[0] - nvir
    fvv = eris.fock[nfullocc:,nfullocc:]
    eris_ovov = np.asarray(eris.ovov)
    Fac  =-2*lib.einsum('kcld,klad,k,l,d->ac', eris_ovov, t2, mo_occ_of, mo_occ_of, mo_occ_fv)
    Fac +=   lib.einsum('kdlc,klad,k,l,d->ac', eris_ovov, t2, mo_occ_of, mo_occ_of, mo_occ_fv)
    Fac -= 2*lib.einsum('kcld,ka,ld,k,l,d->ac', eris_ovov, t1, t1, mo_occ_of, mo_occ_of, mo_occ_fv)
    Fac +=   lib.einsum('kdlc,ka,ld,k,l,d->ac', eris_ovov, t1, t1, mo_occ_of, mo_occ_of, mo_occ_fv)
    Fac += fvv
    return Fac

def foncc_Fov(t1, t2, eris, mo_occ_of, mo_occ_fv):
    nocc, nvir = t1.shape
    nfullocc = eris.fock.shape[0] - nvir
    fov = eris.fock[:nocc,nfullocc:]
    eris_ovov = np.asarray(eris.ovov)
    Fkc  = 2*np.einsum('kcld,ld,l,d->kc', eris_ovov, t1, mo_occ_of, mo_occ_fv)
    Fkc -=   np.einsum('kdlc,ld,l,d->kc', eris_ovov, t1, mo_occ_of, mo_occ_fv)
    Fkc += fov
    return Fkc

def fon_Loo(t1, t2, eris, mo_occ_of, mo_occ_fv):
    nocc, nvir = t1.shape
    nfullocc = eris.fock.shape[0] - nvir
    fov = eris.fock[:nocc,nfullocc:]
    Lki = foncc_Foo(t1, t2, eris, mo_occ_of, mo_occ_fv) + np.einsum('kc,ic,c->ki',fov, t1, mo_occ_fv)
    eris_ovoo = np.asarray(eris.ovoo)
    Lki += 2*np.einsum('lcki,lc,l,c->ki', eris_ovoo, t1, mo_occ_of, mo_occ_fv)
    Lki -=   np.einsum('kcli,lc,l,c->ki', eris_ovoo, t1, mo_occ_of, mo_occ_fv)
    return Lki

def fon_Lvv(t1, t2, eris, mo_occ_of, mo_occ_fv):
    nocc, nvir = t1.shape
    nfullocc = eris.fock.shape[0] - nvir
    fov = eris.fock[:nocc,nfullocc:]
    Lac = foncc_Fvv(t1, t2, eris, mo_occ_of, mo_occ_fv) - np.einsum('kc,ka,k->ac',fov, t1, mo_occ_of)
    eris_ovvv = np.asarray(eris.get_ovvv())
    Lac += 2*np.einsum('kdac,kd,k,d->ac', eris_ovvv, t1, mo_occ_of, mo_occ_fv)
    Lac -=   np.einsum('kcad,kd,k,d->ac', eris_ovvv, t1, mo_occ_of, mo_occ_fv)
    return Lac

def foncc_Woooo(t1, t2, eris, mo_occ_of, mo_occ_fv):
    eris_ovoo = np.asarray(eris.ovoo)
    Wklij  = lib.einsum('lcki,jc,c->klij', eris_ovoo, t1, mo_occ_fv)
    Wklij += lib.einsum('kclj,ic,c->klij', eris_ovoo, t1, mo_occ_fv)
    eris_ovov = np.asarray(eris.ovov)
    Wklij += lib.einsum('kcld,ijcd,c,d->klij', eris_ovov, t2, mo_occ_fv, mo_occ_fv)
    Wklij += lib.einsum('kcld,ic,jd,c,d->klij', eris_ovov, t1, t1, mo_occ_fv, mo_occ_fv)
    Wklij += np.asarray(eris.oooo).transpose(0,2,1,3)
    return Wklij

def foncc_Wvvvv(t1, t2, eris, mo_occ_of, mo_occ_fv):
    from pyscf.cc.rintermediates import _get_vvvv
    # Incore
    eris_ovvv = np.asarray(eris.get_ovvv())
    Wabcd  = lib.einsum('kdac,kb,k->abcd', eris_ovvv,-t1, mo_occ_of)
    Wabcd -= lib.einsum('kcbd,ka,k->abcd', eris_ovvv, t1, mo_occ_of)
    Wabcd += np.asarray(_get_vvvv(eris)).transpose(0,2,1,3)
    return Wabcd

def foncc_Wvoov(t1, t2, eris, mo_occ_of, mo_occ_fv):
    eris_ovvv = np.asarray(eris.get_ovvv())
    eris_ovoo = np.asarray(eris.ovoo)
    Wakic  = lib.einsum('kcad,id,d->akic', eris_ovvv, t1, mo_occ_fv)
    Wakic -= lib.einsum('kcli,la,l->akic', eris_ovoo, t1, mo_occ_of)
    Wakic += np.asarray(eris.ovvo).transpose(2,0,3,1)
    eris_ovov = np.asarray(eris.ovov)
    Wakic -= 0.5*lib.einsum('ldkc,ilda,l,d->akic', eris_ovov, t2, mo_occ_of, mo_occ_fv)
    Wakic -= 0.5*lib.einsum('lckd,ilad,l,d->akic', eris_ovov, t2, mo_occ_of, mo_occ_fv)
    Wakic -= lib.einsum('ldkc,id,la,l,d->akic', eris_ovov, t1, t1, mo_occ_of, mo_occ_fv)
    Wakic += lib.einsum('ldkc,ilad,l,d->akic', eris_ovov, t2, mo_occ_of, mo_occ_fv)
    return Wakic

def foncc_Wvovo(t1, t2, eris, mo_occ_of, mo_occ_fv):
    eris_ovvv = np.asarray(eris.get_ovvv())
    eris_ovoo = np.asarray(eris.ovoo)
    Wakci  = lib.einsum('kdac,id,d->akci', eris_ovvv, t1, mo_occ_fv)
    Wakci -= lib.einsum('lcki,la,l->akci', eris_ovoo, t1, mo_occ_of)
    Wakci += np.asarray(eris.oovv).transpose(2,0,3,1)
    eris_ovov = np.asarray(eris.ovov)
    Wakci -= 0.5*lib.einsum('lckd,ilda,l,d->akci', eris_ovov, t2, mo_occ_of, mo_occ_fv)
    Wakci -= lib.einsum('lckd,id,la,l,d->akci', eris_ovov, t1, t1, mo_occ_of, mo_occ_fv)
    return Wakci

FON_RCCSD = FON_CCSD


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf, cc

    mol = gto.Mole()
    mol.atom = '''
        Ne 0.0 0.0 0.0
    '''
    mol.basis = 'ccpvdz'
    mol.verbose = 0
    mol.build()

    # mf = scf.RHF(mol)
    # mf.kernel()
    # mycc = cc.CCSD(mf)
    # mycc.kernel()

    mf = scf.RHF(mol)
    mf = scf.addons.smearing_(mf, sigma=0.5, method="fermi")
    mf.kernel()
    print(mf.mo_occ)
    mycc = FON_CCSD(mf)
    mycc.kernel()
    print(mycc.e_corr)
    

