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


import numpy as np
from functools import reduce

from pyscf import lib
from pyscf import ao2mo
from pyscf import scf
from pyscf.lib import logger
from pyscf.cc import ccsd, gccsd
from pyscf.cc import gintermediates as imd
from pyscf.cc.addons import spatial2spin, spin2spatial
from pyscf import __config__

#einsum = np.einsum
einsum = lib.einsum

def _make_eris_phys_incore(mycc, mo_coeff=None, ao2mofn=None):
    eris = gccsd._PhysicistsERIs()
    eris._common_init_(mycc, mo_coeff)
    nocc = eris.nocc
    nao, nmo = eris.mo_coeff.shape

    if callable(ao2mofn):
        eri = ao2mofn(eris.mo_coeff).reshape([nmo]*4)
    else:
        assert (eris.mo_coeff.dtype == np.double)
        mo_a = eris.mo_coeff[:nao//2]
        mo_b = eris.mo_coeff[nao//2:]
        orbspin = eris.orbspin
        if orbspin is None:
            eri  = ao2mo.kernel(mycc._scf._eri, mo_a)
            eri += ao2mo.kernel(mycc._scf._eri, mo_b)
            eri1 = ao2mo.kernel(mycc._scf._eri, (mo_a,mo_a,mo_b,mo_b))
            eri += eri1
            eri += eri1.T
        else:
            mo = mo_a + mo_b
            eri = ao2mo.kernel(mycc._scf._eri, mo)
            if eri.size == nmo**4:  # if mycc._scf._eri is a complex array
                eri = eri.reshape(nmo**2, nmo**2)
                sym_forbid = (orbspin[:,None] != orbspin).ravel()
            else:  # 4-fold symmetry
                sym_forbid = (orbspin[:,None] != orbspin)[np.tril_indices(nmo)]
            eri[sym_forbid,:] = 0
            eri[:,sym_forbid] = 0

        if eri.dtype == np.double:
            eri = ao2mo.restore(1, eri, nmo)

    eri = eri.reshape(nmo,nmo,nmo,nmo)
    eri = eri.transpose(0,2,1,3)

    eris.oooo = eri[:nocc,:nocc,:nocc,:nocc].copy()
    eris.ooov = eri[:nocc,:nocc,:nocc,nocc:].copy()
    eris.oovv = eri[:nocc,:nocc,nocc:,nocc:].copy()
    eris.ovov = eri[:nocc,nocc:,:nocc,nocc:].copy()
    eris.ovvo = eri[:nocc,nocc:,nocc:,:nocc].copy()
    eris.ovvv = eri[:nocc,nocc:,nocc:,nocc:].copy()
    eris.vvvv = eri[nocc:,nocc:,nocc:,nocc:].copy()
    return eris

def make_tau_p(t2, t1a, t1b, fac=1, out=None):
    t1t1 = einsum('ia,jb->ijab', fac*0.5*t1a, t1b)
    t1t1 = t1t1 - t1t1.transpose(1,0,2,3)
    tau1 = t1t1 - t1t1.transpose(0,1,3,2)
    return tau1

def cc_Fvv_p(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    fvv = eris.fock[nocc:,nocc:]
    eris_vovv = np.asarray(eris.ovvv).transpose(1,0,3,2)
    t1t1 = einsum('ia,jb->ijab', 0.25*t1, t1)
    t1t1 = t1t1 - t1t1.transpose(1,0,2,3)
    tau_tilde = t1t1 - t1t1.transpose(0,1,3,2)
    Fae = fvv - 0.5*einsum('me,ma->ae',fov, t1)
    Fae += einsum('mf,amef->ae', t1, eris_vovv)
    Fae -= 0.5*einsum('mnaf,mnef->ae', tau_tilde, eris.oovv)
    return Fae

def cc_Foo_p(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    foo = eris.fock[:nocc,:nocc]
    t1t1 = einsum('ia,jb->ijab', 0.25*t1, t1)
    t1t1 = t1t1 - t1t1.transpose(1,0,2,3)
    tau_tilde = t1t1 - t1t1.transpose(0,1,3,2)
    Fmi = ( foo + 0.5*einsum('me,ie->mi',fov, t1)
            + einsum('ne,mnie->mi', t1, eris.ooov)
            + 0.5*einsum('inef,mnef->mi', tau_tilde, eris.oovv) )
    return Fmi

def cc_Woooo(t1, t2, eris):
    tau = imd.make_tau(t2, t1, t1)
    tmp = einsum('je,mnie->mnij', t1, eris.ooov)
    Wmnij = eris.oooo + tmp - tmp.transpose(0,1,3,2)
    Wmnij += 0.5*einsum('ijef,mnef->mnij', tau, eris.oovv)
    return Wmnij

def cc_Woooo_p(t1, t2, eris):
    t1t1 = einsum('ia,jb->ijab', 0.5*t1, t1)
    t1t1 = t1t1 - t1t1.transpose(1,0,2,3)
    tau = t1t1 - t1t1.transpose(0,1,3,2)
    tmp = einsum('je,mnie->mnij', t1, eris.ooov)
    Wmnij = eris.oooo + tmp - tmp.transpose(0,1,3,2)
    Wmnij += 0.5*einsum('ijef,mnef->mnij', tau, eris.oovv)
    return Wmnij

def cc_Wvvvv(t1, t2, eris):
    eris_ovvv = np.asarray(eris.ovvv)
    tmp = einsum('mb,mafe->bafe', t1, eris_ovvv)
    Wabef = np.asarray(eris.vvvv) - tmp + tmp.transpose(1,0,2,3)
    return Wabef

def cc_Wovvo_p(t1, t2, eris):
    eris_ovvo = -np.asarray(eris.ovov).transpose(0,1,3,2)
    eris_oovo = -np.asarray(eris.ooov).transpose(0,1,3,2)
    Wmbej  = einsum('jf,mbef->mbej', t1, eris.ovvv)
    Wmbej -= einsum('nb,mnej->mbej', t1, eris_oovo)
    # Wmbej -= 0.5*einsum('jnfb,mnef->mbej', t2, eris.oovv)
    Wmbej -= einsum('jf,nb,mnef->mbej', t1, t1, eris.oovv)
    Wmbej += eris_ovvo
    return Wmbej

def update_amps(cc, t1, t2, eris):
    assert (isinstance(eris, gccsd._PhysicistsERIs))
    nocc, nvir = t1.shape
    fock = eris.fock

    fov = fock[:nocc,nocc:]
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + cc.level_shift

    tau = imd.make_tau(t2, t1, t1)
    tau_p = make_tau_p(t2, t1, t1)

    Fvv = imd.cc_Fvv(t1, t2, eris)
    Foo = imd.cc_Foo(t1, t2, eris)
    Fov = imd.cc_Fov(t1, t2, eris)

    # Move energy terms to the other side
    Fvv[np.diag_indices(nvir)] -= mo_e_v
    Foo[np.diag_indices(nocc)] -= mo_e_o

    # T1 equation
    t1new  =  einsum('ie,ae->ia', t1, Fvv)
    t1new += -einsum('ma,mi->ia', t1, Foo)
    t1new +=  einsum('imae,me->ia', t2, Fov)
    t1new += -einsum('nf,naif->ia', t1, eris.ovov)
    t1new += -0.5*einsum('imef,maef->ia', t2, eris.ovvv)
    t1new += -0.5*einsum('mnae,mnie->ia', t2, eris.ooov)
    t1new += fov.conj()

    Woooo = cc_Woooo(t1, t2, eris)
    Woooo_p = cc_Woooo_p(t1, t2, eris)
    Wvvvv = cc_Wvvvv(t1, t2, eris)
    Wovvo_p = cc_Wovvo_p(t1, t2, eris)

    Fvv = cc_Fvv_p(t1, t2, eris)
    Foo = cc_Foo_p(t1, t2, eris)
    # Move energy terms to the other side
    Fvv[np.diag_indices(nvir)] -= mo_e_v
    Foo[np.diag_indices(nocc)] -= mo_e_o

    # T2 equation
    Ftmp = Fvv - 0.5*einsum('mb,me->be', t1, Fov)
    tmp = einsum('ijae,be->ijab', t2, Ftmp)
    t2new = tmp - tmp.transpose(0,1,3,2)
    Ftmp = Foo + 0.5*einsum('je,me->mj', t1, Fov)
    tmp = einsum('imab,mj->ijab', t2, Ftmp)
    t2new -= tmp - tmp.transpose(1,0,2,3)
    t2new += np.asarray(eris.oovv).conj()
    t2new += 0.5*einsum('mnab,mnij->ijab', t2, Woooo_p)
    t2new += 0.5*einsum('mnab,mnij->ijab', tau_p, Woooo)
    t2new += 0.5*einsum('ijef,abef->ijab', tau, Wvvvv)
    tmp = einsum('imae,mbej->ijab', t2, Wovvo_p)
    tmp -= -einsum('ie,ma,mbje->ijab', t1, t1, eris.ovov)
    tmp = tmp - tmp.transpose(1,0,2,3)
    tmp = tmp - tmp.transpose(0,1,3,2)
    t2new += tmp
    tmp = einsum('ie,jeba->ijab', t1, np.array(eris.ovvv).conj())
    t2new += (tmp - tmp.transpose(1,0,2,3))
    tmp = einsum('ma,ijmb->ijab', t1, np.asarray(eris.ooov).conj())
    t2new -= (tmp - tmp.transpose(0,1,3,2))

    # A
    Ftmp = 0.5*einsum('inef,mnef->mi', t2, eris.oovv)
    tmp = einsum('imab,mj->ijab', t2, Ftmp)
    Atmp = -tmp + tmp.transpose(1,0,2,3)
    # C
    Ftmp = -0.5*einsum('mnaf,mnef->ae', t2, eris.oovv)
    tmp = einsum('ijae,be->ijab', t2, Ftmp)
    Ctmp = tmp - tmp.transpose(0,1,3,2)
    # B
    tmp = 0.5*einsum('ijef,mnef->mnij', t2, eris.oovv)
    Btmp = 0.5*einsum('mnab,mnij->ijab', t2, tmp)

    if isinstance(cc, pGCCSD):
        tmp1 = -0.5*einsum('jnfb,mnef->mbej', t2, eris.oovv)
        tmp = einsum('imae,mbej->ijab', t2, tmp1)
        tmp = tmp - tmp.transpose(1,0,2,3)
        Dtmp = tmp - tmp.transpose(0,1,3,2)
        t2new += (0.5+0.5*cc.p_mu)*Atmp + cc.p_mu*Btmp + cc.p_sigma*Ctmp + cc.p_sigma*Dtmp
    else: # GDCSD
        tmp1 = -0.5*einsum('jnfb,mnef->mbej', t2, cc.oovv_phys)
        tmp = einsum('imae,mbej->ijab', t2, tmp1)
        tmp = tmp - tmp.transpose(1,0,2,3)
        Dtmp = tmp - tmp.transpose(0,1,3,2)
        t2new += 0.5*Atmp + 0.5*Ctmp + Dtmp

    eia = mo_e_o[:,None] - mo_e_v
    eijab = lib.direct_sum('ia,jb->ijab', eia, eia)
    t1new /= eia
    t2new /= eijab

    return t1new, t2new



class GDCSD(gccsd.GCCSD):
    update_amps = update_amps
    oovv_phys = None
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        super().__init__(mf, frozen, mo_coeff, mo_occ)
        self.oovv_phys = _make_eris_phys_incore(self).oovv

class pGCCSD(GDCSD):
    p_mu = -1.0
    p_sigma = 1.0
    update_amps = update_amps
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, mu=None, sigma=None):
        super().__init__(mf, frozen, mo_coeff, mo_occ)
        if mu is not None:
            self.p_mu = mu
        if sigma is not None:
            self.p_sigma = sigma
    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None,
                     eris=None):
        from pyscf.cc import gdcsd_lambda
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        self.converged_lambda, self.l1, self.l2 = \
                gdcsd_lambda.kernel(self, eris, t1, t2, l1, l2,
                                    max_cycle=self.max_cycle,
                                    tol=self.conv_tol_normt,
                                    verbose=self.verbose)
        return self.l1, self.l2

    def ccsd_t(self, t1=None, t2=None, eris=None):
        raise NotImplementedError

    def ipccsd(self, nroots=1, left=False, koopmans=False, guess=None,
               partition=None, eris=None):
        from pyscf.cc import eom_gdcsd
        return eom_gdcsd.EOMIP(self).kernel(nroots, left, koopmans, guess,
                                            partition, eris)

    def eaccsd(self, nroots=1, left=False, koopmans=False, guess=None,
               partition=None, eris=None):
        from pyscf.cc import eom_gdcsd
        return eom_gdcsd.EOMEA(self).kernel(nroots, left, koopmans, guess,
                                            partition, eris)

    def eeccsd(self, nroots=1, koopmans=False, guess=None, eris=None):
        from pyscf.cc import eom_gdcsd
        return eom_gdcsd.EOMEE(self).kernel(nroots, koopmans, guess, eris)

    def eomip_method(self):
        from pyscf.cc import eom_gdcsd
        return eom_gdcsd.EOMIP(self)

    def eomea_method(self):
        from pyscf.cc import eom_gdcsd
        return eom_gdcsd.EOMEA(self)

    def eomee_method(self):
        from pyscf.cc import eom_gdcsd
        return eom_gdcsd.EOMEE(self)

    def make_rdm1(self, t1=None, t2=None, l1=None, l2=None, ao_repr=False,
                  with_frozen=True, with_mf=True):
        '''Un-relaxed 1-particle density matrix in MO space'''
        from pyscf.cc import gdcsd_rdm
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2
        if l1 is None: l1, l2 = self.solve_lambda(t1, t2)
        return gdcsd_rdm.make_rdm1(self, t1, t2, l1, l2, ao_repr=ao_repr,
                                   with_frozen=with_frozen, with_mf=with_mf)

    def make_rdm2(self, t1=None, t2=None, l1=None, l2=None, ao_repr=False,
                  with_frozen=True, with_dm1=True):
        raise NotImplementedError
    

class GDCD(GDCSD):
    def update_amps(self, t1, t2, eris):
        t1, t2 = update_amps(self, t1, t2, eris)
        return np.zeros_like(t1), t2

    def kernel(self, t2=None, eris=None):
        nocc = self.nocc
        nvir = self.nmo - nocc
        t1 = np.zeros((nocc, nvir))
        GDCSD.kernel(self, t1, t2, eris)
        return self.e_corr, self.t2
    
    def solve_lambda(self, t2=None, l2=None, eris=None):
        from pyscf.cc import gdcsd_lambda
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)

        nocc = self.nocc
        nvir = self.nmo - nocc
        l1 = t1 = np.zeros((nocc, nvir))

        def update_lambda(mycc, t1, t2, l1, l2, eris=None, imds=None):
            l1, l2 = gdcsd_lambda.update_lambda(mycc, t1, t2, l1, l2, eris, imds)
            return np.zeros_like(l1), l2

        self.converged_lambda, self.l1, self.l2 = \
                gdcsd_lambda.kernel(self, eris, t1, t2, l1, l2,
                                   max_cycle=self.max_cycle,
                                   tol=self.conv_tol_normt,
                                   verbose=self.verbose, fupdate=update_lambda)
        return self.l2

class pGCCD(pGCCSD):
    def update_amps(self, t1, t2, eris):
        t1, t2 = update_amps(self, t1, t2, eris)
        return np.zeros_like(t1), t2
    
    def kernel(self, t2=None, eris=None):
        nocc = self.nocc
        nvir = self.nmo - nocc
        t1 = np.zeros((nocc, nvir))
        pGCCSD.kernel(self, t1, t2, eris)
        return self.e_corr, self.t2
    
    def make_rdm2(self, t1=None, t2=None, l1=None, l2=None, ao_repr=False,
                  with_frozen=True, with_dm1=True):
        '''2-particle density matrix in MO space.  The density matrix is
        stored as

        dm2[p,r,q,s] = <p^+ q^+ s r>
        '''
        from pyscf.cc import gdcsd_rdm
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2
        if l1 is None: l1, l2 = self.solve_lambda(t1, t2)
        return gdcsd_rdm.make_rdm2(self, t1, t2, l1, l2, ao_repr=ao_repr,
                                   with_frozen=with_frozen, with_dm1=with_dm1)

if __name__ == '__main__':
    from pyscf import scf, gto
    mol = gto.Mole()
    mol.atom = '''
    O
    H  1  3.0
    H  1  3.0  2 107.6
'''
    mol.unit = "Bohr"
    mol.basis = 'cc-pvdz'
    # mol.charge = 8
    mol.build()
    rhf = scf.RHF(mol)
    rhf.scf()
    fc = 2
    mydc = GDCSD(rhf.to_ghf(),frozen=fc)
    mydc.kernel()
    mypcc = pGCCSD(rhf.to_ghf(),frozen=fc, mu=1.0, sigma=1.0)
    mypcc.kernel()
    mycc = gccsd.GCCSD(rhf.to_ghf(),frozen=fc)
    mycc.kernel()
    assert np.allclose(mycc.e_tot, mypcc.e_tot)