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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
RCCSD for real integrals
8-fold permutation symmetry has been used
(ij|kl) = (ji|kl) = (kl|ij) = ...
'''


import ctypes
from functools import reduce
import numpy
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf.cc import _ccsd, ccsd
from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask, get_e_hf, _mo_without_core
from pyscf import __config__

BLKMIN = getattr(__config__, 'cc_ccsd_blkmin', 4)
MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)

def cc_parameter(mycc):
    # change notation from mu sigma to alpha beta gamma delta
    # to make DCSD and pCCSD more consistent in the code
    if hasattr(mycc, "p_mu"): # pCCSD
        is_dcsd = False
        p_alpha = mycc.p_sigma
        p_delta = mycc.p_sigma
        p_beta = (1.0 + mycc.p_mu) / 2.0
        p_gamma = mycc.p_mu
    else: # DCSD
        is_dcsd = True
        p_alpha = 0.5
        p_beta = 0.5
        p_gamma = 0.0
        p_delta = 1.0
    return p_alpha, p_beta, p_gamma, p_delta, is_dcsd


def update_amps(mycc, t1, t2, eris):
    assert (isinstance(eris, ccsd._ChemistsERIs))
    if isinstance(mycc, pCCSD):
        variant = 'pccsd'
    else:
        variant = 'dcsd'

    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + mycc.level_shift

    t1new = numpy.zeros_like(t1)
    t2new = mycc._add_vvvv(t1, t2, eris, t2sym='jiba')
    t2new *= .5  # *.5 because t2+t2.transpose(1,0,3,2) in the end
    time1 = log.timer_debug1('vvvv', *time0)

#** make_inter_F
    fov = fock[:nocc,nocc:].copy()
    t1new += fov

    foo = fock[:nocc,:nocc] - numpy.diag(mo_e_o)
    foo += .5 * numpy.einsum('ia,ja->ij', fock[:nocc,nocc:], t1)
    ft_ij = foo.copy()

    fvv = fock[nocc:,nocc:] - numpy.diag(mo_e_v)
    fvv -= .5 * numpy.einsum('ia,ib->ab', t1, fock[:nocc,nocc:])

    if mycc.incore_complete:
        fswap = None
    else:
        fswap = lib.H5TmpFile()
    fwVOov, fwVooV = ccsd._add_ovvv_(mycc, t1, t2, eris, fvv, t1new, t2new, fswap)
    ft_ab = fvv.copy()
    time1 = log.timer_debug1('ovvv', *time1)

    woooo = numpy.asarray(eris.oooo).transpose(0,2,1,3).copy()
    woooo_p = woooo.copy()

    unit = nocc**2*nvir*7 + nocc**3 + nocc*nvir**2
    mem_now = lib.current_memory()[0]
    max_memory = max(0, mycc.max_memory - mem_now)
    blksize = min(nvir, max(BLKMIN, int((max_memory*.9e6/8-nocc**4)/unit)))
    log.debug1('max_memory %d MB,  nocc,nvir = %d,%d  blksize = %d',
               max_memory, nocc, nvir, blksize)

    for p0, p1 in lib.prange(0, nvir, blksize):
        wVOov = fwVOov[p0:p1]
        wVooV = fwVooV[p0:p1]
        eris_ovoo = eris.ovoo[:,p0:p1]
        eris_oovv = numpy.empty((nocc,nocc,p1-p0,nvir))
        def load_oovv(p0, p1):
            eris_oovv[:] = eris.oovv[:,:,p0:p1]
        with lib.call_in_background(load_oovv, sync=not mycc.async_io) as prefetch_oovv:
            #:eris_oovv = eris.oovv[:,:,p0:p1]
            prefetch_oovv(p0, p1)
            tmp = numpy.einsum('kc,kcji->ij', 2*t1[:,p0:p1], eris_ovoo)
            tmp += numpy.einsum('kc,icjk->ij',  -t1[:,p0:p1], eris_ovoo)
            foo += tmp
            ft_ij += tmp
            tmp = lib.einsum('la,jaik->lkji', t1[:,p0:p1], eris_ovoo)
            woooo += tmp + tmp.transpose(1,0,3,2)
            woooo_p += tmp + tmp.transpose(1,0,3,2)
            tmp = None

            wVOov -= lib.einsum('jbik,ka->bjia', eris_ovoo, t1)
            t2new[:,:,p0:p1] += wVOov.transpose(1,2,0,3)

            wVooV += lib.einsum('kbij,ka->bija', eris_ovoo, t1)
            eris_ovoo = None
        load_oovv = prefetch_oovv = None

        eris_ovvo = numpy.empty((nocc,p1-p0,nvir,nocc))
        def load_ovvo(p0, p1):
            eris_ovvo[:] = eris.ovvo[:,p0:p1]
        with lib.call_in_background(load_ovvo, sync=not mycc.async_io) as prefetch_ovvo:
            #:eris_ovvo = eris.ovvo[:,p0:p1]
            prefetch_ovvo(p0, p1)
            t1new[:,p0:p1] -= numpy.einsum('jb,jiab->ia', t1, eris_oovv)
            wVooV -= eris_oovv.transpose(2,0,1,3)
            wVOov += wVooV*.5  #: bjia + bija*.5
        load_ovvo = prefetch_ovvo = None

        t2new[:,:,p0:p1] += (eris_ovvo*0.5).transpose(0,3,1,2)
        eris_voov = eris_ovvo.conj().transpose(1,0,3,2)
        t1new[:,p0:p1] += 2*numpy.einsum('jb,aijb->ia', t1, eris_voov)
        eris_ovvo = None

        tmp  = lib.einsum('ic,kjbc->ibkj', t1, eris_oovv)
        tmp += lib.einsum('bjkc,ic->jbki', eris_voov, t1)
        t2new[:,:,p0:p1] -= lib.einsum('ka,jbki->jiba', t1, tmp)
        eris_oovv = tmp = None

        fov[:,p0:p1] += numpy.einsum('kc,aikc->ia', t1, eris_voov) * 2
        fov[:,p0:p1] -= numpy.einsum('kc,akic->ia', t1, eris_voov)

        tau  = numpy.einsum('ia,jb->ijab', t1[:,p0:p1]*.5, t1)
        tau += t2[:,:,p0:p1]
        theta  = tau.transpose(1,0,2,3) * 2
        theta -= tau
        fvv -= lib.einsum('cjia,cjib->ab', theta.transpose(2,1,0,3), eris_voov)
        foo += lib.einsum('aikb,kjab->ij', eris_voov, theta)
        tau = theta = None
        if variant == 'dcsd':
            tau  = numpy.einsum('ia,jb->ijab', t1[:,p0:p1]*.5, t1)
            tau += t2[:,:,p0:p1] * 0.5 # extra factor for DCSD
            theta  = tau.transpose(1,0,2,3) * 2
            theta -= tau
            ft_ab -= lib.einsum('cjia,cjib->ab', theta.transpose(2,1,0,3), eris_voov)
            ft_ij += lib.einsum('aikb,kjab->ij', eris_voov, theta)
            tau = theta = None
        elif variant == 'pccsd':
            tau  = numpy.einsum('ia,jb->ijab', t1[:,p0:p1]*.5, t1)
            tau += t2[:,:,p0:p1] * (mycc.p_mu + 1.0) / 2.0 # extra factor for pCCSD
            theta  = tau.transpose(1,0,2,3) * 2
            theta -= tau
            ft_ij += lib.einsum('aikb,kjab->ij', eris_voov, theta)

            tau  = numpy.einsum('ia,jb->ijab', t1[:,p0:p1]*.5, t1)
            tau += t2[:,:,p0:p1] * mycc.p_sigma # extra factor for pCCSD
            theta  = tau.transpose(1,0,2,3) * 2
            theta -= tau
            ft_ab -= lib.einsum('cjia,cjib->ab', theta.transpose(2,1,0,3), eris_voov)
            tau = theta = None

        tau = numpy.einsum('ia,jb->ijab', t1[:,p0:p1], t1)
        tau_p = tau.copy()
        tau += t2[:,:,p0:p1]
        if variant == 'dcsd':
            # remove t2 for DCSD
            pass
        elif variant == 'pccsd':
            tau_p += t2[:,:,p0:p1] * mycc.p_mu # extra factor for pCCSD
        woooo += lib.einsum('ijab,aklb->ijkl', tau, eris_voov)
        woooo_p += lib.einsum('ijab,aklb->ijkl', tau_p, eris_voov)
        tau = tau_p = None

        if variant != 'dcsd':
            def update_wVooV(q0, q1, tau):
                wVooV[:] += lib.einsum('bkic,jkca->bija', eris_voov[:,:,:,q0:q1], tau) * mycc.p_sigma
            with lib.call_in_background(update_wVooV, sync=not mycc.async_io) as update_wVooV:
                for q0, q1 in lib.prange(0, nvir, blksize):
                    update_wVooV(q0, q1, t2[:,:,q0:q1] * .5)
        for q0, q1 in lib.prange(0, nvir, blksize):
            wVooV[:] += lib.einsum('bkic,jc,ka->bija', eris_voov[:,:,:,q0:q1], t1[:,q0:q1], t1)
        update_wVooV = None

        def update_t2(q0, q1, tmp):
            t2new[:,:,q0:q1] += tmp.transpose(2,0,1,3)
            tmp *= .5
            t2new[:,:,q0:q1] += tmp.transpose(0,2,1,3)
        with lib.call_in_background(update_t2, sync=not mycc.async_io) as update_t2:
            for q0, q1 in lib.prange(0, nvir, blksize):
                tmp = lib.einsum('jkca,ckib->jaib', t2[:,:,p0:p1,q0:q1], wVooV)
                update_t2(q0, q1, tmp)
                tmp = None
        wVooV = None

        wVOov += eris_voov
        eris_VOov = eris_voov.copy()
        eris_VOov -= .5 * eris_voov.transpose(0,2,1,3)
        
        def update_wVOov(q0, q1, tau):
            if variant == 'dcsd':
                wVOov[:,:,:,q0:q1] += .5 * lib.einsum('aikc,kcjb->aijb', eris_voov, tau) # Coulomb eris_voov only
            elif variant == 'pccsd':
                wVOov[:,:,:,q0:q1] += .5 * lib.einsum('aikc,kcjb->aijb', eris_VOov, tau) * mycc.p_sigma
        with lib.call_in_background(update_wVOov, sync=not mycc.async_io) as update_wVOov:
            for q0, q1 in lib.prange(0, nvir, blksize):
                tau  = t2[:,:,q0:q1].transpose(1,3,0,2) * 2
                tau -= t2[:,:,q0:q1].transpose(0,3,1,2)
                update_wVOov(q0, q1, tau)
                tau = None
        for q0, q1 in lib.prange(0, nvir, blksize):
            wVOov[:,:,:,q0:q1] -= .5 * lib.einsum('aikc,kb,jc->aijb', eris_VOov, t1[:,q0:q1]*2, t1)
        eris_voov = eris_VOov = update_wVOov = None

        def update_t2(q0, q1, theta):
            t2new[:,:,q0:q1] += lib.einsum('kica,ckjb->ijab', theta, wVOov)
        with lib.call_in_background(update_t2, sync=not mycc.async_io) as update_t2:
            for q0, q1 in lib.prange(0, nvir, blksize):
                theta  = t2[:,:,p0:p1,q0:q1] * 2
                theta -= t2[:,:,p0:p1,q0:q1].transpose(1,0,2,3)
                update_t2(q0, q1, theta)
                theta = None
        wVOov = None
        time1 = log.timer_debug1('voov [%d:%d]'%(p0, p1), *time1)
    fwVOov = fwVooV = fswap = None

    for p0, p1 in lib.prange(0, nvir, blksize):
        theta = t2[:,:,p0:p1].transpose(1,0,2,3) * 2 - t2[:,:,p0:p1]
        t1new += numpy.einsum('jb,ijba->ia', fov[:,p0:p1], theta)
        t1new -= lib.einsum('jbki,kjba->ia', eris.ovoo[:,p0:p1], theta)

        tau = numpy.einsum('ia,jb->ijab', t1[:,p0:p1], t1)
        t2new[:,:,p0:p1] += .5 * lib.einsum('ijkl,klab->ijab', woooo, tau)
        t2new[:,:,p0:p1] += .5 * lib.einsum('ijkl,klab->ijab', woooo_p, t2[:,:,p0:p1])
        theta = tau = None

    ft_ij += numpy.einsum('ja,ia->ij', .5*t1, fov)
    ft_ab -= numpy.einsum('ia,ib->ab', .5*t1, fov)
    t2new += lib.einsum('ijac,bc->ijab', t2, ft_ab)
    t2new -= lib.einsum('ki,kjab->ijab', ft_ij, t2)
    ft_ab = ft_ij = None

    eia = mo_e_o[:,None] - mo_e_v
    t1new += numpy.einsum('ib,ab->ia', t1, fvv)
    t1new -= numpy.einsum('ja,ji->ia', t1, foo)
    t1new /= eia

    #: t2new = t2new + t2new.transpose(1,0,3,2)
    for i in range(nocc):
        if i > 0:
            t2new[i,:i] += t2new[:i,i].transpose(0,2,1)
            t2new[i,:i] /= lib.direct_sum('a,jb->jab', eia[i], eia[:i])
            t2new[:i,i] = t2new[i,:i].transpose(0,2,1)
        t2new[i,i] = t2new[i,i] + t2new[i,i].T
        t2new[i,i] /= lib.direct_sum('a,b->ab', eia[i], eia[i])

    time0 = log.timer_debug1('update t1 t2', *time0)
    return t1new, t2new


class DCSD(ccsd.CCSD):
    __doc__ = ccsd.CCSD.__doc__

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None, eris=None):
        from pyscf.cc import dcsd_lambda
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        self.converged_lambda, self.l1, self.l2 = \
                dcsd_lambda.kernel(self, eris, t1, t2, l1, l2,
                                   max_cycle=self.max_cycle,
                                   tol=self.conv_tol_normt,
                                   verbose=self.verbose)
        return self.l1, self.l2

    def ccsd_t(self, t1=None, t2=None, eris=None):
        raise NotImplementedError

    def ipccsd(self, nroots=1, left=False, koopmans=False, guess=None,
               partition=None, eris=None):
        from pyscf.cc import eom_rdcsd
        return eom_rdcsd.EOMIP(self).kernel(nroots, left, koopmans, guess,
                                            partition, eris)

    def eaccsd(self, nroots=1, left=False, koopmans=False, guess=None,
               partition=None, eris=None):
        from pyscf.cc import eom_rdcsd
        return eom_rdcsd.EOMEA(self).kernel(nroots, left, koopmans, guess,
                                            partition, eris)

    def eeccsd(self, nroots=1, koopmans=False, guess=None, eris=None):
        raise NotImplementedError("Please use eomee_ccsd_singlet or eomee_ccsd_triplet")

    def eomee_ccsd_singlet(self, nroots=1, koopmans=False, guess=None, eris=None):
        from pyscf.cc import eom_rdcsd
        return eom_rdcsd.EOMEESinglet(self).kernel(nroots, koopmans, guess, eris)

    def eomee_ccsd_triplet(self, nroots=1, koopmans=False, guess=None, eris=None):
        from pyscf.cc import eom_rdcsd
        return eom_rdcsd.EOMEETriplet(self).kernel(nroots, koopmans, guess, eris)

    def eomsf_ccsd(self, nroots=1, koopmans=False, guess=None, eris=None):
        raise NotImplementedError

    def eomip_method(self):
        from pyscf.cc import eom_rdcsd
        return eom_rdcsd.EOMIP(self)

    def eomea_method(self):
        from pyscf.cc import eom_rdcsd
        return eom_rdcsd.EOMEA(self)

    def eomee_method(self):
        raise NotImplementedError("Please use eomee_ccsd_singlet or eomee_ccsd_triplet")

    def make_rdm1(self, t1=None, t2=None, l1=None, l2=None, ao_repr=False,
                  with_frozen=True, with_mf=True):
        '''Un-relaxed 1-particle density matrix in MO space'''
        from pyscf.cc import dcsd_rdm
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2
        if l1 is None: l1, l2 = self.solve_lambda(t1, t2)
        return dcsd_rdm.make_rdm1(self, t1, t2, l1, l2, ao_repr=ao_repr,
                                  with_frozen=with_frozen, with_mf=with_mf)

    def make_rdm2(self, t1=None, t2=None, l1=None, l2=None, ao_repr=False,
                  with_frozen=True, with_dm1=True):
        '''2-particle density matrix in MO space.  The density matrix is
        stored as

        dm2[p,r,q,s] = <p^+ q^+ s r>
        '''
        from pyscf.cc import dcsd_rdm
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2
        if l1 is None: l1, l2 = self.solve_lambda(t1, t2)
        return dcsd_rdm.make_rdm2(self, t1, t2, l1, l2, ao_repr=ao_repr,
                                  with_frozen=with_frozen, with_dm1=with_dm1)

    def nuc_grad_method(self):
        raise NotImplementedError
    
    update_amps = update_amps

class pCCSD(DCSD):
    __doc__ = ccsd.CCSD.__doc__
    p_mu = -1.0
    p_sigma = 1.0
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None, mu=-1.0, sigma=1.0):
        super().__init__(mf, frozen, mo_coeff, mo_occ)
        self.p_mu = mu
        self.p_sigma = sigma

class DCD(DCSD):
    def update_amps(self, t1, t2, eris):
        t1, t2 = update_amps(self, t1, t2, eris)
        return numpy.zeros_like(t1), t2

    def kernel(self, t2=None, eris=None):
        nocc = self.nocc
        nvir = self.nmo - nocc
        t1 = numpy.zeros((nocc, nvir))
        DCSD.kernel(self, t1, t2, eris)
        return self.e_corr, self.t2
    
    def solve_lambda(self, t2=None, l2=None, eris=None):
        from pyscf.cc import dcsd_lambda, ccsd_lambda
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)

        nocc = self.nocc
        nvir = self.nmo - nocc
        l1 = t1 = numpy.zeros((nocc, nvir))

        def update_lambda(mycc, t1, t2, l1, l2, eris=None, imds=None):
            l1, l2 = dcsd_lambda.update_lambda(mycc, t1, t2, l1, l2, eris, imds)
            return numpy.zeros_like(l1), l2

        self.converged_lambda, self.l1, self.l2 = \
                ccsd_lambda.kernel(self, eris, t1, t2, l1, l2,
                                   max_cycle=self.max_cycle,
                                   tol=self.conv_tol_normt,
                                   verbose=self.verbose, 
                                   fintermediates=dcsd_lambda.make_intermediates,
                                   fupdate=update_lambda)
        return self.l2
    
    def make_rdm1(self, t2=None, l2=None, ao_repr=False):
        from pyscf.cc import dcsd_rdm
        '''Un-relaxed 1-particle density matrix in MO space'''
        if t2 is None: t2 = self.t2
        if l2 is None: l2 = self.l2
        if l2 is None: l2 = self.solve_lambda(t2)

        nocc = self.nocc
        nvir = self.nmo - nocc
        l1 = t1 = numpy.zeros((nocc, nvir))

        return dcsd_rdm.make_rdm1(self, t1, t2, l1, l2, ao_repr=ao_repr)

    def make_rdm2(self, t1=None, t2=None, l1=None, l2=None, ao_repr=False):
        from pyscf.cc import dcsd_rdm
        '''2-particle density matrix in MO space.  The density matrix is
        stored as

        dm2[p,r,q,s] = <p^+ q^+ s r>
        '''
        if t2 is None: t2 = self.t2
        if l2 is None: l2 = self.l2
        if l2 is None: l2 = self.solve_lambda(t2)

        nocc = self.nocc
        nvir = self.nmo - nocc
        l1 = t1 = numpy.zeros((nocc, nvir))

        return dcsd_rdm.make_rdm2(self, t1, t2, l1, l2, ao_repr=ao_repr)

class pCCD(pCCSD):
    def update_amps(self, t1, t2, eris):
        t1, t2 = update_amps(self, t1, t2, eris)
        return numpy.zeros_like(t1), t2
    
    def kernel(self, t2=None, eris=None):
        nocc = self.nocc
        nvir = self.nmo - nocc
        t1 = numpy.zeros((nocc, nvir))
        pCCSD.kernel(self, t1, t2, eris)
        return self.e_corr, self.t2
    
    def solve_lambda(self, t2=None, l2=None, eris=None):
        from pyscf.cc import dcsd_lambda, ccsd_lambda
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)

        nocc = self.nocc
        nvir = self.nmo - nocc
        l1 = t1 = numpy.zeros((nocc, nvir))

        def update_lambda(mycc, t1, t2, l1, l2, eris=None, imds=None):
            l1, l2 = dcsd_lambda.update_lambda(mycc, t1, t2, l1, l2, eris, imds)
            return numpy.zeros_like(l1), l2

        self.converged_lambda, self.l1, self.l2 = \
                ccsd_lambda.kernel(self, eris, t1, t2, l1, l2,
                                   max_cycle=self.max_cycle,
                                   tol=self.conv_tol_normt,
                                   verbose=self.verbose, 
                                   fintermediates=dcsd_lambda.make_intermediates,
                                   fupdate=update_lambda)
        return self.l2
    
    def make_rdm1(self, t2=None, l2=None, ao_repr=False):
        from pyscf.cc import dcsd_rdm
        '''Un-relaxed 1-particle density matrix in MO space'''
        if t2 is None: t2 = self.t2
        if l2 is None: l2 = self.l2
        if l2 is None: l2 = self.solve_lambda(t2)

        nocc = self.nocc
        nvir = self.nmo - nocc
        l1 = t1 = numpy.zeros((nocc, nvir))

        return dcsd_rdm.make_rdm1(self, t1, t2, l1, l2, ao_repr=ao_repr)

    def make_rdm2(self, t1=None, t2=None, l1=None, l2=None, ao_repr=False):
        from pyscf.cc import dcsd_rdm
        '''2-particle density matrix in MO space.  The density matrix is
        stored as

        dm2[p,r,q,s] = <p^+ q^+ s r>
        '''
        if t2 is None: t2 = self.t2
        if l2 is None: l2 = self.l2
        if l2 is None: l2 = self.solve_lambda(t2)

        nocc = self.nocc
        nvir = self.nmo - nocc
        l1 = t1 = numpy.zeros((nocc, nvir))

        return dcsd_rdm.make_rdm2(self, t1, t2, l1, l2, ao_repr=ao_repr)

if __name__ == '__main__':
    from pyscf import scf
    mol = gto.Mole()
    mol.atom = '''
    O
    H  1  5.0
    H  1  5.0  2 107.6
'''
    mol.unit = "Bohr"
    mol.basis = 'cc-pvdz'
    mol.build()
    rhf = scf.RHF(mol)
    rhf.scf()
    mcc = DCSD(rhf)
    mcc.kernel()
    assert abs(mcc.e_tot - -75.90048871511547) < 1e-7
    mcc = pCCSD(rhf)
    mcc.kernel()
    assert abs(mcc.e_tot - -75.89848268554668) < 1e-7

    # recover CCSD
    mccp = pCCSD(rhf)
    mccp.p_mu = 1.0
    mccp.p_sigma =  1.0
    mccp.kernel()
    mcc = ccsd.CCSD(rhf)
    mcc.kernel()
    assert abs(mcc.e_tot - mccp.e_tot) < 1e-7

    # exact for two-electron system
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 1.5'
    mol.basis = "ccpvdz"
    mol.unit = "Bohr"
    mol.build()
    rhf = scf.RHF(mol).run()
    mdc = DCSD(rhf)
    mdc.kernel()
    mpcc = pCCSD(rhf)
    mpcc.kernel()
    mcc = ccsd.CCSD(rhf)
    mcc.kernel() 
    assert abs(mcc.e_tot - mdc.e_tot) < 1e-7
    assert abs(mcc.e_tot - mpcc.e_tot) < 1e-7
