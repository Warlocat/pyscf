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
Restricted DCSD implementation for real integrals.  Permutation symmetry for
the 4-index integrals (ij|kl) = (ij|lk) = (ji|kl) are assumed.

Note MO integrals are treated in chemist's notation
'''


from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import dcsd
from pyscf.cc.ccsd_lambda import _cp
from pyscf.cc import ccsd_lambda
from pyscf.cc import _ccsd
from pyscf.cc.ccsd import BLKMIN

def kernel(mycc, eris=None, t1=None, t2=None, l1=None, l2=None,
           max_cycle=50, tol=1e-8, verbose=logger.INFO):
    if eris is None: eris = mycc.ao2mo()
    return ccsd_lambda.kernel(mycc, eris, t1, t2, l1, l2, max_cycle, tol,
                              verbose, make_intermediates, update_lambda)

# l2, t2 as ijab
def make_intermediates(mycc, t1, t2, eris):
    p_alpha, p_beta, p_gamma, p_delta, is_dcsd = dcsd.cc_parameter(mycc)
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    foo = eris.fock[:nocc,:nocc]
    fov = eris.fock[:nocc,nocc:]
    #fvo = eris.fock[nocc:,:nocc]
    fvv = eris.fock[nocc:,nocc:]

    class _IMDS: pass
    imds = _IMDS()
    #TODO: mycc.incore_complete
    imds.ftmp = lib.H5TmpFile()
    imds.woooo = imds.ftmp.create_dataset('woooo', (nocc,nocc,nocc,nocc), 'f8')
    imds.wvooo = imds.ftmp.create_dataset('wvooo', (nvir,nocc,nocc,nocc), 'f8')
    imds.wVOov_p = imds.ftmp.create_dataset('wVOov_p', (nvir,nocc,nocc,nvir), 'f8')
    imds.wvOOv_p = imds.ftmp.create_dataset('wvOOv_p', (nvir,nocc,nocc,nvir), 'f8')
    imds.wvvov = imds.ftmp.create_dataset('wvvov', (nvir,nvir,nocc,nvir), 'f8')
    imds.woooo_p = imds.ftmp.create_dataset('woooo_p', (nocc,nocc,nocc,nocc), 'f8')

    w1 = fvv - numpy.einsum('ja,jb->ba', fov, t1)
    w2 = foo + numpy.einsum('ib,jb->ij', fov, t1)
    w1_p = w1.copy()
    w2_p = w2.copy()
    w3 = numpy.einsum('kc,jkbc->bj', fov, t2) * 2 + fov.T
    w3 -= numpy.einsum('kc,kjbc->bj', fov, t2)
    w3 += lib.einsum('kc,kb,jc->bj', fov, t1, t1)
    w4 = fov.copy()

    unit = nocc*nvir**2*6
    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    blksize = min(nvir, max(BLKMIN, int((max_memory*.95e6/8-nocc**4-nvir*nocc**3)/unit)))
    log.debug1('ccsd lambda make_intermediates: block size = %d, nvir = %d in %d blocks',
               blksize, nvir, int((nvir+blksize-1)//blksize))

    fswap = lib.H5TmpFile()
    for istep, (p0, p1) in enumerate(lib.prange(0, nvir, blksize)):
        eris_ovvv = eris.get_ovvv(slice(None), slice(p0,p1))
        fswap['vvov/%d'%istep] = eris_ovvv.transpose(2,3,0,1)

    woooo = numpy.zeros((nocc,nocc,nocc,nocc))
    woooo_p = numpy.zeros((nocc,nocc,nocc,nocc))
    wvooo = numpy.zeros((nvir,nocc,nocc,nocc))
    for p0, p1 in lib.prange(0, nvir, blksize):
        eris_ovvv = eris.get_ovvv(slice(None), slice(p0,p1))
        eris_vvov = numpy.empty(((p1-p0),nvir,nocc,nvir))
        for istep, (q0, q1) in enumerate(lib.prange(0, nvir, blksize)):
            eris_vvov[:,:,:,q0:q1] = fswap['vvov/%d'%istep][p0:p1]

        tmp = numpy.einsum('jcba,jc->ba', eris_ovvv, t1[:,p0:p1]*2)
        w1 += tmp
        w1_p += tmp
        tmp = numpy.einsum('jabc,jc->ba', eris_ovvv, t1)
        w1[:,p0:p1] -= tmp
        w1_p[:,p0:p1] -= tmp
        tmp = None
        theta = t2[:,:,:,p0:p1] * 2 - t2[:,:,:,p0:p1].transpose(1,0,2,3)
        w3 += lib.einsum('jkcd,kdcb->bj', theta, eris_ovvv)
        theta = None
        wVOov = lib.einsum('jbcd,kd->bjkc', eris_ovvv, t1)
        wvOOv = lib.einsum('cbjd,kd->cjkb', eris_vvov,-t1)
        g2vovv = eris_vvov.transpose(0,2,1,3) * 2 - eris_vvov.transpose(0,2,3,1)
        for i0, i1 in lib.prange(0, nocc, blksize):
            tau = t2[:,i0:i1] + numpy.einsum('ia,jb->ijab', t1, t1[i0:i1])
            wvooo[p0:p1,i0:i1] += lib.einsum('cibd,jkbd->ckij', g2vovv, tau)
        g2vovv = tau = None

        # Watch out memory usage here, due to the t2 transpose
        wvvov  = lib.einsum('jabd,jkcd->abkc', eris_ovvv, t2) * -1.5
        wvvov += eris_vvov.transpose(0,3,2,1) * 2
        wvvov -= eris_vvov

        g2vvov = eris_vvov * 2 - eris_ovvv.transpose(1,2,0,3)
        for i0, i1 in lib.prange(0, nocc, blksize):
            theta = t2[i0:i1] * 2 - t2[i0:i1].transpose(0,1,3,2)
            vackb = lib.einsum('acjd,kjbd->ackb', g2vvov, theta)
            wvvov[:,:,i0:i1] += vackb.transpose(0,3,2,1)
            wvvov[:,:,i0:i1] -= vackb * .5
        g2vvov = eris_ovvv = eris_vvov = theta = None

        eris_ovoo = _cp(eris.ovoo[:,p0:p1])
        tmp = numpy.einsum('kbij,kb->ij', eris_ovoo, t1[:,p0:p1]) * 2
        tmp -= numpy.einsum('ibkj,kb->ij', eris_ovoo, t1[:,p0:p1])
        w2 += tmp
        w2_p += tmp
        theta = t2[:,:,p0:p1].transpose(1,0,2,3) * 2 - t2[:,:,p0:p1]
        w3 -= lib.einsum('lckj,klcb->bj', eris_ovoo, theta)

        tmp = lib.einsum('lc,jcik->ijkl', t1[:,p0:p1], eris_ovoo)
        woooo += tmp
        woooo += tmp.transpose(1,0,3,2)
        woooo_p += tmp
        woooo_p += tmp.transpose(1,0,3,2)
        theta = tmp = None

        wvOOv += lib.einsum('lbjk,lc->bjkc', eris_ovoo, t1)
        wVOov -= lib.einsum('jbkl,lc->bjkc', eris_ovoo, t1)
        wvooo[p0:p1] += eris_ovoo.transpose(1,3,2,0) * 2
        wvooo[p0:p1] -= eris_ovoo.transpose(1,0,2,3)
        wvooo -= lib.einsum('klbc,iblj->ckij', t2[:,:,p0:p1], eris_ovoo*1.5)

        g2ovoo = eris_ovoo * 2 - eris_ovoo.transpose(2,1,0,3)
        theta = t2[:,:,:,p0:p1]*2 - t2[:,:,:,p0:p1].transpose(1,0,2,3)
        vcjik = lib.einsum('jlcb,lbki->cjki', theta, g2ovoo)
        wvooo += vcjik.transpose(0,3,2,1)
        wvooo -= vcjik*.5
        theta = g2ovoo = None

        eris_voov = _cp(eris.ovvo[:,p0:p1]).transpose(1,0,3,2)
        tau = t2[:,:,p0:p1] + numpy.einsum('ia,jb->ijab', t1[:,p0:p1], t1)
        woooo += lib.einsum('cijd,klcd->ijkl', eris_voov, tau)
        tau = t2[:,:,p0:p1] * p_gamma + numpy.einsum('ia,jb->ijab', t1[:,p0:p1], t1)
        woooo_p += lib.einsum('cijd,klcd->ijkl', eris_voov, tau) 
        tau = None

        g2voov = eris_voov*2 - eris_voov.transpose(0,2,1,3)
        tmpw4 = numpy.einsum('ckld,ld->kc', g2voov, t1)
        tmp = lib.einsum('ckja,kjcb->ba', g2voov, t2[:,:,p0:p1])
        w1 -= tmp
        w1_p -= tmp * p_alpha
        tmp = numpy.einsum('ja,jb->ba', tmpw4, t1)
        w1[:,p0:p1] -= tmp
        w1_p[:,p0:p1] -= tmp
        tmp = lib.einsum('jkbc,bikc->ij', t2[:,:,p0:p1], g2voov)
        w2 += tmp
        w2_p += tmp * p_beta
        tmp = numpy.einsum('ib,jb->ij', tmpw4, t1[:,p0:p1])
        w2 += tmp
        w2_p += tmp
        tmp = None
        w3 += reduce(numpy.dot, (t1.T, tmpw4, t1[:,p0:p1].T))
        w4[:,p0:p1] += tmpw4

        wvOOv += lib.einsum('bljd,kd,lc->bjkc', eris_voov, t1, t1)
        wVOov -= lib.einsum('bjld,kd,lc->bjkc', eris_voov, t1, t1)
        wvOOv_p = wvOOv.copy()
        wVOov_p = wVOov.copy()

        VOov  = lib.einsum('bjld,klcd->bjkc', g2voov, t2)
        VOov -= lib.einsum('bjld,kldc->bjkc', eris_voov, t2)
        VOov += eris_voov
        vOOv = lib.einsum('bljd,kldc->bjkc', eris_voov, t2)
        vOOv -= _cp(eris.oovv[:,:,p0:p1]).transpose(2,1,0,3)
        wVOov += VOov
        wvOOv += vOOv

        if is_dcsd:
            VOov_p  = lib.einsum('bjld,klcd->bjkc', eris_voov*2, t2)
            VOov_p -= lib.einsum('bjld,kldc->bjkc', eris_voov, t2)
            VOov_p += eris_voov
            vOOv_p = -_cp(eris.oovv[:,:,p0:p1]).transpose(2,1,0,3)
        else:
            VOov_p  = lib.einsum('bjld,klcd->bjkc', g2voov, t2) * p_delta
            VOov_p -= lib.einsum('bjld,kldc->bjkc', eris_voov, t2) * p_delta
            VOov_p += eris_voov
            vOOv_p = lib.einsum('bljd,kldc->bjkc', eris_voov, t2) * p_delta
            vOOv_p -= _cp(eris.oovv[:,:,p0:p1]).transpose(2,1,0,3)
        wVOov_p += VOov_p
        wvOOv_p += vOOv_p
        imds.wVOov_p[p0:p1] = wVOov_p
        imds.wvOOv_p[p0:p1] = wvOOv_p

        ov1 = vOOv*2 + VOov
        ov2 = VOov*2 + vOOv
        vOOv = VOov = wVOov_p = vOOv_p = None
        wvooo -= lib.einsum('jb,bikc->ckij', t1[:,p0:p1], ov1)
        wvooo += lib.einsum('kb,bijc->ckij', t1[:,p0:p1], ov2)
        w3 += numpy.einsum('ckjb,kc->bj', ov2, t1[:,p0:p1])

        wvvov += lib.einsum('ajkc,jb->abkc', ov1, t1)
        wvvov -= lib.einsum('ajkb,jc->abkc', ov2, t1)

        eris_ovoo = _cp(eris.ovoo[:,p0:p1])
        g2ovoo = eris_ovoo * 2 - eris_ovoo.transpose(2,1,0,3)
        tau = t2 + numpy.einsum('ia,jb->ijab', t1, t1)
        wvvov += lib.einsum('laki,klbc->abic', g2ovoo, tau)
        imds.wvvov[p0:p1] = wvvov
        wvvov = ov1 = ov2 = g2ovoo = None

    woooo += _cp(eris.oooo).transpose(0,2,1,3)
    woooo_p += _cp(eris.oooo).transpose(0,2,1,3)
    imds.woooo[:] = woooo
    imds.woooo_p[:] = woooo_p
    imds.wvooo[:] = wvooo
    woooo = wvooo = woooo_p = None

    w3 += numpy.einsum('bc,jc->bj', w1, t1)
    w3 -= numpy.einsum('kj,kb->bj', w2, t1)

    fswap = None

    imds.w1 = w1
    imds.w2 = w2
    imds.w1_p = w1_p
    imds.w2_p = w2_p
    imds.w3 = w3
    imds.w4 = w4
    imds.ftmp.flush()
    return imds


# update L1, L2
def update_lambda(mycc, t1, t2, l1, l2, eris=None, imds=None):
    if imds is None: imds = make_intermediates(mycc, t1, t2, eris)
    p_alpha, p_beta, p_gamma, p_delta, is_dcsd = dcsd.cc_parameter(mycc)
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + mycc.level_shift

    theta = t2*2 - t2.transpose(0,1,3,2)
    mba = lib.einsum('klca,klcb->ba', l2, theta)
    mij = lib.einsum('ikcd,jkcd->ij', l2, theta)
    theta = None
    mba1 = numpy.einsum('jc,jb->bc', l1, t1) + mba
    mij1 = numpy.einsum('kb,jb->kj', l1, t1) + mij
    mba1_p = numpy.einsum('jc,jb->bc', l1, t1) + mba * p_alpha
    mij1_p = numpy.einsum('kb,jb->kj', l1, t1) + mij * p_beta
    mia1 = t1 + numpy.einsum('kc,jkbc->jb', l1, t2) * 2
    mia1 -= numpy.einsum('kc,jkcb->jb', l1, t2)
    mia1 -= reduce(numpy.dot, (t1, l1.T, t1))
    mia1 -= numpy.einsum('bd,jd->jb', mba, t1)
    mia1 -= numpy.einsum('lj,lb->jb', mij, t1)

    l2new = mycc._add_vvvv(None, l2, eris, with_ovvv=False, t2sym='jiba')
    l1new  = numpy.einsum('ijab,jb->ia', l2new, t1) * 2
    l1new -= numpy.einsum('jiab,jb->ia', l2new, t1)
    l2new *= .5  # *.5 because of l2+l2.transpose(1,0,3,2) in the end
    tmp = None

    w1 = imds.w1 - numpy.diag(mo_e_v)
    w2 = imds.w2 - numpy.diag(mo_e_o)
    w1_p = imds.w1_p - numpy.diag(mo_e_v)
    w2_p = imds.w2_p - numpy.diag(mo_e_o)

    l1new += fov
    l1new += numpy.einsum('ib,ba->ia', l1, w1)
    l1new -= numpy.einsum('ja,ij->ia', l1, w2)
    l1new -= numpy.einsum('ik,ka->ia', mij, imds.w4)
    l1new -= numpy.einsum('ca,ic->ia', mba, imds.w4)
    l1new += numpy.einsum('ijab,bj->ia', l2, imds.w3) * 2
    l1new -= numpy.einsum('ijba,bj->ia', l2, imds.w3)

    l2new += numpy.einsum('ia,jb->ijab', l1, imds.w4)
    l2new += lib.einsum('jibc,ca->jiba', l2, w1_p)
    l2new -= lib.einsum('jk,kiba->jiba', w2_p, l2)

    eris_ovoo = _cp(eris.ovoo)
    l1new -= numpy.einsum('iajk,kj->ia', eris_ovoo, mij1) * 2
    l1new += numpy.einsum('jaik,kj->ia', eris_ovoo, mij1)
    l2new -= lib.einsum('jbki,ka->jiba', eris_ovoo, l1)
    eris_ovoo = None

    tau = _ccsd.make_tau(t2, t1, t1)
    l2tau = lib.einsum('ijcd,klcd->ijkl', l2, tau)
    tau = None
    tau_p = numpy.einsum('ia,jb->ijab', t1, t1)
    tau_p += t2*p_gamma
    l2tau_p = lib.einsum('ijcd,klcd->ijkl', l2, tau_p)
    tau_p = None
    l2t1 = lib.einsum('jidc,kc->ijkd', l2, t1)

    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    unit = nocc*nvir**2*5
    blksize = min(nocc, max(BLKMIN, int(max_memory*.95e6/8/unit)))
    log.debug1('block size = %d, nocc = %d is divided into %d blocks',
               blksize, nocc, int((nocc+blksize-1)/blksize))

    l1new -= numpy.einsum('jb,jiab->ia', l1, _cp(eris.oovv))
    for p0, p1 in lib.prange(0, nvir, blksize):
        eris_ovvv = eris.get_ovvv(slice(None), slice(p0,p1))
        l1new[:,p0:p1] += numpy.einsum('iabc,bc->ia', eris_ovvv, mba1) * 2
        l1new -= numpy.einsum('ibca,bc->ia', eris_ovvv, mba1[p0:p1])
        l2new[:,:,p0:p1] += lib.einsum('jbac,ic->jiba', eris_ovvv, l1)
        m4 = lib.einsum('ijkd,kadb->ijab', l2t1, eris_ovvv)
        l2new[:,:,p0:p1] -= m4
        l1new[:,p0:p1] -= numpy.einsum('ijab,jb->ia', m4, t1) * 2
        l1new -= numpy.einsum('ijab,ia->jb', m4, t1[:,p0:p1]) * 2
        l1new[:,p0:p1] += numpy.einsum('jiab,jb->ia', m4, t1)
        l1new += numpy.einsum('jiab,ia->jb', m4, t1[:,p0:p1])
        eris_ovvv = m4 = None

        eris_voov = _cp(eris.ovvo[:,p0:p1].transpose(1,0,3,2))
        l1new[:,p0:p1] += numpy.einsum('jb,aijb->ia', l1, eris_voov) * 2
        l2new[:,:,p0:p1] += eris_voov.transpose(1,2,0,3) * .5
        l2new[:,:,p0:p1] -= lib.einsum('bjic,ca->jiba', eris_voov, mba1_p)
        l2new[:,:,p0:p1] -= lib.einsum('bjka,ik->jiba', eris_voov, mij1_p)
        l1new[:,p0:p1] += numpy.einsum('aijb,jb->ia', eris_voov, mia1) * 2
        l1new -= numpy.einsum('bija,jb->ia', eris_voov, mia1[:,p0:p1])
        m4 = lib.einsum('ijkl,aklb->ijab', l2tau, eris_voov)
        l2new[:,:,p0:p1] += lib.einsum('ijkl,aklb->ijab', l2tau_p, eris_voov) * .5 
        l1new[:,p0:p1] += numpy.einsum('ijab,jb->ia', m4, t1) * 2
        l1new -= numpy.einsum('ijba,jb->ia', m4, t1[:,p0:p1])

        saved_wvooo = _cp(imds.wvooo[p0:p1])
        l1new -= lib.einsum('ckij,jkca->ia', saved_wvooo, l2[:,:,p0:p1])
        saved_wvovv = _cp(imds.wvvov[p0:p1])
        # Watch out memory usage here, due to the l2 transpose
        l1new[:,p0:p1] += lib.einsum('abkc,kibc->ia', saved_wvovv, l2)
        saved_wvooo = saved_wvovv = None

        saved_wvOOv_p = _cp(imds.wvOOv_p[p0:p1])
        tmp_voov_p = _cp(imds.wVOov_p[p0:p1]) * 2
        tmp_voov_p += saved_wvOOv_p
        tmp = l2.transpose(0,2,1,3) - l2.transpose(0,3,1,2)*.5
        l2new[:,:,p0:p1] += lib.einsum('iakc,bjkc->jiba', tmp, tmp_voov_p)
        tmp = None

        tmp = lib.einsum('jkca,bikc->jiba', l2, saved_wvOOv_p)
        l2new[:,:,p0:p1] += tmp
        l2new[:,:,p0:p1] += tmp.transpose(1,0,2,3) * .5
        saved_wvOOv_p = tmp = None

    saved_woooo = _cp(imds.woooo)
    m3 = lib.einsum('ijkl,klab->ijab', saved_woooo, l2)
    l1new += numpy.einsum('ijab,jb->ia', m3, t1) * 2
    l1new -= numpy.einsum('ijba,jb->ia', m3, t1)

    saved_woooo_p = _cp(imds.woooo_p)
    m3 = lib.einsum('ijkl,klab->ijab', saved_woooo_p, l2)
    l2new += m3 * .5
    saved_woooo = saved_woooo_p = m3 = None
    #time1 = log.timer_debug1('lambda pass [%d:%d]'%(p0, p1), *time1)

    eia = lib.direct_sum('i-a->ia', mo_e_o, mo_e_v)
    l1new /= eia

    for i in range(nocc):
        if i > 0:
            l2new[i,:i] += l2new[:i,i].transpose(0,2,1)
            l2new[i,:i] /= lib.direct_sum('a,jb->jab', eia[i], eia[:i])
            l2new[:i,i] = l2new[i,:i].transpose(0,2,1)
        l2new[i,i] = l2new[i,i] + l2new[i,i].T
        l2new[i,i] /= lib.direct_sum('a,b->ab', eia[i], eia[i])

    time0 = log.timer_debug1('update l1 l2', *time0)
    return l1new, l2new



if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo
    from pyscf.cc import dcsd_test, dcsd_rdm, ccsd, ccsd_rdm

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = '631g'
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1.0
    mf.kernel()

    mycc = dcsd.pCCSD(mf, mu = -2, sigma = 3.5)
    mycc.conv_tol = 1e-12
    mycc.kernel()
    dm1 = mycc.make_rdm1()
    dm2 = mycc.make_rdm2()
    nmo = mf.mo_coeff.shape[1]
    eri = ao2mo.kernel(mf._eri, mf.mo_coeff, compact=False).reshape([nmo]*4)
    hcore = mf.get_hcore()
    h1 = reduce(numpy.dot, (mf.mo_coeff.T, hcore, mf.mo_coeff))
    e1 = numpy.einsum('ij,ji', h1, dm1)
    e1+= numpy.einsum('ijkl,ijkl', eri, dm2) * .5
    e1+= mol.energy_nuc()
    print(e1 - mycc.e_tot)