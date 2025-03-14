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


import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.cc import ccsd, ccsd_rdm, dcsd

def _gamma2_outcore(mycc, t1, t2, l1, l2, h5fobj, compress_vvvv=False):
    p_alpha, p_beta, p_gamma, p_delta, is_dcsd = dcsd.cc_parameter(mycc)
    if is_dcsd:
        raise NotImplementedError
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    nvir_pair = nvir * (nvir+1) //2
    dtype = numpy.result_type(t1, t2, l1, l2).char
    if compress_vvvv:
        dvvvv = h5fobj.create_dataset('dvvvv', (nvir_pair,nvir_pair), dtype)
    else:
        dvvvv = h5fobj.create_dataset('dvvvv', (nvir,nvir,nvir,nvir), dtype)
    dovvo = h5fobj.create_dataset('dovvo', (nocc,nvir,nvir,nocc), dtype,
                                  chunks=(nocc,1,nvir,nocc))
    fswap = lib.H5TmpFile()

    time1 = logger.process_clock(), logger.perf_counter()
    pvOOv = lib.einsum('ikca,jkcb->aijb', l2, t2)
    moo = numpy.einsum('dljd->jl', pvOOv) * 2
    mvv = numpy.einsum('blld->db', pvOOv) * 2
    gooov = lib.einsum('kc,cija->jkia', t1, pvOOv)
    fswap['mvOOv'] = pvOOv
    pvOOv = None

    pvoOV = -lib.einsum('ikca,jkbc->aijb', l2, t2)
    theta = t2 * 2 - t2.transpose(0,1,3,2)
    pvoOV += lib.einsum('ikac,jkbc->aijb', l2, theta)
    moo += numpy.einsum('dljd->jl', pvoOV)
    mvv += numpy.einsum('blld->db', pvoOV)
    gooov -= lib.einsum('jc,cika->jkia', t1, pvoOV)
    fswap['mvoOV'] = pvoOV
    pvoOV = theta = None

    mia =(numpy.einsum('kc,ikac->ia', l1, t2) * 2 -
          numpy.einsum('kc,ikca->ia', l1, t2))
    mab = numpy.einsum('kc,kb->cb', l1, t1)
    mij = numpy.einsum('kc,jc->jk', l1, t1)

    tau = numpy.einsum('ia,jb->ijab', t1, t1)
    tau += t2
    goooo = lib.einsum('ijab,klab->ijkl', tau, l2)*.5
    h5fobj['doooo'] = (goooo.transpose(0,2,1,3)*2 -
                       goooo.transpose(0,3,1,2)).conj()
    tau_p = numpy.einsum('ia,jb->ijab', t1, t1)
    tau_p += t2*p_gamma
    goooo_p = lib.einsum('ijab,klab->ijkl', tau_p, l2)*.5

    gooov += numpy.einsum('ji,ka->jkia', -.5*moo, t1)
    gooov += lib.einsum('la,jkil->jkia', 2*t1, goooo)
    gooov -= lib.einsum('ib,jkba->jkia', l1, tau)
    gooov = gooov.conj()
    gooov -= lib.einsum('jkba,ib->jkia', l2, t1)
    h5fobj['dooov'] = gooov.transpose(0,2,1,3)*2 - gooov.transpose(1,2,0,3)
    tau = None
    time1 = log.timer_debug1('rdm intermediates pass1', *time1)

    goovv = numpy.einsum('ia,jb->ijab', mia.conj(), t1.conj())
    if is_dcsd:
        goovv_dcsd = numpy.zeros_like(goovv)
    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    unit = nocc**2*nvir*6
    blksize = min(nocc, nvir, max(ccsd.BLKMIN, int(max_memory*.95e6/8/unit)))
    doovv = h5fobj.create_dataset('doovv', (nocc,nocc,nvir,nvir), dtype,
                                  chunks=(nocc,nocc,1,nvir))

    log.debug1('rdm intermediates pass 2: block size = %d, nvir = %d in %d blocks',
               blksize, nvir, int((nvir+blksize-1)/blksize))
    for p0, p1 in lib.prange(0, nvir, blksize):
        tau = numpy.einsum('ia,jb->ijab', t1[:,p0:p1], t1)
        tau += t2[:,:,p0:p1]
        tmpoovv  = lib.einsum('ijkl,ka,lb->ijab', goooo, t1[:,p0:p1], t1)
        tmpoovv += lib.einsum('ijkl,klab->ijab', goooo_p, t2[:,:,p0:p1])
        tmpoovv -= lib.einsum('jk,ikab->ijab', mij, tau)
        tmpoovv -= lib.einsum('jk,ikab->ijab', moo*.5, t2[:,:,p0:p1]) * p_beta
        tmpoovv -= lib.einsum('jk,ia,kb->ijab', moo*.5, t1[:,p0:p1], t1)
        tmpoovv -= lib.einsum('cb,ijac->ijab', mab, t2[:,:,p0:p1])
        tmpoovv -= lib.einsum('bd,ijad->ijab', mvv*.5, t2[:,:,p0:p1]) * p_alpha
        tmpoovv -= lib.einsum('bd,ia,jd->ijab', mvv*.5, t1[:,p0:p1], t1)
        tmpoovv += .5 * tau
        tmpoovv = tmpoovv.conj()
        tmpoovv += .5 * l2[:,:,p0:p1]
        goovv[:,:,p0:p1] += tmpoovv
        tau = None

        pvOOv = fswap['mvOOv'][p0:p1]
        pvoOV = fswap['mvoOV'][p0:p1]
        gOvvO = lib.einsum('kiac,jc,kb->iabj', l2[:,:,p0:p1], t1, t1)
        gOvvO += numpy.einsum('aijb->iabj', pvOOv)
        govVO = numpy.einsum('ia,jb->iabj', l1[:,p0:p1], t1)
        govVO -= lib.einsum('ikac,jc,kb->iabj', l2[:,:,p0:p1], t1, t1)
        govVO += numpy.einsum('aijb->iabj', pvoOV)
        dovvo[:,p0:p1] = 2*govVO + gOvvO
        doovv[:,:,p0:p1] = (-2*gOvvO - govVO).transpose(3,0,1,2).conj()
        gOvvO = govVO = None
        if is_dcsd:
            # Something wrong here
            for q0, q1 in lib.prange(0, nvir, blksize):
                goovv[:,:,q0:q1,:] += lib.einsum('dlib,jd,la->ijab', pvOOv, t1[:,p0:p1], t1[:,q0:q1]).conj()
                goovv[:,:,:,q0:q1] -= lib.einsum('dlia,jd,lb->ijab', pvoOV, t1[:,p0:p1], t1[:,q0:q1]).conj()
                goovv_dcsd[:,:,q0:q1,:] += lib.einsum('dlib,jlda->ijab', pvOOv, t2[:,:,p0:p1,q0:q1]).conj()
                goovv_dcsd[:,:,:,q0:q1] -= lib.einsum('dlia,jldb->ijab', pvoOV, t2[:,:,p0:p1,q0:q1]).conj()
                tmp = pvoOV[:,:,:,q0:q1] + pvOOv[:,:,:,q0:q1]*.5
                goovv_dcsd[:,:,q0:q1,:] += lib.einsum('dlia,jlbd->ijab', tmp, t2[:,:,:,p0:p1]).conj()
        else:
            tau_delta = numpy.einsum('ia,jb->ijab', t1[:,p0:p1], t1)
            tau_delta += t2[:,:,p0:p1] * .5 * p_delta
            for q0, q1 in lib.prange(0, nvir, blksize):
                goovv[:,:,q0:q1,:] += lib.einsum('dlib,jlda->ijab', pvOOv, tau_delta[:,:,:,q0:q1]).conj()
                goovv[:,:,:,q0:q1] -= lib.einsum('dlia,jldb->ijab', pvoOV, tau_delta[:,:,:,q0:q1]).conj()
                tmp = pvoOV[:,:,:,q0:q1] + pvOOv[:,:,:,q0:q1]*.5
                goovv[:,:,q0:q1,:] += lib.einsum('dlia,jlbd->ijab', tmp, t2[:,:,:,p0:p1]).conj() * p_delta
            tau_delta = None
        pvOOv = pvoOV = None
        time1 = log.timer_debug1('rdm intermediates pass2 [%d:%d]'%(p0, p1), *time1)
    
    if is_dcsd:
        h5fobj['dovov'] = goovv.transpose(0,2,1,3) * 2 - goovv.transpose(1,2,0,3) + goovv_dcsd.transpose(0,2,1,3) * 2
        goovv_dcsd = None
    else:
        h5fobj['dovov'] = goovv.transpose(0,2,1,3) * 2 - goovv.transpose(1,2,0,3)
    goovv = goooo = None

    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    unit = max(nocc**2*nvir*2+nocc*nvir**2*3,
               nvir**3*2+nocc*nvir**2*2+nocc**2*nvir*2)
    blksize = min(nvir, max(ccsd.BLKMIN, int(max_memory*.9e6/8/unit)))
    log.debug1('rdm intermediates pass 3: block size = %d, nvir = %d in %d blocks',
               blksize, nocc, int((nvir+blksize-1)/blksize))
    dovvv = h5fobj.create_dataset('dovvv', (nocc,nvir,nvir,nvir), dtype,
                                  chunks=(nocc,min(nocc,nvir),1,nvir))
    time1 = logger.process_clock(), logger.perf_counter()
    for istep, (p0, p1) in enumerate(lib.prange(0, nvir, blksize)):
        l2tmp = l2[:,:,p0:p1]
        gvvvv = lib.einsum('ijab,ijcd->abcd', l2tmp, t2)
        jabc = lib.einsum('ijab,ic->jabc', l2tmp, t1)
        gvvvv += lib.einsum('jabc,jd->abcd', jabc, t1)
        l2tmp = jabc = None

        if compress_vvvv:
            # symmetrize dvvvv because it does not affect the results of ccsd_grad
            # dvvvv = gvvvv.transpose(0,2,1,3)-gvvvv.transpose(0,3,1,2)*.5
            # dvvvv = (dvvvv+dvvvv.transpose(0,1,3,2)) * .5
            # dvvvv = (dvvvv+dvvvv.transpose(1,0,2,3)) * .5
            # now dvvvv == dvvvv.transpose(0,1,3,2) == dvvvv.transpose(1,0,3,2)
            tmp = numpy.empty((nvir,nvir,nvir))
            tmpvvvv = numpy.empty((p1-p0,nvir,nvir_pair))
            for i in range(p1-p0):
                vvv = gvvvv[i].conj().transpose(1,0,2)
                tmp[:] = vvv - vvv.transpose(2,1,0)*.5
                lib.pack_tril(tmp+tmp.transpose(0,2,1), out=tmpvvvv[i])
            # tril of (dvvvv[p0:p1,p0:p1]+dvvvv[p0:p1,p0:p1].T)
            for i in range(p0, p1):
                for j in range(p0, i):
                    tmpvvvv[i-p0,j] += tmpvvvv[j-p0,i]
                tmpvvvv[i-p0,i] *= 2
            for i in range(p1, nvir):
                off = i * (i+1) // 2
                dvvvv[off+p0:off+p1] = tmpvvvv[:,i]
            for i in range(p0, p1):
                off = i * (i+1) // 2
                if p0 > 0:
                    tmpvvvv[i-p0,:p0] += dvvvv[off:off+p0]
                dvvvv[off:off+i+1] = tmpvvvv[i-p0,:i+1] * .25
            tmp = tmpvvvv = None
        else:
            for i in range(p0, p1):
                vvv = gvvvv[i-p0].conj().transpose(1,0,2)
                dvvvv[i] = vvv - vvv.transpose(2,1,0)*.5

        gvovv = lib.einsum('adbc,id->aibc', gvvvv, -t1)
        gvvvv = None

        gvovv += lib.einsum('akic,kb->aibc', fswap['mvoOV'][p0:p1], t1)
        gvovv -= lib.einsum('akib,kc->aibc', fswap['mvOOv'][p0:p1], t1)

        gvovv += lib.einsum('ja,jibc->aibc', l1[:,p0:p1], t2)
        gvovv += lib.einsum('ja,jb,ic->aibc', l1[:,p0:p1], t1, t1)
        gvovv += numpy.einsum('ba,ic->aibc', mvv[:,p0:p1]*.5, t1)
        gvovv = gvovv.conj()
        gvovv += lib.einsum('ja,jibc->aibc', t1[:,p0:p1], l2)

        dovvv[:,:,p0:p1] = gvovv.transpose(1,3,0,2)*2 - gvovv.transpose(1,2,0,3)
        gvovv = None
        time1 = log.timer_debug1('rdm intermediates pass3 [%d:%d]'%(p0, p1), *time1)

    fswap = None
    dvvov = None
    return (h5fobj['dovov'], h5fobj['dvvvv'], h5fobj['doooo'], h5fobj['doovv'],
            h5fobj['dovvo'], dvvov          , h5fobj['dovvv'], h5fobj['dooov'])

make_rdm1 = ccsd_rdm.make_rdm1

def make_rdm2(mycc, t1, t2, l1, l2, ao_repr=False, with_frozen=True, with_dm1=True):
    r'''
    Spin-traced two-particle density matrix in MO basis

    dm2[p,q,r,s] = \sum_{sigma,tau} <p_sigma^\dagger r_tau^\dagger s_tau q_sigma>

    Note the contraction between ERIs (in Chemist's notation) and rdm2 is
    E = einsum('pqrs,pqrs', eri, rdm2)
    '''
    d1 = ccsd_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2)
    f = lib.H5TmpFile()
    d2 = _gamma2_outcore(mycc, t1, t2, l1, l2, f, False)
    return ccsd_rdm._make_rdm2(mycc, d1, d2, with_dm1=with_dm1, with_frozen=with_frozen,
                      ao_repr=ao_repr)
