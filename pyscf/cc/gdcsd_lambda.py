#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
from pyscf.cc import ccsd_lambda, gdcsd

einsum = lib.einsum

def kernel(mycc, eris=None, t1=None, t2=None, l1=None, l2=None,
           max_cycle=50, tol=1e-8, verbose=logger.INFO):
    if eris is None: eris = mycc.ao2mo()
    return ccsd_lambda.kernel(mycc, eris, t1, t2, l1, l2, max_cycle, tol,
                              verbose, make_intermediates, update_lambda)

# l2, t2 as ijab
def make_intermediates(mycc, t1, t2, eris):
    if hasattr(mycc, "p_mu"): # pCCSD
        # change notation
        p_alpha = mycc.p_sigma
        p_delta = mycc.p_sigma
        p_beta = (1.0 + mycc.p_mu) / 2.0
        p_gamma = mycc.p_mu
        is_dcsd = False
    elif hasattr(mycc, "oovv_phys"): # DCSD
        p_alpha = 0.5
        p_beta = 0.5
        p_delta = 1.0
        p_gamma = 0.0
        is_dcsd = True
    else:
        raise ValueError("Unknown method for gdcsd GDCSD-Lambda")
    nocc, nvir = t1.shape
    foo = eris.fock[:nocc,:nocc]
    fov = eris.fock[:nocc,nocc:]
    fvo = eris.fock[nocc:,:nocc]
    fvv = eris.fock[nocc:,nocc:]

    tau_ccsd = t2 + einsum('ia,jb->ijab', t1, t1) * 2

    # Fab
    v1 = fvv - einsum('ja,jb->ba', fov, t1)
    v1-= numpy.einsum('jbac,jc->ba', eris.ovvv, t1)
    v1+= einsum('jkca,jkbc->ba', eris.oovv, tau_ccsd) * .5
    v1_p = v1.copy()
    v1_p += einsum('jkca,jkbc->ba', eris.oovv, t2) * .5 * (p_alpha - 1.0)

    # Fij
    v2 = foo + einsum('ib,jb->ij', fov, t1)
    v2-= numpy.einsum('kijb,kb->ij', eris.ooov, t1)
    v2 += einsum('ikbc,jkbc->ij', eris.oovv, tau_ccsd) * .5
    v2_p = v2.copy()
    v2_p += einsum('ikbc,jkbc->ij', eris.oovv, t2) * .5 * (p_beta - 1.0)

    v4 = einsum('ljdb,klcd->jcbk', eris.oovv, t2)
    v4+= numpy.asarray(eris.ovvo)

    v5 = fvo + numpy.einsum('kc,jkbc->bj', fov, t2)
    tmp = fov - numpy.einsum('kldc,ld->kc', eris.oovv, t1)
    v5+= numpy.einsum('kc,kb,jc->bj', tmp, t1, t1, optimize=True)
    v5-= einsum('kljc,klbc->bj', eris.ooov, t2) * .5
    v5+= einsum('kbdc,jkcd->bj', eris.ovvv, t2) * .5

    w3 = v5 + numpy.einsum('jcbk,jb->ck', v4, t1)
    w3 += numpy.einsum('cb,jb->cj', v1, t1)
    w3 -= numpy.einsum('jk,jb->bk', v2, t1)

    woooo = numpy.asarray(eris.oooo) * .5
    woooo+= einsum('jilc,kc->jilk', eris.ooov, t1)
    woooo+= einsum('ijcd,klcd->ijkl', eris.oovv, tau_ccsd) * .25
    woooo_p = woooo.copy()
    woooo_p+= einsum('ijcd,klcd->ijkl', eris.oovv, t2) * .25 * (p_gamma - 1.0)

    wovvo = - numpy.einsum('ljdb,lc,kd->jcbk', eris.oovv, t1, t1, optimize=True)
    wovvo-= einsum('ljkb,lc->jcbk', eris.ooov, t1)
    wovvo+= einsum('jcbd,kd->jcbk', eris.ovvv, t1)
    wovvo_p = wovvo.copy()
    wovvo+= v4
    wovvo_p += numpy.asarray(eris.ovvo)
    if is_dcsd:
        wovvo_p += einsum('ljdb,klcd->jcbk', mycc.oovv_phys, t2)
    else:
        wovvo_p += einsum('ljdb,klcd->jcbk', eris.oovv, t2) * p_delta


    wovoo = einsum('icdb,jkdb->icjk', eris.ovvv, tau_ccsd) * .25
    wovoo+= numpy.einsum('jkic->icjk', numpy.asarray(eris.ooov).conj()) * .5
    wovoo+= einsum('icbk,jb->icjk', v4, t1)
    wovoo-= einsum('lijb,klcb->icjk', eris.ooov, t2)

    wvvvo = einsum('jcak,jb->bcak', v4, t1)
    wvvvo+= einsum('jlka,jlbc->bcak', eris.ooov, tau_ccsd) * .25
    wvvvo-= numpy.einsum('jacb->bcaj', numpy.asarray(eris.ovvv).conj()) * .5
    wvvvo+= einsum('kbad,jkcd->bcaj', eris.ovvv, t2)

    class _IMDS: pass
    imds = _IMDS()
    imds.ftmp = lib.H5TmpFile()
    dtype = numpy.result_type(t2, eris.vvvv).char
    imds.woooo = imds.ftmp.create_dataset('woooo', (nocc,nocc,nocc,nocc), dtype)
    imds.wovvo = imds.ftmp.create_dataset('wovvo', (nocc,nvir,nvir,nocc), dtype)
    imds.wovoo = imds.ftmp.create_dataset('wovoo', (nocc,nvir,nocc,nocc), dtype)
    imds.wvvvo = imds.ftmp.create_dataset('wvvvo', (nvir,nvir,nvir,nocc), dtype)
    imds.woooo_p = imds.ftmp.create_dataset('woooo_p', (nocc,nocc,nocc,nocc), dtype)
    imds.wovvo_p = imds.ftmp.create_dataset('wovvo_p', (nocc,nvir,nvir,nocc), dtype)
    imds.woooo[:] = woooo
    imds.wovvo[:] = wovvo
    imds.wovoo[:] = wovoo
    imds.wvvvo[:] = wvvvo
    imds.woooo_p[:] = woooo_p
    imds.wovvo_p[:] = wovvo_p
    imds.v1 = v1
    imds.v2 = v2
    imds.v1_p = v1_p
    imds.v2_p = v2_p
    imds.w3 = w3
    imds.p_alpha = p_alpha
    imds.p_beta = p_beta
    imds.p_gamma = p_gamma
    imds.p_delta = p_delta
    imds.ftmp.flush()
    return imds


# update L1, L2
def update_lambda(mycc, t1, t2, l1, l2, eris, imds):
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + mycc.level_shift
    v1 = imds.v1 - numpy.diag(mo_e_v)
    v2 = imds.v2 - numpy.diag(mo_e_o)
    v1_p = imds.v1_p - numpy.diag(mo_e_v)
    v2_p = imds.v2_p - numpy.diag(mo_e_o)

    l1new = numpy.zeros_like(l1)
    l2new = numpy.zeros_like(l2)

    # oooo and vvvv for l2
    m3 = einsum('klab,ijkl->ijab', l2, numpy.asarray(imds.woooo_p))
    tau_gamma = einsum('ia,jb->ijab', t1, t1) * 2 + t2 * imds.p_gamma
    tmp = einsum('ijcd,klcd->ijkl', l2, tau_gamma)
    oovv = numpy.asarray(eris.oovv)
    m3 += einsum('klab,ijkl->ijab', oovv, tmp) * .25
    tmp = einsum('ijcd,kd->ijck', l2, t1)
    l2_tmp = -einsum('kcba,ijck->ijab', eris.ovvv, tmp)
    l2_tmp += einsum('ijcd,cdab->ijab', l2, eris.vvvv) * .5
    l2new += oovv + m3 + l2_tmp

    fov1 = fov + einsum('kjcb,kc->jb', oovv, t1)
    tmp = einsum('ia,jb->ijab', l1, fov1)
    tmp+= einsum('kica,jcbk->ijab', l2, numpy.asarray(imds.wovvo_p))
    tmp = tmp - tmp.transpose(1,0,2,3)
    l2new += tmp - tmp.transpose(0,1,3,2)

    mba = einsum('klca,klcb->ba', l2, t2) * .5
    tmp = einsum('ka,ijkb->ijab', l1, eris.ooov)
    tmp+= einsum('ijca,cb->ijab', l2, v1_p)
    tmp1vv = einsum('ka,kb->ba', l1, t1) + mba * imds.p_alpha
    tmp+= einsum('ca,ijcb->ijab', tmp1vv, oovv)
    l2new -= tmp - tmp.transpose(0,1,3,2)

    mij = einsum('kicd,kjcd->ij', l2, t2) * .5
    tmp = einsum('ic,jcba->jiba', l1, eris.ovvv)
    tmp+= einsum('kiab,jk->ijab', l2, v2_p)
    tmp1oo = einsum('ic,kc->ik', l1, t1) + mij * imds.p_beta
    tmp-= einsum('ik,kjab->ijab', tmp1oo, oovv)
    l2new += tmp - tmp.transpose(1,0,2,3)

    m3 = einsum('klab,ijkl->ijab', l2, numpy.asarray(imds.woooo))
    tau_ccsd = t2 + einsum('ia,jb->ijab', t1, t1) * 2
    tmp = einsum('ijcd,klcd->ijkl', l2, tau_ccsd)
    m3 += einsum('klab,ijkl->ijab', oovv, tmp) * .25
    m3 += l2_tmp

    tmp1vv = mba + einsum('ka,kb->ba', l1, t1)
    tmp1oo = mij + einsum('ic,kc->ik', l1, t1)

    l1new += fov
    l1new += einsum('jb,ibaj->ia', l1, eris.ovvo)
    l1new += einsum('ib,ba->ia', l1, v1)
    l1new -= einsum('ja,ij->ia', l1, v2)
    l1new -= einsum('kjca,icjk->ia', l2, imds.wovoo)
    l1new -= einsum('ikbc,bcak->ia', l2, imds.wvvvo)
    l1new += einsum('ijab,jb->ia', m3, t1)
    l1new += einsum('jiba,bj->ia', l2, imds.w3)
    tmp =(t1 + einsum('kc,kjcb->jb', l1, t2)
          - einsum('bd,jd->jb', tmp1vv, t1)
          - einsum('lj,lb->jb', mij, t1))
    l1new += numpy.einsum('jiba,jb->ia', oovv, tmp)
    l1new += numpy.einsum('icab,bc->ia', eris.ovvv, tmp1vv)
    l1new -= numpy.einsum('jika,kj->ia', eris.ooov, tmp1oo)
    tmp = fov - einsum('kjba,jb->ka', oovv, t1)
    l1new -= numpy.einsum('ik,ka->ia', mij, tmp)
    l1new -= numpy.einsum('ca,ic->ia', mba, tmp)

    eia = lib.direct_sum('i-j->ij', mo_e_o, mo_e_v)
    l1new /= eia
    l2new /= lib.direct_sum('ia+jb->ijab', eia, eia)

    time0 = log.timer_debug1('update l1 l2', *time0)
    return l1new, l2new


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import ao2mo
    from pyscf.cc import gdcsd_rdm, gdcsd
    from functools import reduce

    mol = gto.Mole()
    mol.atom = [
        ["N" , (0. , 0.     , 0.)],
        ["N" , (0. , -1.2 , 0.0)]]
    mol.basis = '631g'
    mol.build()
    mf = scf.RHF(mol).run(tol_con=1.)
    ghf = scf.addons.convert_to_ghf(mf)

    mcc = gdcsd.pGCCSD(ghf)
    mcc.conv_tol = 1e-12
    ecc, t1, t2 = mcc.kernel()
    conv, l1, l2 = kernel(mcc, t1=t1, t2=t2, tol=1e-8)

    dm1 = gdcsd_rdm.make_rdm1(mcc, t1, t2, l1, l2)
    dm2 = gdcsd_rdm.make_rdm2(mcc, t1, t2, l1, l2)
    nao = mol.nao_nr()
    mo_a = ghf.mo_coeff[:nao]
    mo_b = ghf.mo_coeff[nao:]
    nmo = mo_a.shape[1]
    eri = ao2mo.kernel(ghf._eri, mo_a+mo_b, compact=False).reshape([nmo]*4)
    orbspin = ghf.mo_coeff.orbspin
    sym_forbid = (orbspin[:,None] != orbspin)
    eri[sym_forbid,:,:] = 0
    eri[:,:,sym_forbid] = 0
    hcore = scf.RHF(mol).get_hcore()
    from functools import reduce
    h1 = reduce(numpy.dot, (mo_a.T.conj(), hcore, mo_a))
    h1+= reduce(numpy.dot, (mo_b.T.conj(), hcore, mo_b))
    e1 = numpy.einsum('ij,ji', h1, dm1)
    e1+= numpy.einsum('ijkl,ijkl', eri, dm2) * .5
    e1+= mol.energy_nuc()
    print(e1 - mcc.e_tot)


    
    