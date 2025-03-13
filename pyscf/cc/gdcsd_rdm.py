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
#         Jun Yang
#

import numpy
from pyscf import lib
from pyscf.cc import gdcsd, gccsd_rdm

#einsum = numpy.einsum
einsum = lib.einsum

def _gamma2_intermediates(mycc, t1, t2, l1, l2):
    assert hasattr(mycc, "p_mu"), "DCSD two-particle density matrices are not well-defined"
    tau = t2 + einsum('ia,jb->ijab', t1, t1) * 2
    miajb = einsum('ikac,kjcb->iajb', l2, t2)

    goovv = 0.25 * (l2.conj() + tau)
    tmp = einsum('kc,kica->ia', l1, t2)
    goovv += einsum('ia,jb->ijab', tmp, t1)
    tmp = einsum('kc,kb->cb', l1, t1)
    goovv += einsum('cb,ijca->ijab', tmp, t2) * .5
    tmp = einsum('kc,jc->kj', l1, t1)
    goovv += einsum('kiab,kj->ijab', tau, tmp) * .5
    tmp = numpy.einsum('ldjd->lj', miajb)
    tau_p = t2 * (0.5+0.5*mycc.p_mu) + einsum('ia,jb->ijab', t1, t1) * 2
    goovv -= einsum('lj,liba->ijab', tmp, tau_p) * .25
    tmp = numpy.einsum('ldlb->db', miajb)
    tau_p = t2 * mycc.p_sigma + einsum('ia,jb->ijab', t1, t1) * 2
    goovv -= einsum('db,jida->ijab', tmp, tau_p) * .25
    tau_p = t2 * mycc.p_sigma + einsum('ia,jb->ijab', t1, t1) * 2
    goovv -= einsum('ldia,ljbd->ijab', miajb, tau_p) * .5
    tmp = einsum('klcd,ijcd->ijkl', l2, tau) * .25**2
    goovv += einsum('ijkl,ka,lb->ijab', tmp, t1, t1) * 2

    tau_p = t2 * mycc.p_mu + einsum('ia,jb->ijab', t1, t1) * 2
    tmp = einsum('klcd,ijcd->ijkl', l2, tau_p) * .25**2
    goovv += einsum('ijkl,klab->ijab', tmp, t2)
    goovv = goovv.conj()

    gvvvv = einsum('ijab,ijcd->abcd', tau, l2) * 0.125
    goooo = einsum('klab,ijab->klij', l2, tau) * 0.125

    gooov  = einsum('jkba,ib->jkia', tau, l1) * -0.25
    gooov += einsum('iljk,la->jkia', goooo, t1)
    tmp = numpy.einsum('icjc->ij', miajb) * .25
    gooov -= einsum('ij,ka->jkia', tmp, t1)
    gooov += einsum('icja,kc->jkia', miajb, t1) * .5
    gooov = gooov.conj()
    gooov += einsum('jkab,ib->jkia', l2, t1) * .25

    govvo  = einsum('ia,jb->ibaj', l1, t1)
    govvo += numpy.einsum('iajb->ibaj', miajb)
    govvo -= einsum('ikac,jc,kb->ibaj', l2, t1, t1)

    govvv  = einsum('ja,ijcb->iacb', l1, tau) * .25
    govvv += einsum('bcad,id->iabc', gvvvv, t1)
    tmp = numpy.einsum('kakb->ab', miajb) * .25
    govvv += einsum('ab,ic->iacb', tmp, t1)
    govvv += einsum('kaib,kc->iabc', miajb, t1) * .5
    govvv = govvv.conj()
    govvv += einsum('ijbc,ja->iabc', l2, t1) * .25

    dovov = goovv.transpose(0,2,1,3) - goovv.transpose(0,3,1,2)
    dvvvv = gvvvv.transpose(0,2,1,3) - gvvvv.transpose(0,3,1,2)
    doooo = goooo.transpose(0,2,1,3) - goooo.transpose(0,3,1,2)
    dovvv = govvv.transpose(0,2,1,3) - govvv.transpose(0,3,1,2)
    dooov = gooov.transpose(0,2,1,3) - gooov.transpose(1,2,0,3)
    dovvo = govvo.transpose(0,2,1,3)
    dovov =(dovov + dovov.transpose(2,3,0,1)) * .5
    dvvvv = dvvvv + dvvvv.transpose(1,0,3,2).conj()
    doooo = doooo + doooo.transpose(1,0,3,2).conj()
    dovvo =(dovvo + dovvo.transpose(3,2,1,0).conj()) * .5
    doovv = None # = -dovvo.transpose(0,3,2,1)
    dvvov = None
    return (dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov)

make_rdm1 = gccsd_rdm.make_rdm1

def make_rdm2(mycc, t1, t2, l1, l2, ao_repr=False, with_frozen=True, with_dm1=True):
    r'''
    Two-particle density matrix in the molecular spin-orbital representation

    dm2[p,q,r,s] = <p^\dagger r^\dagger s q>

    where p,q,r,s are spin-orbitals. p,q correspond to one particle and r,s
    correspond to another particle.  The contraction between ERIs (in
    Chemist's notation) and rdm2 is
    E = einsum('pqrs,pqrs', eri, rdm2)
    '''
    d1 = gccsd_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2)
    d2 = _gamma2_intermediates(mycc, t1, t2, l1, l2)
    return gccsd_rdm._make_rdm2(mycc, d1, d2, with_dm1=with_dm1, with_frozen=with_frozen,
                      ao_repr=ao_repr)