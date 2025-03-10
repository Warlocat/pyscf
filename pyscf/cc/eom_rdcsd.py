#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
# Author: Chaoqun Zhang <cq_zhang@outlook.com>
#         Qiming Sun <osirpt.sun@gmail.com>
#


import numpy as np

from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger, module_method
from pyscf.cc import ccsd, eom_rccsd, dcsd
from pyscf.cc import rintermediates as imd
from pyscf import __config__

einsum = lib.einsum

def Loo_p(t1, t2, eris, beta):
    nocc, nvir = t1.shape
    foo = eris.fock[:nocc,:nocc]
    eris_ovov = np.asarray(eris.ovov)
    Lki  = 2*beta*einsum('kcld,ilcd->ki', eris_ovov, t2)
    Lki -=   beta*einsum('kdlc,ilcd->ki', eris_ovov, t2)
    Lki += 2*einsum('kcld,ic,ld->ki', eris_ovov, t1, t1)
    Lki -=   einsum('kdlc,ic,ld->ki', eris_ovov, t1, t1)
    Lki += foo
    
    fov = eris.fock[:nocc,nocc:]
    Lki += np.einsum('kc,ic->ki',fov, t1)
    eris_ovoo = np.asarray(eris.ovoo)
    Lki += 2*np.einsum('lcki,lc->ki', eris_ovoo, t1)
    Lki -=   np.einsum('kcli,lc->ki', eris_ovoo, t1)
    return Lki

def Lvv_p(t1, t2, eris, alpha):
    nocc, nvir = t1.shape
    fvv = eris.fock[nocc:,nocc:]
    eris_ovov = np.asarray(eris.ovov)
    Lac  =-2*alpha*einsum('kcld,klad->ac', eris_ovov, t2)
    Lac +=   alpha*einsum('kdlc,klad->ac', eris_ovov, t2)
    Lac -= 2*einsum('kcld,ka,ld->ac', eris_ovov, t1, t1)
    Lac +=   einsum('kdlc,ka,ld->ac', eris_ovov, t1, t1)
    Lac += fvv

    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    Lac -= np.einsum('kc,ka->ac',fov, t1)
    eris_ovvv = np.asarray(eris.get_ovvv())
    Lac += 2*np.einsum('kdac,kd->ac', eris_ovvv, t1)
    Lac -=   np.einsum('kcad,kd->ac', eris_ovvv, t1)
    return Lac

def Woooo_p(t1, t2, eris, gamma):
    eris_ovov = np.asarray(eris.ovov)
    Wklij  = einsum('kcld,ijcd->klij', eris_ovov, t2) * gamma
    Wklij += einsum('kcld,ic,jd->klij', eris_ovov, t1, t1)
    eris_ovoo = np.asarray(eris.ovoo)
    Wklij += einsum('ldki,jd->klij', eris_ovoo, t1)
    Wklij += einsum('kclj,ic->klij', eris_ovoo, t1)
    Wklij += np.asarray(eris.oooo).transpose(0,2,1,3)
    return Wklij

def Wvvvv_p(t1, t2, eris, gamma):
    eris_ovov = np.asarray(eris.ovov)
    Wabcd  = einsum('kcld,klab->abcd', eris_ovov, t2) * gamma
    Wabcd += einsum('kcld,ka,lb->abcd', eris_ovov, t1, t1)
    Wabcd += np.asarray(imd._get_vvvv(eris)).transpose(0,2,1,3)
    eris_ovvv = np.asarray(eris.get_ovvv())
    Wabcd -= einsum('ldac,lb->abcd', eris_ovvv, t1)
    Wabcd -= einsum('kcbd,ka->abcd', eris_ovvv, t1)
    return Wabcd

def Wovvo_p(t1, t2, eris, delta, is_dcsd):
    Wkaci = imd.W2ovvo(t1, t2, eris)
    eris_ovov = np.asarray(eris.ovov)
    Wkaci += 2*einsum('kcld,ilad->kaci', eris_ovov, t2) * delta
    Wkaci +=  -einsum('kcld,liad->kaci', eris_ovov, t2) * delta
    if is_dcsd:
        assert delta == 1
        pass
    else:
        Wkaci +=  -einsum('kdlc,ilad->kaci', eris_ovov, t2) * delta
    Wkaci += np.asarray(eris.ovvo).transpose(0,2,1,3)
    return Wkaci

def Wovov_p(t1, t2, eris, delta, is_dcsd):
    Wkbid = imd.W2ovov(t1, t2, eris)
    eris_ovov = np.asarray(eris.ovov)
    if is_dcsd:
        pass
    else:
        Wkbid -= einsum('kcld,ilcb->kbid', eris_ovov, t2) * delta
    Wkbid += np.asarray(eris.oovv).transpose(0,2,1,3)
    return Wkbid

def _make_tau_p(t2, t1, r1, fac=1, t2_fac=1, out=None):
    tau = np.einsum('ia,jb->ijab', t1, r1)
    tau = tau + tau.transpose(1,0,3,2)
    tau *= fac * .5
    tau += t2 * t2_fac
    return tau

########################################
# EOM-IP-DCSD
########################################

def ipccsd_matvec(eom, vector, imds=None, diag=None):
    # Ref: Nooijen and Snijders, J. Chem. Phys. 102, 1681 (1995) Eqs.(8)-(9)
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    r1, r2 = eom.vector_to_amplitudes(vector, nmo, nocc)

    # 1h-1h block
    Hr1 = -np.einsum('ki,k->i', imds.Loo, r1)
    #1h-2h1p block
    Hr1 += 2*np.einsum('ld,ild->i', imds.Fov, r2)
    Hr1 +=  -np.einsum('kd,kid->i', imds.Fov, r2)
    Hr1 += -2*np.einsum('klid,kld->i', imds.Wooov, r2)
    Hr1 +=    np.einsum('lkid,kld->i', imds.Wooov, r2)

    # 2h1p-1h block
    Hr2 = -np.einsum('kbij,k->ijb', imds.Wovoo, r1)
    # 2h1p-2h1p block
    if eom.partition == 'mp':
        fock = imds.eris.fock
        foo = fock[:nocc,:nocc]
        fvv = fock[nocc:,nocc:]
        Hr2 += einsum('bd,ijd->ijb', fvv, r2)
        Hr2 += -einsum('ki,kjb->ijb', foo, r2)
        Hr2 += -einsum('lj,ilb->ijb', foo, r2)
    elif eom.partition == 'full':
        diag_matrix2 = eom.vector_to_amplitudes(diag, nmo, nocc)[1]
        Hr2 += diag_matrix2 * r2
    else:
        Hr2 += einsum('bd,ijd->ijb', imds.Lvv_p, r2)
        Hr2 += -einsum('ki,kjb->ijb', imds.Loo_p, r2)
        Hr2 += -einsum('lj,ilb->ijb', imds.Loo_p, r2)
        Hr2 +=  einsum('klij,klb->ijb', imds.Woooo_p, r2)
        Hr2 += 2*einsum('lbdj,ild->ijb', imds.Wovvo_p, r2)
        Hr2 +=  -einsum('kbdj,kid->ijb', imds.Wovvo_p, r2)
        Hr2 +=  -einsum('lbjd,ild->ijb', imds.Wovov_p, r2) #typo in Ref
        Hr2 +=  -einsum('kbid,kjd->ijb', imds.Wovov_p, r2)
        tmp = 2*np.einsum('lkdc,kld->c', imds.Woovv, r2)
        tmp += -np.einsum('kldc,kld->c', imds.Woovv, r2)
        Hr2 += -np.einsum('c,ijcb->ijb', tmp, imds.t2) * imds.p_alpha

    vector = eom.amplitudes_to_vector(Hr1, Hr2)
    return vector

def lipccsd_matvec(eom, vector, imds=None, diag=None):
    raise NotImplementedError

def ipccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    dtype = np.result_type(t1, t2)
    nocc, nvir = t1.shape
    fock = imds.eris.fock
    foo = fock[:nocc,:nocc]
    fvv = fock[nocc:,nocc:]

    Hr1 = -np.diag(imds.Loo)
    Hr2 = np.zeros((nocc,nocc,nvir), dtype)
    for i in range(nocc):
        for j in range(nocc):
            for b in range(nvir):
                if eom.partition == 'mp':
                    Hr2[i,j,b] += fvv[b,b]
                    Hr2[i,j,b] += -foo[i,i]
                    Hr2[i,j,b] += -foo[j,j]
                else:
                    Hr2[i,j,b] += imds.Lvv_p[b,b]
                    Hr2[i,j,b] += -imds.Loo_p[i,i]
                    Hr2[i,j,b] += -imds.Loo_p[j,j]
                    Hr2[i,j,b] +=  imds.Woooo_p[i,j,i,j]
                    Hr2[i,j,b] +=2*imds.Wovvo_p[j,b,b,j]
                    Hr2[i,j,b] += -imds.Wovvo_p[i,b,b,i]*(i==j)
                    Hr2[i,j,b] += -imds.Wovov_p[j,b,j,b]
                    Hr2[i,j,b] += -imds.Wovov_p[i,b,i,b]
                    Hr2[i,j,b] += -2*np.dot(imds.Woovv[j,i,b,:], t2[i,j,:,b]) * imds.p_alpha
                    Hr2[i,j,b] += np.dot(imds.Woovv[i,j,b,:], t2[i,j,:,b]) * imds.p_alpha

    vector = eom.amplitudes_to_vector(Hr1, Hr2)
    return vector

def ipccsd_star_contract(eom, ipccsd_evals, ipccsd_evecs, lipccsd_evecs, imds=None):
    raise NotImplementedError

class EOMIP(eom_rccsd.EOMIP):
    matvec = ipccsd_matvec
    l_matvec = lipccsd_matvec
    get_diag = ipccsd_diag
    ccsd_star_contract = ipccsd_star_contract

    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris=eris)
        imds.make_ip(self.partition)
        return imds

########################################
# EOM-EA-DCSD
########################################

def eaccsd_matvec(eom, vector, imds=None, diag=None):
    # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1995) Eqs.(30)-(31)
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    nvir = nmo - nocc
    r1, r2 = eom.vector_to_amplitudes(vector, nmo, nocc)

    # Eq. (37)
    # 1p-1p block
    Hr1 =  np.einsum('ac,c->a', imds.Lvv, r1)
    # 1p-2p1h block
    Hr1 += np.einsum('ld,lad->a', 2.*imds.Fov, r2)
    Hr1 += np.einsum('ld,lda->a',   -imds.Fov, r2)
    Hr1 += np.einsum('alcd,lcd->a', 2.*imds.Wvovv-imds.Wvovv.transpose(0,1,3,2), r2)
    # Eq. (38)
    # 2p1h-1p block
    Hr2 = np.einsum('abcj,c->jab', imds.Wvvvo, r1)
    # 2p1h-2p1h block
    if eom.partition == 'mp':
        fock = imds.eris.fock
        foo = fock[:nocc,:nocc]
        fvv = fock[nocc:,nocc:]
        Hr2 +=  einsum('ac,jcb->jab', fvv, r2)
        Hr2 +=  einsum('bd,jad->jab', fvv, r2)
        Hr2 += -einsum('lj,lab->jab', foo, r2)
    elif eom.partition == 'full':
        diag_matrix2 = eom.vector_to_amplitudes(diag, nmo, nocc)[1]
        Hr2 += diag_matrix2 * r2
    else:
        Hr2 +=  einsum('ac,jcb->jab', imds.Lvv_p, r2)
        Hr2 +=  einsum('bd,jad->jab', imds.Lvv_p, r2)
        Hr2 += -einsum('lj,lab->jab', imds.Loo_p, r2)
        Hr2 += einsum('lbdj,lad->jab', 2.*imds.Wovvo_p-imds.Wovov_p.transpose(0,1,3,2), r2)
        Hr2 += -einsum('lajc,lcb->jab', imds.Wovov_p, r2)
        Hr2 += -einsum('lbcj,lca->jab', imds.Wovvo_p, r2)
        for a in range(nvir):
            Hr2[:,a,:] += einsum('bcd,jcd->jb', imds.Wvvvv_p[a], r2)
        tmp = np.einsum('klcd,lcd->k', 2.*imds.Woovv-imds.Woovv.transpose(0,1,3,2), r2)
        Hr2 += -np.einsum('k,kjab->jab', tmp, imds.t2) * imds.p_beta

    vector = eom.amplitudes_to_vector(Hr1,Hr2)
    return vector

def leaccsd_matvec(eom, vector, imds=None, diag=None):
    raise NotImplementedError

def eaccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    dtype = np.result_type(t1, t2)
    nocc, nvir = t1.shape

    fock = imds.eris.fock
    foo = fock[:nocc,:nocc]
    fvv = fock[nocc:,nocc:]

    Hr1 = np.diag(imds.Lvv)
    Hr2 = np.zeros((nocc,nvir,nvir), dtype)
    for a in range(nvir):
        if eom.partition != 'mp':
            _Wvvvva = np.array(imds.Wvvvv_p[a])
        for b in range(nvir):
            for j in range(nocc):
                if eom.partition == 'mp':
                    Hr2[j,a,b] += fvv[a,a]
                    Hr2[j,a,b] += fvv[b,b]
                    Hr2[j,a,b] += -foo[j,j]
                else:
                    Hr2[j,a,b] += imds.Lvv_p[a,a]
                    Hr2[j,a,b] += imds.Lvv_p[b,b]
                    Hr2[j,a,b] += -imds.Loo_p[j,j]
                    Hr2[j,a,b] += 2*imds.Wovvo_p[j,b,b,j]
                    Hr2[j,a,b] += -imds.Wovov_p[j,b,j,b]
                    Hr2[j,a,b] += -imds.Wovov_p[j,a,j,a]
                    Hr2[j,a,b] += -imds.Wovvo_p[j,b,b,j]*(a==b)
                    Hr2[j,a,b] += _Wvvvva[b,a,b]
                    Hr2[j,a,b] += -2*np.dot(imds.Woovv[:,j,a,b], t2[:,j,a,b]) * imds.p_beta
                    Hr2[j,a,b] += np.dot(imds.Woovv[:,j,b,a], t2[:,j,a,b]) * imds.p_beta

    vector = eom.amplitudes_to_vector(Hr1,Hr2)
    return vector

def eaccsd_star_contract(eom, eaccsd_evals, eaccsd_evecs, leaccsd_evecs, imds=None):
    raise NotImplementedError

class EOMEA(eom_rccsd.EOMEA):
    matvec = eaccsd_matvec
    l_matvec = leaccsd_matvec
    get_diag = eaccsd_diag
    ccsd_star_contract = eaccsd_star_contract

    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris=eris)
        imds.make_ea(self.partition)
        return imds

########################################
# EOM-EE-DCSD
########################################

def eeccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    eris = imds.eris
    t1, t2 = imds.t1, imds.t2
    dtype = np.result_type(t1, t2)
    nocc, nvir = t1.shape

    Fo = imds.Foo.diagonal()
    Fv = imds.Fvv.diagonal()
    Wovab = np.einsum('iaai->ia', imds.woVVo)
    Wovaa = Wovab + np.einsum('iaai->ia', imds.woVvO)

    eia = lib.direct_sum('-i+a->ia', Fo, Fv)
    Hr1aa = eia + Wovaa
    Hr1ab = eia + Wovab

    Fo = imds.Foo_p.diagonal()
    Fv = imds.Fvv_p.diagonal()
    Wovab = np.einsum('iaai->ia', imds.woVVo_p)
    Wovaa = Wovab + np.einsum('iaai->ia', imds.woVvO_p)
    
    eris_ovov = np.asarray(eris.ovov)
    tau_p = _make_tau_p(t2, t1, t1, t2_fac=imds.p_gamma)
    Wvvab = np.einsum('mnab,manb->ab', tau_p, eris_ovov)
    Wvvaa = .5*Wvvab - .5*np.einsum('mnba,manb->ab', tau_p, eris_ovov)

    ijb = np.einsum('iejb,ijeb->ijb', eris_ovov, t2)
    Hr2ab = lib.direct_sum('iJB+a->iJaB',-ijb, Fv)
    jab = np.einsum('kajb,kjab->jab', eris_ovov, t2)
    Hr2ab+= lib.direct_sum('-i-jab->ijab', Fo, jab)

    jib = np.einsum('iejb,ijbe->jib', eris_ovov, t2)
    jib = jib + jib.transpose(1,0,2)
    jib-= ijb + ijb.transpose(1,0,2)
    jba = np.einsum('kajb,jkab->jba', eris_ovov, t2)
    jba = jba + jba.transpose(0,2,1)
    jba-= jab + jab.transpose(0,2,1)
    Hr2aa = lib.direct_sum('jib+a->jiba', jib, Fv)
    Hr2aa+= lib.direct_sum('-i+jba->ijba', Fo, jba)
    eris_ovov = None

    Hr2baaa = lib.direct_sum('ijb+a->ijba',-ijb, Fv)
    Hr2baaa += Wovaa.reshape(1,nocc,1,nvir)
    Hr2baaa += Wovab.reshape(nocc,1,1,nvir)
    Hr2baaa = Hr2baaa + Hr2baaa.transpose(0,1,3,2)
    Hr2baaa+= lib.direct_sum('-i+jab->ijab', Fo, jba)
    Hr2baaa-= Fo.reshape(1,-1,1,1)
    Hr2aaba = lib.direct_sum('-i-jab->ijab', Fo, jab)
    Hr2aaba += Wovaa.reshape(1,nocc,1,nvir)
    Hr2aaba += Wovab.reshape(1,nocc,nvir,1)
    Hr2aaba = Hr2aaba + Hr2aaba.transpose(1,0,2,3)
    Hr2aaba+= lib.direct_sum('ijb+a->ijab', jib, Fv)
    Hr2aaba+= Fv.reshape(1,1,1,-1)
    Hr2ab += Wovaa.reshape(1,nocc,1,nvir)
    Hr2ab += Wovab.reshape(nocc,1,1,nvir)
    Hr2ab = Hr2ab + Hr2ab.transpose(1,0,3,2)
    Hr2aa += Wovaa.reshape(1,nocc,1,nvir) * 2
    Hr2aa = Hr2aa + Hr2aa.transpose(0,1,3,2)
    Hr2aa = Hr2aa + Hr2aa.transpose(1,0,2,3)
    Hr2aa *= .5

    Wooab = np.einsum('ijij->ij', imds.woOoO_p)
    Wooaa = Wooab - np.einsum('ijji->ij', imds.woOoO_p)
    Hr2aa += Wooaa.reshape(nocc,nocc,1,1)
    Hr2ab += Wooab.reshape(nocc,nocc,1,1)
    Hr2baaa += Wooab.reshape(nocc,nocc,1,1)
    Hr2aaba += Wooaa.reshape(nocc,nocc,1,1)

    #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocc*nvir,nvir**2)).reshape(nocc,nvir,nvir,nvir)
    #:tmp = np.einsum('mb,mbaa->ab', t1, eris_ovvv)
    #:Wvvaa += np.einsum('mb,maab->ab', t1, eris_ovvv)
    mem_now = lib.current_memory()[0]
    max_memory = max(0, eom.max_memory - mem_now)
    blksize = min(nocc, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvir**3*3))))
    tmp = np.zeros((nvir,nvir), dtype=dtype)
    for p0,p1 in lib.prange(0, nocc, blksize):
        ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
        tmp += np.einsum('mb,mbaa->ab', t1[p0:p1], ovvv)
        Wvvaa += np.einsum('mb,maab->ab', t1[p0:p1], ovvv)
        ovvv = None
    Wvvaa -= tmp
    Wvvab -= tmp
    Wvvab -= tmp.T
    Wvvaa = Wvvaa + Wvvaa.T
    if eris.vvvv is None: # AO-direct CCSD, vvvv is not generated.
        pass
    elif eris.vvvv.ndim == 4:
        eris_vvvv = ao2mo.restore(1,np.asarray(eris.vvvv), t1.shape[1])
        tmp = np.einsum('aabb->ab', eris_vvvv)
        Wvvaa += tmp
        Wvvaa -= np.einsum('abba->ab', eris_vvvv)
        Wvvab += tmp
    else:
        for i in range(nvir):
            i0 = i*(i+1)//2
            vvv = lib.unpack_tril(np.asarray(eris.vvvv[i0:i0+i+1]))
            tmp = np.einsum('bb->b', vvv[i])
            Wvvaa[i] += tmp
            Wvvab[i] += tmp
            tmp = np.einsum('bb->b', vvv[:,:i+1,i])
            Wvvaa[i,:i+1] -= tmp
            Wvvaa[:i  ,i] -= tmp[:i]
            vvv = None

    Hr2aa += Wvvaa.reshape(1,1,nvir,nvir)
    Hr2ab += Wvvab.reshape(1,1,nvir,nvir)
    Hr2baaa += Wvvaa.reshape(1,1,nvir,nvir)
    Hr2aaba += Wvvab.reshape(1,1,nvir,nvir)

    vec_eeS = eom_rccsd.amplitudes_to_vector_singlet(Hr1aa, Hr2ab)
    vec_eeT = eom_rccsd.amplitudes_to_vector_triplet(Hr1aa, (Hr2aa,Hr2ab))
    vec_sf = eom_rccsd.amplitudes_to_vector_eomsf(Hr1ab, (Hr2baaa,Hr2aaba))
    return vec_eeS, vec_eeT, vec_sf

def eeccsd_matvec_singlet(eom, vector, imds=None):
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    nvir = nmo - nocc

    r1, r2 = eom.vector_to_amplitudes(vector, nmo, nocc)
    t1, t2, eris = imds.t1, imds.t2, imds.eris
    nocc, nvir = t1.shape

    Hr1  = einsum('ae,ie->ia', imds.Fvv, r1)
    Hr1 -= einsum('mi,ma->ia', imds.Foo, r1)
    Hr1 += np.einsum('me,imae->ia',imds.Fov, r2) * 2
    Hr1 -= np.einsum('me,imea->ia',imds.Fov, r2)

    #:eris_vvvv = ao2mo.restore(1,np.asarray(eris.vvvv), t1.shape[1])
    #:Hr2 += einsum('ijef,aebf->ijab', tau2, eris_vvvv) * .5
    tau2 = _make_tau_p(r2, r1, t1, fac=2)
    Hr2 = eom._cc._add_vvvv(None, tau2, eris, with_ovvv=False, t2sym='jiba')

    woOoO_p = np.asarray(imds.woOoO_p)
    Hr2 += einsum('mnij,mnab->ijab', woOoO_p, r2)
    Hr2 *= .5
    woOoO_p = None

    Hr2 += einsum('be,ijae->ijab', imds.Fvv_p , r2)
    Hr2 -= einsum('mj,imab->ijab', imds.Foo_p , r2)

    mem_now = lib.current_memory()[0]
    max_memory = max(0, eom.max_memory - mem_now - Hr2.size*8e-6)
    blksize = min(nocc, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvir**3*3))))
    for p0,p1 in lib.prange(0, nocc, blksize):
        ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
        theta = r2[p0:p1] * 2 - r2[p0:p1].transpose(0,1,3,2)
        Hr1 += einsum('mfae,mife->ia', ovvv, theta)
        theta = None
        tmp = einsum('meaf,ijef->maij', ovvv, tau2)
        Hr2 -= einsum('ma,mbij->ijab', t1[p0:p1], tmp)
        tmp  = einsum('meaf,me->af', ovvv, r1[p0:p1]) * 2
        tmp -= einsum('mfae,me->af', ovvv, r1[p0:p1])
        Hr2 += einsum('af,ijfb->ijab', tmp, t2)
        ovvv = tmp = None
    tau2 = None
    Hr2 -= einsum('mbij,ma->ijab', imds.woVoO, r1)

    blksize = min(nvir, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nocc*nvir**2*2))))
    for p0, p1 in lib.prange(0, nvir, nocc):
        Hr2 += einsum('ejab,ie->ijab', np.asarray(imds.wvOvV[p0:p1]), r1[:,p0:p1])

    woVVo_p = np.asarray(imds.woVVo_p)
    tmp = einsum('mbej,imea->jiab', woVVo_p, r2)
    Hr2 += tmp
    tmp *= .5
    Hr2 += tmp.transpose(0,1,3,2)
    tmp = None
    woVvO_p = woVVo_p * .5
    woVVo_p = None
    woVvO_p += np.asarray(imds.woVvO_p)
    theta = r2*2 - r2.transpose(0,1,3,2)
    Hr2 += einsum('mbej,imae->ijab', woVvO_p, theta)
    woVvO_p = None

    woVvO = np.asarray(imds.woVVo) * .5
    woVvO += np.asarray(imds.woVvO)
    Hr1 += np.einsum('maei,me->ia', woVvO, r1) * 2
    woVvO = None

    woOoV = np.asarray(imds.woOoV)
    Hr1-= einsum('mnie,mnae->ia', woOoV, theta)
    tmp = einsum('nmie,me->ni', woOoV, r1) * 2
    tmp-= einsum('mnie,me->ni', woOoV, r1)
    Hr2 -= einsum('ni,njab->ijab', tmp, t2)
    tmp = woOoV = None

    eris_ovov = np.asarray(eris.ovov)
    tmp  = np.einsum('mfne,mf->en', eris_ovov, r1) * 2
    tmp -= np.einsum('menf,mf->en', eris_ovov, r1)
    tmp  = np.einsum('en,nb->eb', tmp, t1)
    tmp += einsum('menf,mnbf->eb', eris_ovov, theta)*imds.p_alpha
    Hr2 -= einsum('eb,ijea->jiab', tmp, t2)
    tmp = None

    tmp = einsum('nemf,imef->ni', eris_ovov, theta)
    Hr1 -= einsum('na,ni->ia', t1, tmp)
    Hr2 -= einsum('mj,miab->ijba', tmp, t2)*imds.p_beta
    tmp = theta = None

    tau2 = _make_tau_p(r2, r1, t1, fac=2, t2_fac=0)
    tmp = einsum('menf,ijef->mnij', eris_ovov, tau2)
    tau2 = None
    tau = _make_tau_p(t2, t1, t1)
    tau *= .5
    Hr2 += einsum('mnij,mnab->ijab', tmp, tau)
    tau = tmp = None

    tmp = einsum('menf,ijef->mnij', eris_ovov, r2)
    tau = _make_tau_p(t2, t1, t1, t2_fac=imds.p_gamma)
    tau *= .5
    Hr2 += einsum('mnij,mnab->ijab', tmp, tau)
    tau = tmp = eris_ovov = None

    Hr2 = Hr2 + Hr2.transpose(1,0,3,2)
    vector = eom.amplitudes_to_vector(Hr1, Hr2)
    return vector

def eeccsd_matvec_triplet(eom, vector, imds=None):
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    nvir = nmo - nocc

    r1, r2 = eom.vector_to_amplitudes(vector, nmo, nocc)
    r2aa, r2ab = r2
    t1, t2, eris = imds.t1, imds.t2, imds.eris
    nocc, nvir = t1.shape

    Hr1  = einsum('ae,ie->ia', imds.Fvv, r1)
    Hr1 -= einsum('mi,ma->ia', imds.Foo, r1)
    Hr1 += np.einsum('me,imae->ia',imds.Fov, r2aa)
    Hr1 += np.einsum('ME,iMaE->ia',imds.Fov, r2ab)

    tau2ab = np.einsum('ia,jb->ijab', r1, t1)
    tau2ab-= np.einsum('ia,jb->ijab', t1, r1)
    tau2ab+= r2ab
    tau2aa = np.einsum('ia,jb->ijab', r1, t1)
    tau2aa-= np.einsum('ia,jb->jiab', r1, t1)
    tau2aa = tau2aa - tau2aa.transpose(0,1,3,2)
    tau2aa+= r2aa

    #:eris_vvvv = ao2mo.restore(1,np.asarray(eris.vvvv), t1.shape[1])
    #:Hr2aa += einsum('ijef,aebf->ijab', tau2aa, eris_vvvv) * .25
    #:Hr2ab += einsum('ijef,aebf->ijab', tau2ab, eris_vvvv) * .5
    Hr2aa = eom._cc._add_vvvv(None, tau2aa, eris, with_ovvv=False, t2sym='jiba')
    Hr2ab = eom._cc._add_vvvv(None, tau2ab, eris, with_ovvv=False, t2sym='-jiba')

    woOoO_p = np.asarray(imds.woOoO_p)
    Hr2aa += einsum('mnij,mnab->ijab', woOoO_p, r2aa)
    Hr2ab += einsum('mNiJ,mNaB->iJaB', woOoO_p, r2ab)
    Hr2aa *= .25
    Hr2ab *= .5
    woOoO_p = None

    Hr2aa += einsum('be,ijae->ijab', imds.Fvv_p*.5, r2aa)
    Hr2aa -= einsum('mj,imab->ijab', imds.Foo_p*.5, r2aa)
    Hr2ab += einsum('BE,iJaE->iJaB', imds.Fvv_p, r2ab)
    Hr2ab -= einsum('MJ,iMaB->iJaB', imds.Foo_p, r2ab)

    mem_now = lib.current_memory()[0]
    max_memory = max(0, eom.max_memory - mem_now - Hr2aa.size*8e-6)
    blksize = min(nocc, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvir**3*3))))
    tmp1 = np.zeros((nvir,nvir), dtype=r1.dtype)
    for p0,p1 in lib.prange(0, nocc, blksize):
        ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
        theta = r2aa[:,p0:p1] + r2ab[:,p0:p1]
        Hr1 += einsum('mfae,imef->ia', ovvv, theta)
        theta = None
        tmpaa = einsum('meaf,ijef->maij', ovvv, tau2aa)
        tmpab = einsum('meAF,iJeF->mAiJ', ovvv, tau2ab)
        tmp1 += einsum('mfae,me->af', ovvv, r1[p0:p1])
        Hr2aa+= einsum('mb,maij->ijab', t1[p0:p1]*.5, tmpaa)
        Hr2ab-= einsum('mb,mAiJ->iJbA', t1[p0:p1], tmpab)
        ovvv = tmpaa = tmpab = None
    tau2aa = tau2ab = None

    woVVo = np.asarray(imds.woVVo)
    Hr1 += np.einsum('maei,me->ia', woVVo, r1)
    woVVo = None
    woVVo_p = np.asarray(imds.woVVo_p)
    Hr2aa += einsum('mbej,imae->ijba', woVVo_p, r2ab)
    Hr2ab += einsum('MBEJ,iMEa->iJaB', woVVo_p, r2aa)
    Hr2ab += einsum('MbeJ,iMeA->iJbA', woVVo_p, r2ab)

    wovvo_p = woVVo_p + np.asarray(imds.woVvO_p)
    theta = r2aa + r2ab
    tmp = einsum('mbej,imae->ijab', wovvo_p, theta)
    woVVo_p = wovvo_p = None

    woOoV = np.asarray(imds.woOoV)
    Hr1 -= einsum('mnie,mnae->ia', woOoV, theta)
    tmpa = einsum('mnie,me->ni', woOoV, r1)
    tmp += einsum('ni,njab->ijab', tmpa, t2)
    tmp -= einsum('af,ijfb->ijab', tmp1, t2)
    tmp -= einsum('mbij,ma->ijab', imds.woVoO, r1)

    blksize = min(nvir, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nocc*nvir**2*2))))
    for p0,p1 in lib.prange(0, nvir, blksize):
        tmp += einsum('ejab,ie->ijab', np.asarray(imds.wvOvV[p0:p1]), r1[:,p0:p1])

    Hr2aa += tmp
    Hr2ab += tmp
    tmp = woOoV = None

    eris_ovov = np.asarray(eris.ovov)
    tmpa = -einsum('menf,imfe->ni', eris_ovov, theta)
    Hr1 += einsum('na,ni->ia', t1, tmpa)
    tmp  = einsum('mj,imab->ijab', tmpa, t2) * imds.p_beta
    tmp1 = np.einsum('menf,mf->en', eris_ovov, r1)
    tmpa = einsum('en,nb->eb', tmp1, t1)
    tmpa-= einsum('menf,mnbf->eb', eris_ovov, theta) * imds.p_alpha
    tmp += einsum('eb,ijae->ijab', tmpa, t2)
    Hr2aa += tmp
    Hr2ab -= tmp
    tmp = theta = tmp1 = tmpa = None

    tau2aa = np.einsum('ia,jb->ijab', r1, t1)
    tau2aa-= np.einsum('ia,jb->jiab', r1, t1)
    tau2aa = tau2aa - tau2aa.transpose(0,1,3,2)
    tmpaa = einsum('menf,ijef->mnij', eris_ovov, tau2aa)
    tau2aa = None
    tmpaa *= .25
    tau = _make_tau_p(t2, t1, t1)
    Hr2aa += einsum('mnij,mnab->ijab', tmpaa, tau)
    tmpaa = tau = None

    tmpaa = einsum('menf,ijef->mnij', eris_ovov, r2aa)
    tmpaa *= .25
    tau = _make_tau_p(t2, t1, t1, t2_fac=imds.p_gamma)
    Hr2aa += einsum('mnij,mnab->ijab', tmpaa, tau)
    tmpaa = tau = None

    tau2ab = np.einsum('ia,jb->ijab', r1, t1)
    tau2ab-= np.einsum('ia,jb->ijab', t1, r1)
    tmpab = einsum('meNF,iJeF->mNiJ', eris_ovov, tau2ab)
    tau2ab = None
    tmpab *= .5
    tau = _make_tau_p(t2, t1, t1)
    Hr2ab += einsum('mNiJ,mNaB->iJaB', tmpab, tau)
    tmpab = tau = None

    tmpab = einsum('meNF,iJeF->mNiJ', eris_ovov, r2ab)
    tmpab *= .5
    tau = _make_tau_p(t2, t1, t1, t2_fac=imds.p_gamma)
    Hr2ab += einsum('mNiJ,mNaB->iJaB', tmpab, tau)
    tmpab = tau = None
    eris_ovov = None

    Hr2aa = Hr2aa - Hr2aa.transpose(0,1,3,2)
    Hr2aa = Hr2aa - Hr2aa.transpose(1,0,2,3)
    Hr2ab = Hr2ab - Hr2ab.transpose(1,0,3,2)
    vector = eom.amplitudes_to_vector(Hr1, (Hr2aa,Hr2ab))
    return vector

class EOMEESinglet(eom_rccsd.EOMEESinglet):
    matvec = eeccsd_matvec_singlet
    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris)
        imds.make_ee()
        return imds
    def get_diag(self, imds=None):
        return eeccsd_diag(self, imds=None)[0]
class EOMEETriplet(eom_rccsd.EOMEETriplet):
    matvec = eeccsd_matvec_triplet
    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris)
        imds.make_ee()
        return imds
    def get_diag(self, imds=None):
        return eeccsd_diag(self, imds=None)[1]
    
dcsd.DCSD.EOMEESinglet = lib.class_as_method(EOMEESinglet)
dcsd.pCCSD.EOMEESinglet = lib.class_as_method(EOMEESinglet)
dcsd.DCSD.EOMEA = lib.class_as_method(EOMEA)
dcsd.pCCSD.EOMEA = lib.class_as_method(EOMEA)
dcsd.DCSD.EOMIP = lib.class_as_method(EOMIP)
dcsd.pCCSD.EOMIP = lib.class_as_method(EOMIP)

class _IMDS:
    def __init__(self, cc, eris=None):
        self.verbose = cc.verbose
        self.stdout = cc.stdout
        self.max_memory = cc.max_memory
        self.t1 = cc.t1
        self.t2 = cc.t2
        if eris is None:
            eris = cc.ao2mo()
        self.eris = eris
        self._made_shared_2e = False
        if hasattr(cc, "p_mu"): # pCCSD
            self.dcsd = False
            # change notation
            self.p_alpha = cc.p_sigma
            self.p_delta = cc.p_sigma
            self.p_beta = (1.0 + cc.p_mu) / 2.0
            self.p_gamma = cc.p_mu
        else: # DCSD
            self.dcsd = True
            self.p_alpha = 0.5
            self.p_beta = 0.5
            self.p_gamma = 0.0
            self.p_delta = 1.0

    def _make_shared_1e(self):
        cput0 = (logger.process_clock(), logger.perf_counter())

        t1, t2, eris = self.t1, self.t2, self.eris
        self.Loo = imd.Loo(t1, t2, eris)
        self.Lvv = imd.Lvv(t1, t2, eris)
        self.Fov = imd.cc_Fov(t1, t2, eris)
        self.Lvv_p = Lvv_p(t1, t2, eris, self.p_alpha)
        self.Loo_p = Loo_p(t1, t2, eris, self.p_beta)

        logger.timer_debug1(self, 'EOM-DCSD shared one-electron '
                            'intermediates', *cput0)
        return self

    def _make_shared_2e(self):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris
        # 2 virtuals
        self.Wovov_p = Wovov_p(t1, t2, eris, self.p_delta, self.dcsd)
        self.Wovvo_p = Wovvo_p(t1, t2, eris, self.p_delta, self.dcsd)
        self.Woovv = np.asarray(eris.ovov).transpose(0,2,1,3)

        self._made_shared_2e = True
        log.timer_debug1('EOM-DCSD shared two-electron intermediates', *cput0)
        return self

    def make_ip(self, ip_partition=None):
        self._make_shared_1e()
        if not self._made_shared_2e and ip_partition != 'mp':
            self._make_shared_2e()

        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris

        # 0 or 1 virtuals
        if ip_partition != 'mp':
            self.Woooo_p = Woooo_p(t1, t2, eris, self.p_gamma)
        self.Wooov = imd.Wooov(t1, t2, eris)
        self.Wovoo = imd.Wovoo(t1, t2, eris)
        log.timer_debug1('EOM-DCSD IP intermediates', *cput0)
        return self

    def make_t3p2_ip(self, cc, ip_partition=None):
        raise NotImplementedError
        return self

    def make_ea(self, ea_partition=None):
        self._make_shared_1e()
        if not self._made_shared_2e and ea_partition != 'mp':
            self._make_shared_2e()

        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris

        # 3 or 4 virtuals
        self.Wvovv = imd.Wvovv(t1, t2, eris)
        if ea_partition == 'mp':
            self.Wvvvo = imd.Wvvvo(t1, t2, eris)
        else:
            self.Wvvvv_p = Wvvvv_p(t1, t2, eris, self.p_gamma)
            self.Wvvvo = imd.Wvvvo(t1, t2, eris)
        log.timer_debug1('EOM-DCSD EA intermediates', *cput0)
        return self

    def make_t3p2_ea(self, cc, ea_partition=None):
        raise NotImplementedError
        return self

    def make_ee(self):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)

        t1, t2, eris = self.t1, self.t2, self.eris
        dtype = np.result_type(t1, t2)
        if np.iscomplexobj(t2):
            raise NotImplementedError('Complex integrals are not supported in EOM-EE-DCSD')

        nocc, nvir = t1.shape

        fswap = lib.H5TmpFile()
        self.saved = lib.H5TmpFile()
        self.wvOvV = self.saved.create_dataset('wvOvV', (nvir,nocc,nvir,nvir), dtype.char)
        self.woVvO = self.saved.create_dataset('woVvO', (nocc,nvir,nvir,nocc), dtype.char)
        self.woVVo = self.saved.create_dataset('woVVo', (nocc,nvir,nvir,nocc), dtype.char)
        self.woOoV = self.saved.create_dataset('woOoV', (nocc,nocc,nocc,nvir), dtype.char)
        self.woVvO_p = self.saved.create_dataset('woVvO_p', (nocc,nvir,nvir,nocc), dtype.char)
        self.woVVo_p = self.saved.create_dataset('woVVo_p', (nocc,nvir,nvir,nocc), dtype.char)

        foo = eris.fock[:nocc,:nocc]
        fov = eris.fock[:nocc,nocc:]
        fvv = eris.fock[nocc:,nocc:]

        self.Fov = np.zeros((nocc,nvir), dtype=dtype)
        self.Foo = np.zeros((nocc,nocc), dtype=dtype)
        self.Fvv = np.zeros((nvir,nvir), dtype=dtype)
        self.Foo_p = np.zeros((nocc,nocc), dtype=dtype)
        self.Fvv_p = np.zeros((nvir,nvir), dtype=dtype)

        #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocc*nvir,nvir**2)).reshape(nocc,nvir,nvir,nvir)
        #:self.Fvv  = np.einsum('mf,mfae->ae', t1, eris_ovvv) * 2
        #:self.Fvv -= np.einsum('mf,meaf->ae', t1, eris_ovvv)
        #:self.woVvO = einsum('jf,mebf->mbej', t1, eris_ovvv)
        #:self.woVVo = einsum('jf,mfbe->mbej',-t1, eris_ovvv)
        #:tau = _make_tau(t2, t1, t1)
        #:self.woVoO  = 0.5 * einsum('mebf,ijef->mbij', eris_ovvv, tau)
        #:self.woVoO += 0.5 * einsum('mfbe,ijfe->mbij', eris_ovvv, tau)
        eris_ovoo = np.asarray(eris.ovoo)
        woVoO = np.empty((nocc,nvir,nocc,nocc), dtype=dtype)
        tau_ccsd = _make_tau_p(t2, t1, t1)
        theta = t2*2 - t2.transpose(0,1,3,2)

        mem_now = lib.current_memory()[0]
        max_memory = max(0, self.max_memory - mem_now)
        blksize = min(nocc, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nvir**3*3))))
        for seg, (p0,p1) in enumerate(lib.prange(0, nocc, blksize)):
            ovvv = eris.get_ovvv(slice(p0,p1))  # ovvv = eris.ovvv[p0:p1]
            # transform integrals (ia|bc) -> (ac|ib)
            fswap['ebmf/%d'%seg] = np.einsum('mebf->ebmf', ovvv)

            tmp = np.einsum('mf,mfae->ae', t1[p0:p1], ovvv) * 2 - \
                  np.einsum('mf,meaf->ae', t1[p0:p1], ovvv)
            self.Fvv += tmp
            self.Fvv_p += tmp
            tmp = None
            woVoO[p0:p1] = einsum('mebf,ijef->mbij', ovvv, tau_ccsd)
            woVvO = einsum('jf,mebf->mbej', t1, ovvv)
            woVVo = einsum('jf,mfbe->mbej',-t1, ovvv)
            ovvv = None

            eris_ovov = np.asarray(eris.ovov[p0:p1])
            woOoV = einsum('if,mfne->mnie', t1, eris_ovov)
            woOoV+= eris_ovoo[:,:,p0:p1].transpose(2,0,3,1)
            self.woOoV[p0:p1] = woOoV
            woOoV = None

            ovoo = einsum('menf,jf->menj', eris_ovov, t1)
            woVvO -= einsum('nb,menj->mbej', t1, ovoo)
            ovoo = einsum('mfne,jf->menj', eris_ovov, t1)
            woVVo += einsum('nb,menj->mbej', t1, ovoo)
            ovoo = None

            ovov = eris_ovov * 2 - eris_ovov.transpose(0,3,2,1)
            self.Fov[p0:p1] = np.einsum('nf,menf->me', t1, ovov)
            tilab = np.einsum('ia,jb->ijab', t1[p0:p1], t1) * .5
            tilab += t2[p0:p1] * self.p_beta
            self.Foo_p += einsum('mief,menf->ni', tilab, ovov)
            tilab += t2[p0:p1] * (self.p_alpha - self.p_beta)
            self.Fvv_p -= einsum('mnaf,menf->ae', tilab, ovov)

            tilab += t2[p0:p1] * (1.0 - self.p_alpha)
            self.Foo += einsum('mief,menf->ni', tilab, ovov)
            self.Fvv -= einsum('mnaf,menf->ae', tilab, ovov)
            ovov = tilab = None

            woVvO -= einsum('nb,menj->mbej', t1, eris_ovoo[p0:p1,:,:])
            woVVo += einsum('nb,nemj->mbej', t1, eris_ovoo[:,:,p0:p1])

            woVvO += np.asarray(eris.ovvo[p0:p1]).transpose(0,2,1,3)
            woVVo -= np.asarray(eris.oovv[p0:p1]).transpose(0,2,3,1)

            woVvO_p = woVvO.copy()
            woVVo_p = woVVo.copy()

            ovov = eris_ovov * 2 - eris_ovov.transpose(0,3,2,1)
            woVvO += einsum('njfb,menf->mbej', theta, ovov) * .5
            if self.dcsd:
                woVvO_p += einsum('njfb,menf->mbej', theta, eris_ovov)
            else:
                woVvO_p += einsum('njfb,menf->mbej', theta, ovov) * .5 * self.p_delta
            ovov = None            

            tmp = einsum('njbf,mfne->mbej', t2, eris_ovov)
            woVvO -= tmp * .5
            woVVo += tmp
            if self.dcsd:
                pass ######
            else:
                woVvO_p -= tmp * .5 * self.p_delta
                woVVo_p += tmp * self.p_delta
            tmp = eris_ovov = None

            self.woVvO[p0:p1] = woVvO
            self.woVVo[p0:p1] = woVVo
            self.woVvO_p[p0:p1] = woVvO_p
            self.woVVo_p[p0:p1] = woVVo_p

        self.Fov += fov
        tmp = 0.5*np.einsum('me,ie->mi', self.Fov+fov, t1)
        self.Foo += foo + tmp
        self.Foo_p += foo + tmp
        tmp = 0.5*np.einsum('me,ma->ae', self.Fov+fov, t1)
        self.Fvv += fvv - tmp
        self.Fvv_p += fvv - tmp

        # 0 or 1 virtuals
        woOoO = einsum('je,nemi->mnij', t1, eris_ovoo)
        woOoO = woOoO + woOoO.transpose(1,0,3,2)
        woOoO += np.asarray(eris.oooo).transpose(0,2,1,3)

        tmp = einsum('meni,jneb->mbji', eris_ovoo, t2)
        woVoO -= tmp.transpose(0,1,3,2) * .5
        woVoO -= tmp
        tmp = None
        ovoo = eris_ovoo*2 - eris_ovoo.transpose(2,1,0,3)
        woVoO += einsum('nemi,njeb->mbij', ovoo, theta) * .5
        tmp = np.einsum('ne,nemi->mi', t1, ovoo)
        self.Foo += tmp
        self.Foo_p += tmp
        ovoo = tmp = None

        eris_ovov = np.asarray(eris.ovov)
        tau_p = _make_tau_p(t2, t1, t1, t2_fac=self.p_gamma)
        woOoO_p = woOoO.copy()
        woOoO += einsum('ijef,menf->mnij', tau_ccsd, eris_ovov)
        woOoO_p += einsum('ijef,menf->mnij', tau_p, eris_ovov)
        self.woOoO = self.saved['woOoO'] = woOoO
        self.woOoO_p = self.saved['woOoO_p'] = woOoO_p
        woVoO -= einsum('nb,mnij->mbij', t1, woOoO)
        woOoO = None

        tmpoovv = einsum('njbf,nemf->ejmb', t2, eris_ovov)
        ovov = eris_ovov*2 - eris_ovov.transpose(0,3,2,1)
        eris_ovov = None

        tmpovvo = einsum('nifb,menf->eimb', theta, ovov)
        ovov = None

        tmpovvo *= -.5
        tmpovvo += tmpoovv * .5
        woVoO -= einsum('ie,ejmb->mbij', t1, tmpovvo)
        woVoO -= einsum('ie,ejmb->mbji', t1, tmpoovv)
        woVoO += eris_ovoo.transpose(3,1,2,0)

        # 3 or 4 virtuals
        eris_ovvo = np.asarray(eris.ovvo)
        tmpovvo -= eris_ovvo.transpose(1,3,0,2)
        fswap['ovvo'] = tmpovvo
        tmpovvo = None

        eris_oovv = np.asarray(eris.oovv)
        tmpoovv -= eris_oovv.transpose(3,1,0,2)
        fswap['oovv'] = tmpoovv
        tmpoovv = None

        woVoO += einsum('mebj,ie->mbij', eris_ovvo, t1)
        woVoO += einsum('mjbe,ie->mbji', eris_oovv, t1)
        woVoO += einsum('me,ijeb->mbij', self.Fov, t2)
        self.woVoO = self.saved['woVoO'] = woVoO
        woVoO = eris_ovvo = eris_oovv = None

        #:theta = t2*2 - t2.transpose(0,1,3,2)
        #:eris_ovvv = lib.unpack_tril(np.asarray(eris.ovvv).reshape(nocc*nvir,nvir**2)).reshape(nocc,nvir,nvir,nvir)
        #:ovvv = eris_ovvv*2 - eris_ovvv.transpose(0,3,2,1)
        #:tmpab = einsum('mebf,miaf->eiab', eris_ovvv, t2)
        #:tmpab = tmpab + tmpab.transpose(0,1,3,2) * .5
        #:tmpab-= einsum('mfbe,mifa->eiba', ovvv, theta) * .5
        #:self.wvOvV += eris_ovvv.transpose(2,0,3,1).conj()
        #:self.wvOvV -= tmpab
        nsegs = len(fswap['ebmf'])
        def load_ebmf(slice):
            dat = [fswap['ebmf/%d'%i][slice] for i in range(nsegs)]
            return np.concatenate(dat, axis=2)

        mem_now = lib.current_memory()[0]
        max_memory = max(0, self.max_memory - mem_now)
        blksize = min(nocc, max(ccsd.BLKMIN, int(max_memory*1e6/8/(nocc*nvir**2*4))))
        for p0, p1 in lib.prange(0, nvir, blksize):
            #:wvOvV  = einsum('mebf,miaf->eiab', ovvv, t2)
            #:wvOvV += einsum('mfbe,miaf->eiba', ovvv, t2)
            #:wvOvV -= einsum('mfbe,mifa->eiba', ovvv, t2)*2
            #:wvOvV += einsum('mebf,mifa->eiba', ovvv, t2)

            ebmf = load_ebmf(slice(p0, p1))
            wvOvV = einsum('ebmf,miaf->eiab', ebmf, t2)
            wvOvV = -.5 * wvOvV.transpose(0,1,3,2) - wvOvV

            # Using the permutation symmetry (em|fb) = (em|bf)
            efmb = load_ebmf((slice(None), slice(p0, p1)))
            wvOvV += np.einsum('ebmf->bmfe', efmb.conj())

            # tmp = (mf|be) - (me|bf)*.5
            tmp = -.5 * ebmf
            tmp += efmb.transpose(1,0,2,3)
            ebmf = None
            wvOvV += einsum('efmb,mifa->eiba', tmp, theta)
            tmp = None

            wvOvV += einsum('meni,mnab->eiab', eris_ovoo[:,p0:p1], tau_ccsd)
            wvOvV -= einsum('me,miab->eiab', self.Fov[:,p0:p1], t2)
            wvOvV += einsum('ma,eimb->eiab', t1, fswap['ovvo'][p0:p1])
            wvOvV += einsum('ma,eimb->eiba', t1, fswap['oovv'][p0:p1])

            self.wvOvV[p0:p1] = wvOvV

        self.made_ee_imds = True
        log.timer('EOM-DCSD EE intermediates', *cput0)
        return self
