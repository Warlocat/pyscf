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
from pyscf.cc import ccsd, gdcsd
from pyscf.cc import gintermediates as imd
from pyscf.cc import eom_gccsd
from pyscf import __config__

#einsum = np.einsum
einsum = lib.einsum


'''
    EOM-DCSD for GDCSD
    Ref: J. Chem. Phys. 146, 144104 (2017)
'''
def make_tau_p(t2, t1, fac_t2=1.0, fac_t1=1, out=None):
    t1t1 = einsum('ia,jb->ijab', fac_t1*t1, t1)
    tau = t1t1 - t1t1.transpose(0,1,3,2)
    if fac_t2 != 0.0:
        tau += fac_t2*t2
    return tau

def Fvv_p(t1, t2, eris, alpha):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    fvv = eris.fock[nocc:,nocc:]
    eris_vovv = np.asarray(eris.ovvv).transpose(1,0,3,2)
    tau_p = make_tau_p(t2, t1, fac_t2=alpha)
    Fae = fvv - einsum('me,ma->ae',fov, t1)
    Fae += einsum('mf,amef->ae', t1, eris_vovv)
    Fae -= 0.5*einsum('mnaf,mnef->ae', tau_p, eris.oovv)
    return Fae

def Foo_p(t1, t2, eris, beta):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    foo = eris.fock[:nocc,:nocc]
    tau_p = make_tau_p(t2, t1, fac_t2=beta)
    Fmi = ( foo + einsum('me,ie->mi',fov, t1)
            + einsum('ne,mnie->mi', t1, eris.ooov)
            + 0.5*einsum('inef,mnef->mi', tau_p, eris.oovv) )
    return Fmi

def Woooo_p(t1, t2, eris, p_gamma):
    tau_p = make_tau_p(t2, t1, fac_t2=p_gamma)
    tmp = einsum('je,mnie->mnij', t1, eris.ooov)
    Wmnij = eris.oooo + tmp - tmp.transpose(0,1,3,2)
    Wmnij += 0.5*einsum('ijef,mnef->mnij', tau_p, np.asarray(eris.oovv))
    return Wmnij

def Wvvvv_p(t1, t2, eris, p_gamma):
    tau_p = make_tau_p(t2, t1, fac_t2=p_gamma)
    eris_ovvv = np.asarray(eris.ovvv)
    tmp = einsum('mb,mafe->bafe', t1, eris_ovvv)
    Wabef = np.asarray(eris.vvvv) - tmp + tmp.transpose(1,0,2,3)
    Wabef += einsum('mnab,mnef->abef', tau_p, 0.5*np.asarray(eris.oovv))
    return Wabef

def Wovvo_p(t1, t2, eris, delta, eris_oovv_phys = None):
    eris_ovvo = -np.asarray(eris.ovov).transpose(0,1,3,2)
    eris_oovo = -np.asarray(eris.ooov).transpose(0,1,3,2)
    Wmbej  = einsum('jf,mbef->mbej', t1, eris.ovvv)
    Wmbej -= einsum('nb,mnej->mbej', t1, eris_oovo)
    if eris_oovv_phys is not None:
        assert delta == 1.0
        Wmbej -= einsum('jnfb,mnef->mbej', t2, eris_oovv_phys)
    else:
        Wmbej -= delta*einsum('jnfb,mnef->mbej', t2, eris.oovv)
    Wmbej -= einsum('jf,nb,mnef->mbej', t1, t1, eris.oovv)
    Wmbej += eris_ovvo
    return Wmbej

########################################
# EOM-IP-DCSD
########################################

def ipccsd_matvec(eom, vector, imds=None, diag=None):
    # Derived from EOMEE
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    r1, r2 = eom.vector_to_amplitudes(vector, nmo, nocc)

    # Eq. (8)
    Hr1 = -np.einsum('mi,m->i', imds.Foo, r1)
    Hr1 += np.einsum('me,mie->i', imds.Fov, r2)
    Hr1 += -0.5*np.einsum('nmie,mne->i', imds.Wooov, r2)
    # Eq. (9)
    Hr2 =  lib.einsum('ae,ije->ija', imds.Fvv_p, r2)
    tmp1 = lib.einsum('mi,mja->ija', imds.Foo_p, r2)
    Hr2 -= tmp1 - tmp1.transpose(1,0,2)
    Hr2 -= np.einsum('maji,m->ija', imds.Wovoo, r1)
    Hr2 += 0.5*lib.einsum('mnij,mna->ija', imds.Woooo_p, r2)
    tmp2 = lib.einsum('maei,mje->ija', imds.Wovvo_p, r2)
    Hr2 += tmp2 - tmp2.transpose(1,0,2)
    Hr2 += 0.5*lib.einsum('mnef,mnf,ijae->ija', imds.Woovv, r2, imds.t2)*imds.p_alpha

    vector = eom.amplitudes_to_vector(Hr1, Hr2)
    return vector

def ipccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    nocc, nvir = t1.shape

    Hr1 = -np.diag(imds.Foo)
    Hr2 = np.zeros((nocc,nocc,nvir), dtype=t1.dtype)
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvir):
                Hr2[i,j,a] += imds.Fvv_p[a,a]
                Hr2[i,j,a] += -imds.Foo_p[i,i]
                Hr2[i,j,a] += -imds.Foo_p[j,j]
                Hr2[i,j,a] += 0.5*(imds.Woooo_p[i,j,i,j]-imds.Woooo_p[j,i,i,j])
                Hr2[i,j,a] += imds.Wovvo[i,a,a,i] * imds.p_beta
                Hr2[i,j,a] += imds.Wovvo[j,a,a,j] * imds.p_beta
                Hr2[i,j,a] += 0.5*(np.dot(imds.Woovv[i,j,:,a], t2[i,j,a,:]) -
                                   np.dot(imds.Woovv[j,i,:,a], t2[i,j,a,:]))

    vector = eom.amplitudes_to_vector(Hr1, Hr2)
    return vector

class EOMIP(eom_gccsd.EOMIP):
    matvec = ipccsd_matvec
    get_diag = ipccsd_diag

    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris)
        imds.make_ip()
        return imds

########################################
# EOM-EA-DCSD
########################################

def eaccsd_matvec(eom, vector, imds=None, diag=None):
    # Ref: Nooijen and Bartlett, J. Chem. Phys. 102, 3629 (1994) Eqs.(30)-(31)
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    r1, r2 = eom.vector_to_amplitudes(vector, nmo, nocc)

    # Eq. (30)
    Hr1  = np.einsum('ac,c->a', imds.Fvv, r1)
    Hr1 += np.einsum('ld,lad->a', imds.Fov, r2)
    Hr1 += 0.5*np.einsum('alcd,lcd->a', imds.Wvovv, r2)
    # Eq. (31)
    Hr2 = np.einsum('abcj,c->jab', imds.Wvvvo, r1)
    tmp1 = lib.einsum('ac,jcb->jab', imds.Fvv_p, r2)
    Hr2 += tmp1 - tmp1.transpose(0,2,1)
    Hr2 -= lib.einsum('lj,lab->jab', imds.Foo_p, r2)
    tmp2 = lib.einsum('lbdj,lad->jab', imds.Wovvo_p, r2)
    Hr2 += tmp2 - tmp2.transpose(0,2,1)
    Hr2 += 0.5*lib.einsum('abcd,jcd->jab', imds.Wvvvv_p, r2)
    Hr2 -= 0.5*lib.einsum('klcd,lcd,kjab->jab', imds.Woovv, r2, imds.t2)*imds.p_beta

    vector = eom.amplitudes_to_vector(Hr1, Hr2)
    return vector

def eaccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    nocc, nvir = t1.shape

    Hr1 = np.diag(imds.Fvv)
    Hr2 = np.zeros((nocc,nvir,nvir),dtype=t1.dtype)
    for a in range(nvir):
        _Wvvvva = np.array(imds.Wvvvv_p[a])
        for b in range(a):
            for j in range(nocc):
                Hr2[j,a,b] += imds.Fvv_p[a,a]
                Hr2[j,a,b] += imds.Fvv_p[b,b]
                Hr2[j,a,b] += -imds.Foo_p[j,j]
                Hr2[j,a,b] += imds.Wovvo[j,b,b,j] * imds.p_alpha
                Hr2[j,a,b] += imds.Wovvo[j,a,a,j] * imds.p_alpha
                Hr2[j,a,b] += 0.5*(_Wvvvva[b,a,b]-_Wvvvva[b,b,a])
                Hr2[j,a,b] -= 0.5*(np.dot(imds.Woovv[:,j,a,b], t2[:,j,a,b]) -
                                   np.dot(imds.Woovv[:,j,b,a], t2[:,j,a,b]))

    vector = eom.amplitudes_to_vector(Hr1, Hr2)
    return vector

class EOMEA(eom_gccsd.EOMEA):
    matvec = eaccsd_matvec
    get_diag = eaccsd_diag

    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris)
        imds.make_ea()
        return imds

########################################
# EOM-EE-DCSD
########################################

def eeccsd_matvec(eom, vector, imds=None, diag=None):
    # Ref: J. Chem. Phys. 146, 144104 (2017)
    # Almost the same as the normal EOM-CCSD but with 
    # prefactors for terms in r2 -> Hr2
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    r1, r2 = eom.vector_to_amplitudes(vector, nmo, nocc)

    Hr1  = lib.einsum('ae,ie->ia', imds.Fvv, r1)
    Hr1 -= lib.einsum('mi,ma->ia', imds.Foo, r1)
    Hr1 += lib.einsum('me,imae->ia', imds.Fov, r2)
    Hr1 += lib.einsum('maei,me->ia', imds.Wovvo, r1)
    Hr1 -= 0.5*lib.einsum('mnie,mnae->ia', imds.Wooov, r2)
    Hr1 += 0.5*lib.einsum('amef,imef->ia', imds.Wvovv, r2)

    tmpab = lib.einsum('be,ijae->ijab', imds.Fvv_p, r2)
    tmp = 0.5*lib.einsum('mnef,mnbf->eb', imds.Woovv, r2)*imds.p_alpha
    tmpab -= lib.einsum('eb,ijae->ijab', tmp, imds.t2)
    tmpab -= lib.einsum('mbij,ma->ijab', imds.Wovoo, r1)
    tmpab -= lib.einsum('amef,ijfb,me->ijab', imds.Wvovv, imds.t2, r1)
    tmpij  = lib.einsum('mj,imab->ijab', -imds.Foo_p, r2)
    tmp = 0.5*lib.einsum('mnef,jnef->mj', imds.Woovv, r2)*imds.p_beta
    tmpij -= lib.einsum('mj,imab->ijab', tmp, imds.t2)
    tmpij += lib.einsum('abej,ie->ijab', imds.Wvvvo, r1)
    tmpij += lib.einsum('mnie,njab,me->ijab', imds.Wooov, imds.t2, r1)

    tmpabij = lib.einsum('mbej,imae->ijab', imds.Wovvo_p, r2)
    tmpabij = tmpabij - tmpabij.transpose(1,0,2,3)
    tmpabij = tmpabij - tmpabij.transpose(0,1,3,2)
    Hr2 = tmpabij

    Hr2 += tmpab - tmpab.transpose(0,1,3,2)
    Hr2 += tmpij - tmpij.transpose(1,0,2,3)
    Hr2 += 0.5*lib.einsum('mnij,mnab->ijab', imds.Woooo_p, r2)
    Hr2 += 0.5*lib.einsum('abef,ijef->ijab', imds.Wvvvv_p, r2)

    vector = eom.amplitudes_to_vector(Hr1, Hr2)
    return vector

def eeccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    nocc, nvir = t1.shape

    Hr1 = np.zeros((nocc,nvir), dtype=t1.dtype)
    Hr2 = np.zeros((nocc,nocc,nvir,nvir), dtype=t1.dtype)
    for i in range(nocc):
        for a in range(nvir):
            Hr1[i,a] = imds.Fvv[a,a] - imds.Foo[i,i] + imds.Wovvo[i,a,a,i]
    for a in range(nvir):
        tmp = 0.5*(np.einsum('ijeb,ijbe->ijb', imds.Woovv, t2) -
                   np.einsum('jieb,ijbe->ijb', imds.Woovv, t2))
        Hr2[:,:,:,a] += imds.Fvv_p[a,a] + tmp * imds.p_alpha
        Hr2[:,:,a,:] += imds.Fvv_p[a,a] + tmp * imds.p_alpha
        _Wvvvva = np.array(imds.Wvvvv_p[a])
        for b in range(a):
            Hr2[:,:,a,b] += 0.5*(_Wvvvva[b,a,b]-_Wvvvva[b,b,a])
        for i in range(nocc):
            tmp = imds.Wovvo_p[i,a,a,i]
            Hr2[:,i,:,a] += tmp
            Hr2[i,:,:,a] += tmp
            Hr2[:,i,a,:] += tmp
            Hr2[i,:,a,:] += tmp
    for i in range(nocc):
        tmp = 0.5*(np.einsum('kjab,jkab->jab', imds.Woovv, t2) -
                   np.einsum('kjba,jkab->jab', imds.Woovv, t2))
        Hr2[:,i,:,:] += -imds.Foo_p[i,i] + tmp * imds.p_beta
        Hr2[i,:,:,:] += -imds.Foo_p[i,i] + tmp * imds.p_beta
        for j in range(i):
            Hr2[i,j,:,:] += 0.5*(imds.Woooo_p[i,j,i,j]-imds.Woooo_p[j,i,i,j])

    vector = eom.amplitudes_to_vector(Hr1, Hr2)
    return vector

class EOMEE(eom_gccsd.EOMEE):
    matvec = eeccsd_matvec
    get_diag = eeccsd_diag
    def gen_matvec(self, imds=None, **kwargs):
        if imds is None: imds = self.make_imds()
        diag = self.get_diag(imds)
        matvec = lambda xs: [self.matvec(x, imds) for x in xs]
        return matvec, diag
    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris)
        imds.make_ee()
        return imds
    
gdcsd.GDCSD.EOMEE = lib.class_as_method(EOMEE)
gdcsd.pGCCSD.EOMEE = lib.class_as_method(EOMEE)
gdcsd.GDCD.EOMEA = lib.class_as_method(EOMEA)
gdcsd.pGCCSD.EOMEA = lib.class_as_method(EOMEA)
gdcsd.GDCSD.EOMIP = lib.class_as_method(EOMIP)
gdcsd.pGCCSD.EOMIP = lib.class_as_method(EOMIP)


class _IMDS:
    def __init__(self, cc, eris=None):
        self.verbose = cc.verbose
        self.stdout = cc.stdout
        self.t1 = cc.t1
        self.t2 = cc.t2
        if eris is None:
            eris = cc.ao2mo()
        self.eris = eris
        self._made_shared = False
        self.made_ip_imds = False
        self.made_ea_imds = False
        self.made_ee_imds = False
        if hasattr(cc, "p_mu"): # pCCSD
            # change notation
            self.p_alpha = cc.p_sigma
            self.p_delta = cc.p_sigma
            self.p_beta = (1.0 + cc.p_mu) / 2.0
            self.p_gamma = cc.p_mu
            self.oovv_phys = None
        elif hasattr(cc, "oovv_phys"): # DCSD
            self.p_alpha = 0.5
            self.p_beta = 0.5
            self.p_delta = 1.0
            self.p_gamma = 0.0
            self.oovv_phys = cc.oovv_phys
        else:
            raise ValueError("Unknown method for gdcsd EOM-CC")

    def _make_shared(self):
        cput0 = (logger.process_clock(), logger.perf_counter())

        t1, t2, eris = self.t1, self.t2, self.eris
        self.Foo_p = Foo_p(t1, t2, eris, self.p_beta)
        self.Fvv_p = Fvv_p(t1, t2, eris, self.p_alpha)
        self.Foo = imd.Foo(t1, t2, eris)
        self.Fvv = imd.Fvv(t1, t2, eris)
        self.Fov = imd.Fov(t1, t2, eris)

        # 2 virtuals
        self.Wovvo_p = Wovvo_p(t1, t2, eris, self.p_delta, self.oovv_phys)
        self.Wovvo = imd.Wovvo(t1, t2, eris)
        self.Woovv = eris.oovv

        self._made_shared = True
        logger.timer_debug1(self, 'EOM-GDCSD shared intermediates', *cput0)
        return self

    def make_ip(self):
        if not self._made_shared:
            self._make_shared()

        cput0 = (logger.process_clock(), logger.perf_counter())

        t1, t2, eris = self.t1, self.t2, self.eris

        # 0 or 1 virtuals
        self.Woooo_p = Woooo_p(t1, t2, eris, self.p_gamma)
        self.Wooov = imd.Wooov(t1, t2, eris)
        self.Wovoo = imd.Wovoo(t1, t2, eris)

        self.made_ip_imds = True
        logger.timer_debug1(self, 'EOM-DCSD IP intermediates', *cput0)
        return self

    def make_t3p2_ip(self, cc):
        raise NotImplementedError
        return self


    def make_ea(self):
        if not self._made_shared:
            self._make_shared()

        cput0 = (logger.process_clock(), logger.perf_counter())

        t1, t2, eris = self.t1, self.t2, self.eris

        # 3 or 4 virtuals
        self.Wvovv = imd.Wvovv(t1, t2, eris)
        self.Wvvvv_p = Wvvvv_p(t1, t2, eris, self.p_gamma)
        self.Wvvvo = imd.Wvvvo(t1, t2, eris)

        self.made_ea_imds = True
        logger.timer_debug1(self, 'EOM-DCSD EA intermediates', *cput0)
        return self

    def make_t3p2_ea(self, cc):
        raise NotImplementedError
        return self


    def make_ee(self):
        if not self._made_shared:
            self._make_shared()

        cput0 = (logger.process_clock(), logger.perf_counter())

        t1, t2, eris = self.t1, self.t2, self.eris

        if not self.made_ip_imds:
            # 0 or 1 virtuals
            self.Woooo_p = Woooo_p(t1, t2, eris, self.p_gamma)
            self.Wooov = imd.Wooov(t1, t2, eris)
            self.Wovoo = imd.Wovoo(t1, t2, eris)
        if not self.made_ea_imds:
            # 3 or 4 virtuals
            self.Wvovv = imd.Wvovv(t1, t2, eris)
            self.Wvvvv_p = Wvvvv_p(t1, t2, eris, self.p_gamma)
            self.Wvvvo = imd.Wvvvo(t1, t2, eris)

        self.made_ee_imds = True
        logger.timer(self, 'EOM-DCSD EE intermediates', *cput0)
        return self


if __name__ == '__main__':
    from pyscf import gto, scf, cc, dft
    from pyscf.cc import eom_rccsd
    import numpy as np
    from pyscf.data.nist import HARTREE2EV
    
    mol = gto.Mole()
    re = 2.13713
    mol.atom = f'''
C 0.0 0.0 0.0
H 0.0 0.0 {5.0}
'''
    mol.basis ={'H': gto.basis.parse('''
H S
    19.2406  0.032828
     2.8992  0.231208
     0.6534  0.817238
H S
     0.1776  1.0
H S
     0.0250  1.0
H P
     1.0000  1.0
''', optimize=True), 'C': gto.basis.parse('''
C S
  4231.61   0.002029
   634.882   0.015535
   146.097   0.075411
    42.4974  0.257121
    14.1892  0.596555
     1.9666  0.242517
C S
     5.1477  1.0
C S
     0.4962  1.0
C S
     0.1533  1.0
C S
     0.0150  1.0
C P
    18.1557  0.018534
     3.9864  0.115442
     1.1429  0.386206
     0.3594  0.640089
C P
     0.1146  1.0
C P
     0.0110  1.0
C D
     0.7500  1.0
                                          ''' , optimize=True)}
    mol.unit = "bohr"
    mol.cart = True
    mol.charge = 1
    mol.symmetry = False
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    mycc = cc.CCSD(mf, frozen=1)
    mycc.conv_tol = 1e-10
    mycc.run()
    print("RCCSD")
    print(mycc.e_corr)
    myeom = eom_rccsd.EOMEETriplet(mycc)
    myeom.kernel(nroots=10)
    print(myeom.e*HARTREE2EV)
    myeom = eom_rccsd.EOMEESinglet(mycc)
    myeom.conv_tol = 1e-8
    myeom.kernel(nroots=10)
    print(myeom.e*HARTREE2EV)
    # exit()
    print("GDCSD")
    mycc = gdcsd.GDCSD(mf.to_ghf(), frozen=2).run()
    print(mycc.e_corr)
    myeom = EOMEE(mycc)
    myeom.kernel(nroots=60)
    print(myeom.e*HARTREE2EV)
    print("2CC")
    mycc = gdcsd.pGCCSD(mf.to_ghf(), frozen=2, mu=1.0, sigma=0.0).run()
    print(mycc.e_corr)
    myeom = EOMEE(mycc)
    myeom.kernel(nroots=60)
    print(myeom.e*HARTREE2EV)
    print("pGCCSD")
    mycc = gdcsd.pGCCSD(mf.to_ghf(), frozen=2).run()
    print(mycc.e_corr)
    myeom = EOMEE(mycc)
    myeom.kernel(nroots=60)
    print(myeom.e*HARTREE2EV)

    # myeom = eom_gccsd.EOMEE(mycc)
    # myeom.kernel(nroots=100)
    # print(myeom.e*HARTREE2EV)
