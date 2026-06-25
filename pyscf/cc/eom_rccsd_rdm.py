#!/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Spatial (closed-shell RHF) EOM-IP/EA-CCSD unrelaxed <L|R> one-particle density
matrices and Dyson orbitals, computed directly from eom_rccsd amplitudes
(r1,r2,l1,l2,t1,t2) plus the ground-state Lambda (gl1,gl2) for phi^L.

gamma[p,q] = <Psi_L| p^dag q |Psi_R> in the MO basis.  L,R are biorthonormalized
to the spatial metric  <L|R> = l1.r1 + 2 l2_mnp r2_mnp - l2_mnp r2_nmp  (IP)
                                 l1.r1 + 2 l2_iab r2_iab - l2_iab r2_iba  (EA).

The formulas come from spin-integration of Stanton's spin-orbital EOM-CC density
(Wick&d), reduced to closed-shell spatial amplitudes.  Each block is the alpha+beta
spin trace; the spin-block amplitude intermediates below (t2aa, r_oov, l_voo, ...)
encode the closed-shell relations, e.g. t_oovv(aa) = t2 - t2.swapab, t_oOvV = t2.
"""

import numpy as np
from pyscf import lib

einsum = lib.einsum


def _ip_intermediates(t1, t2, r1, r2, l1, l2):
    """Closed-shell spin-block amplitude intermediates for EOM-IP."""
    t2aa = t2 - t2.transpose(0, 1, 3, 2)
    r_oov = r2.transpose(1, 0, 2) - r2
    r_oOV = -r2
    l_voo = (l2.transpose(1, 0, 2) - l2).transpose(2, 0, 1)
    l_VoO = (-l2).transpose(2, 0, 1)
    return t2aa, r_oov, r_oOV, l_voo, l_VoO


def _ea_intermediates(t1, t2, r1, r2, l1, l2):
    """Closed-shell spin-block amplitude intermediates for EOM-EA."""
    t2aa = t2 - t2.transpose(0, 1, 3, 2)
    r_ovv = r2 - r2.transpose(0, 2, 1)
    r_OvV = r2
    l_vvo = (l2 - l2.transpose(0, 2, 1)).transpose(1, 2, 0)
    l_vVO = l2.transpose(1, 2, 0)
    return t2aa, r_ovv, r_OvV, l_vvo, l_vVO


def _z_intermediates(gl1, gl2):
    """Ground-state Lambda spin-blocks (de-excitation) used in the Dyson phi^L."""
    z_vo = gl1.T
    z_vvoo = (gl2 - gl2.transpose(0, 1, 3, 2)).transpose(2, 3, 0, 1)
    z_vVoO = gl2.transpose(2, 3, 0, 1)
    return z_vo, z_vvoo, z_vVoO


def make_rdm1_ip(t1, t2, r1, r2, l1, l2):
    """Spatial RHF EOM-IP 1-RDM gamma[p,q]. r1(o),r2(o,o,v); l1(o),l2(o,o,v)."""
    nocc, nvir = r1.shape[0], t1.shape[1]
    t2aa, r_oov, r_oOV, l_voo, l_VoO = _ip_intermediates(t1, t2, r1, r2, l1, l2)
    lr = l1.dot(r1) + 2*einsum('mnp,mnp', l2, r2) - einsum('mnp,nmp', l2, r2)

    # oo
    doo = -einsum('j,i->ij', l1, r1)
    doo += einsum('ajk,k,ia->ij', l_voo, r1, t1)
    doo -= einsum('ajk,ika->ij', l_voo, r_oov)
    doo -= einsum('AjI,iIA->ij', l_VoO, r_oOV)
    doo += -einsum('AiJ,i,IA->IJ', l_VoO, r1, t1) - einsum('AiJ,iIA->IJ', l_VoO, r_oOV)
    # ov
    dov = -einsum('j,i,ja->ia', l1, r1, t1)
    dov += einsum('j,j,ia->ia', l1, r1, t1)
    dov -= einsum('j,ija->ia', l1, r_oov)
    dov -= 0.5*einsum('bjk,i,jkab->ia', l_voo, r1, t2aa)
    dov += einsum('bjk,j,ikab->ia', l_voo, r1, t2aa)
    dov -= einsum('bjk,j,ib,ka->ia', l_voo, r1, t1, t1)
    dov -= 0.5*einsum('bjk,jka,ib->ia', l_voo, r_oov, t1)
    dov += einsum('bjk,ijb,ka->ia', l_voo, r_oov, t1)
    dov += 0.5*einsum('bjk,jkb,ia->ia', l_voo, r_oov, t1)
    dov -= einsum('AjI,i,jIaA->ia', l_VoO, r1, t2)
    dov += einsum('AjI,j,iIaA->ia', l_VoO, r1, t2)
    dov -= einsum('AjI,iIA,ja->ia', l_VoO, r_oOV, t1)
    dov += einsum('AjI,jIA,ia->ia', l_VoO, r_oOV, t1)
    dov += einsum('i,i,IA->IA', l1, r1, t1)
    dov += einsum('i,iIA->IA', l1, r_oOV)
    dov += einsum('aij,i,jIaA->IA', l_voo, r1, t2)
    dov += 0.5*einsum('aij,ija,IA->IA', l_voo, r_oov, t1)
    dov += einsum('BiJ,i,IJAB->IA', l_VoO, r1, t2aa)
    dov -= einsum('BiJ,i,IB,JA->IA', l_VoO, r1, t1, t1)
    dov -= einsum('BiJ,iJA,IB->IA', l_VoO, r_oOV, t1)
    dov -= einsum('BiJ,iIB,JA->IA', l_VoO, r_oOV, t1)
    dov += einsum('BiJ,iJB,IA->IA', l_VoO, r_oOV, t1)
    # vo
    dvo = -einsum('aij,j->ai', l_voo, r1) + einsum('AiI,i->AI', l_VoO, r1)
    # vv
    dvv = einsum('aij,i,jb->ab', l_voo, r1, t1)
    dvv += 0.5*einsum('aij,ijb->ab', l_voo, r_oov)
    dvv += einsum('AiI,i,IB->AB', l_VoO, r1, t1)
    dvv += einsum('AiI,iIB->AB', l_VoO, r_oOV)

    g = np.zeros((nocc+nvir, nocc+nvir))
    g[:nocc, :nocc] = doo + 2*np.eye(nocc)*lr
    g[:nocc, nocc:] = dov
    g[nocc:, :nocc] = dvo
    g[nocc:, nocc:] = dvv
    return g


def make_rdm1_ea(t1, t2, r1, r2, l1, l2):
    """Spatial RHF EOM-EA 1-RDM gamma[p,q]. r1(v),r2(o,v,v); l1(v),l2(o,v,v)."""
    nocc, nvir = t1.shape[0], r1.shape[0]
    t2aa, r_ovv, r_OvV, l_vvo, l_vVO = _ea_intermediates(t1, t2, r1, r2, l1, l2)
    lr = l1.dot(r1) + 2*einsum('iab,iab', l2, r2) - einsum('iab,iba', l2, r2)

    # oo
    doo = -einsum('abj,a,ib->ij', l_vvo, r1, t1)
    doo -= 0.5*einsum('abj,iab->ij', l_vvo, r_ovv)
    doo += -einsum('aAJ,a,IA->IJ', l_vVO, r1, t1) - einsum('aAJ,IaA->IJ', l_vVO, r_OvV)
    # ov
    dov = -einsum('b,a,ib->ia', l1, r1, t1)
    dov -= einsum('b,iab->ia', l1, r_ovv)
    dov += einsum('b,b,ia->ia', l1, r1, t1)
    dov -= 0.5*einsum('bcj,a,ijbc->ia', l_vvo, r1, t2aa)
    dov += einsum('bcj,jab,ic->ia', l_vvo, r_ovv, t1)
    dov += einsum('bcj,b,ijac->ia', l_vvo, r1, t2aa)
    dov -= einsum('bcj,b,ic,ja->ia', l_vvo, r1, t1, t1)
    dov -= 0.5*einsum('bcj,ibc,ja->ia', l_vvo, r_ovv, t1)
    dov += 0.5*einsum('bcj,jbc,ia->ia', l_vvo, r_ovv, t1)
    dov -= einsum('bAI,a,iIbA->ia', l_vVO, r1, t2)
    dov -= einsum('bAI,IaA,ib->ia', l_vVO, r_OvV, t1)
    dov += einsum('bAI,b,iIaA->ia', l_vVO, r1, t2)
    dov += einsum('bAI,IbA,ia->ia', l_vVO, r_OvV, t1)
    dov += einsum('a,a,IA->IA', l1, r1, t1)
    dov += einsum('a,IaA->IA', l1, r_OvV)
    dov += einsum('abi,a,iIbA->IA', l_vvo, r1, t2)
    dov += 0.5*einsum('abi,iab,IA->IA', l_vvo, r_ovv, t1)
    dov += einsum('aBJ,a,IJAB->IA', l_vVO, r1, t2aa)
    dov -= einsum('aBJ,a,IB,JA->IA', l_vVO, r1, t1, t1)
    dov -= einsum('aBJ,JaA,IB->IA', l_vVO, r_OvV, t1)
    dov -= einsum('aBJ,IaB,JA->IA', l_vVO, r_OvV, t1)
    dov += einsum('aBJ,JaB,IA->IA', l_vVO, r_OvV, t1)
    # vo
    dvo = -einsum('abi,b->ai', l_vvo, r1) + einsum('aAI,a->AI', l_vVO, r1)
    # vv
    dvv = einsum('a,b->ab', l1, r1)
    dvv += einsum('aci,ibc->ab', l_vvo, r_ovv)
    dvv -= einsum('aci,c,ib->ab', l_vvo, r1, t1)
    dvv += einsum('aAI,IbA->ab', l_vVO, r_OvV)
    dvv += einsum('aAI,a,IB->AB', l_vVO, r1, t1)
    dvv += einsum('aAI,IaB->AB', l_vVO, r_OvV)

    g = np.zeros((nocc+nvir, nocc+nvir))
    g[:nocc, :nocc] = doo + 2*np.eye(nocc)*lr
    g[:nocc, nocc:] = dov
    g[nocc:, :nocc] = dvo
    g[nocc:, nocc:] = dvv
    return g


def make_dyson_ip(t1, t2, r1, r2, l1, l2, gl1, gl2):
    """Spatial RHF EOM-IP Dyson orbitals (alpha component), length nmo each.

    Returns (phi_R, phi_L):
      phi_R[p] = <Psi_k^{N-1}| a_p |Psi_0>      (right Dyson, from EOM-left l1,l2)
      phi_L[p] = <Psi_0| a_p^dag |Psi_k^{N-1}>  (left Dyson, from EOM-right r1,r2
                                                 and ground-state Lambda gl1,gl2)
    """
    t2aa, r_oov, r_oOV, l_voo, l_VoO = _ip_intermediates(t1, t2, r1, r2, l1, l2)
    z_vo, z_vvoo, z_vVoO = _z_intermediates(gl1, gl2)

    phiR_occ = l1.copy()
    phiR_vir = einsum('i,ia->a', l1, t1)
    phiR_vir += 0.5*einsum('bij,ijab->a', l_voo, t2aa)
    phiR_vir += einsum('AiI,iIaA->a', l_VoO, t2)

    phiL_occ = r1.copy()
    phiL_occ -= einsum('j,ia,aj->i', r1, t1, z_vo)
    phiL_occ -= 0.5*einsum('j,ikab,abjk->i', r1, t2aa, z_vvoo)
    phiL_occ -= einsum('j,iIaA,aAjI->i', r1, t2, z_vVoO)
    phiL_occ += einsum('ija,aj->i', r_oov, z_vo)
    phiL_occ += 0.5*einsum('jka,ib,abjk->i', r_oov, t1, z_vvoo)
    phiL_occ += einsum('iIA,AI->i', r_oOV, z_vo)
    phiL_occ -= einsum('jIA,ia,aAjI->i', r_oOV, t1, z_vVoO)
    phiL_vir = einsum('i,ai->a', r1, z_vo)
    phiL_vir += 0.5*einsum('ijb,abij->a', r_oov, z_vvoo)
    phiL_vir += einsum('iIA,aAiI->a', r_oOV, z_vVoO)

    phiR = np.concatenate([phiR_occ, phiR_vir])
    phiL = np.concatenate([phiL_occ, phiL_vir])
    return phiR, phiL


def make_dyson_ea(t1, t2, r1, r2, l1, l2, gl1, gl2):
    """Spatial RHF EOM-EA Dyson orbitals (alpha component), length nmo each.

    Returns (phi_R, phi_L):
      phi_R[p] = <Psi_k^{N+1}| a_p^dag |Psi_0>  (right Dyson, from EOM-left l1,l2)
      phi_L[p] = <Psi_0| a_p |Psi_k^{N+1}>      (left Dyson, from EOM-right r1,r2
                                                 and ground-state Lambda gl1,gl2)
    """
    t2aa, r_ovv, r_OvV, l_vvo, l_vVO = _ea_intermediates(t1, t2, r1, r2, l1, l2)
    z_vo, z_vvoo, z_vVoO = _z_intermediates(gl1, gl2)

    phiR_occ = -einsum('a,ia->i', l1, t1)
    phiR_occ -= 0.5*einsum('abj,ijab->i', l_vvo, t2aa)
    phiR_occ -= einsum('aAI,iIaA->i', l_vVO, t2)
    phiR_vir = l1.copy()

    phiL_occ = -einsum('a,ai->i', r1, z_vo)
    phiL_occ -= 0.5*einsum('jab,abij->i', r_ovv, z_vvoo)
    phiL_occ -= einsum('IaA,aAiI->i', r_OvV, z_vVoO)
    phiL_vir = r1.copy()
    phiL_vir += einsum('iab,bi->a', r_ovv, z_vo)
    phiL_vir += einsum('IaA,AI->a', r_OvV, z_vo)
    phiL_vir -= einsum('b,ia,bi->a', r1, t1, z_vo)
    phiL_vir -= 0.5*einsum('b,ijac,bcij->a', r1, t2aa, z_vvoo)
    phiL_vir -= einsum('b,iIaA,bAiI->a', r1, t2, z_vVoO)
    phiL_vir += 0.5*einsum('ibc,ja,bcij->a', r_ovv, t1, z_vvoo)
    phiL_vir -= einsum('IbA,ia,bAiI->a', r_OvV, t1, z_vVoO)

    phiR = np.concatenate([phiR_occ, phiR_vir])
    phiL = np.concatenate([phiL_occ, phiL_vir])
    return phiR, phiL


# ---- EOM-class interface (vector unpacking + biorthonormalization) ----
def ipccsd_rdm1_LR(myeom, r, l, t1=None, t2=None):
    '''1-RDM for an EOM-IP-CCSD state, LR part only: <L|p^dag q|R>.

    This is the <L|R> expectation density (Tr = N-1), NOT the true unrelaxed
    property density -- the latter additionally needs the zeta (Z-vector) term.
    r,l: right/left eigenvectors (packed).'''
    cc = myeom._cc
    if t1 is None:
        t1, t2 = cc.t1, cc.t2
    nocc, nmo = myeom.nocc, myeom.nmo
    r1, r2 = myeom.vector_to_amplitudes(r, nmo, nocc)
    l1, l2 = myeom.vector_to_amplitudes(l, nmo, nocc)
    lr = np.dot(l1, r1) + 2*einsum('mnp,mnp', l2, r2) - einsum('mnp,nmp', l2, r2)
    return make_rdm1_ip(t1, t2, r1, r2, l1/lr, l2/lr)


def eaccsd_rdm1_LR(myeom, r, l, t1=None, t2=None):
    '''1-RDM for an EOM-EA-CCSD state, LR part only: <L|p^dag q|R> (Tr = N+1).

    The <L|R> expectation density, NOT the zeta-relaxed property density.'''
    cc = myeom._cc
    if t1 is None:
        t1, t2 = cc.t1, cc.t2
    nocc, nmo = myeom.nocc, myeom.nmo
    r1, r2 = myeom.vector_to_amplitudes(r, nmo, nocc)
    l1, l2 = myeom.vector_to_amplitudes(l, nmo, nocc)
    lr = np.dot(l1, r1) + 2*einsum('iab,iab', l2, r2) - einsum('iab,iba', l2, r2)
    return make_rdm1_ea(t1, t2, r1, r2, l1/lr, l2/lr)


def _ground_lambda(cc):
    '''Return ground-state Lambda amplitudes, solving if necessary.'''
    if getattr(cc, 'l1', None) is None or getattr(cc, 'l2', None) is None:
        cc.solve_lambda()
    return cc.l1, cc.l2


def ipccsd_dyson(myeom, r, l, t1=None, t2=None):
    '''Spatial Dyson orbitals (phi_R, phi_L) for an EOM-IP-CCSD state.'''
    cc = myeom._cc
    if t1 is None:
        t1, t2 = cc.t1, cc.t2
    gl1, gl2 = _ground_lambda(cc)
    nocc, nmo = myeom.nocc, myeom.nmo
    r1, r2 = myeom.vector_to_amplitudes(r, nmo, nocc)
    l1, l2 = myeom.vector_to_amplitudes(l, nmo, nocc)
    lr = np.dot(l1, r1) + 2*einsum('mnp,mnp', l2, r2) - einsum('mnp,nmp', l2, r2)
    return make_dyson_ip(t1, t2, r1, r2, l1/lr, l2/lr, gl1, gl2)


def eaccsd_dyson(myeom, r, l, t1=None, t2=None):
    '''Spatial Dyson orbitals (phi_R, phi_L) for an EOM-EA-CCSD state.'''
    cc = myeom._cc
    if t1 is None:
        t1, t2 = cc.t1, cc.t2
    gl1, gl2 = _ground_lambda(cc)
    nocc, nmo = myeom.nocc, myeom.nmo
    r1, r2 = myeom.vector_to_amplitudes(r, nmo, nocc)
    l1, l2 = myeom.vector_to_amplitudes(l, nmo, nocc)
    lr = np.dot(l1, r1) + 2*einsum('iab,iab', l2, r2) - einsum('iab,iba', l2, r2)
    return make_dyson_ea(t1, t2, r1, r2, l1/lr, l2/lr, gl1, gl2)


def ipccsd_tdm1(myeom, rk, lk, rkp, lkp, t1=None, t2=None):
    '''State-to-state transition 1-RDM <Psi_k| p^dag q |Psi_k'> between two
    EOM-IP states k and k'.  Bra uses the left vector lk biorthonormalized to its
    own right partner rk (<lk|rk>=1); ket uses the right vector rkp.'''
    cc = myeom._cc
    if t1 is None:
        t1, t2 = cc.t1, cc.t2
    nocc, nmo = myeom.nocc, myeom.nmo
    l1, l2 = myeom.vector_to_amplitudes(lk, nmo, nocc)
    lr1, lr2 = myeom.vector_to_amplitudes(rk, nmo, nocc)
    r1, r2 = myeom.vector_to_amplitudes(rkp, nmo, nocc)
    lr = np.dot(l1, lr1) + 2*einsum('mnp,mnp', l2, lr2) - einsum('mnp,nmp', l2, lr2)
    return make_rdm1_ip(t1, t2, r1, r2, l1/lr, l2/lr)


def eaccsd_tdm1(myeom, rk, lk, rkp, lkp, t1=None, t2=None):
    '''State-to-state transition 1-RDM between two EOM-EA states k and k'.'''
    cc = myeom._cc
    if t1 is None:
        t1, t2 = cc.t1, cc.t2
    nocc, nmo = myeom.nocc, myeom.nmo
    l1, l2 = myeom.vector_to_amplitudes(lk, nmo, nocc)
    lr1, lr2 = myeom.vector_to_amplitudes(rk, nmo, nocc)
    r1, r2 = myeom.vector_to_amplitudes(rkp, nmo, nocc)
    lr = np.dot(l1, lr1) + 2*einsum('iab,iab', l2, lr2) - einsum('iab,iba', l2, lr2)
    return make_rdm1_ea(t1, t2, r1, r2, l1/lr, l2/lr)


# ================== spatial RHF EOM-EE (singlet), LR part ==================
def make_rdm1_ee(t1, t2, le1, le2, r1, r2):
    """Spatial RHF singlet EOM-EE LR 1-RDM <L|p^dag q|R>, gamma[p,q].

    le1,le2 = EOM-EE left amplitudes; r1,r2 = EOM-EE right (singlet, with the
    symmetry x2[i,j,a,b] = x2[j,i,b,a]).  Derived by spin-integration of the
    spin-orbital EOM-EE density, verified vs the GHF spin trace (~1e-15).
    """
    nocc, nvir = t1.shape
    t2aa = t2 - t2.transpose(0, 1, 3, 2)
    r2aa = r2 - r2.transpose(0, 1, 3, 2)
    le2aa = le2 - le2.transpose(0, 1, 3, 2)
    lvo = le1.T
    lvvoo = le2aa.transpose(2, 3, 0, 1)
    lvVoO = le2.transpose(2, 3, 0, 1)
    lr = 2*einsum('ia,ia', le1, r1) + einsum('ijab,ijab', le2, 2*r2 - r2.transpose(0, 1, 3, 2))

    doo = -einsum('aj,ia->ij', lvo, r1)
    doo += einsum('abjk,ka,ib->ij', lvvoo, r1, t1)
    doo -= 0.5*einsum('abjk,ikab->ij', lvvoo, r2aa)
    doo -= einsum('aAjI,iIaA->ij', lvVoO, r2)
    doo -= einsum('aAjI,IA,ia->ij', lvVoO, r1, t1)

    dov = -einsum('bj,ja,ib->ia', lvo, r1, t1)
    dov += einsum('bj,ijab->ia', lvo, r2aa)
    dov -= einsum('bj,ib,ja->ia', lvo, r1, t1)
    dov += einsum('bj,jb,ia->ia', lvo, r1, t1)
    dov -= 0.5*einsum('bcjk,ja,ikbc->ia', lvvoo, r1, t2aa)
    dov += 0.5*einsum('bcjk,jkab,ic->ia', lvvoo, r2aa, t1)
    dov -= 0.5*einsum('bcjk,ib,jkac->ia', lvvoo, r1, t2aa)
    dov += einsum('bcjk,jb,ikac->ia', lvvoo, r1, t2aa)
    dov -= einsum('bcjk,jb,ic,ka->ia', lvvoo, r1, t1, t1)
    dov += 0.5*einsum('bcjk,ijbc,ka->ia', lvvoo, r2aa, t1)
    dov += 0.25*einsum('bcjk,jkbc,ia->ia', lvvoo, r2aa, t1)
    dov -= einsum('bAjI,ja,iIbA->ia', lvVoO, r1, t2)
    dov -= einsum('bAjI,jIaA,ib->ia', lvVoO, r2, t1)
    dov -= einsum('bAjI,ib,jIaA->ia', lvVoO, r1, t2)
    dov += einsum('bAjI,jb,iIaA->ia', lvVoO, r1, t2)
    dov -= einsum('bAjI,iIbA,ja->ia', lvVoO, r2, t1)
    dov += einsum('bAjI,jIbA,ia->ia', lvVoO, r2, t1)
    dov += einsum('bAjI,IA,ijab->ia', lvVoO, r1, t2aa)
    dov -= einsum('bAjI,IA,ib,ja->ia', lvVoO, r1, t1, t1)
    dov += einsum('AI,iIaA->ia', lvo, r2)
    dov += einsum('AI,IA,ia->ia', lvo, r1, t1)
    dov += einsum('ABIJ,IA,iJaB->ia', lvvoo, r1, t2)
    dov += 0.25*einsum('ABIJ,IJAB,ia->ia', lvvoo, r2aa, t1)

    dvo = einsum('abij,jb->ai', lvvoo, r1)
    dvo += einsum('aAiI,IA->ai', lvVoO, r1)

    dvv = einsum('ai,ib->ab', lvo, r1)
    dvv += 0.5*einsum('acij,ijbc->ab', lvvoo, r2aa)
    dvv -= einsum('acij,ic,jb->ab', lvvoo, r1, t1)
    dvv += einsum('aAiI,iIbA->ab', lvVoO, r2)
    dvv += einsum('aAiI,IA,ib->ab', lvVoO, r1, t1)

    g = np.zeros((nocc+nvir, nocc+nvir))
    g[:nocc, :nocc] = doo + np.eye(nocc)*lr
    g[:nocc, nocc:] = dov
    g[nocc:, :nocc] = dvo
    g[nocc:, nocc:] = dvv
    return g


def make_tdm1_ee_0k(t1, t2, r1, r2, gl1, gl2):
    """RHF singlet ground<-excited (absorption) transition density
    <Psi_0| p^dag q |Psi_k> = <0|(1+Lam) e^-T p^dag q e^T R_k|0>, spatial."""
    nocc, nvir = t1.shape
    t2aa = t2 - t2.transpose(0, 1, 3, 2)
    r2aa = r2 - r2.transpose(0, 1, 3, 2)
    gl2aa = gl2 - gl2.transpose(0, 1, 3, 2)
    zvo = gl1.T
    zvvoo = gl2aa.transpose(2, 3, 0, 1)
    zvVoO = gl2.transpose(2, 3, 0, 1)
    ov = 2*einsum('ia,ia', gl1, r1) + einsum('ijab,ijab', gl2, 2*r2 - r2.transpose(0, 1, 3, 2))

    doo = -einsum('ia,aj->ij', r1, zvo)
    doo += einsum('ka,ib,abjk->ij', r1, t1, zvvoo)
    doo -= 0.5*einsum('ikab,abjk->ij', r2aa, zvvoo)
    doo -= einsum('iIaA,aAjI->ij', r2, zvVoO)
    doo -= einsum('IA,ia,aAjI->ij', r1, t1, zvVoO)

    dov = r1.copy()
    dov -= einsum('ja,ib,bj->ia', r1, t1, zvo)
    dov -= 0.5*einsum('ja,ikbc,bcjk->ia', r1, t2aa, zvvoo)
    dov -= einsum('ja,iIbA,bAjI->ia', r1, t2, zvVoO)
    dov += einsum('ijab,bj->ia', r2aa, zvo)
    dov += 0.5*einsum('jkab,ic,bcjk->ia', r2aa, t1, zvvoo)
    dov += einsum('iIaA,AI->ia', r2, zvo)
    dov -= einsum('jIaA,ib,bAjI->ia', r2, t1, zvVoO)
    dov -= einsum('ib,ja,bj->ia', r1, t1, zvo)
    dov -= 0.5*einsum('ib,jkac,bcjk->ia', r1, t2aa, zvvoo)
    dov -= einsum('ib,jIaA,bAjI->ia', r1, t2, zvVoO)
    dov += einsum('jb,ia,bj->ia', r1, t1, zvo)
    dov += einsum('jb,ikac,bcjk->ia', r1, t2aa, zvvoo)
    dov += einsum('jb,iIaA,bAjI->ia', r1, t2, zvVoO)
    dov -= einsum('jb,ic,ka,bcjk->ia', r1, t1, t1, zvvoo)
    dov += 0.5*einsum('ijbc,ka,bcjk->ia', r2aa, t1, zvvoo)
    dov += 0.25*einsum('jkbc,ia,bcjk->ia', r2aa, t1, zvvoo)
    dov -= einsum('iIbA,ja,bAjI->ia', r2, t1, zvVoO)
    dov += einsum('jIbA,ia,bAjI->ia', r2, t1, zvVoO)
    dov += einsum('IA,ia,AI->ia', r1, t1, zvo)
    dov += einsum('IA,ijab,bAjI->ia', r1, t2aa, zvVoO)
    dov += einsum('IA,iJaB,ABIJ->ia', r1, t2, zvvoo)
    dov -= einsum('IA,ib,ja,bAjI->ia', r1, t1, t1, zvVoO)
    dov += 0.25*einsum('IJAB,ia,ABIJ->ia', r2aa, t1, zvvoo)

    dvo = einsum('jb,abij->ai', r1, zvvoo)
    dvo += einsum('IA,aAiI->ai', r1, zvVoO)

    dvv = einsum('ib,ai->ab', r1, zvo)
    dvv += 0.5*einsum('ijbc,acij->ab', r2aa, zvvoo)
    dvv += einsum('iIbA,aAiI->ab', r2, zvVoO)
    dvv -= einsum('ic,jb,acij->ab', r1, t1, zvvoo)
    dvv += einsum('IA,ib,aAiI->ab', r1, t1, zvVoO)

    g = np.zeros((nocc+nvir, nocc+nvir))
    g[:nocc, :nocc] = doo + np.eye(nocc)*ov
    g[:nocc, nocc:] = dov
    g[nocc:, :nocc] = dvo
    g[nocc:, nocc:] = dvv
    return np.sqrt(2.0) * g


def make_tdm1_ee_k0(t1, t2, le1, le2):
    """RHF singlet excited<-ground (emission) transition density
    <Psi_k| p^dag q |Psi_0> = <0|L_k e^-T p^dag q e^T|0>, spatial."""
    nocc, nvir = t1.shape
    t2aa = t2 - t2.transpose(0, 1, 3, 2)
    le2aa = le2 - le2.transpose(0, 1, 3, 2)
    lvo = le1.T
    lvvoo = le2aa.transpose(2, 3, 0, 1)
    lvVoO = le2.transpose(2, 3, 0, 1)

    doo = -einsum('aj,ia->ij', lvo, t1)
    doo -= 0.5*einsum('abjk,ikab->ij', lvvoo, t2aa)
    doo -= einsum('aAjI,iIaA->ij', lvVoO, t2)

    dov = einsum('bj,ijab->ia', lvo, t2aa)
    dov -= einsum('bj,ib,ja->ia', lvo, t1, t1)
    dov -= 0.5*einsum('bcjk,ja,ikbc->ia', lvvoo, t1, t2aa)
    dov -= 0.5*einsum('bcjk,ib,jkac->ia', lvvoo, t1, t2aa)
    dov -= einsum('bAjI,ja,iIbA->ia', lvVoO, t1, t2)
    dov -= einsum('bAjI,ib,jIaA->ia', lvVoO, t1, t2)
    dov += einsum('AI,iIaA->ia', lvo, t2)

    dvo = lvo.copy()

    dvv = einsum('ai,ib->ab', lvo, t1)
    dvv += 0.5*einsum('acij,ijbc->ab', lvvoo, t2aa)
    dvv += einsum('aAiI,iIbA->ab', lvVoO, t2)

    g = np.zeros((nocc+nvir, nocc+nvir))
    g[:nocc, :nocc] = doo
    g[:nocc, nocc:] = dov
    g[nocc:, :nocc] = dvo
    g[nocc:, nocc:] = dvv
    return np.sqrt(2.0) * g


def _ee_overlap(l1, l2, r1, r2):
    '''Singlet EE biorthogonal metric <L|R>.'''
    return einsum('ia,ia', l1, r1) + 0.5*einsum('ijab,ijab', l2, 2*r2 - r2.transpose(0, 1, 3, 2))


def _ee_metric(myeom, lk, rk):
    nocc, nmo = myeom.nocc, myeom.nmo
    l1, l2 = myeom.vector_to_amplitudes(lk, nmo, nocc)
    r1, r2 = myeom.vector_to_amplitudes(rk, nmo, nocc)
    # <L|R> (so the normalized state has Tr(gamma)=N)
    return l1, l2, _ee_overlap(l1, l2, r1, r2)


def _ee_r0(gl1, gl2, r1, r2):
    '''Reference (vacuum) weight of a singlet EOM-EE right vector: r0 = -<Lambda|R>.
    The sqrt(2) is the singlet single-excitation spin-adaptation factor.'''
    return -np.sqrt(2.0) * _ee_overlap(gl1, gl2, r1, r2)


def eeccsd_rdm1_LR(myeom, r, l, t1=None, t2=None, with_r0=True):
    '''RHF singlet EOM-EE state 1-RDM <Psi_k|p^dag q|Psi_k> (Tr = N).

    This is the <L|R> expectation density, NOT the zeta-relaxed property density.
    with_r0=True (default) includes the reference-weight term of the full right
    eigenvector R_hat = r0*1 + R: + r0 * <L|p^dag q|1>, with r0 = -<Lambda|R>
    (Stanton-Gauss).  r0 is nonzero only for totally-symmetric states; with_r0=False
    drops it and gives only the bare R1+R2 (excitation-manifold) density.'''
    cc = myeom._cc
    if t1 is None:
        t1, t2 = cc.t1, cc.t2
    nocc, nmo = myeom.nocc, myeom.nmo
    r1, r2 = myeom.vector_to_amplitudes(r, nmo, nocc)
    l1, l2, lr = _ee_metric(myeom, l, r)
    l1, l2 = l1/lr, l2/lr
    g = make_rdm1_ee(t1, t2, l1, l2, r1, r2)
    if with_r0:
        gl1, gl2 = _ground_lambda(cc)
        g = g + _ee_r0(gl1, gl2, r1, r2) * make_tdm1_ee_k0(t1, t2, l1, l2)
    return g


def eeccsd_tdm1(myeom, rk, lk, rkp, lkp, t1=None, t2=None, with_r0=True):
    '''State-to-state transition 1-RDM between two RHF singlet EOM-EE states.

    with_r0=True (default) includes the reference-weight term of the ket
    eigenvector k' (r0_k' = -<Lambda|R_k'>), required for the true transition
    density when the ket state is totally-symmetric.  See eeccsd_rdm1_LR.'''
    cc = myeom._cc
    if t1 is None:
        t1, t2 = cc.t1, cc.t2
    nocc, nmo = myeom.nocc, myeom.nmo
    l1, l2, lr = _ee_metric(myeom, lk, rk)
    l1, l2 = l1/lr, l2/lr
    r1, r2 = myeom.vector_to_amplitudes(rkp, nmo, nocc)
    g = make_rdm1_ee(t1, t2, l1, l2, r1, r2)
    if with_r0:
        gl1, gl2 = _ground_lambda(cc)
        g = g + _ee_r0(gl1, gl2, r1, r2) * make_tdm1_ee_k0(t1, t2, l1, l2)
    return g


def eeccsd_tdm1_ground(myeom, r, l=None, t1=None, t2=None, with_r0=True):
    '''RHF singlet ground<->excited EOM-EE transition 1-RDMs (oscillator strengths).

    Returns (tdm_0k, tdm_k0):
      tdm_0k = <Psi_0| p^dag q |Psi_k>   (absorption, ground Lambda + right R)
      tdm_k0 = <Psi_k| p^dag q |Psi_0>   (emission, left L)
    `r` = right vector; `l` = left vector (defaults to r).  The right vector is used
    unit-normalized and the left is biorthonormalized to it (<L|R>=1).
    with_r0=True (default) adds the r0*<Psi_0|p^dag q|Psi_0> term to tdm_0k, making
    <Psi_0|Psi_k>=0 (Tr(tdm_0k)=0) so the transition moment is origin-independent
    (the physical value, matching FCI).  tdm_k0 is unaffected (left l0=0).  r0 is
    nonzero only for totally-symmetric excited states; with_r0=False gives the bare
    R1+R2 transition density (origin-dependent when r0 != 0).'''
    cc = myeom._cc
    if t1 is None:
        t1, t2 = cc.t1, cc.t2
    gl1, gl2 = _ground_lambda(cc)
    nocc, nmo = myeom.nocc, myeom.nmo
    r1, r2 = myeom.vector_to_amplitudes(r, nmo, nocc)
    lvec = r if l is None else l
    l1, l2, lr = _ee_metric(myeom, lvec, r)
    tdm_0k = make_tdm1_ee_0k(t1, t2, r1, r2, gl1, gl2)
    tdm_k0 = make_tdm1_ee_k0(t1, t2, l1/lr, l2/lr)
    if with_r0:
        tdm_0k = tdm_0k + _ee_r0(gl1, gl2, r1, r2) * cc.make_rdm1()
    return tdm_0k, tdm_k0


from pyscf.cc import eom_rccsd as _eom_rccsd
_eom_rccsd.EOMIP.make_rdm1_LR = ipccsd_rdm1_LR
_eom_rccsd.EOMEA.make_rdm1_LR = eaccsd_rdm1_LR
_eom_rccsd.EOMIP.make_dyson = ipccsd_dyson
_eom_rccsd.EOMEA.make_dyson = eaccsd_dyson
_eom_rccsd.EOMIP.make_tdm1 = ipccsd_tdm1
_eom_rccsd.EOMEA.make_tdm1 = eaccsd_tdm1
_eom_rccsd.EOMEESinglet.make_rdm1_LR = eeccsd_rdm1_LR
_eom_rccsd.EOMEESinglet.make_tdm1 = eeccsd_tdm1
_eom_rccsd.EOMEESinglet.make_tdm1_ground = eeccsd_tdm1_ground
