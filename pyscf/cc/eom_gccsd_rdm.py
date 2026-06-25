#!/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Spin-orbital (GHF) EOM-IP/EA-CCSD unrelaxed <L|R> one-particle density matrices
and Dyson orbitals, from eom_gccsd amplitudes (r1,r2,l1,l2,t1,t2) plus the
ground-state Lambda (gl1,gl2) for phi^L.

gamma[p,q] = <Psi_L| p^dag q |Psi_R>.  L,R are biorthonormalized to the metric
   <L|R> = l1.r1 + 0.5 l2_ija r2_ija   (IP),
   <L|R> = l1.r1 + 0.5 l2_iab r2_iab   (EA).

These are the canonical Stanton spin-orbital expressions (Wick&d-derived);
amplitudes use the pyscf eom_gccsd packing (r2/l2 antisymmetric in the two like
indices).  Cross-checked against the spin-adapted eom_rccsd_rdm module via the
spatial2spin spin trace (~1e-15).
"""

import numpy as np
from pyscf import lib

einsum = lib.einsum


def make_rdm1_ip(t1, t2, r1, r2, l1, l2):
    """Spin-orbital EOM-IP 1-RDM gamma[p,q]. r1(o),r2(o,o,v) antisym(oo); l likewise."""
    nocc, nvir = r1.shape[0], t1.shape[1]
    lvoo = l2.transpose(2, 0, 1)
    lr = l1.dot(r1) + 0.5*einsum('ija,ija', l2, r2)
    oo, vv = slice(0, nocc), slice(nocc, nocc+nvir)

    doo = -einsum('j,i->ij', l1, r1)
    doo += einsum('ajk,k,ia->ij', lvoo, r1, t1)
    doo -= einsum('ajk,ika->ij', lvoo, r2)

    dov = -einsum('j,i,ja->ia', l1, r1, t1)
    dov += einsum('j,j,ia->ia', l1, r1, t1)
    dov -= einsum('j,ija->ia', l1, r2)
    dov -= 0.5*einsum('bjk,i,jkab->ia', lvoo, r1, t2)
    dov += einsum('bjk,j,ikab->ia', lvoo, r1, t2)
    dov -= einsum('bjk,j,ib,ka->ia', lvoo, r1, t1, t1)
    dov -= 0.5*einsum('bjk,jka,ib->ia', lvoo, r2, t1)
    dov += einsum('bjk,ijb,ka->ia', lvoo, r2, t1)
    dov += 0.5*einsum('bjk,jkb,ia->ia', lvoo, r2, t1)

    dvo = -einsum('aij,j->ai', lvoo, r1)

    dvv = einsum('aij,i,jb->ab', lvoo, r1, t1)
    dvv += 0.5*einsum('aij,ijb->ab', lvoo, r2)

    g = np.zeros((nocc+nvir, nocc+nvir), dtype=np.result_type(t1, r1, l1))
    g[oo, oo] = doo + np.eye(nocc)*lr
    g[oo, vv] = dov
    g[vv, oo] = dvo
    g[vv, vv] = dvv
    return g


def make_rdm1_ea(t1, t2, r1, r2, l1, l2):
    """Spin-orbital EOM-EA 1-RDM gamma[p,q]. r1(v),r2(o,v,v) antisym(vv); l likewise."""
    nocc, nvir = t1.shape[0], r1.shape[0]
    lvvo = l2.transpose(1, 2, 0)
    lr = l1.dot(r1) + 0.5*einsum('iab,iab', l2, r2)
    oo, vv = slice(0, nocc), slice(nocc, nocc+nvir)

    doo = -einsum('abj,a,ib->ij', lvvo, r1, t1)
    doo -= 0.5*einsum('abj,iab->ij', lvvo, r2)

    dov = -einsum('b,a,ib->ia', l1, r1, t1)
    dov -= einsum('b,iab->ia', l1, r2)
    dov += einsum('b,b,ia->ia', l1, r1, t1)
    dov -= 0.5*einsum('bcj,a,ijbc->ia', lvvo, r1, t2)
    dov += einsum('bcj,jab,ic->ia', lvvo, r2, t1)
    dov += einsum('bcj,b,ijac->ia', lvvo, r1, t2)
    dov -= einsum('bcj,b,ic,ja->ia', lvvo, r1, t1, t1)
    dov -= 0.5*einsum('bcj,ibc,ja->ia', lvvo, r2, t1)
    dov += 0.5*einsum('bcj,jbc,ia->ia', lvvo, r2, t1)

    dvo = -einsum('abi,b->ai', lvvo, r1)

    dvv = einsum('a,b->ab', l1, r1)
    dvv += einsum('aci,ibc->ab', lvvo, r2)
    dvv -= einsum('aci,c,ib->ab', lvvo, r1, t1)

    g = np.zeros((nocc+nvir, nocc+nvir), dtype=np.result_type(t1, r1, l1))
    g[oo, oo] = doo + np.eye(nocc)*lr
    g[oo, vv] = dov
    g[vv, oo] = dvo
    g[vv, vv] = dvv
    return g


def _z_blocks(gl1, gl2):
    """Ground-state Lambda spin-orbital blocks for the left Dyson orbital."""
    z_vo = gl1.T
    z_vvoo = gl2.transpose(2, 3, 0, 1)
    return z_vo, z_vvoo


def make_dyson_ip(t1, t2, r1, r2, l1, l2, gl1, gl2):
    """Spin-orbital EOM-IP Dyson orbitals (phi_R, phi_L), length nmo each.

      phi_R[p] = <Psi_k^{N-1}| a_p |Psi_0>      (from EOM-left l1,l2)
      phi_L[p] = <Psi_0| a_p^dag |Psi_k^{N-1}>  (from EOM-right r1,r2 and ground Lambda)
    """
    lvoo = l2.transpose(2, 0, 1)
    z_vo, z_vvoo = _z_blocks(gl1, gl2)

    phiR_occ = l1.copy()
    phiR_vir = einsum('i,ia->a', l1, t1) + 0.5*einsum('bij,ijab->a', lvoo, t2)

    phiL_occ = r1.copy()
    phiL_occ -= einsum('j,ia,aj->i', r1, t1, z_vo)
    phiL_occ -= 0.5*einsum('j,ikab,abjk->i', r1, t2, z_vvoo)
    phiL_occ += einsum('ija,aj->i', r2, z_vo)
    phiL_occ += 0.5*einsum('jka,ib,abjk->i', r2, t1, z_vvoo)
    phiL_vir = einsum('i,ai->a', r1, z_vo)
    phiL_vir += 0.5*einsum('ijb,abij->a', r2, z_vvoo)

    phiR = np.concatenate([phiR_occ, phiR_vir])
    phiL = np.concatenate([phiL_occ, phiL_vir])
    return phiR, phiL


def make_dyson_ea(t1, t2, r1, r2, l1, l2, gl1, gl2):
    """Spin-orbital EOM-EA Dyson orbitals (phi_R, phi_L), length nmo each.

      phi_R[p] = <Psi_k^{N+1}| a_p^dag |Psi_0>  (from EOM-left l1,l2)
      phi_L[p] = <Psi_0| a_p |Psi_k^{N+1}>      (from EOM-right r1,r2 and ground Lambda)
    """
    lvvo = l2.transpose(1, 2, 0)
    z_vo, z_vvoo = _z_blocks(gl1, gl2)

    phiR_occ = -einsum('a,ia->i', l1, t1) - 0.5*einsum('abj,ijab->i', lvvo, t2)
    phiR_vir = l1.copy()

    phiL_occ = -einsum('a,ai->i', r1, z_vo)
    phiL_occ -= 0.5*einsum('jab,abij->i', r2, z_vvoo)
    phiL_vir = r1.copy()
    phiL_vir += einsum('iab,bi->a', r2, z_vo)
    phiL_vir -= einsum('b,ia,bi->a', r1, t1, z_vo)
    phiL_vir -= 0.5*einsum('b,ijac,bcij->a', r1, t2, z_vvoo)
    phiL_vir += 0.5*einsum('ibc,ja,bcij->a', r2, t1, z_vvoo)

    phiR = np.concatenate([phiR_occ, phiR_vir])
    phiL = np.concatenate([phiL_occ, phiL_vir])
    return phiR, phiL


# ---- EOM-class interface (vector unpacking + biorthonormalization) ----
def ipccsd_rdm1_LR(myeom, r, l, t1=None, t2=None):
    '''1-RDM for an EOM-IP-CCSD (GHF) state, LR part only: <L|p^dag q|R>.

    The <L|R> expectation density (Tr = N-1), NOT the zeta-relaxed property density.'''
    cc = myeom._cc
    if t1 is None:
        t1, t2 = cc.t1, cc.t2
    nocc, nmo = myeom.nocc, myeom.nmo
    r1, r2 = myeom.vector_to_amplitudes(r, nmo, nocc)
    l1, l2 = myeom.vector_to_amplitudes(l, nmo, nocc)
    lr = np.dot(l1, r1) + 0.5*einsum('ija,ija', l2, r2)
    return make_rdm1_ip(t1, t2, r1, r2, l1/lr, l2/lr)


def eaccsd_rdm1_LR(myeom, r, l, t1=None, t2=None):
    '''1-RDM for an EOM-EA-CCSD (GHF) state, LR part only: <L|p^dag q|R>.

    The <L|R> expectation density (Tr = N+1), NOT the zeta-relaxed property density.'''
    cc = myeom._cc
    if t1 is None:
        t1, t2 = cc.t1, cc.t2
    nocc, nmo = myeom.nocc, myeom.nmo
    r1, r2 = myeom.vector_to_amplitudes(r, nmo, nocc)
    l1, l2 = myeom.vector_to_amplitudes(l, nmo, nocc)
    lr = np.dot(l1, r1) + 0.5*einsum('iab,iab', l2, r2)
    return make_rdm1_ea(t1, t2, r1, r2, l1/lr, l2/lr)


def _ground_lambda(cc):
    if getattr(cc, 'l1', None) is None or getattr(cc, 'l2', None) is None:
        cc.solve_lambda()
    return cc.l1, cc.l2


def ipccsd_dyson(myeom, r, l, t1=None, t2=None):
    '''Spin-orbital Dyson orbitals (phi_R, phi_L) for an EOM-IP-CCSD (GHF) state.'''
    cc = myeom._cc
    if t1 is None:
        t1, t2 = cc.t1, cc.t2
    gl1, gl2 = _ground_lambda(cc)
    nocc, nmo = myeom.nocc, myeom.nmo
    r1, r2 = myeom.vector_to_amplitudes(r, nmo, nocc)
    l1, l2 = myeom.vector_to_amplitudes(l, nmo, nocc)
    lr = np.dot(l1, r1) + 0.5*einsum('ija,ija', l2, r2)
    return make_dyson_ip(t1, t2, r1, r2, l1/lr, l2/lr, gl1, gl2)


def eaccsd_dyson(myeom, r, l, t1=None, t2=None):
    '''Spin-orbital Dyson orbitals (phi_R, phi_L) for an EOM-EA-CCSD (GHF) state.'''
    cc = myeom._cc
    if t1 is None:
        t1, t2 = cc.t1, cc.t2
    gl1, gl2 = _ground_lambda(cc)
    nocc, nmo = myeom.nocc, myeom.nmo
    r1, r2 = myeom.vector_to_amplitudes(r, nmo, nocc)
    l1, l2 = myeom.vector_to_amplitudes(l, nmo, nocc)
    lr = np.dot(l1, r1) + 0.5*einsum('iab,iab', l2, r2)
    return make_dyson_ea(t1, t2, r1, r2, l1/lr, l2/lr, gl1, gl2)


def ipccsd_tdm1(myeom, rk, lk, rkp, lkp, t1=None, t2=None):
    '''State-to-state transition 1-RDM between two EOM-IP (GHF) states k and k'.'''
    cc = myeom._cc
    if t1 is None:
        t1, t2 = cc.t1, cc.t2
    nocc, nmo = myeom.nocc, myeom.nmo
    l1, l2 = myeom.vector_to_amplitudes(lk, nmo, nocc)
    lr1, lr2 = myeom.vector_to_amplitudes(rk, nmo, nocc)
    r1, r2 = myeom.vector_to_amplitudes(rkp, nmo, nocc)
    lr = np.dot(l1, lr1) + 0.5*einsum('ija,ija', l2, lr2)
    return make_rdm1_ip(t1, t2, r1, r2, l1/lr, l2/lr)


def eaccsd_tdm1(myeom, rk, lk, rkp, lkp, t1=None, t2=None):
    '''State-to-state transition 1-RDM between two EOM-EA (GHF) states k and k'.'''
    cc = myeom._cc
    if t1 is None:
        t1, t2 = cc.t1, cc.t2
    nocc, nmo = myeom.nocc, myeom.nmo
    l1, l2 = myeom.vector_to_amplitudes(lk, nmo, nocc)
    lr1, lr2 = myeom.vector_to_amplitudes(rk, nmo, nocc)
    r1, r2 = myeom.vector_to_amplitudes(rkp, nmo, nocc)
    lr = np.dot(l1, lr1) + 0.5*einsum('iab,iab', l2, lr2)
    return make_rdm1_ea(t1, t2, r1, r2, l1/lr, l2/lr)


# ================== EOM-EE (Stanton-Gauss spin-orbital, LR part) ==================
# le1,le2 = EOM-EE left amplitudes; r1,r2 = EOM-EE right; gl1,gl2 = ground Lambda.
# <L|R>_EE = l1.r1 + 0.25 l2.r2.  Verified vs Fock-space oracle to ~1e-16.

def make_rdm1_ee(t1, t2, le1, le2, r1, r2):
    """EOM-EE LR 1-RDM <L|p^dag q|R> (state RDM if same root; state-to-state if
    cross).  All amplitudes shape (o,v)/(o,o,v,v); l2,r2,t2 antisymmetric."""
    nocc, nvir = t1.shape
    lvo = le1.T
    lvvoo = le2.transpose(2, 3, 0, 1)
    lr = einsum('ia,ia', le1, r1) + 0.25*einsum('ijab,ijab', le2, r2)
    oo, vv = slice(0, nocc), slice(nocc, nocc+nvir)
    g = np.zeros((nocc+nvir,)*2, dtype=np.result_type(t1, r1, le1))
    doo = -einsum('aj,ia->ij', lvo, r1)
    doo += einsum('abjk,ka,ib->ij', lvvoo, r1, t1)
    doo -= 0.5*einsum('abjk,ikab->ij', lvvoo, r2)
    dov = -einsum('bj,ja,ib->ia', lvo, r1, t1)
    dov += einsum('bj,ijab->ia', lvo, r2)
    dov -= einsum('bj,ib,ja->ia', lvo, r1, t1)
    dov += einsum('bj,jb,ia->ia', lvo, r1, t1)
    dov -= 0.5*einsum('bcjk,ja,ikbc->ia', lvvoo, r1, t2)
    dov += 0.5*einsum('bcjk,jkab,ic->ia', lvvoo, r2, t1)
    dov -= 0.5*einsum('bcjk,ib,jkac->ia', lvvoo, r1, t2)
    dov += einsum('bcjk,jb,ikac->ia', lvvoo, r1, t2)
    dov -= einsum('bcjk,jb,ic,ka->ia', lvvoo, r1, t1, t1)
    dov += 0.5*einsum('bcjk,ijbc,ka->ia', lvvoo, r2, t1)
    dov += 0.25*einsum('bcjk,jkbc,ia->ia', lvvoo, r2, t1)
    dvo = einsum('abij,jb->ai', lvvoo, r1)
    dvv = einsum('ai,ib->ab', lvo, r1)
    dvv += 0.5*einsum('acij,ijbc->ab', lvvoo, r2)
    dvv -= einsum('acij,ic,jb->ab', lvvoo, r1, t1)
    g[oo, oo] = doo + np.eye(nocc)*lr
    g[oo, vv] = dov
    g[vv, oo] = dvo
    g[vv, vv] = dvv
    return g


def make_tdm1_ee_0k(t1, t2, r1, r2, gl1, gl2):
    """Ground<-excited (absorption) transition density <0|(1+Lam) p^dag q |R_k>."""
    nocc, nvir = t1.shape
    zvo = gl1.T
    zvvoo = gl2.transpose(2, 3, 0, 1)
    ov = einsum('ia,ia', gl1, r1) + 0.25*einsum('ijab,ijab', gl2, r2)
    oo, vv = slice(0, nocc), slice(nocc, nocc+nvir)
    g = np.zeros((nocc+nvir,)*2, dtype=np.result_type(t1, r1, gl1))
    doo = -einsum('ia,aj->ij', r1, zvo)
    doo += einsum('ka,ib,abjk->ij', r1, t1, zvvoo)
    doo -= 0.5*einsum('ikab,abjk->ij', r2, zvvoo)
    dov = r1.copy()
    dov -= einsum('ja,ib,bj->ia', r1, t1, zvo)
    dov -= 0.5*einsum('ja,ikbc,bcjk->ia', r1, t2, zvvoo)
    dov += einsum('ijab,bj->ia', r2, zvo)
    dov += 0.5*einsum('jkab,ic,bcjk->ia', r2, t1, zvvoo)
    dov -= einsum('ib,ja,bj->ia', r1, t1, zvo)
    dov -= 0.5*einsum('ib,jkac,bcjk->ia', r1, t2, zvvoo)
    dov += einsum('jb,ia,bj->ia', r1, t1, zvo)
    dov += einsum('jb,ikac,bcjk->ia', r1, t2, zvvoo)
    dov -= einsum('jb,ic,ka,bcjk->ia', r1, t1, t1, zvvoo)
    dov += 0.5*einsum('ijbc,ka,bcjk->ia', r2, t1, zvvoo)
    dov += 0.25*einsum('jkbc,ia,bcjk->ia', r2, t1, zvvoo)
    dvo = einsum('jb,abij->ai', r1, zvvoo)
    dvv = einsum('ib,ai->ab', r1, zvo)
    dvv += 0.5*einsum('ijbc,acij->ab', r2, zvvoo)
    dvv -= einsum('ic,jb,acij->ab', r1, t1, zvvoo)
    g[oo, oo] = doo + np.eye(nocc)*ov
    g[oo, vv] = dov
    g[vv, oo] = dvo
    g[vv, vv] = dvv
    return g


def make_tdm1_ee_k0(t1, t2, le1, le2):
    """Excited<-ground (emission) transition density <L_k| p^dag q |0>."""
    nocc, nvir = t1.shape
    lvo = le1.T
    lvvoo = le2.transpose(2, 3, 0, 1)
    oo, vv = slice(0, nocc), slice(nocc, nocc+nvir)
    g = np.zeros((nocc+nvir,)*2, dtype=np.result_type(t1, le1))
    doo = -einsum('aj,ia->ij', lvo, t1)
    doo -= 0.5*einsum('abjk,ikab->ij', lvvoo, t2)
    dov = einsum('bj,ijab->ia', lvo, t2)
    dov -= einsum('bj,ib,ja->ia', lvo, t1, t1)
    dov -= 0.5*einsum('bcjk,ja,ikbc->ia', lvvoo, t1, t2)
    dov -= 0.5*einsum('bcjk,ib,jkac->ia', lvvoo, t1, t2)
    dvo = lvo.copy()
    dvv = einsum('ai,ib->ab', lvo, t1)
    dvv += 0.5*einsum('acij,ijbc->ab', lvvoo, t2)
    g[oo, oo] = doo
    g[oo, vv] = dov
    g[vv, oo] = dvo
    g[vv, vv] = dvv
    return g


def _ee_r0(gl1, gl2, r1, r2):
    '''Reference (vacuum) weight of an EOM-EE right vector: r0 = -<Lambda|R>.
    Makes the full R_hat = r0*1 + R biorthogonal to the ground-state left <(1+Lam)|.'''
    return -(einsum('ia,ia', gl1, r1) + 0.25*einsum('ijab,ijab', gl2, r2))


def eeccsd_rdm1_LR(myeom, r, l, t1=None, t2=None, with_r0=True):
    '''EOM-EE-CCSD state 1-RDM <Psi_k|p^dag q|Psi_k> (Tr = N).

    NOTE this is the <L|R> expectation density, NOT the true unrelaxed property
    density -- the latter additionally needs the zeta (Z-vector) contribution.

    with_r0=True (default) includes the reference-weight (r0) term of the full right
    eigenvector R_hat = r0*1 + R (Stanton-Gauss): + r0 * <L|p^dag q|1>, with
    r0 = -<Lambda|R>.  r0 is nonzero only for totally-symmetric states; with_r0=False
    drops it and gives only the bare R1+R2 (excitation-manifold) density.'''
    cc = myeom._cc
    if t1 is None:
        t1, t2 = cc.t1, cc.t2
    nocc, nmo = myeom.nocc, myeom.nmo
    r1, r2 = myeom.vector_to_amplitudes(r, nmo, nocc)
    l1, l2 = myeom.vector_to_amplitudes(l, nmo, nocc)
    lr = einsum('ia,ia', l1, r1) + 0.25*einsum('ijab,ijab', l2, r2)
    l1, l2 = l1/lr, l2/lr
    g = make_rdm1_ee(t1, t2, l1, l2, r1, r2)
    if with_r0:
        gl1, gl2 = _ground_lambda(cc)
        g = g + _ee_r0(gl1, gl2, r1, r2) * make_tdm1_ee_k0(t1, t2, l1, l2)
    return g


def eeccsd_tdm1(myeom, rk, lk, rkp, lkp, t1=None, t2=None, with_r0=True):
    '''State-to-state transition 1-RDM between two EOM-EE states k and k'.

    with_r0=True (default) includes the reference-weight term of the ket
    eigenvector k' (r0_k' = -<Lambda|R_k'>), required for the true transition
    density when the ket state is totally-symmetric.  See eeccsd_rdm1_LR.'''
    cc = myeom._cc
    if t1 is None:
        t1, t2 = cc.t1, cc.t2
    nocc, nmo = myeom.nocc, myeom.nmo
    l1, l2 = myeom.vector_to_amplitudes(lk, nmo, nocc)
    lr1, lr2 = myeom.vector_to_amplitudes(rk, nmo, nocc)
    r1, r2 = myeom.vector_to_amplitudes(rkp, nmo, nocc)
    lr = einsum('ia,ia', l1, lr1) + 0.25*einsum('ijab,ijab', l2, lr2)
    l1, l2 = l1/lr, l2/lr
    g = make_rdm1_ee(t1, t2, l1, l2, r1, r2)
    if with_r0:
        gl1, gl2 = _ground_lambda(cc)
        g = g + _ee_r0(gl1, gl2, r1, r2) * make_tdm1_ee_k0(t1, t2, l1, l2)
    return g


def eeccsd_tdm1_ground(myeom, r, l=None, t1=None, t2=None, with_r0=True):
    '''Ground<->excited EOM-EE transition 1-RDMs (for oscillator strengths).

    Returns (tdm_0k, tdm_k0):
      tdm_0k = <Psi_0| p^dag q |Psi_k>   (absorption, uses ground Lambda + R)
      tdm_k0 = <Psi_k| p^dag q |Psi_0>   (emission, uses EOM-left L)
    `r` is the right (R) vector; `l` the left (L) vector (defaults to r if omitted).

    with_r0=True (default) adds the r0 reference-weight term to tdm_0k: + r0 * D_ground,
    with r0 = -<Lambda|R>.  This restores Tr(tdm_0k)=0 so the transition moment is
    origin-independent (the physical value, matching FCI).  tdm_k0 is unaffected
    (l0=0).  r0 is nonzero only for totally-symmetric states; with_r0=False gives the
    bare R1+R2 transition density (origin-dependent when r0 != 0).'''
    cc = myeom._cc
    if t1 is None:
        t1, t2 = cc.t1, cc.t2
    gl1, gl2 = _ground_lambda(cc)
    nocc, nmo = myeom.nocc, myeom.nmo
    r1, r2 = myeom.vector_to_amplitudes(r, nmo, nocc)
    lv = r if l is None else l
    l1, l2 = myeom.vector_to_amplitudes(lv, nmo, nocc)
    # biorthonormalize the left vector to its right partner (<L|R>=1) so tdm_k0
    # has the same normalization as tdm_0k (the EOM solver returns unit-norm vecs).
    lr = einsum('ia,ia', l1, r1) + 0.25*einsum('ijab,ijab', l2, r2)
    tdm_0k = make_tdm1_ee_0k(t1, t2, r1, r2, gl1, gl2)
    tdm_k0 = make_tdm1_ee_k0(t1, t2, l1/lr, l2/lr)
    if with_r0:
        tdm_0k = tdm_0k + _ee_r0(gl1, gl2, r1, r2) * cc.make_rdm1()
    return tdm_0k, tdm_k0


from pyscf.cc import eom_gccsd as _eom_gccsd
_eom_gccsd.EOMIP.make_rdm1_LR = ipccsd_rdm1_LR
_eom_gccsd.EOMEA.make_rdm1_LR = eaccsd_rdm1_LR
_eom_gccsd.EOMIP.make_dyson = ipccsd_dyson
_eom_gccsd.EOMEA.make_dyson = eaccsd_dyson
_eom_gccsd.EOMIP.make_tdm1 = ipccsd_tdm1
_eom_gccsd.EOMEA.make_tdm1 = eaccsd_tdm1
_eom_gccsd.EOMEE.make_rdm1_LR = eeccsd_rdm1_LR
_eom_gccsd.EOMEE.make_tdm1 = eeccsd_tdm1
_eom_gccsd.EOMEE.make_tdm1_ground = eeccsd_tdm1_ground
