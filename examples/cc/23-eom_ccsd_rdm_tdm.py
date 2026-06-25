#!/usr/bin/env python

'''
EOM-CCSD reduced/transition density matrices and transition dipole moments
for an RHF reference (pyscf.cc.eom_rccsd).

Importing pyscf.cc.eom_rccsd_rdm attaches these methods to the EOM classes:

    EOMEESinglet : make_rdm1_LR, make_tdm1_ground, make_tdm1
    EOMIP/EOMEA  : make_rdm1_LR, make_dyson, make_tdm1

They evaluate the <L|p^dag q|R> expectation density in the MO basis (NOT the
zeta-relaxed property density).  Each needs both a right and a left EOM
eigenvector; the left vectors come from kernel(..., left=True).  Transition
densities and Dyson orbitals also need the ground-state Lambda amplitudes
(call mycc.solve_lambda() first).

The EE densities include a reference-weight (r0) term by default (with_r0=True);
see the comments at the make_tdm1_ground / make_tdm1 calls below.  IP/EA have no
r0 term (the reference and the target live in different particle-number sectors).
'''

import numpy as np
from pyscf import gto, scf, cc
from pyscf.cc import eom_rccsd
from pyscf.cc import eom_rccsd_rdm  # noqa: F401  attaches the methods used below
from pyscf.data.nist import HARTREE2EV as au2ev

mol = gto.M(
    atom='''O 0.0000000000  0.0000000000  0.0000000000
            H 0.0000000000  0.7568775067 -0.5860544071
            H 0.0000000000 -0.7568775067 -0.5860544071''',
    basis='cc-pvdz')
mf = scf.RHF(mol).run()
mycc = cc.CCSD(mf).run()
mycc.solve_lambda()          # ground-state Lambda, needed below

# Electronic dipole operator in the MO basis (for transition dipoles).
with mol.with_common_orig((0, 0, 0)):
    ao_dip = mol.intor_symmetric('int1e_r', comp=3)
dip_mo = np.einsum('xpq,pi,qj->xij', ao_dip, mf.mo_coeff, mf.mo_coeff)


def trans_dip(gamma):
    '''Electronic transition dipole (a.u.) from a transition 1-RDM in MO basis.'''
    return -np.einsum('xpq,qp->x', dip_mo, gamma)


################################################################################
# EOM-EE-CCSD (singlet): state 1-RDM, ground<->excited and state-to-state TDMs
################################################################################
eom_ee = eom_rccsd.EOMEESinglet(mycc)
nroots = 3
e_ee, r_ee = eom_ee.kernel(nroots=nroots)             # right eigenvectors
_,    l_ee = eom_ee.kernel(nroots=nroots, left=True)  # left eigenvectors

# (1) State 1-RDM of an excited state (trace = N).  with_r0=True (the default)
#     includes the reference weight of the right eigenvector, which is required for
#     the true state density of a totally-symmetric excited state (it is zero by
#     symmetry otherwise, so the default is harmless for other states).
dm1 = eom_ee.make_rdm1_LR(r_ee[0], l_ee[0])
print('EE state 1: Tr(rdm1) = %.6f  (N = %d)' % (np.trace(dm1).real, mol.nelectron))

# (2) Ground <-> excited transition densities -> transition dipole, oscillator
#     strength.  make_tdm1_ground returns (tdm_0k, tdm_k0) (absorption, emission).
#     The dipole strength mu_0k . mu_k0 (and thus f) is independent of the EOM
#     normalization gauge.
#
#     r0 reference-weight term (make_tdm1_ground(..., with_r0=...), EE only):
#     the full right excited eigenvector is R_hat = r0 + R1 + R2, r0 = -<Lambda|R>.
#     - with_r0=True (default, used here): adds r0 * <0|p^dag q|0> to tdm_0k,
#       enforcing <Psi_0|Psi_k>=0 (Tr(tdm_0k)=0).  This makes the transition moment
#       origin-independent -- the physical value (it matches FCI).
#     - with_r0=False: the bare R1+R2 transition density, which is origin-dependent
#       when r0 != 0.
#     r0 is nonzero only for totally-symmetric excited states, and only affects
#     tdm_0k; the emission side tdm_k0 (left vector, l0=0) is unchanged.
print('\nEE ground -> excited absorption:')
for k in range(nroots):
    tdm_0k, tdm_k0 = eom_ee.make_tdm1_ground(r_ee[k], l_ee[k])
    mu_0k, mu_k0 = trans_dip(tdm_0k), trans_dip(tdm_k0)
    fosc = 2./3. * e_ee[k] * np.dot(mu_0k, mu_k0)
    print('  0 -> %d   omega = %7.4f eV   f = %.5f' % (k + 1, e_ee[k] * au2ev, fosc))

# (3) State-to-state transition dipole between excited states 1 and 2.
#     with_r0=True (the default) includes the *ket* state's reference weight
#     (r0_k' * <k|p^dag q|0>), needed for the true state-to-state density when the
#     ket excited state is totally-symmetric (r0 != 0); harmless otherwise.
g_12 = eom_ee.make_tdm1(r_ee[0], l_ee[0], r_ee[1], l_ee[1])   # <1|p^dag q|2>
print('\nEE 1 -> 2 transition dipole (a.u.):',
      np.array2string(trans_dip(g_12), precision=5, suppress_small=True))


################################################################################
# EOM-IP-CCSD: state 1-RDM, Dyson orbitals, state-to-state TDM
################################################################################
eom_ip = eom_rccsd.EOMIP(mycc)
e_ip, r_ip = eom_ip.kernel(nroots=2)
_,    l_ip = eom_ip.kernel(nroots=2, left=True)

dm1_ip = eom_ip.make_rdm1_LR(r_ip[0], l_ip[0])
print('\nIP state 1: Tr(rdm1) = %.6f  (N-1 = %d)'
      % (np.trace(dm1_ip).real, mol.nelectron - 1))

# Dyson orbital (phi_R, phi_L) in the MO basis; pole strength = <phi_L|phi_R>.
phi_R, phi_L = eom_ip.make_dyson(r_ip[0], l_ip[0])
print('IP state 1: Dyson pole strength = %.4f' % np.dot(phi_L, phi_R))

# State-to-state transition density between IP states 1 and 2 (traceless).
g_ip = eom_ip.make_tdm1(r_ip[0], l_ip[0], r_ip[1], l_ip[1])
print('IP 1 -> 2 transition-density trace = %.1e (orthogonal -> ~0)' % np.trace(g_ip))


################################################################################
# EOM-EA-CCSD: state 1-RDM, Dyson orbitals
################################################################################
eom_ea = eom_rccsd.EOMEA(mycc)
e_ea, r_ea = eom_ea.kernel(nroots=2)
_,    l_ea = eom_ea.kernel(nroots=2, left=True)

dm1_ea = eom_ea.make_rdm1_LR(r_ea[0], l_ea[0])
print('\nEA state 1: Tr(rdm1) = %.6f  (N+1 = %d)'
      % (np.trace(dm1_ea).real, mol.nelectron + 1))

phi_R, phi_L = eom_ea.make_dyson(r_ea[0], l_ea[0])
print('EA state 1: Dyson pole strength = %.4f' % np.dot(phi_L, phi_R))
