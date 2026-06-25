#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

'''RHF EOM-EE(singlet) state / transition 1-RDMs and transition dipole moments.

Reference system: H2O / cc-pVDZ, molecule in the yz-plane (x is perpendicular
to the molecular plane, z is the C2 axis).  The first three singlet roots are
1B1 (n->3s, x-polarised, bright), 1A2 (dark from the ground state) and 1A1.

The signed components of a transition dipole depend on the (arbitrary) phase of
the EOM left/right eigenvectors, so the tests assert phase-invariant quantities:
the dipole strength mu_{0k}.mu_{k0} and oscillator strength, and the magnitude
of the dominant component.
'''

import unittest
import numpy as np

from pyscf import gto, scf, cc
from pyscf.cc import eom_rccsd, eom_rccsd_rdm  # noqa: F401  attaches make_tdm1*


def setUpModule():
    global mol, mf, mycc, eom, e, rv, lv, dip_mo
    mol = gto.M(
        atom='''O 0.0000000000  0.0000000000  0.0000000000
                H 0.0000000000  0.7568775067 -0.5860544071
                H 0.0000000000 -0.7568775067 -0.5860544071''',
        basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mycc = cc.CCSD(mf).run(conv_tol=1e-10)
    mycc.solve_lambda()

    # electronic position operator in the MO basis
    C = mf.mo_coeff
    with mol.with_common_orig((0, 0, 0)):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    dip_mo = np.einsum('xpq,pi,qj->xij', ao_dip, C, C)

    eom = eom_rccsd.EOMEESinglet(mycc)
    e, rv = eom.kernel(nroots=3)
    _, lv = eom.kernel(nroots=3, left=True)


def tearDownModule():
    global mol, mf, mycc, eom, e, rv, lv, dip_mo
    del mol, mf, mycc, eom, e, rv, lv, dip_mo


def tdip(gamma):
    '''electronic transition dipole vector from a (spatial) 1-TDM (a.u.)'''
    return -np.einsum('xpq,qp->x', dip_mo, gamma)


class KnownValues(unittest.TestCase):
    def test_excitation_energies(self):
        self.assertAlmostEqual(e[0], 0.3008196143, 6)
        self.assertAlmostEqual(e[1], 0.3761600080, 6)
        self.assertAlmostEqual(e[2], 0.3978780648, 6)

    def test_tdm_ground_to_excited(self):
        # 0 <-> 1 : ground <-> first excited (1B1), x-polarised
        d0k, dk0 = eom.make_tdm1_ground(rv[0], lv[0])
        mu_01 = tdip(d0k)   # <0|mu|1>  absorption
        mu_10 = tdip(dk0)   # <1|mu|0>  emission
        # transition is purely x-polarised (B1)
        self.assertAlmostEqual(abs(mu_01[0]), 0.365020815, 6)
        self.assertAlmostEqual(abs(mu_10[0]), 0.369064250, 6)
        self.assertAlmostEqual(np.linalg.norm(mu_01[1:]), 0, 7)
        self.assertAlmostEqual(np.linalg.norm(mu_10[1:]), 0, 7)
        # phase-invariant dipole strength and oscillator strength
        strength = float(np.dot(mu_01, mu_10))
        self.assertAlmostEqual(strength, 0.1347165574, 6)
        f_osc = 2.0/3.0 * e[0] * strength
        self.assertAlmostEqual(f_osc, 0.0270168368, 7)

    def test_tdm_state_to_state(self):
        # 1 <-> 2 : first excited (1B1) <-> second excited (1A2), y-polarised
        g_12 = eom.make_tdm1(rv[0], lv[0], rv[1], lv[1])   # <1|p^dag q|2>
        g_21 = eom.make_tdm1(rv[1], lv[1], rv[0], lv[0])   # <2|p^dag q|1>
        mu_12 = tdip(g_12)
        mu_21 = tdip(g_21)
        self.assertAlmostEqual(abs(mu_12[1]), 1.756637433, 6)
        self.assertAlmostEqual(abs(mu_21[1]), 1.750542162, 6)
        self.assertAlmostEqual(np.linalg.norm(mu_12[[0, 2]]), 0, 7)
        self.assertAlmostEqual(np.linalg.norm(mu_21[[0, 2]]), 0, 7)
        self.assertAlmostEqual(float(np.dot(mu_12, mu_21)), 3.0750686, 5)

    def test_rdm_traces(self):
        nelec = mol.nelectron
        # state 1-RDM (LR) is normalised to N
        g11 = eom.make_rdm1_LR(rv[0], lv[0])
        self.assertAlmostEqual(np.trace(g11).real, nelec, 7)
        # ground<->excited transition density is traceless (orthogonal states)
        d0k, dk0 = eom.make_tdm1_ground(rv[0], lv[0])
        self.assertAlmostEqual(np.trace(d0k), 0, 8)
        self.assertAlmostEqual(np.trace(dk0), 0, 8)
        # state-to-state transition density is traceless
        g_12 = eom.make_tdm1(rv[0], lv[0], rv[1], lv[1])
        self.assertAlmostEqual(np.trace(g_12), 0, 8)

    def test_tdm_ground_r0_orthogonality(self):
        # with_r0 enforces <Psi_0|Psi_k>=0 i.e. Tr(tdm_0k)=0 for every state,
        # including totally-symmetric ones where r0 != 0.
        for k in range(3):
            d0k_r0, _ = eom.make_tdm1_ground(rv[k], lv[k], with_r0=True)
            self.assertAlmostEqual(np.trace(d0k_r0), 0, 9)


if __name__ == '__main__':
    print('Full Tests for RHF EOM-EE transition densities / dipoles')
    unittest.main()
