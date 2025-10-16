import numpy
import numpy as np
from pyscf import scf
from pyscf.cc import ccsd
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf.cc.ccsd import _ChemistsERIs,BLKMIN,MEMORYMIN

IP_CONTINUUM, EA_CONTINUUM = [0,1]

class _CONTINUUM_ORBITAL_HF_:
    def __init__(self, mf, continuum_orbital):
        assert isinstance(mf, scf.hf.RHF)
        self._scf = mf
        self.__dict__.update(mf.__dict__)
        self.init_guess = "hcore"
        self.continuum_orbital = continuum_orbital.lower()
        assert self.continuum_orbital in ["occupied", "virtual", "ip", "ea"]
        if self.continuum_orbital == "occupied" or self.continuum_orbital == "ea":
            mol_tmp = mf.mol.copy()
            mol_tmp.charge -= 2
            mol_tmp.build()
            self.mol = mol_tmp
            self._scf.mol = mol_tmp
    
    def get_hcore(self, mol=None):
        hcore_deriv = self._scf.get_hcore(mol)
        n = hcore_deriv.shape[0]
        hcore = numpy.zeros((n+1, n+1))
        hcore[:n, :n] = hcore_deriv
        return hcore
    def get_ovlp(self, mol=None):
        ovlp_deriv = self._scf.get_ovlp(mol)
        n = ovlp_deriv.shape[0]
        ovlp = numpy.zeros((n+1, n+1))
        ovlp[:n, :n] = ovlp_deriv
        ovlp[n, n] = 1
        return ovlp
    def get_jk(self, mol, dm, *args, **kwargs):
        n = dm.shape[0]
        dm_tmp = dm[:n-1, :n-1]
        jk_deriv = self._scf.get_jk(mol, dm_tmp, *args, **kwargs)
        n = jk_deriv[0].shape[0]
        jk = numpy.zeros((2, n+1, n+1))
        jk[0, :n, :n] = jk_deriv[0]
        jk[1, :n, :n] = jk_deriv[1]
        return jk
    def get_occ(self, mo_energy, *args, **kwargs):
        occ_deriv = self._scf.get_occ(mo_energy, *args, **kwargs)
        if mo_energy is None:
            return occ_deriv
        nocc = sum(occ_deriv > 0)
        # check if the continuum orbital is occupied
        for i in range(len(mo_energy)):
            if abs(mo_energy[i]) < 1e-6:
                co_index = i
                break
        if self.continuum_orbital == "occupied" or self.continuum_orbital == "ea":
            if occ_deriv[co_index] == 0:
                occ_deriv[co_index] = 2.0
                occ_deriv[nocc - 1] = 0.0
        elif self.continuum_orbital == "virtual" or self.continuum_orbital == "ip":
            if occ_deriv[co_index] > 0:
                occ_deriv[co_index] = 0.0
                occ_deriv[nocc] = 2.0
        return occ_deriv
    def reorder_mo(self):
        for i in range(len(self.mo_energy)):
            if abs(self.mo_energy[i]) < 1e-6:
                co_index = i
                break
        nocc = sum(self.mo_occ > 0)
        def exchange(i, j):
            tmp = self.mo_coeff[:,i].copy()
            self.mo_coeff[:,i] = self.mo_coeff[:,j]
            self.mo_coeff[:,j] = tmp
            tmp = self.mo_energy[i]
            self.mo_energy[i] = self.mo_energy[j]
            self.mo_energy[j] = tmp
            tmp = self.mo_occ[i]
            self.mo_occ[i] = self.mo_occ[j]
            self.mo_occ[j] = tmp
            return
        if (self.continuum_orbital == "occupied" or self.continuum_orbital == "ea"):
            for i in range(co_index-1, 0, -1):
                if self.mo_occ[i] == 2:
                    break
                else:
                    exchange(i, i+1)
        elif (self.continuum_orbital == "virtual" or self.continuum_orbital == "ip"):
            for i in range(co_index+1, len(self.mo_occ)):
                if self.mo_occ[i] == 0:
                    break
                else:
                    exchange(i, i-1)
        else:
            return
        # reorder the mo




class _CONTINUUM_ORBITAL_CC_:
    def __init__(self, mycc):
        assert isinstance(mycc._scf, _CONTINUUM_ORBITAL_HF_)
        self._cc = mycc
        self.__dict__.update(mycc.__dict__)
        # find continuum index
        nfrozen = len(self._scf.mo_energy) - mycc.get_nmo()
        for i in range(len(self._scf.mo_energy) - nfrozen):
            if abs(self._scf.mo_energy[i + nfrozen]) < 1e-6:
                self.co_index = i
                break

    def ao2mo(self, mo_coeff = None):
        nmo = self.nmo
        nao = self.mo_coeff.shape[0]
        nmo_pair = nmo * (nmo+1) // 2
        nao_pair = nao * (nao+1) // 2
        mem_incore = (max(nao_pair**2, nmo**4) + nmo_pair**2) * 8/1e6
        mem_now = lib.current_memory()[0]
        if (self._scf._eri is not None and
            (mem_incore+mem_now < self.max_memory or self.incore_complete)):
            return _make_eris_incore_co(self, mo_coeff)
        elif getattr(self._scf, 'with_df', None):
            raise NotImplementedError
        else:
            return _make_eris_outcore_co(self, mo_coeff)
        
    def update_amps(self, t1, t2, eris):
        t1new, t2new = self._cc.update_amps(t1,t2,eris)
        # find continuum index and set to zero
        # to stablize convergence
        nocc = self.nocc
        co_index = self.co_index
        if co_index < nocc:
            t1new[co_index,:] = 0.0
            t2new[co_index,:,:,:] = 0.0
            t2new[:,co_index,:,:] = 0.0
        else:
            co_index -= nocc
            t1new[:,co_index] = 0.0
            t2new[:,:,co_index,:] = 0.0
            t2new[:,:,:,co_index] = 0.0
        return t1new, t2new

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None, eris=None):
        raise NotImplementedError

class _CONTINUUM_ORBITAL_EOMCC_:
    def __init__(self, eom):
        assert isinstance(eom._cc, _CONTINUUM_ORBITAL_CC_)
        self.__dict__.update(eom.__dict__)
        self._eom = eom

    def get_init_guess(self, nroots=1, koopmans=True, diag=None):
        # the initial guess must have continuum index
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', np.double)
        co_index = self._cc.co_index
        nocc = self._cc.nocc
        nvir = self._cc.nmo - nocc
        if (co_index < nocc and nroots > nvir)\
            or (co_index >= nocc and nroots > nocc):
            raise NotImplementedError

        guess = []
        for i in range(nroots):
            g = np.zeros(size, dtype)
            r1, r2 = self.vector_to_amplitudes(g)
            if co_index < nocc:
                r1[co_index,i] = 1.0
            else:
                r1[i,co_index-nocc] = 1.0
            g = self.amplitudes_to_vector(r1,r2)
            guess.append(g)
        return guess

    def matvec(self, vector, imds=None):
        def conti_mask(r1,r2,co_index,nocc):
            if co_index < nocc:
                r2[co_index,co_index,:,:] = 0.0
                for i in range(r1.shape[0]):
                    if i != co_index:
                        r1[i,:] = 0.0
                    for j in range(r1.shape[0]):
                        if i != co_index and j != co_index:
                            r2[i,j,:,:] = 0.0
            else:
                co_index -= nocc
                r2[:,:,co_index,co_index] = 0.0
                for i in range(r1.shape[1]):
                    if i != co_index:
                        r1[:,i] = 0.0
                    for j in range(r1.shape[1]):
                        if i != co_index and j != co_index:
                            r2[:,:,i,j] = 0.0
            return r1, r2
        nocc = imds.eris.nocc
        co_index = self._cc.co_index
        r1, r2 = self._eom.vector_to_amplitudes(vector)
        r1, r2 = conti_mask(r1,r2,co_index,nocc)
        vector_new = self._eom.matvec(self._eom.amplitudes_to_vector(r1,r2), imds)
        r1, r2 = self._eom.vector_to_amplitudes(vector_new)
        r1, r2 = conti_mask(r1,r2,co_index,nocc)
        return self._eom.amplitudes_to_vector(r1,r2)

def _make_eris_incore_co(mycc, mo_coeff):
    cput0 = (logger.process_clock(), logger.perf_counter())
    eris = _ChemistsERIs()
    mo_coeff = mycc.mo_coeff[:mycc.mo_coeff.shape[0]-1,:]
    eris._common_init_(mycc, mo_coeff)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc

    eris.mo_coeff = eris.mo_coeff[:eris.mo_coeff.shape[0]-1,:]
    eri1 = ao2mo.incore.full(mycc._scf._eri, eris.mo_coeff)
    if eri1.ndim == 4:
        eri1 = ao2mo.restore(4, eri1, nmo)

    nvir_pair = nvir * (nvir+1) // 2
    eris.oooo = numpy.empty((nocc,nocc,nocc,nocc))
    eris.ovoo = numpy.empty((nocc,nvir,nocc,nocc))
    eris.ovvo = numpy.empty((nocc,nvir,nvir,nocc))
    eris.ovov = numpy.empty((nocc,nvir,nocc,nvir))
    eris.ovvv = numpy.empty((nocc,nvir,nvir_pair))
    eris.vvvv = numpy.empty((nvir_pair,nvir_pair))

    ij = 0
    outbuf = numpy.empty((nmo,nmo,nmo))
    oovv = numpy.empty((nocc,nocc,nvir,nvir))
    for i in range(nocc):
        buf = lib.unpack_tril(eri1[ij:ij+i+1], out=outbuf[:i+1])
        for j in range(i+1):
            eris.oooo[i,j] = eris.oooo[j,i] = buf[j,:nocc,:nocc]
            oovv[i,j] = oovv[j,i] = buf[j,nocc:,nocc:]
        ij += i + 1
    eris.oovv = oovv
    oovv = None

    ij1 = 0
    for i in range(nocc,nmo):
        buf = lib.unpack_tril(eri1[ij:ij+i+1], out=outbuf[:i+1])
        eris.ovoo[:,i-nocc] = buf[:nocc,:nocc,:nocc]
        eris.ovvo[:,i-nocc] = buf[:nocc,nocc:,:nocc]
        eris.ovov[:,i-nocc] = buf[:nocc,:nocc,nocc:]
        eris.ovvv[:,i-nocc] = lib.pack_tril(buf[:nocc,nocc:,nocc:])
        dij = i - nocc + 1
        lib.pack_tril(buf[nocc:i+1,nocc:,nocc:],
                      out=eris.vvvv[ij1:ij1+dij])
        ij += i + 1
        ij1 += dij
    logger.timer(mycc, 'CCSD integral transformation', *cput0)
    return eris

def _make_eris_outcore_co(mycc, mo_coeff):
    from pyscf.scf.hf import RHF
    assert isinstance(mycc._scf, RHF)
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(mycc.stdout, mycc.verbose)
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    mol = mycc.mol
    mo_coeff = numpy.asarray(eris.mo_coeff[:eris.mo_coeff.shape[0]-1,:], order='F')
    nocc = eris.nocc
    nao, nmo = mo_coeff.shape
    nvir = nmo - nocc
    orbo = mo_coeff[:,:nocc]
    orbv = mo_coeff[:,nocc:]
    nvpair = nvir * (nvir+1) // 2
    eris.feri1 = lib.H5TmpFile()
    eris.oooo = eris.feri1.create_dataset('oooo', (nocc,nocc,nocc,nocc), 'f8')
    eris.oovv = eris.feri1.create_dataset('oovv', (nocc,nocc,nvir,nvir), 'f8', chunks=(nocc,nocc,1,nvir))
    eris.ovoo = eris.feri1.create_dataset('ovoo', (nocc,nvir,nocc,nocc), 'f8', chunks=(nocc,1,nocc,nocc))
    eris.ovvo = eris.feri1.create_dataset('ovvo', (nocc,nvir,nvir,nocc), 'f8', chunks=(nocc,1,nvir,nocc))
    eris.ovov = eris.feri1.create_dataset('ovov', (nocc,nvir,nocc,nvir), 'f8', chunks=(nocc,1,nocc,nvir))
    eris.ovvv = eris.feri1.create_dataset('ovvv', (nocc,nvir,nvpair), 'f8')

    def save_occ_frac(p0, p1, eri):
        eri = eri.reshape(p1-p0,nocc,nmo,nmo)
        eris.oooo[p0:p1] = eri[:,:,:nocc,:nocc]
        eris.oovv[p0:p1] = eri[:,:,nocc:,nocc:]

    def save_vir_frac(p0, p1, eri):
        eri = eri.reshape(p1-p0,nocc,nmo,nmo)
        eris.ovoo[:,p0:p1] = eri[:,:,:nocc,:nocc].transpose(1,0,2,3)
        eris.ovvo[:,p0:p1] = eri[:,:,nocc:,:nocc].transpose(1,0,2,3)
        eris.ovov[:,p0:p1] = eri[:,:,:nocc,nocc:].transpose(1,0,2,3)
        vvv = lib.pack_tril(eri[:,:,nocc:,nocc:].reshape((p1-p0)*nocc,nvir,nvir))
        eris.ovvv[:,p0:p1] = vvv.reshape(p1-p0,nocc,nvpair).transpose(1,0,2)

    cput1 = logger.process_clock(), logger.perf_counter()
    if not mycc.direct:
        max_memory = max(MEMORYMIN, mycc.max_memory-lib.current_memory()[0])
        eris.feri2 = lib.H5TmpFile()
        ao2mo.full(mol, orbv, eris.feri2, max_memory=max_memory, verbose=log)
        eris.vvvv = eris.feri2['eri_mo']
        cput1 = log.timer_debug1('transforming vvvv', *cput1)

    fswap = lib.H5TmpFile()
    max_memory = max(MEMORYMIN, mycc.max_memory-lib.current_memory()[0])
    int2e = mol._add_suffix('int2e')
    ao2mo.outcore.half_e1(mol, (mo_coeff,orbo), fswap, int2e,
                          's4', 1, max_memory, verbose=log)

    ao_loc = mol.ao_loc_nr()
    nao_pair = nao * (nao+1) // 2
    blksize = int(min(8e9,max_memory*.5e6)/8/(nao_pair+nmo**2)/nocc)
    blksize = min(nmo, max(BLKMIN, blksize))
    log.debug1('blksize %d', blksize)
    cput2 = cput1

    fload = ao2mo.outcore._load_from_h5g
    buf = numpy.empty((blksize*nocc,nao_pair))
    buf_prefetch = numpy.empty_like(buf)
    def load(buf_prefetch, p0, rowmax):
        if p0 < rowmax:
            p1 = min(rowmax, p0+blksize)
            fload(fswap['0'], p0*nocc, p1*nocc, buf_prefetch)

    outbuf = numpy.empty((blksize*nocc,nmo**2))
    with lib.call_in_background(load, sync=not mycc.async_io) as prefetch:
        prefetch(buf_prefetch, 0, nocc)
        for p0, p1 in lib.prange(0, nocc, blksize):
            buf, buf_prefetch = buf_prefetch, buf
            prefetch(buf_prefetch, p1, nocc)

            nrow = (p1 - p0) * nocc
            dat = ao2mo._ao2mo.nr_e2(buf[:nrow], mo_coeff, (0,nmo,0,nmo),
                                     's4', 's1', out=outbuf, ao_loc=ao_loc)
            save_occ_frac(p0, p1, dat)
        cput2 = log.timer_debug1('transforming oopp', *cput2)

        prefetch(buf_prefetch, nocc, nmo)
        for p0, p1 in lib.prange(0, nvir, blksize):
            buf, buf_prefetch = buf_prefetch, buf
            prefetch(buf_prefetch, nocc+p1, nmo)

            nrow = (p1 - p0) * nocc
            dat = ao2mo._ao2mo.nr_e2(buf[:nrow], mo_coeff, (0,nmo,0,nmo),
                                     's4', 's1', out=outbuf, ao_loc=ao_loc)
            save_vir_frac(p0, p1, dat)
            cput2 = log.timer_debug1('transforming ovpp [%d:%d]'%(p0,p1), *cput2)

    cput1 = log.timer_debug1('transforming oppp', *cput1)
    log.timer('CCSD integral transformation', *cput0)
    return eris


def continuum_orbital_mf(mf, continuum_orbital):
    '''For the given SCF object, update the matrix constructor with
    corresponding integrals with continuum orbitals.
    '''
    from pyscf import df
    from pyscf.scf import dhf
    assert (isinstance(mf, scf.hf.RHF))

    comf = _CONTINUUM_ORBITAL_HF_(mf, continuum_orbital)
    return lib.set_class(comf, (_CONTINUUM_ORBITAL_HF_, mf.__class__))

def continuum_orbital_cc(mycc):
    assert (isinstance(mycc._scf, scf.hf.RHF))
    cocc = _CONTINUUM_ORBITAL_CC_(mycc)
    return lib.set_class(cocc, (_CONTINUUM_ORBITAL_CC_, mycc.__class__))

def continuum_orbital_eom(mycc):
    assert (isinstance(mycc._scf, scf.hf.RHF))
    eom = mycc.EOMEESinglet()
    coeom = _CONTINUUM_ORBITAL_EOMCC_(eom)
    return lib.set_class(coeom, (_CONTINUUM_ORBITAL_EOMCC_, eom.__class__))


if __name__ == '__main__':
    from pyscf import gto, scf
    from pyscf.cc import dcsd
    import numpy as np
    
    mol = gto.Mole()
    mol.atom = """
    Au 0.0000000 0.0000000 -1.2651508
    Au 0.0000000 0.0000000 1.2651508
    """
    mol.basis = "def2-tzvpp"
    mol.ecp = "def2-tzvpp"
    mol.verbose = 4
    mol.build()

    # mf = scf.RHF(mol)
    # mf.kernel()
    # mycc = dcsd.DCSD(mf,frozen=2)
    # mycc.kernel()
    # myeom = mycc.EOMEA()
    # myeom.kernel(nroots=10)
    # print(myeom.e)
    # myeom = mycc.EOMIP()
    # myeom.kernel(nroots=10)
    # print(myeom.e)

    mf = scf.RHF(mol)
    mf_con = continuum_orbital_mf(mf,"ea")
    mf_con.kernel()
    mf_con.reorder_mo()
    print(mf_con.mo_energy)
    print(mf_con.mo_occ)
    mycc = continuum_orbital_cc(dcsd.DCSD(mf_con,frozen=2))
    mycc.kernel()
    myeom = continuum_orbital_eom(mycc)
    myeom.kernel(nroots=5)
    print(myeom.e)
