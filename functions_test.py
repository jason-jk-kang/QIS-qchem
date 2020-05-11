import os
from numpy import array, concatenate, zeros
import numpy as np
from numpy.random import randn
from scipy.optimize import minimize
from openfermion.config import *
from openfermionprojectq import *
from openfermion.hamiltonians import MolecularData
from openfermion.transforms import jordan_wigner, get_fermion_operator, get_sparse_operator
from openfermion.utils import uccsd_singlet_paramsize
from projectq.ops import X, All, Measure
from projectq.backends import CommandPrinter, CircuitDrawer, IBMBackend
from pyscf import mp, fci
import matplotlib.pyplot as plt
from openfermionpyscf import run_pyscf
from openfermion.utils._unitary_cc import (uccsd_generator,
                                           uccsd_singlet_generator,
                                           uccsd_singlet_get_packed_amplitudes,
                                           uccsd_singlet_paramsize)


def energy_objective(packed_amplitudes, molecule, qubit_hamiltonian, compiler_engine):
    """Evaluate the energy of a UCCSD singlet wavefunction with packed_amplitudes
    Args:
        packed_amplitudes(ndarray): Compact array that stores the unique
            amplitudes for a UCCSD singlet wavefunction.
    Returns:
        energy(float): Energy corresponding to the given amplitudes
    """
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    # Set Jordan-Wigner initial state with correct number of electrons
    wavefunction = compiler_engine.allocate_qureg(molecule.n_qubits)
    for i in range(molecule.n_electrons):
        X | wavefunction[i]

    # Build the circuit and act it on the wavefunction
    evolution_operator = uccsd_singlet_evolution(packed_amplitudes,
                                                 molecule.n_qubits,
                                                 molecule.n_electrons)
    evolution_operator | wavefunction
    compiler_engine.flush()

    # Evaluate the energy and reset wavefunction
    energy = compiler_engine.backend.get_expectation_value(qubit_hamiltonian, wavefunction)
    All(Measure) | wavefunction
    compiler_engine.flush()
    return energy
    

basis = 'sto-3g'
spin = 1

# Set calculation parameters.
run_scf = 1
run_mp2 = 1
run_cisd = 1
run_ccsd = 1
run_fci = 1

geometry = [('H', (0., 0., 0)),
                    ('H', (0., 0., 1)),
                    ('H', (0., 0., 2))]

# Generate and populate instance of MolecularData.
molecule = MolecularData(geometry, basis, spin, charge = 1, description="h3")

molecule = run_pyscf(molecule,
                         run_scf=run_scf,
                         run_mp2=run_mp2,
                         run_cisd=run_cisd,
                         run_ccsd=run_ccsd,
                         run_fci=run_fci)

print(f'We have {molecule.n_electrons} electrons')
# Use a Jordan-Wigner encoding, and compress to remove 0 imaginary components
molecular_hamiltonian = molecule.get_molecular_hamiltonian()

fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)
qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
qubit_hamiltonian.compress()
compiler_engine = uccsd_trotter_engine()

packed_amplitudes = uccsd_singlet_get_packed_amplitudes(
            molecule.ccsd_single_amps,
            molecule.ccsd_double_amps,
            molecule.n_qubits,
            molecule.n_electrons)

opt_amplitudes = np.array(packed_amplitudes)*(-1.)

initial_energy = energy_objective(opt_amplitudes, molecule, qubit_hamiltonian, compiler_engine)
print ('Initial energy UCCSD', molecule.ccsd_energy)
print ('Initial amplitude UCCSD', opt_amplitudes)

# Run VQE Optimization to find new CCSD parameters
opt_result = minimize(energy_objective, opt_amplitudes, (molecule, qubit_hamiltonian, compiler_engine), method="CG", options={'disp':True})

opt_energy, opt_amplitudes = opt_result.fun, opt_result.x
    
print ('Final energy VQE', opt_energy)
print ('Final amplitude VQE', opt_amplitudes)

print('FCI energy', molecule.fci_energy)
print('CISD energy', molecule.cisd_energy)
print('SCF (Hartree-Fock) energy', molecule.hf_energy)

