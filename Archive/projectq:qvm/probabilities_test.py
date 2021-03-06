# Attempt to pass commands from ProjectQ algorithm to ibm_backend

import os
from numpy import array, concatenate, zeros
from numpy.random import randn
from scipy.optimize import minimize
from openfermion.config import *
from openfermionprojectq import *
from openfermion.hamiltonians import MolecularData
from openfermion.transforms import jordan_wigner
from openfermion.utils import uccsd_singlet_paramsize
from projectq.ops import X, All, Measure
from projectq import MainEngine
from projectq.backends import CommandPrinter, CircuitDrawer, IBMBackend

from openfermionpyscf import run_pyscf


def energy_objective(packed_amplitudes):
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

    All(Measure) | wavefunction

    compiler_engine.flush()

    # new_engine = MainEngine(IBMBackend())
    # new_engine.backend.receive()

    results = compiler_engine.backend.get_probabilities(wavefunction)

    for state in results:
        print("Measured {} with p = {}".format(state, results[state]))

    return results


# Load saved file for H3.
basis = 'sto-3g'
spin = 2

# Set calculation parameters.
run_scf = 1
run_mp2 = 1
run_cisd = 0
run_ccsd = 0
run_fci = 1
delete_input = True
delete_output = True


# Set Hamiltonian parameters.
active_space_start = 1
active_space_stop = 3
geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.7414)), ('H', (0., 0., 3.3))]

# Generate and populate instance of MolecularData.
molecule = MolecularData(geometry, basis, spin, description="h3")

molecule = run_pyscf(molecule,
                     run_scf=run_scf,
                     run_mp2=run_mp2,
                     run_cisd=run_cisd,
                     run_ccsd=run_ccsd,
                     run_fci=run_fci)

# Use a Jordan-Wigner encoding, and compress to remove 0 imaginary components
qubit_hamiltonian = jordan_wigner(molecule.get_molecular_hamiltonian())
qubit_hamiltonian.compress()
# compiler_engine = uccsd_trotter_engine(CommandPrinter())
compiler_engine = uccsd_trotter_engine(compiler_backend=IBMBackend(user="jason_kang@college.harvard.edu",
                                                  password="987412365",
                                                  use_hardware=False, num_runs=1024,
                                                  verbose=False))

n_amplitudes = uccsd_singlet_paramsize(molecule.n_qubits, molecule.n_electrons)
initial_amplitudes = [0.01] * n_amplitudes
energy_objective(initial_amplitudes)






# # Run VQE Optimization to find new CCSD parameters
# opt_result = minimize(energy_objective, initial_amplitudes,
#                       method="CG", options={'disp':True})
#
# opt_energy, opt_amplitudes = opt_result.fun, opt_result.x
#
# print("\n Results for {}:".format(molecule.name))
# print("Optimal UCCSD Singlet Energy: {}".format(opt_energy))
# print("Optimal UCCSD Singlet Amplitudes: {}".format(opt_amplitudes))
# print("Classical CCSD Energy: {} Hartrees".format(molecule.ccsd_energy))
# print("Exact FCI Energy: {} Hartrees".format(molecule.fci_energy))
# print("Initial Energy of UCCSD with CCSD amplitudes: {} Hartrees".format(initial_energy))



# compiler_engine = uccsd_trotter_engine(CommandPrinter())

# wavefunction = compiler_engine.allocate_qureg(molecule.n_qubits)
# for i in range(molecule.n_electrons):
#     X | wavefunction[i]
#
# # Build the circuit and act it on the wavefunction
# evolution_operator = uccsd_singlet_evolution(opt_amplitudes,
#                                              molecule.n_qubits,
#                                              molecule.n_electrons)
# evolution_operator | wavefunction
# compiler_engine.flush()
# print("Measured {}".format(int(wavefunction)))
