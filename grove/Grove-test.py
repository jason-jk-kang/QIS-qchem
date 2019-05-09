from pyquil.quil import Program
import pyquil.api as api
from pyquil.gates import *

from openfermion.hamiltonians import MolecularData
from openfermion.transforms import jordan_wigner, get_fermion_operator, get_sparse_operator

from grove.pyvqe.vqe import VQE
from scipy.optimize import minimize
import numpy as np

from openfermionpyscf import run_pyscf
from forestopenfermion import qubitop_to_pyquilpauli


qvm = api.QVMConnection()

def small_ansatz(params):
    return Program(RX(params[0], 0))


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

pauli_hamiltonian = qubitop_to_pyquilpauli(qubit_hamiltonian)

vqe_inst = VQE(minimizer=minimize,
               minimizer_kwargs={'method': 'nelder-mead'})

# angle = 2.0
# result = vqe_inst.expectation(small_ansatz([angle]), pauli_hamiltonian, None, qvm)
# print(result)

angle_range = np.linspace(0.0, 2 * np.pi, 20)
data = [vqe_inst.expectation(small_ansatz([angle]), pauli_hamiltonian, None, qvm)
        for angle in angle_range]

import matplotlib.pyplot as plt
plt.xlabel('Angle [radians]')
plt.ylabel('Expectation value')
plt.plot(angle_range, data)
plt.show()










#
# n_amplitudes = uccsd_singlet_paramsize(molecule.n_qubits, molecule.n_electrons)
# initial_amplitudes = [0.01] * n_amplitudes
# initial_energy = energy_objective(initial_amplitudes)
#
# # Run VQE Optimization to find new CCSD parameters
# opt_result = minimize(energy_objective, initial_amplitudes,
#                       method="CG", options={'disp':True})
#
# opt_energy, opt_amplitudes = opt_result.fun, opt_result.x
#
#
# compiler_engine = uccsd_trotter_engine(compiler_backend=IBMBackend(user="jason_kang@college.harvard.edu",
#                                                   password="987412365",
#                                                   use_hardware=False, num_runs=1024,
#                                                   verbose=False))
#
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
