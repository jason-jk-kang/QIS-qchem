from pyquil.quil import Program
import pyquil.api as api
from pyquil.gates import *

from openfermion.hamiltonians import MolecularData
from openfermion.transforms import jordan_wigner, get_fermion_operator, get_sparse_operator

from grove.pyvqe.vqe import VQE
from scipy.optimize import minimize
import numpy as np



qvm = api.QVMConnection()

basis = 'sto-3g'
spin = 2

# Set Hamiltonian parameters.
active_space_start = 1
active_space_stop = 3

geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.7414)), ('H', (0., 0., 3.3))]

# Generate and populate instance of MolecularData.
molecule = MolecularData(geometry, basis, spin, description="h2+h")
molecule.load()

# Get the Hamiltonian in an active space.
molecular_hamiltonian = molecule.get_molecular_hamiltonian(
    occupied_indices=range(active_space_start),
    active_indices=range(active_space_start, active_space_stop))

# Map operator to fermions and qubits.
fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)
qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)

# compress removes 0 entries. qubit_hamiltonian is a qubit_operator
qubit_hamiltonian.compress()



vqe_inst = VQE(minimizer=minimize,
               minimizer_kwargs={'method': 'nelder-mead'})


#
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
