import numpy
import scipy
import scipy.linalg
import sys
sys.path.append("../")

from numpy import array, concatenate, zeros
from numpy.random import randn
from scipy.optimize import minimize

from openfermion.config import *
from openfermionprojectq import *

from openfermion.hamiltonians import MolecularData
from openfermion.transforms import jordan_wigner, get_fermion_operator, get_sparse_operator
from openfermion.utils import uccsd_singlet_paramsize

from projectq import MainEngine

from projectq.ops import X, All, Measure
from projectq.backends import CommandPrinter, CircuitDrawer

import matplotlib.pyplot as plt
from openqasm import OpenQASMEngine

from openfermionpyscf import run_pyscf

from functions import sep_uccsd_singlet_evolution


"""
This is the original h3 calculation using a simple counter that moved the center nuclei
over a range of values. No propogation
"""

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
    evolution_operators = sep_uccsd_singlet_evolution(packed_amplitudes,
                                                 molecule.n_qubits,
                                                 molecule.n_electrons)
    print(evolution_operators["singles"])
    print(evolution_operators["doubles"])
    
    evolution_operator = evolution_operators["singles"]
    
    evolution_operator | wavefunction
    compiler_engine.flush()
    
    qc = backend.circuit
    print(len(qc))
    print(type(qc))
    print(qc.count_ops())

    # Evaluate the energy and reset wavefunction
    energy = compiler_engine.backend.get_expectation_value(qubit_hamiltonian, wavefunction)
    All(Measure) | wavefunction
    compiler_engine.flush()
    return energy


# Load saved file for H2 + H.
basis = 'sto-3g'
spin = 2
n_points = 40

# Set Hamiltonian parameters.
active_space_start = 1
active_space_stop = 3

# Set calculation parameters.
run_scf = 1
run_mp2 = 1
run_cisd = 0
run_ccsd = 0
run_fci = 1
delete_input = True
delete_output = True

geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.6)), ('H', (0., 0., 3.3))]
# geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.6))]


# Generate and populate instance of MolecularData.
molecule = MolecularData(geometry, basis, spin, description=str(round(0.6, 2)))

molecule = run_pyscf(molecule,
                     run_scf=run_scf,
                     run_mp2=run_mp2,
                     run_cisd=run_cisd,
                     run_ccsd=run_ccsd,
                     run_fci=run_fci)

# Get the Hamiltonian in an active space.
molecular_hamiltonian = molecule.get_molecular_hamiltonian(
    occupied_indices=range(active_space_start),
    active_indices=range(active_space_start, active_space_stop))

# Map operator to fermions and qubits.
fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)
qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)

# compress removes 0 entries. qubit_hamiltonian is a qubit_operator
qubit_hamiltonian.compress()



# Test OpenQASM
qc_list = []

def _process(circuit):
    qc_list.append(circuit)
        
backend = OpenQASMEngine(_process)
compiler_engine = uccsd_trotter_engine(compiler_backend=backend)





n_amplitudes = uccsd_singlet_paramsize(molecule.n_qubits, molecule.n_electrons)
initial_amplitudes = [0.01] * n_amplitudes
initial_energy = energy_objective(initial_amplitudes)




# Run VQE Optimization to find new CCSD parameters
opt_result = minimize(energy_objective, amplitudes,
                      method="CG", options={'disp':True})
opt_energy, opt_amplitudes = opt_result.fun, opt_result.x
amplitudes = opt_amplitudes



#qc = backend.circuit
#print(qc)
