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
from projectq.backends import CommandPrinter, CircuitDrawer

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
    compiler_engine.flush()

    # Evaluate the energy and reset wavefunction
    energy = compiler_engine.backend.get_expectation_value(qubit_hamiltonian, wavefunction)
    All(Measure) | wavefunction
    compiler_engine.flush()
    return energy






# Load saved file for H2 + H.
basis = 'sto-3g'
spin = 2
n_points = 40
bond_length_interval = 3.0 / n_points
bond_lengths = []

# Set Hamiltonian parameters.
active_space_start = 1
active_space_stop = 3

for point in range(1, n_points + 1):
    bond_length = bond_length_interval * float(point) + 0.2
    bond_lengths += [bond_length]
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., bond_length)), ('H', (0., 0., 3.3))]

    # Generate and populate instance of MolecularData.
    molecule = MolecularData(geometry, basis, spin, description=str(round(bond_length, 2)))
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

    compiler_engine = uccsd_trotter_engine()

    n_amplitudes = uccsd_singlet_paramsize(molecule.n_qubits, molecule.n_electrons)
    initial_amplitudes = [0, 0.05677]
    initial_energy = energy_objective(initial_amplitudes)

    # Run VQE Optimization to find new CCSD parameters
    opt_result = minimize(energy_objective, initial_amplitudes,
                          method="CG", options={'disp':True})

    opt_energy, opt_amplitudes = opt_result.fun, opt_result.x
    print("\nResults for {}:".format(molecule.name))
    print("Optimal UCCSD Singlet Energy: {}".format(opt_energy))
    print("Optimal UCCSD Singlet Amplitudes: {}".format(opt_amplitudes))
    print("Classical CCSD Energy: {} Hartrees".format(molecule.ccsd_energy))
    print("Exact FCI Energy: {} Hartrees".format(molecule.fci_energy))
    print("Initial Energy of UCCSD with CCSD amplitudes: {} Hartrees".format(initial_energy))
