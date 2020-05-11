import numpy
import scipy
import scipy.linalg

from numpy import array, concatenate, zeros
from numpy.random import randn
from scipy.optimize import minimize

from openfermion.config import *
from openfermionprojectq import *

from openfermion.hamiltonians import MolecularData
from openfermion.transforms import jordan_wigner, get_fermion_operator, get_sparse_operator
from openfermion.utils import uccsd_singlet_paramsize, uccsd_singlet_get_packed_amplitudes

from projectq.ops import X, All, Measure
from projectq.backends import CommandPrinter, CircuitDrawer

import matplotlib.pyplot as plt

from openfermionpyscf import run_pyscf

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

fci_energies = []
VQE_energies = []

# Set Hamiltonian parameters.
active_space_start = 1
active_space_stop = 3

# Set calculation parameters.
run_scf = 1
run_mp2 = 1
run_cisd = 0
run_ccsd = 1
run_fci = 1
delete_input = True
delete_output = True


'''
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Different ways to choose our initial Amplitudes
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
'''
# Naive initial guess
amplitudes = [0.01] * 5
'# # # # # # # # # # # # # # # # # # # # # # # # # # # #'


f = open('ProjectQ-h3-results.txt', 'w')

for point in range(1, n_points + 1):
    bond_length = bond_length_interval * float(point) + 0.2
    bond_lengths += [bond_length]
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., bond_length)), ('H', (0., 0., 3.3))]

    # Generate and populate instance of MolecularData.
    molecule = MolecularData(geometry, basis, spin, charge = 0, description="h3")

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

    compiler_engine = uccsd_trotter_engine()

    '''
    # This resets the amplitudes to be naive guess every time
    n_amplitudes = uccsd_singlet_paramsize(molecule.n_qubits, molecule.n_electrons)
    initial_amplitudes = [0.01] * n_amplitudes
    '''
    initial_energy = energy_objective(amplitudes)

    # Choosing UCCSD amplitudes, nearly perfect
    packed_amplitudes = uccsd_singlet_get_packed_amplitudes(
               molecule.ccsd_single_amps,
               molecule.ccsd_double_amps,
               molecule.n_qubits,
               molecule.n_electrons)

    amplitudes = array(packed_amplitudes)*(-1.)

    # Run VQE Optimization to find new CCSD parameters
    opt_result = minimize(energy_objective, amplitudes,
                          method="CG", options={'disp':True})

    opt_energy, opt_amplitudes = opt_result.fun, opt_result.x

    'This chooses our initial guess for the next iteration as the optimal amplitudes of the last iteration'
    amplitudes = opt_amplitudes
    fci_energies += [float(molecule.fci_energy)]
    VQE_energies += [float(opt_energy)]

    # Write Results
    f = open('ProjectQ-h3-results.txt', 'a')
    f.write("Results for {} at bond length {}:\n".format(molecule.name, str(bond_length)))
    f.write("Optimal VQE Energy: {} \n".format(str(opt_energy)))
    f.write("Optimal VQE Amplitudes: ")
    for item in opt_amplitudes:
        f.write("{} ".format(str(item)))
    f.write("\n")

    f.write("Exact FCI Energy: {} Hartrees \n".format(str(float(molecule.fci_energy))))
    f.write("Initial Energy of VQE: {} Hartrees \n\n".format(str(float(initial_energy))))

    f.close

    # Print Results
    print("\nResults for {} at bond length {}:".format(molecule.name, str(bond_length)))
    print("Optimal VQE Energy: {}".format(opt_energy))
    print("Optimal VQE Amplitudes: {}".format(opt_amplitudes))
    print("Classical CCSD Energy: {} Hartrees".format(molecule.ccsd_energy))
    print("Exact FCI Energy: {} Hartrees".format(molecule.fci_energy))
    print("Initial Energy: {} Hartrees\n\n".format(initial_energy))



# plot energies
f1 = plt.figure(0)
plt.plot(bond_lengths, fci_energies, ':', color = 'blue', label = "FCI")
plt.plot(bond_lengths, VQE_energies, '-', color = 'orange', label = "VQE")
plt.ylabel('Energy in Hartree')
plt.xlabel('Position of h2 atom in angstrom')
plt.legend()

plt.savefig("ProjectQ-Original-energy-graph", dpi=400, orientation='portrait')

plt.show()


energy_delta = [b - a for a, b in zip(fci_energies, fci_energies[1:])]
print ("This is the energy_delta:", energy_delta)

length_delta = [b - a for a, b in zip(bond_lengths, bond_lengths[1:])]
print ("This is the length_delta:", length_delta)

force = [-a/b for a, b in zip(energy_delta, length_delta)]
print ("This is the force:", force)

lengths = [a + 1/2*(b - a) for a, b in zip(bond_lengths, bond_lengths[1:])]
print ("This is the lengths:",lengths)

f2 = plt.figure(0)
plt.plot(lengths, force, '-')
plt.ylabel('Force in Hartree / angstrom')
plt.xlabel('Position of h2 atom in angstrom')

plt.savefig("ProjectQ-Original-force-graph", dpi=400, orientation='portrait')

plt.show()
