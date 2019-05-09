# This runs the projectQ algorithm iterating over nuclei positions, plots results

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
from openfermion.utils import uccsd_singlet_paramsize

from projectq.ops import X, All, Measure
from projectq.backends import CommandPrinter, CircuitDrawer

import matplotlib.pyplot as plt

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
UCCSD_energies = []

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


f = open('ProjectQ-h3-results.txt', 'w')

for point in range(1, n_points + 1):
    bond_length = bond_length_interval * float(point) + 0.2
    bond_lengths += [bond_length]
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., bond_length)), ('H', (0., 0., 3.3))]

    # Generate and populate instance of MolecularData.
    molecule = MolecularData(geometry, basis, spin, description=str(round(bond_length, 2)))


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

    n_amplitudes = uccsd_singlet_paramsize(molecule.n_qubits, molecule.n_electrons)
    initial_amplitudes = [0.01] * n_amplitudes
    initial_energy = energy_objective(initial_amplitudes)

    # Run VQE Optimization to find new CCSD parameters
    opt_result = minimize(energy_objective, initial_amplitudes,
                          method="CG", options={'disp':True})

    opt_energy, opt_amplitudes = opt_result.fun, opt_result.x

    fci_energies += [float(molecule.fci_energy)]
    UCCSD_energies += [float(opt_energy)]

    print(fci_energies)
    print(UCCSD_energies)
    print(type(fci_energies))
    print(type(UCCSD_energies))

    # write results into txt file
    f = open('ProjectQ-h3-results.txt', 'a')
    f.write("Results for {}: \n".format(molecule.name))

    f.write("Optimal UCCSD Singlet Energy: {} \n".format(str(opt_energy)))
    f.write("Optimal UCCSD Singlet Amplitudes: ")
    for item in opt_amplitudes:
        f.write("{} ".format(str(item)))
    f.write("\n")

    f.write("Classical CCSD Energy: {} Hartrees \n".format(str(float(item))))
    f.write("Exact FCI Energy: {} Hartrees \n".format(str(float(molecule.fci_energy))))
    f.write("Initial Energy of UCCSD with CCSD amplitudes: {} Hartrees \n\n".format(str(float(initial_energy))))

    f.close

    print("\nResults for {}:".format(molecule.name))
    print("Optimal UCCSD Singlet Energy: {}".format(opt_energy))
    print("Optimal UCCSD Singlet Amplitudes: {}".format(opt_amplitudes))
    print("Classical CCSD Energy: {} Hartrees".format(molecule.ccsd_energy))
    print("Exact FCI Energy: {} Hartrees".format(molecule.fci_energy))
    print("Initial Energy of UCCSD with CCSD amplitudes: {} Hartrees\n\n".format(initial_energy))



# plot energies
plt.figure(0)
plt.plot(bond_lengths, fci_energies, 'x-')
plt.plot(bond_lengths, UCCSD_energies, 'o-')
plt.ylabel('Energy in Hartree')
plt.xlabel('Bond length in angstrom')

plt.savefig("ProjectQ-h3-graph", dpi=400, orientation='portrait')

plt.show()
