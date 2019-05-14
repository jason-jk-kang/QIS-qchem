import os
from numpy import array, concatenate, zeros
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

# Set record list for plotting. Existing information input from ProjectQ simulator.
bond_lengths = [0.73]
fci_energies = [-1.603565128035238, -1.6004199263636436]
UCCSD_energies = [-1.5836999664044602, -1.5771459927119653]

#initial force as calculated by fci
force_list = [0.041936022287925389248029717233852330706289645136440941719]

distance_counter = 0.8

initial_velocity = 1.10305*10**(-30)

mass = 1.6735575*10**(-27)

time = 1

bond_lengths += [distance_counter]

force = force_list[-1]

acceleration = (force/mass * 4.359744650*(10**(-28)) * ((10**10)**2) *
    (2.41888*10**(-17))**2)

distance_counter += acceleration*1/2*time**2 + initial_velocity*time


geometry = [('H', (0., 0., 0.)), ('H', (0., 0., distance_counter)),
            ('H', (0., 0., 3.3))]

# Generate and populate instance of MolecularData.
molecule = MolecularData(geometry, basis, spin, description="h3")

molecule = run_pyscf(molecule,
                     run_scf=run_scf,
                     run_mp2=run_mp2,
                     run_cisd=run_cisd,
                     run_ccsd=run_ccsd,
                     run_fci=run_fci)

# Use a Jordan-Wigner encoding, and compress to remove 0 imaginary components
molecular_hamiltonian = molecule.get_molecular_hamiltonian(
    occupied_indices=range(active_space_start),
    active_indices=range(active_space_start, active_space_stop))

fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)
qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
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
# UCCSD_energies += [float(opt_energy)]

force_list += [(fci_energies[-1] - fci_energies[-2])/(distance_counter - bond_lengths[-1])]

print("\n Results for {}:".format(molecule.name))
print("Optimal UCCSD Singlet Energy: {}".format(opt_energy))
print("Optimal UCCSD Singlet Amplitudes: {}".format(opt_amplitudes))
print("Classical CCSD Energy: {} Hartrees".format(molecule.ccsd_energy))
print("Exact FCI Energy: {} Hartrees".format(molecule.fci_energy))
print("Initial Energy of UCCSD with CCSD amplitudes: {} Hartrees".format(initial_energy))


print("This is bond lengths:", bond_lengths)

print("This is force lists:", force_list)

adjusted_lengths = [1/2(b - a) for a, b in zip(bond_lengths, bond_lengths[1:])]

print("This is adjusted lengths:", adjusted_lengths)

f1 = plt.figure(0)
plt.plot(adjusted_lengths, force_list, 'x-')
plt.ylabel('Force in Hartree/angstrom')
plt.xlabel('Bond length in angstrom')

plt.savefig("Force-Propogation-h3-graph", dpi=400, orientation='portrait')

plt.show()