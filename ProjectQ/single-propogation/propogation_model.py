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
fci_force_list = [0]
UCCSD_force_list = [0]
bond_lengths = [1.37005]
fci_energies = []
UCCSD_energies = []
opt_amplitudes = [-5.7778375420113214e-08, -1.6441896890657683e-06, 9.223967507357728e-08, 0.03732738061624315, 1.5707960798368998]

# Initial Information Computed by FCI on nuclei position 1.51178 bohrs. velocity at 300K approx .394
velocity = [0.036]
mass = 1836
time = .5
counter = 1

while (counter < 1500) and (abs(bond_lengths[-1]) < 8):

    # Update lists
    if len(fci_energies) > 1:
        distance_delta = bond_lengths[-1] - bond_lengths[-2]
        fci_force_list += [-(fci_energies[-1] - fci_energies[-2])/distance_delta]
        UCCSD_force_list += [-(UCCSD_energies[-1] - UCCSD_energies[-2])/distance_delta]

    # Compute distance after force propogation
    bond_lengths += [bond_lengths[-1] + time*velocity[-1] + 0.5 * fci_force_list[-1]/mass * (time**2)]
    velocity += [velocity[-1] + fci_force_list[-1]/mass * time]

    # Print Simulation Information
    print("\nThis is function_run #{}".format(counter))
    print("distance: {} bohrs".format(bond_lengths[-1]))
    print("force:{}".format(fci_force_list[-1]))
    print("velocity:{}".format(velocity[-2]))

    # Begin Running Simulation, Convert distance_counter to angstroms
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., bond_lengths[-1] * 0.529177249)),
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
    print(fermion_hamiltonian)
    qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
    qubit_hamiltonian.compress()
    compiler_engine = uccsd_trotter_engine(CommandPrinter())
    initial_energy = energy_objective(opt_amplitudes)

    # Run VQE Optimization to find new CCSD parameters
    opt_result = minimize(energy_objective, opt_amplitudes,
                          method="CG", options={'disp':True})

    opt_energy, opt_amplitudes = opt_result.fun, opt_result.x

    fci_energies += [float(molecule.fci_energy)]
    UCCSD_energies += [float(opt_energy)]

    # Print Results
    print("\nResults for {}:".format(molecule.name))
    print("Optimal UCCSD Singlet Energy: {}".format(opt_energy))
    print("Exact FCI Energy: {} Hartrees".format(molecule.fci_energy))

    # Iterate counter
    counter += 1


# Adjust lists
fci_force_list = fci_force_list[1:]
UCCSD_force_list = UCCSD_force_list[1:]
bond_lengths = bond_lengths[1:]
adjusted_lengths = [a+1/2*(b - a) for a, b in zip(bond_lengths, bond_lengths[1:])]




# Plot Force Over Length
f0 = plt.figure(0)
plt.plot(adjusted_lengths[1:], fci_force_list, '-')
plt.plot(adjusted_lengths[1:], UCCSD_force_list, color='orange')
plt.ylabel('Force in Hartree / Bohrs')
plt.xlabel('Bond length in bohrs')

plt.savefig("FP-0-Force", dpi=400, orientation='portrait')

plt.show()



# Plot Energy Over Length
f2 = plt.figure(1)
plt.plot(bond_lengths, fci_energies, '-')
plt.plot(bond_lengths, UCCSD_energies, '-', color='orange')
plt.ylabel('Energy in Hartree')
plt.xlabel('Bond length in bohr')

plt.savefig("FP-0-Energy", dpi=400, orientation='portrait')

plt.show()



# Plot Distance Over Time
clock = [time]
for x in range(len(bond_lengths) - 1):
    clock += [clock[-1] + time]

f2 = plt.figure(1)
plt.plot(clock, bond_lengths, '-')
plt.ylabel('Distance in bohrs')
plt.xlabel('Time in au')

plt.savefig("FP-0-distance", dpi=400, orientation='portrait')

plt.show()





clock = [time]
for x in range(len(velocity) - 1):
    clock += [clock[-1] + time]

f3 = plt.figure(4)
plt.plot(clock, velocity, '-')
plt.ylabel('Velocity')
plt.xlabel('Time in au')

plt.savefig("FP-0-Velocity", dpi=400, orientation='portrait')

plt.show()
