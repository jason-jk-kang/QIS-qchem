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
    

def run_simulation (system, indx):
    # Load saved file for H3.
    basis = 'sto-3g'
    spin = 2

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

    # Begin Running Simulation, Convert distance_counter to angstroms
    if indx == None:
        geometry = [('H', (0., 0., system.atoms[0].position[-1] * 0.529177249)),
                    ('H', (0., 0., system.atoms[1].position[-1] * 0.529177249)),
                    ('H', (0., 0., system.atoms[2].position[-1] * 0.529177249))]
    elif indx == 0:
        geometry = [('H', (0., 0., system.atoms[0].stand_by_position * 0.529177249)),
                    ('H', (0., 0., system.atoms[1].position[-1] * 0.529177249)),
                    ('H', (0., 0., system.atoms[2].position[-1] * 0.529177249))]
                    
    elif indx == 1:
        geometry = [('H', (0., 0., system.atoms[0].position[-1] * 0.529177249)),
                    ('H', (0., 0., system.atoms[1].stand_by_position * 0.529177249)),
                    ('H', (0., 0., system.atoms[2].position[-1] * 0.529177249))]
                        
    elif indx == 2: 
        geometry = [('H', (0., 0., system.atoms[0].position[-1] * 0.529177249)),
                    ('H', (0., 0., system.atoms[1].position[-1] * 0.529177249)),
                    ('H', (0., 0., system.atoms[2].stand_by_position * 0.529177249))]

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
    initial_energy = energy_objective(system.opt_amplitudes, molecule, qubit_hamiltonian, compiler_engine)

    # Run VQE Optimization to find new CCSD parameters
    opt_result = minimize(energy_objective, system.opt_amplitudes, (molecule, qubit_hamiltonian, compiler_engine),
                          method="CG", options={'disp':True})

    opt_energy, system.opt_amplitudes = opt_result.fun, opt_result.x

    return ({"Name" : molecule.name, "UCCSD Energy" : opt_energy,
             "FCI Energy" : molecule.fci_energy})
