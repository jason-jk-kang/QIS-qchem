import os
from numpy import array, concatenate, zeros
from numpy.random import randn
from scipy.optimize import minimize
from openfermion.config import *
from openfermionprojectq import *
from openfermion.hamiltonians import MolecularData
from openfermion.transforms import jordan_wigner, get_fermion_operator, get_sparse_operator
from openfermion.utils import uccsd_singlet_paramsize, uccsd_singlet_get_packed_amplitudes
from projectq.ops import X, All, Measure
from projectq.backends import CommandPrinter, CircuitDrawer, IBMBackend, Simulator
from pyscf import mp, fci
import matplotlib.pyplot as plt
from openfermionpyscf import run_pyscf
from projectq.meta import insert_engine

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

    # Prepares the wavefunction on our packed_amplitudes, generates a circuit
    evolution_operator = uccsd_singlet_evolution(packed_amplitudes,
                                                 molecule.n_qubits,
                                                 molecule.n_electrons)

    # print("what is this,", type(evolution_operator))
    # print("Evolution Operator")
    # print(evolution_operator)
    evolution_operator | wavefunction
    compiler_engine.flush()

    # Evaluate the energy and reset wavefunction
    # print("Qubit Hamiltonian", qubit_hamiltonian)
    energy = compiler_engine.backend.get_expectation_value(qubit_hamiltonian, wavefunction)
    All(Measure) | wavefunction
    compiler_engine.flush()
    return energy
    

def run_simulation (system, indx):
    # Load saved file for H3.
    basis = 'sto-3g'
    spin = 1

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
    molecule = MolecularData(geometry, basis, spin, charge = 1, description="h3")
    
    molecule = run_pyscf(molecule,
                         run_scf=run_scf,
                         run_mp2=run_mp2,
                         run_cisd=run_cisd,
                         run_ccsd=run_ccsd,
                         run_fci=run_fci)

    # Use a Jordan-Wigner encoding, and compress to remove 0 imaginary components
    molecular_hamiltonian = molecule.get_molecular_hamiltonian()

    fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)
    # print(fermion_hamiltonian)
    qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
    qubit_hamiltonian.compress()
    
    # # Messing with Engines
    # backend = IBMBackend()
    # compiler_engine = uccsd_trotter_engine(backend)
    # cmd_printer = CommandPrinter()
    # insert_engine(backend, cmd_printer)
    
    compiler_engine = uccsd_trotter_engine()
    
    packed_amplitudes = uccsd_singlet_get_packed_amplitudes(
           molecule.ccsd_single_amps,
           molecule.ccsd_double_amps,
           molecule.n_qubits,
           molecule.n_electrons)    
           
    ## Initialize the VQE with UCCSD amplitudes
    UCCSD_amplitudes = array(packed_amplitudes)*(-1.)
    initial_energy = energy_objective(UCCSD_amplitudes, molecule, qubit_hamiltonian, compiler_engine)
    print ('    Initial energy UCCSD:', molecule.ccsd_energy)
    print ('    Initial amplitude UCCSD:', UCCSD_amplitudes)

    # Run VQE Optimization to find new CCSD parameters
    opt_result = minimize(energy_objective, UCCSD_amplitudes, (molecule, qubit_hamiltonian, compiler_engine), method="CG", options={'disp':True})
    
    opt_energy, system.opt_amplitudes = opt_result.fun, opt_result.x    
    print ('    Final energy VQE', opt_energy)
    print ('    Final amplitude VQE', system.opt_amplitudes)
    print ('    FCI energy', molecule.fci_energy)
    print ('    CISD energy', molecule.cisd_energy)
    print ('    SCF (Hartree-Fock) energy', molecule.hf_energy)
    
    
    
    
    
    # Print commands. But this circuit is only up to the point where you prepare the wavefunction for the optimized amplitudes
    print("\n \nCommand Printer \n")
    compiler_engine = uccsd_trotter_engine(CommandPrinter())
    wavefunction = compiler_engine.allocate_qureg(molecule.n_qubits)
    for i in range(molecule.n_electrons):
        X | wavefunction[i]
    evolution_operator = uccsd_singlet_evolution(system.opt_amplitudes, 
                                                 molecule.n_qubits, 
                                                 molecule.n_electrons)
    evolution_operator | wavefunction
    compiler_engine.flush()
    
    
    
    
    
    


    return ({"Name" : molecule.name, "VQE Energy" : opt_energy,
             "FCI Energy" : molecule.fci_energy, "UCCSD Energy": molecule.ccsd_energy})
             
