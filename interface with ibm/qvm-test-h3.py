#This I use to test QVMConnection


from openfermion.hamiltonians import MolecularData
from openfermion.transforms import get_fermion_operator, get_sparse_operator, jordan_wigner
from openfermion.utils import get_ground_state
from forestopenfermion import exponentiate
from pyquil.quil import Program
from pyquil.api import QVMConnection, WavefunctionSimulator
from pyquil.gates import *
from openfermionpyscf import run_pyscf

import numpy
import scipy
import scipy.linalg

# Load saved file for H2 + H.
basis = 'sto-3g'
spin = 2
n_points = 40
bond_length_interval = 3.0 / n_points
bond_lengths = []

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

for point in range(1, n_points + 1):
    bond_length = bond_length_interval * float(point) + 0.2
    bond_lengths += [bond_length]
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., bond_length)), ('H', (0., 0., 3.3))]

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

    pyquil_program = exponentiate(qubit_hamiltonian)
    qvm = WavefunctionSimulator()
    wf = qvm.wavefunction(pyquil_program)
    print('The {} has wavefunction amplitudes of'.format(molecule.name))
    print(wf.amplitudes)

    f = open('h3.wavefunction-amplitudes.txt', 'a')
    f.write('The {} has wavefunction aplitudes of'.format(molecule.name))
    for item in wf.amplitudes:
        f.write("%s\n" % item)
    f.close()

    print("success")
