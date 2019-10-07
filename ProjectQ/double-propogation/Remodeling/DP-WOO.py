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
from Init import *
import matplotlib.pyplot as plt
from openfermionpyscf import run_pyscf

def single_propogation():
    opt_amplitudes = [-5.7778375420113214e-08, -1.6441896890657683e-06, 9.223967507357728e-08, 0.03732738061624315, 1.5707960798368998]
    counter = 1
    
    # Initalize System
    H1 = Atom('H', 1, 0, 0, False)
    H2 = Atom('H', 2, 0.036, 1.37, True)
    H3 = Atom('H', 3, 0, 6.23609576231, False)
    Sys = System([H1, H2, H3])
    
    while counter < 1500:
        Sys.propogation_calculate_energy(1)
        Sys.update_propogation()
        counter += 1
        print(counter)
    
    Sys.print_n_write()

def double_propogation():
    opt_amplitudes = [-5.7778375420113214e-08, -1.6441896890657683e-06, 9.223967507357728e-08, 0.03732738061624315, 1.5707960798368998]
    counter = 1
    
    # Initalize System
    H1 = Atom('H', 1, 0.036, 0, True)
    H2 = Atom('H', 2, 0, 1.37, False)
    H3 = Atom('H', 3, -0.036, 6.23609576231, True)
    Sys = System([H1, H2, H3])
    
    while counter < 1500:
        Sys.propogation_calculate_energy(0)
        Sys.propogation_calculate_energy(2)
        Sys.update_propogation()
        Sys.calculate_energy()
        counter += 1
        print(counter)
    
    Sys.print_n_write()
        



single_propogation()
