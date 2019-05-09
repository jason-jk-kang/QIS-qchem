from pyquil.quil import Program
import pyquil.api as api
from pyquil.gates import *
from pyquil.paulis import sZ
from grove.pyvqe.vqe import VQE
from scipy.optimize import minimize
import numpy as np

qvm = api.QVMConnection()

def small_ansatz(params):
    return Program(RX(params[0], 0))

initial_angle = [0.0]
# Our Hamiltonian is just \sigma_z on the zeroth qubit
hamiltonian = sZ(0)

vqe_inst = VQE(minimizer=minimize,
               minimizer_kwargs={'method': 'nelder-mead'})

angle = 2.0
result = vqe_inst.expectation(small_ansatz([angle]), hamiltonian, None, qvm)

print (result)
