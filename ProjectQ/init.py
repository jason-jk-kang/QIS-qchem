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
from functions import *
from datetime import datetime

now = datetime.now().strftime("%m/%d/%Y %H:%M")

class Atom():
    # Initial Information Computed by FCI on nuclei position 1.51178 bohrs. velocity at 300K approx .394
    # Position in au, but converted to angstroms when I run my VQE
    def __init__(self, name, indx, velocity, position, condition):
        self.name = name
        self.indx = indx
        self.condition = condition
        self.velocity = [velocity]
        self.stand_by_velocity = None
        self.position = [position]
        self.adjusted_position = []
        self.stand_by_position = None
        self.fci_energies = []
        self.fci_forces = [0]
        self.UCCSD_energies = []
        self.UCCSD_forces = [0]
        self.time = 0.5
        self.mass = 1836

    def update_forces(self):
        if len(self.position) > 1:
            distance_delta = self.position[-1] - self.position[-2]
            self.fci_forces.append(-(self.fci_energies[-1] - self.fci_energies[-2])/distance_delta)
            self.UCCSD_forces.append(-(self.UCCSD_energies[-1] - self.UCCSD_energies[-2])/distance_delta)
        else:
            pass

    def propogate(self):
        if self.condition:
            self.stand_by_position = self.position[-1] + self.time * self.velocity[-1] + 0.5 * self.fci_forces[-1]/self.mass * (self.time ** 2)
            self.stand_by_velocity = self.velocity[-1] + self.fci_forces[-1]/self.mass * self.time
        else:
            pass

    def update_propogation(self):
        if self.stand_by_position is not None:
            self.position.append(self.stand_by_position)
        if self.stand_by_velocity is not None:
            self.velocity.append(self.stand_by_velocity)
    
    def print_n_write(self):
        print("Velocities: {}, #{}".format(self.velocity, len(self.velocity)))
        print("Positions: {}, #{}".format(self.position, len(self.position)))
        print("Ajusted Positions: {}, #{}".format(self.adjusted_position, len(self.adjusted_position)))
        print("FCI Energy: {}, #{}".format(self.fci_energies, len(self.fci_energies)))
        print("FCI Forces: {}, #{}".format(self.fci_forces, len(self.fci_forces)))
        print("UCCSD Energy: {}, #{}".format(self.UCCSD_energies, len(self.UCCSD_energies)))
        print("UCCSD Forces: {}, #{}".format(self.UCCSD_forces, len(self.UCCSD_forces)))
        
        f = open("{}{}_{}_Results.txt".format(self.name,self.indx,self.velocity[0]), 'a')
        f.write('''
                ****************************************************************
                Experiment results for {}{} | Initial Velocity = {} | {}
                **************************************************************** \n'''
                .format(self.name, self.indx, self.velocity[0], now))
        f.write("Velocities: \n{}, #{} \n \n".format(self.velocity, len(self.velocity)))
        f.write("Positions: \n{}, #{} \n \n".format(self.position, len(self.position)))
        f.write("Ajusted Positions: \n{}, #{} \n \n".format(self.adjusted_position, len(self.adjusted_position)))
        f.write("FCI Energy: \n{}, #{} \n \n".format(self.fci_energies, len(self.fci_energies)))
        f.write("FCI Forces: \n{}, #{} \n \n".format(self.fci_forces, len(self.fci_forces)))
        f.write("UCCSD Energy: \n{}, #{} \n \n".format(self.UCCSD_energies, len(self.UCCSD_energies)))
        f.write("UCCSD Forces: \n{}, #{} \n \n".format(self.UCCSD_forces, len(self.UCCSD_forces)))
        
    def plot(self):
        if self.condition: 
            self.adjusted_position = [a+1/2*(b - a) for a, b in zip(self.position, self.position[1:])]
            
            # Plot Force Over Length
            f0 = plt.figure(0)
            plt.plot(self.adjusted_position[-len(self.fci_forces):], self.fci_forces, '-', label='FCI Forces')
            plt.plot(self.adjusted_position[-len(self.UCCSD_forces):], self.UCCSD_forces, color='orange', label='UCCSD Forces')
            plt.ylabel('Force in Hartree / Bohrs')
            plt.xlabel('Nuclei Position in au')
            plt.legend(frameon=False, loc='upper center', ncol=2, prop={'size': 10})
            plt.savefig("{}{}-{}-Force.png".format(self.name, self.indx, self.velocity[0]), dpi=400, orientation='portrait')
            plt.close()

            # Plot Energy Over Length
            f1 = plt.figure(1)
            plt.plot(self.position[-len(self.fci_energies):], self.fci_energies, '-', label='FCI Energy')
            plt.plot(self.position[-len(self.UCCSD_energies):], self.UCCSD_energies, '-', color='orange', label='UCCSD Energy')
            plt.ylabel('Energy in Hartree')
            plt.xlabel('Nuclei Position in au')
            plt.legend(frameon=False, loc='upper center', ncol=2, prop={'size': 10})
            plt.savefig("{}{}-{}-Energy.png".format(self.name, self.indx, self.velocity[0]), dpi=400, orientation='portrait')
            plt.close()

            # Plot Distance Over Time
            clock = [self.time]
            for x in range(len(self.position) - 1):
                clock += [clock[-1] + self.time]
                
            f2 = plt.figure(2)
            plt.plot(clock, self.position, '-')
            plt.ylabel('Distance in bohrs')
            plt.xlabel('Time in au')
            plt.savefig("{}{}-{}-Position.png".format(self.name, self.indx, self.velocity[0]), dpi=400, orientation='portrait')
            plt.close()

            clock = [self.time]
            for x in range(len(self.velocity) - 1):
                clock += [clock[-1] + self.time]

            f3 = plt.figure(3)
            plt.plot(clock, self.velocity, '-')
            plt.ylabel('Velocity')
            plt.xlabel('Time in au')
            plt.savefig("{}{}-{}-Velocity.png".format(self.name, self.indx, self.velocity[0]), dpi=400, orientation='portrait')
            plt.close()       

    
class System():
    def __init__(self, atoms):
        self.atoms = atoms
        self.name = "".join(atom.name for atom in self.atoms)
        self.opt_amplitudes = [-5.7778375420113214e-08, -1.6441896890657683e-06, 9.223967507357728e-08, 0.03732738061624315, 1.5707960798368998]
        self.fci_energies = []
        self.UCCSD_energies = []
        self.time = self.atoms[0].time
        self.velocity = max(atom.velocity[-1] for atom in self.atoms)
    
    def initalize_energy(self):
        results = run_simulation(self, None)    
        for atom in self.atoms:
                atom.fci_energies.append(results["FCI Energy"])
                atom.UCCSD_energies.append(results["UCCSD Energy"])
        
        self.fci_energies.append(results["FCI Energy"])
        self.UCCSD_energies.append(results["UCCSD Energy"])
        print("""Energy for system {} initialized at:
        FCI: {}
        UCCSD: {}""".format(self.name, results["FCI Energy"], results["UCCSD Energy"]))
    
    # update every atom in the system to their standby info
    def update_propogation(self):
        for atom in self.atoms:
            atom.update_propogation()
    
    # for a single index, propogate this atom and calculate the energy level
    def calculate_individual_energy(self, indx):
        self.atoms[indx].update_forces()
        self.atoms[indx].propogate()
            
        results = run_simulation(self, indx)

        self.atoms[indx].fci_energies.append(results["FCI Energy"])
        self.atoms[indx].UCCSD_energies.append(results["UCCSD Energy"])
    
    def calculate_system_energy(self):
        results = run_simulation(self, None)
        
        self.fci_energies.append(results["FCI Energy"])
        self.UCCSD_energies.append(results["UCCSD Energy"])
    
    def write_n_plot(self):
        for atom in self.atoms:
            atom.print_n_write()
            atom.plot()
        
        if len(self.fci_energies) > 2:
            print("Velocity: {}".format(self.velocity))
            print("FCI Energy: {}, #{}".format(self.fci_energies, len(self.fci_energies)))
            print("UCCSD Energy: {}, #{}".format(self.UCCSD_energies, len(self.UCCSD_energies)))
            
            f = open("{}_{}_Results.txt".format(self.name, self.velocity), 'a')
            f.write('''
                    ****************************************************************
                    Experiment results for {} | Initial Velocity = {} | {}
                    **************************************************************** \n'''.format(self.name, self.velocity, now))
            f.write("Velocities: \n {}\n \n".format(self.velocity))
            f.write("FCI Energy: \n{} \n \n".format(self.fci_energies, len(self.fci_energies)))
            f.write("UCCSD Energy: \n{} \n \n".format(self.UCCSD_energies, len(self.UCCSD_energies)))
            
            # Plot Energy Over Time
            clock = [self.time]
            for x in range(len(self.fci_energies) - 1):
                clock += [clock[-1] + self.time]
                
            f1 = plt.figure(1)
            plt.plot(clock, self.fci_energies, '-', label='FCI Energy')
            plt.plot(clock, self.UCCSD_energies, '-', color='orange', label='UCCSD Energy')
            plt.ylabel('Energy in Hartree')
            plt.xlabel('Time in au')
            plt.legend(frameon=False, loc='upper center', ncol=2, prop={'size': 10})
            plt.savefig("{}-System-{}-Energy.png".format(self.name, self.velocity), dpi=400, orientation='portrait')
            plt.close()
    
    def in_boundary(self):
        return all([abs(atom.position[-1]) < 10 for atom in self.atoms])