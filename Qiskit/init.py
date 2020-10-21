import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functions import *
from datetime import datetime

H5PY_DEFAULT_READONLY=1
matplotlib.use('Agg')
now = datetime.now().strftime("%m/%d/%Y %H:%M")

class Atom():
    # Initial Information Computed by exact on nuclei position 1.51178 bohrs. velocity at 300K approx .394
    # Position in au, but converted to angstroms when I load into openfermion
    def __init__(self, name, indx, velocity, position, condition):
        self.name = name
        self.indx = indx
        self.condition = condition
        self.velocity = [velocity]
        self.stand_by_velocity = None
        self.position = [position]
        self.adjusted_position = []
        self.stand_by_position = None
        self.exact_energies = []
        self.exact_forces = [0]
        self.VQE_energies = []
        self.VQE_forces = [0]
        self.time = 0.5
        self.mass = 1836

    def update_forces(self):
        if len(self.position) > 1:
            distance_delta = self.position[-1] - self.position[-2]
            self.exact_forces.append(-(self.exact_energies[-1] - self.exact_energies[-2])/distance_delta)
            self.VQE_forces.append(-(self.VQE_energies[-1] - self.VQE_energies[-2])/distance_delta)
        else:
            pass

    def propogate(self):
        if self.condition:
            self.stand_by_position = self.position[-1] + self.time * self.velocity[-1] + 0.5 * self.exact_forces[-1]/self.mass * (self.time ** 2)
            self.stand_by_velocity = self.velocity[-1] + self.exact_forces[-1]/self.mass * self.time
        else:
            pass

    def fill_standby(self):
        if self.stand_by_position is not None:
            self.position.append(self.stand_by_position)
        if self.stand_by_velocity is not None:
            self.velocity.append(self.stand_by_velocity)

    def write_n_plot(self):
        if self.condition:
            self.adjusted_position = [a+1/2*(b - a) for a, b in zip(self.position, self.position[1:])]

            f = open("{}-{}{}-Results.txt".format(self.velocity[0], self.name, self.indx), 'a')
            f.write('''
                    ****************************************************************
                    Experiment results for {}{} | Initial Velocity = {} | {}
                    **************************************************************** \n'''
                    .format(self.name, self.indx, self.velocity[0], now))
            f.write("Velocities: \n{}, #{} \n \n".format(self.velocity, len(self.velocity)))
            f.write("Positions: \n{}, #{} \n \n".format(self.position, len(self.position)))
            f.write("Adjusted Positions: \n{}, #{} \n \n".format(self.adjusted_position, len(self.adjusted_position)))
            f.write("Exact Energy: \n{}, #{} \n \n".format(self.exact_energies, len(self.exact_energies)))
            f.write("Exact Forces: \n{}, #{} \n \n".format(self.exact_forces, len(self.exact_forces)))
            f.write("VQE Energy: \n{}, #{} \n \n".format(self.VQE_energies, len(self.VQE_energies)))
            f.write("VQE Forces: \n{}, #{} \n \n".format(self.VQE_forces, len(self.VQE_forces)))

            # Plot Force Over Length
            f0 = plt.figure(0)
            plt.plot(self.adjusted_position[-len(self.VQE_forces):], self.VQE_forces, color='orange', label='VQE Forces', linestyle = '-')
            plt.plot(self.adjusted_position[-len(self.exact_forces):], self.exact_forces, color='blue', label='Exact Forces', linestyle = ':')
            plt.ylabel('Force in Hartree / Bohrs')
            plt.xlabel('Nuclei Position in au')
            plt.legend(frameon=False, loc='upper center', ncol=2, prop={'size': 10})
            plt.savefig("{}-{}{}-Force.png".format(self.velocity[0], self.name, self.indx), dpi=400, orientation='portrait', bbox_inches='tight')
            plt.close()

            # Plot Energy Over Length
            f1 = plt.figure(1)
            plt.plot(self.position[-len(self.VQE_energies):], self.VQE_energies, color='orange', label='VQE Energy', linestyle = '-')
            plt.plot(self.position[-len(self.exact_energies):], self.exact_energies, color='blue', label='Exact Energy', linestyle = ':')
            plt.ylabel('Energy in Hartree')
            plt.xlabel('Nuclei Position in au')
            plt.legend(frameon=False, loc='upper center', ncol=2, prop={'size': 10})
            plt.savefig("{}-{}{}-Energy.png".format(self.velocity[0], self.name, self.indx), dpi=400, orientation='portrait', bbox_inches='tight')
            plt.close()

            # Plot Position Over Time
            clock = [self.time * iter for iter in range(1, len(self.position) + 1)]

            f2 = plt.figure(2)
            plt.plot(clock, self.position)
            plt.ylabel('Position in bohrs')
            plt.xlabel('Time in au')
            plt.savefig("{}-{}{}-Position.png".format(self.velocity[0], self.name, self.indx), dpi=400, orientation='portrait', bbox_inches='tight')
            plt.close()

            # clock = [self.time]
            # for x in range(len(self.velocity) - 1):
            #     clock += [clock[-1] + self.time]

            f3 = plt.figure(3)
            plt.plot(clock, self.velocity)
            plt.ylabel('Velocity')
            plt.xlabel('Time in au')
            plt.savefig("{}-{}{}-Velocity.png".format(self.velocity[0], self.name, self.indx), dpi=400, orientation='portrait', bbox_inches='tight')
            plt.close()


class System():
    def __init__(self, atoms, noise = False):
        self.atoms = atoms
        self.name = "".join(atom.name for atom in self.atoms)
        self.opt_amplitudes = None
        self.exact_energies = []
        self.VQE_energies = []
        self.CCSD_energies = []
        self.time = self.atoms[0].time
        self.velocity = max(atom.velocity[-1] for atom in self.atoms)
        self.noise = noise

    def initalize_energy(self):
        print("Initializing System Energy")

        results = run_simulation(self, None, noise = self.noise)
        for atom in self.atoms:
            atom.exact_energies.append(results["Exact Energy"])
            atom.VQE_energies.append(results["VQE Energy"])

        self.exact_energies.append(results["Exact Energy"])
        self.VQE_energies.append(results["VQE Energy"])
        print("""Energy for system {} initialized at: \n exact: {} \n VQE: {}""".format(self.name, results["Exact Energy"], results["VQE Energy"]))


    # update every atom in the system to their standby info
    def fill_standby(self):
        for atom in self.atoms:
            atom.fill_standby()

    # for a single index, propogate this atom and calculate the energy level
    def calculate_individual_energy(self, indx, commandprinter = False):
        self.atoms[indx].update_forces()
        self.atoms[indx].propogate()

        results = run_simulation(self, indx, commandprinter, noise = self.noise)

        self.atoms[indx].exact_energies.append(results["Exact Energy"])
        self.atoms[indx].VQE_energies.append(results["VQE Energy"])

    def calculate_system_energy(self):
        results = run_simulation(self, None, noise = self.noise)

        self.exact_energies.append(results["Exact Energy"])
        self.VQE_energies.append(results["VQE Energy"])

    def in_boundary(self):
        return all([abs(atom.position[-1]) < 10 for atom in self.atoms])

    def max_iters(self):
        self.iters = max([len(atom.position) for atom in self.atoms])

    def write_n_plot(self):
        self.max_iters()

        # Plot Atom Position Over Time
        clock = [self.time * i for i in range(1, self.iters + 1)]

        for atom in self.atoms:
            atom.write_n_plot()
            atom.position = atom.position + [atom.position[-1]] * (len(clock) - len(atom.position))

        f1 = plt.figure(1)
        plt.plot(clock, self.atoms[0].position, 'blue', label='Nuclei 1')
        plt.plot(clock, self.atoms[1].position, 'cyan', label='Nuclei 2')
        plt.plot(clock, self.atoms[2].position, 'magenta', label='Nuclei 3')
        plt.ylabel('Position in bohrs')
        plt.xlabel('Time in au')
        plt.legend(frameon=False, loc='upper center', ncol=2, prop={'size': 10})
        plt.savefig("{}-System-{}-Position.png".format(self.velocity, self.name), dpi=400, orientation='portrait', bbox_inches='tight')
        plt.close()

        if len(self.exact_energies) > 2:
            f = open("{}-{}-Results.txt".format(self.velocity, self.name), 'a')
            f.write('''
                    ****************************************************************
                    Experiment results for {} | Initial Velocity = {} | {}
                    **************************************************************** \n'''.format(self.name, self.velocity, now))
            f.write("Velocities: \n {}\n \n".format(self.velocity))
            f.write("Exact Energy: \n{} \n \n".format(self.exact_energies, len(self.exact_energies)))
            f.write("VQE Energy: \n{} \n \n".format(self.VQE_energies, len(self.VQE_energies)))

            # Plot Energy Over Time
            clock = [self.time]
            for x in range(len(self.exact_energies) - 1):
                clock += [clock[-1] + self.time]

            f2 = plt.figure(2)
            plt.plot(clock, self.VQE_energies, color='orange', label='VQE Energy', linestyle = '-')
            plt.plot(clock, self.exact_energies, color='blue', label='exact Energy', linestyle = ':')
            # plt.plot(clock, self.CCSD_energies, color='magenta', label='CCSD Energy', linestyle = ':')
            plt.ylabel('Energy in Hartree')
            plt.xlabel('Time in au')
            plt.legend(frameon=False, loc='upper center', ncol=2, prop={'size': 10})
            plt.savefig("{}-System-{}-Energy.png".format(self.velocity, self.name), dpi=400, orientation='portrait', bbox_inches='tight')
            plt.close()
