from functions import *

class Atom():
    # Initial Information Computed by FCI on nuclei position 1.51178 bohrs. velocity at 300K approx .394
    def __init__(self, name, indx, velocity, position, condition):
        self.name = name
        self.indx = indx
        self.condition = condition
        self.velocity = [velocity]
        self.stand_by_velocity = None
        self.position = [position]
        self.adjusted_position = []
        self.stand_by_position = None
        self.fci_energies = [0]
        self.fci_force_list = [0]
        self.UCCSD_energies = [0]
        self.UCCSD_force_list = [0]
        self.time = 0.5
        self.mass = 1836

    def update_forces(self):
        if len(self.position) > 1:
            distance_delta = self.position[-1] - self.position[-2]
            self.fci_force_list.append(-(self.fci_energies[-1] - self.fci_energies[-2])/distance_delta)
            self.UCCSD_force_list.append(-(self.UCCSD_energies[-1] - self.UCCSD_energies[-2])/distance_delta)
        else:
            pass

    def propogate(self):
        if self.condition:
            self.stand_by_position = self.position[-1] + self.time * self.velocity[-1] + 0.5 * self.fci_force_list[-1]/self.mass * (self.time ** 2)
            self.stand_by_velocity = self.velocity[-1] + self.fci_force_list[-1]/self.mass * self.time
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
        print("FCI Energy: {}, #{}".format(self.fci_energies, len(self.fci_energies)))
        print("FCI Forces: {}, #{}".format(self.fci_force_list, len(self.fci_force_list)))
        print("UCCSD Energy: {}, #{}".format(self.UCCSD_energies, len(self.UCCSD_energies)))
        print("UCCSD Forces: {}, #{}".format(self.UCCSD_force_list, len(self.UCCSD_force_list)))
        
        f = open("Hydrogen_{}_Results.txt".format(self.indx), 'a')
        f.write('''
                ****************************************************************
                Experiment results`
                **************************************************************** \n''')
        f.write("Velocities: \n{}, #{} \n \n".format(self.velocity, len(self.velocity)))
        f.write("Positions: \n{}, #{} \n \n".format(self.position, len(self.position)))
        f.write("FCI Energy: \n{}, #{} \n \n".format(self.fci_energies, len(self.fci_energies)))
        f.write("FCI Forces: \n{}, #{} \n \n".format(self.fci_force_list, len(self.fci_force_list)))
        f.write("UCCSD Energy: \n{}, #{} \n \n".format(self.UCCSD_energies, len(self.UCCSD_energies)))
        f.write("UCCSD Forces: \n{}, #{} \n \n".format(self.UCCSD_force_list, len(self.UCCSD_force_list)))
        
    def adjust_lists(self):
        self.adjusted_position = [a+1/2*(b - a) for a, b in zip(self.position, self.position[1:])]
    
    def plot(self):
        # Plot Force Over Length
        f0 = plt.figure(0)
        plt.plot(self.adjusted_position[-len(self.fci_force_list):], self.fci_force_list, '-')
        plt.plot(self.adjusted_position[-len(self.UCCSD_force_list):], self.UCCSD_force_list, color='orange')
        plt.ylabel('Force in Hartree / Bohrs')
        plt.xlabel('Bond length in bohrs')
        plt.savefig("{}{}-{}-Force".format(self.name, self.indx, self.velocity), dpi=400, orientation='portrait')
        plt.show()

        # Plot Energy Over Length
        f1 = plt.figure(1)
        plt.plot(self.position[-len(self.fci_energies):], self.fci_energies, '-')
        plt.plot(self.position[-len(self.UCCSD_energies):], self.UCCSD_energies, '-', color='orange')
        plt.ylabel('Energy in Hartree')
        plt.xlabel('Bond length in bohr')
        plt.savefig("{}{}-{}-Energy".format(self.name, self.indx, self.velocity), dpi=400, orientation='portrait')
        plt.show()

        # Plot Distance Over Time
        clock = [self.time]
        for x in range(len(self.position) - 1):
            clock += [clock[-1] + self.time]
            
        f2 = plt.figure(2)
        plt.plot(clock, self.position, '-')
        plt.ylabel('Distance in bohrs')
        plt.xlabel('Time in au')
        plt.savefig("{}{}-{}-Position".format(self.name, self.indx, self.velocity), dpi=400, orientation='portrait')
        plt.show()

        clock = [self.time]
        for x in range(len(self.velocity) - 1):
            clock += [clock[-1] + self.time]

        f3 = plt.figure(3)
        plt.plot(clock, self.velocity, '-')
        plt.ylabel('Velocity')
        plt.xlabel('Time in au')
        plt.savefig("{}{}-{}-Velocity".format(self.name, self.indx, self.velocity), dpi=400, orientation='portrait')
        plt.show()

    
class System():
    def __init__(self, atoms):
        self.atoms = atoms
        self.name = "".join(atom.name for atom in atoms)
        self.opt_amplitudes = [-5.7778375420113214e-08, -1.6441896890657683e-06, 9.223967507357728e-08, 0.03732738061624315, 1.5707960798368998]
        self.fci_energies = [0]
        self.UCCSD_energies = [0]
        self.time = self.atoms[0].time
        self.velocity = max([atom.velocity for atom in atoms])
        
    # update every atom in the system to their standby info
    def update_propogation(self):
        for atom in self.atoms:
            atom.update_propogation()
    
    # for a single index, propogate this atom and calculate the energy level
    def propogation_calculate_energy(self, indx):
        self.atoms[indx].update_forces()
        self.atoms[indx].propogate()
            
        results = run_simulation(self, indx)

        self.atoms[indx].fci_energies.append(results["FCI Energy"])
        self.atoms[indx].UCCSD_energies.append(results["UCCSD Energy"])
    
    def calculate_system_energy(self):
        results = run_simulation(self, None)
        
        self.fci_energies.append(results["FCI Energy"])
        self.UCCSD_energies.append(results["UCCSD Energy"])
    
    def print_n_write(self):
        for atom in self.atoms:
            atom.print_n_write()
        print("Velocity: {}".format(self.velocity))
        print("FCI Energy: {}, #{}".format(self.fci_energies, len(self.fci_energies)))
        print("UCCSD Energy: {}, #{}".format(self.UCCSD_energies, len(self.UCCSD_energies)))
        
        f = open("{}_Results.txt".format(self.name), 'a')
        f.write('''
                ****************************************************************
                Experiment results`
                **************************************************************** \n''')
        f.write("Velocities: {}".format(self.velocity))
        f.write("FCI Energy: \n{} \n \n".format(self.fci_energies, len(self.fci_energies)))
        f.write("UCCSD Energy: \n{} \n \n".format(self.UCCSD_energies, len(self.UCCSD_energies)))
    
    def in_boundary(self):
        return all([atom.position[-1] < 8 for atom in self.atoms])
        
    def plot(self):
        # Plot Energy Over Time
        clock = [self.time]
        for x in range(len(self.fci_energies) - 1):
            clock += [clock[-1] + self.time]
            
        f1 = plt.figure(1)
        plt.plot(clock, self.fci_energies, '-')
        plt.plot(clock, self.UCCSD_energies, '-', color='orange')
        plt.ylabel('Energy in Hartree')
        plt.xlabel('Bond length in bohr')
        plt.savefig("{}-System-{}-Energy".format(self.name, self.velocity), dpi=400, orientation='portrait')
        plt.show()