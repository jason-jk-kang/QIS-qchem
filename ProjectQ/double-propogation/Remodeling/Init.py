class Atom():
	# Initial Information Computed by FCI on nuclei position 1.51178 bohrs. velocity at 300K approx .394
	def __init__(self, name, indx, velocity, position, condition):
		self.name = name
		self.indx = indx
		self.condition = condition
		self.velocity = [velocity]
		self.stand_by_velocity = None
		self.position = [position]
		self.stand_by_position = None
		self.fci_energies = [0]
		self.fci_force_list = [0]
		self.UCCSD_energies = [0]
		self.UCCSD_force_list = [0]
		self.time = 0.5
		self.mass = 1836
	
	def update_forces(self):
		if len(self.fci_energies) > 1:
            distance_delta = self.position[-1] - self.position[-2]
            self.fci_force_list.append(-(self.fci_energies[-1] - self.fci_energies[-2])/distance_delta)
            self.UCCSD_force_list.append(-(self.UCCSD_energies[-1] - self.UCCSD_energies[-2])/distance_delta)
		else:
			pass
			
	def propogate(self):
		new_position = self.position[-1] + self.time * self.velocity[-1] + 
					   0.5 * self.fci_force_list[-1]/self.mass * (self.time ** 2)
		self.stand_by_position = new_position
		new_velocity = self.velocity[-1] + self.fci_force_list[-1]/self.mass * self.time
		self.stand_by_velocity = new_velocity
		
	def update_propogation(self):
		if self.stand_by_position !None:
			self.position.append(self.stand_by_position)
		if self.stand_by_velocity !None:
			self.velocity.append(self.stand_by_velocity)

class System():
	def __init__(self, atoms):
		self.atoms = atoms
		self.opt_amplitudes = [-5.7778375420113214e-08, -1.6441896890657683e-06, 9.223967507357728e-08, 0.03732738061624315, 1.5707960798368998]
		
	def update_propogation(self):
		for atom in atoms:
			atom.update_propogation


		

 
