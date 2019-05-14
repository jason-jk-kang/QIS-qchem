force_list = []
fci_energies = [-1.603565128035238, -1.6004199263636436]
distance_counter = 0.8
bond_lengths = [0.725]


force_list += [(fci_energies[-1] - fci_energies[-2])/(distance_counter - bond_lengths[-1])]

print(force_list)
