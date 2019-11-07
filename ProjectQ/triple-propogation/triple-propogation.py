import sys
sys.path.append("../")

from init import *

def triple_propogation(velocity_1,velocity_2,velocity_3):
    counter = 1
    
    # Initalize System
    H1 = Atom('H', 1, velocity_1, 0, True)
    H2 = Atom('H', 2, velocity_2, 1.37, True)
    H3 = Atom('H', 3, velocity_3, 6.23609576231, True)
    Sys = System([H1, H2, H3])
    Sys.initalize_energy()
    
    while counter < 750 and Sys.in_boundary():
        print("\n\nRunning Propogation #{}".format(counter))
        Sys.calculate_individual_energy(0)
        Sys.calculate_individual_energy(1)
        Sys.calculate_individual_energy(2)
        Sys.fill_standby()
        Sys.calculate_system_energy()
        counter += 1
    
    Sys.write_n_plot()
        

triple_propogation(0.0008, -0.0003, -0.0005)