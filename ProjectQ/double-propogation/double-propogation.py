import sys
sys.path.append("../")

from init import *

def double_propogation(velocity):
    counter = 1
    
    # Initalize System
    H1 = Atom('H', 1, velocity, 0, True)
    H2 = Atom('H', 2, 0, 1.37, False)
    H3 = Atom('H', 3, -velocity, 6.23609576231, True)
    Sys = System([H1, H2, H3])
    Sys.initalize_energy()
    
    while counter < 750 and Sys.in_boundary():
        print("\n\nRunning Propogation #{}".format(counter))
        Sys.calculate_individual_energy(0)
        Sys.calculate_individual_energy(2)
        Sys.fill_standby()
        Sys.calculate_system_energy()
        counter += 1
    
    Sys.write_n_plot()
        


double_propogation(0.0036)
double_propogation(0.001)
double_propogation(0.0008)
double_propogation(0.0005)
double_propogation(0.00036)
double_propogation(0.0001)
