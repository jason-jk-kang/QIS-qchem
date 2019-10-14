import sys
sys.path.append("../")

from init import *

def single_propogation(velocity):
    opt_amplitudes = [-5.7778375420113214e-08, -1.6441896890657683e-06, 9.223967507357728e-08, 0.03732738061624315, 1.5707960798368998]
    counter = 1
    
    # Initalize System, position is in au
    H1 = Atom('H', 1, 0, 0, False)
    H2 = Atom('H', 2, velocity, 1.37, True)
    H3 = Atom('H', 3, 0, 6.23609576231, False)
    Sys = System([H1, H2, H3])
    Sys.initalize_energy()
    
    while counter < 1000 and Sys.in_boundary():
        Sys.calculate_individual_energy(1)
        Sys.update_propogation()
        counter += 1
        print(counter)
    
    Sys.write_n_plot()

single_propogation(0.036)
single_propogation(0.005)
single_propogation(0.001)
