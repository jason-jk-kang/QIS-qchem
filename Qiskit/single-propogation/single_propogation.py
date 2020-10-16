import sys
sys.path.append("../")
from init import *

def single_propogation(velocity, noise):
    counter = 1

    # Initalize System, position is in au
    H1 = Atom('H', 1, 0, 0, False)
    H2 = Atom('H', 2, velocity, 1.37, True)
    H3 = Atom('H', 3, 0, 6.23609576231, False)
    Sys = System([H1, H2, H3], noise = noise)
    Sys.initalize_energy()

    while counter < 1000 and Sys.in_boundary():
        print("\n\nRunning Propogation #{}".format(counter))
        Sys.calculate_individual_energy(1)
        Sys.fill_standby()
        counter += 1

    Sys.write_n_plot()


noise = True


print(f"{len(sys.argv) - 1} velocities registered: {sys.argv[1:]}")
for velocity in sys.argv[1:]:
    print('############')
    print(f'Beginning Single Propogation: {velocity} \n')
    single_propogation(float(velocity), noise)
    time.sleep(1)
    print(f'Finished Single Propogation: {velocity} \n \n')
    print('############')
