import glob
import os
import sys

def get_egg_file(module_name):
    def f(packages):
        return glob.glob(
            os.path.join(os.path.dirname(os.path.dirname(sys.executable)),
                         'lib', 'python*', packages, module_name + '.egg-link'))

    return f('site-packages') or f('dist-packages')


egg_file = get_egg_file('openfermionpyscf')
if egg_file:
    os.remove(egg_file[0])