from main import *
import matplotlib.pyplot as plt
import os
import sys
import phono3py._phono3py as phono3c
from phonopy.interface.calculator import read_crystal_structure
from phono3py import Phono3py


os.environ['OPENBLAS_NUM_THREADS'] = sys.argv[1] # set the number of threads to 4
max_threads = phono3c.omp_max_threads()

print(max_threads)

unitcell, _ = read_crystal_structure("POSCAR-unitcell", interface_mode='vasp')
ph3 = Phono3py(unitcell, supercell_matrix=[2, 3, 3], primitive_matrix='auto')
ph3.generate_displacements()

forces = np.load('forces.npy')
ph3.forces = forces
ph3.produce_fc3()
ph3.mesh_numbers = 24

ph3.init_phph_interaction()
ph3.run_thermal_conductivity(temperatures=np.arange(150, 501, 10, dtype="double"), write_gamma=True, write_pp=True, output_filename='ph')
therm_cond = ph3.get_thermal_conductivity()

np.savetxt('cond.dat', therm_cond.kappa[0,:,:])
np.savetxt('temp.dat', therm_cond.temperatures)

np.save('cond.npy', therm_cond.kappa)

plt.semilogy(therm_cond.temperatures, therm_cond.kappa[0,:,0])
plt.ylim((2, 2e3))
plt.xlabel('Temperature (K)')
plt.ylabel('Thermal Conductivity (W/m-K)')
plt.title('Thermal Conductivity vs Temperature')
plt.show()
