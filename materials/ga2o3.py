
# %%
from main import *
import os
if __name__ == '__main__':
    os.chdir('/home/jinho93/ml/gallium-oxide/alloy-in/beta/thcond')
    unitcell, _ = read_crystal_structure("POSCAR-unitcell", interface_mode='vasp')
    ph3 = Phono3py(unitcell, supercell_matrix=[2, 3, 3], primitive_matrix='auto')
    ph3.generate_displacements()
    efs_calculator = EnergyForceStress(ff_settings=load_snap())

    forces = []

    def calculate_ef(atoms):
        pymatgen_structure = Structure(
            lattice=atoms.cell,
            species=atoms.symbols,
            coords=atoms.positions,
            coords_are_cartesian=True)
        _, f, _ = efs_calculator.calculate([pymatgen_structure])[0]
        logging.info(f'Force calculation completed')
        return f

    logging.basicConfig(filename='example.log', level=logging.INFO)
    num_processes = 52  # the number of processes you want to create
    pool = multiprocessing.Pool(processes=num_processes)
    results = []

    results = [pool.apply_async(calculate_ef, args=(atoms,)) \
            for atoms in ph3.supercells_with_displacements]

    forces = [result.get() for result in results]

    pool.close()
    pool.join()
    np.save('forces.npy', forces)
# %%

fpath = '/home/jinho93/ml/gallium-oxide/pristine/beta/thcond/'
forces = np.load(fpath + 'forces.npy')
ph3.forces = forces

ph3.produce_fc3()

ph3.mesh_numbers = 12

ph3.init_phph_interaction()
ph3.run_thermal_conductivity()
therm_cond = ph3.get_thermal_conductivity()

import matplotlib.pyplot as plt
ran =np.where((150 < t) & (t < 500))
plt.semilogy(therm_cond.temperatures[ran], therm_cond.kappa[0,:,0][ran])
plt.ylim((2, 2e3))

plt.xlabel('Temperature (K)')
plt.ylabel('Thermal Conductivity (W/m-K)')
plt.title('Thermal Conductivity vs Temperature')
plt.show()

# %%
ph3.get_phonon_data()
#%%
