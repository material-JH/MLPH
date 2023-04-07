#%%

import sys
from main import *
import logging
import multiprocessing
from phonopy.interface.calculator import read_crystal_structure
from phono3py import Phono3py
from maml.apps.pes import EnergyForceStress
from pymatgen.core import Structure

if __name__ == '__main__':
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
    num_processes = int(sys.argv[1])  # the number of processes you want to create
    pool = multiprocessing.Pool(processes=num_processes)
    results = []

    results = [pool.apply_async(calculate_ef, args=(atoms,)) \
            for atoms in ph3.supercells_with_displacements]

    forces = [result.get() for result in results]

    pool.close()
    pool.join()
    np.save('forces.npy', forces)
