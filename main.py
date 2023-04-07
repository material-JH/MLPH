#%%
from pymatgen.io.vasp.outputs import Vasprun
import numpy as np
from maml.utils import pool_from, convert_docs

from maml.base import SKLModel
from maml.describers import BispectrumCoefficients
from sklearn.linear_model import LinearRegression
from maml.apps.pes import SNAPotential
from pymatgen.core import Structure

def get_from_abn(file):
    structures = []
    forces = []
    energies = []
    stresses = []
    with open("ML_ABN", "r") as file:
        lines = file.readlines()  # Read all the lines in the file into a list
        for i in range(len(lines)):
            if "The maximum number of atoms per system" in lines[i]:
                n_atoms = int(lines[i+2])
            if "Configuration num" in lines[i]:
                specie = []
                for j in range(i+16, i+20):
                    specie.append(lines[j].split())
                species = []
                for m, n in specie:
                    species.extend([m]*int(n))

                lattice = []
                for j in range(i+27, i+30):  # Read the next 25 lines after the pattern
                    lattice.append(list(map(float, lines[j].split())))
                position = []
                for j in range(i+33, i+33+n_atoms):
                    position.append(list(map(float, lines[j].split())))
                    
                structure = Structure(lattice, species, position, coords_are_cartesian=True)
                structures.append(structure)
                j = 0
                while "Total energy" not in lines[i+j]:
                    j += 1
                energy = float(lines[i+j+2])
                energies.append(energy)

                force = []
                for j in range(i+j+6, i+j+6+n_atoms):
                    force.append(list(map(float, lines[j].split())))
                forces.append(np.array(force))

                virial_stress = []
                virial_stress.extend(list(map(float, lines[j+6].split())))
                virial_stress.extend(list(map(float, lines[j+10].split())))
                virial_stress = np.array(virial_stress)

                vasp_stress_order = ['xx', 'yy', 'zz', 'xy', 'yz', 'xz']
                snap_stress_order = ['xx', 'yy', 'zz', 'yz', 'xz', 'xy']

                stresses.append([virial_stress[vasp_stress_order.index(n)] * 0.1 for n in snap_stress_order])
    return structures, energies, forces, stresses

def get_document(n, vrun):
    structure = vrun.ionic_steps[n]['structure']
    energy = vrun.ionic_steps[n]['e_wo_entrp']
    forces = vrun.ionic_steps[n]['forces']
    virial_stress = vrun.ionic_steps[n]['stress']
    vasp_stress_order = ['xx', 'yy', 'zz', 'xy', 'yz', 'xz']
    snap_stress_order = ['xx', 'yy', 'zz', 'yz', 'xz', 'xy']
    
    return structure, energy, forces, [virial_stress[vasp_stress_order.index(n)] * 0.1 for n in snap_stress_order]

def get_train_data(vrun_file):
    train_structures = []
    train_energies = []
    train_forces = []
    train_stresses = []

    vrun = Vasprun(vrun_file)
    for v in range(vrun.nionic_steps):
        s, e, f, str = get_document(v, vrun)
        train_structures.append(s)
        train_energies.append(e)
        train_forces.append(f)
        train_stresses.append(str)
    
    return train_structures, train_energies, train_forces, train_stresses

def get_weights(train_structures, train_energies, train_forces, train_stresses):

    train_pool = pool_from(train_structures, train_energies, train_forces, train_stresses)
    _, df = convert_docs(train_pool)

    weights = np.ones(len(df['dtype']), )

    # set the weights for energy equal to 100
    weights[df['dtype'] == 'energy'] = 100
    weights[df['dtype'] == 'force'] = 1
    weights[df['dtype'] == 'stress'] = 0.01
    return weights

def snap_training(train_structures, train_energies, train_forces, train_stresses, weights, elem):

    element_profile = elem
    describer = BispectrumCoefficients(rcutfac=0.5, twojmax=6, element_profile=element_profile, 
                                    quadratic=False, pot_fit=True)
    model = SKLModel(describer=describer, model=LinearRegression())
    snap = SNAPotential(model=model)

    snap.train(train_structures, train_energies, train_forces, train_stresses, sample_weight=weights)

    snap.write_param()

def load_snap():
    snap = SNAPotential.from_config('SNAPotential.snapparam', 'SNAPotential.snapcoeff')
    return snap
