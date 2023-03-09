#%%
from pymatgen.io.vasp.outputs import Vasprun
import numpy as np
from maml.utils import pool_from, convert_docs

from maml.base import SKLModel
from maml.describers import BispectrumCoefficients
from sklearn.linear_model import LinearRegression
from maml.apps.pes import SNAPotential


def get_document(n, vrun):
    structure = vrun.ionic_steps[n]['structure']
    energy = vrun.ionic_steps[n]['e_wo_entrp']
    forces = vrun.ionic_steps[n]['forces']
    return structure, energy, forces

def get_train_data(vrun_file):
    train_structures = []
    train_energies = []
    train_forces = []

    # vrun_files = glob("vasprun*")
    vrun = Vasprun(vrun_file)
    for v in range(vrun.nionic_steps):
        s, e, f = get_document(v, vrun)
        train_structures.append(s)
        train_energies.append(e)
        train_forces.append(f)
    
    return train_structures, train_energies, train_forces



def get_weights(train_structures, train_energies, train_forces):

    train_pool = pool_from(train_structures, train_energies, train_forces)
    _, df = convert_docs(train_pool)

    weights = np.ones(len(df['dtype']), )

    # set the weights for energy equal to 100
    weights[df['dtype'] == 'energy'] = 10
    
    return weights

def snap_training(train_structures, train_energies, train_forces, weights, elem):

    element_profile = elem
    describer = BispectrumCoefficients(rcutfac=0.5, twojmax=6, element_profile=element_profile, 
                                    quadratic=False, pot_fit=True)
    model = SKLModel(describer=describer, model=LinearRegression())
    snap = SNAPotential(model=model)

    snap.train(train_structures, train_energies, train_forces, sample_weight=weights)

    snap.write_param()

def load_snap():
    snap = SNAPotential.from_config('SNAPotential.snapparam', 'SNAPotential.snapcoeff')
    return snap
