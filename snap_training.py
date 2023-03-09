from main import *

train_structures, train_energies, train_forces = get_train_data('../../mlff/training/vasprun.xml')

w = get_weights(train_structures, train_energies, train_forces)

elem = {'Ga': {'r': 5.0, 'w': 1}, 
        'In': {'r': 5.0, 'w': 1}, 
        'O': {'r': 5.0, 'w': 1}}

snap_training(train_structures, train_energies, train_forces, weights=w, elem=elem)
# %%
