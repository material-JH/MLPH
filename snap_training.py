#%%
from main import *

train_structures, train_energies, train_forces, train_stresses = get_from_abn('ML_ABN')

w = get_weights(train_structures, train_energies, train_forces, train_stresses)

elem = {'Sr': {'r': 5.0, 'w': 1}, 
        'Ba': {'r': 5.0, 'w': 1}, 
        'Ti': {'r': 5.0, 'w': 1}, 
        'O': {'r': 5.0, 'w': 1}}

snap_training(train_structures, train_energies, train_forces, train_stresses, weights=w, elem=elem)
# %%
