# Example file for the mindice library
#
# Run it as:
# $ python example.py
from mindice.mindice import mindice
import numpy as np
import json


with open('./mindice_defs.json', 'r') as f:
        definitions = json.load(f)

spec    = np.loadtxt('NGC1052.spec', unpack = True)
indices = ['Fe4531', 'Fe4668', 'Fe5015', 'Mg1', 'Mg2', 'Mgb']

for indice in indices:
    mindice(spec[0], spec[1], ind = indice,
            definitions = definitions, plot = True)
