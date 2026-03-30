# Example file for the mindice library
#
# Run it as:
# $ python example.py
from mindice import mindice
import numpy as np
import json

spec    = np.loadtxt('NGC1052.spec', unpack = True)
indices = ['Fe4531', 'Hb', 'Fe4668', 'Fe5015', 'Mg1', 'Mg2', 'Mgb']

for indice in indices:
    mindice(spec[0], spec[1], ind = indice, plot = True)
