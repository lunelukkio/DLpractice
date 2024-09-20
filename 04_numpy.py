# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 20:16:46 2024

@author: lunel
"""

import numpy as np

x = np.array([[1, 2, 3],[4, 5, 6]])

y = np.array([[1, 1, 1],[1, 1, 1]])

print(x + y)

print("")
print(np.hstack((x, y)))

rotated = np.rot90(x)
print("")
print(rotated)

print(x[1][2])


print("")
neuron = {'excitatory':5, 'inhibitory':3}

print(neuron.items())
print(neuron.keys())
print(neuron.values())

print(neuron['inhibitory'])

print('number of neurons = ' + str(neuron['excitatory']))