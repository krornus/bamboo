from voxel import Block

import matplotlib.pyplot as plt
import numpy as np

puzzle = dict()

# t block
x, y, z = np.indices((3, 1, 2))
puzzle['t'] = Block((z == 0) | (x == 1))

# l block
x, y, z = np.indices((3, 2, 1))
puzzle['l'] = Block((x == 0) | (y == 1))

# z block
x, y, z = np.indices((3, 1, 2))
puzzle['z'] = Block(((x < 2) & (z == 0)) | ((x > 0) & (z == 1)))

# z squiggle 1
x, y, z = np.indices((2, 2, 2))
puzzle['s1'] = Block( ((x == 0) & ((y == 1) | (z == 1))) | ((x == 1) & (z == 0) & (y == 1)))

# z squiggle 2
x, y, z = np.indices((2, 2, 2))
puzzle['s2'] = Block( ((x == 1) & ((y == 1) | (z == 1))) | ((x == 0) & (z == 0) & (y == 1)))

# corner block
x, y, z = np.indices((2, 2, 2))
puzzle['c'] = Block(((x == 0) & (y == 1)) | ((z == 0) & (x == 0)) | ((y == 1) & (z == 0)))

# right angle block
x, y, z = np.indices((2, 2, 1))
puzzle['r'] = Block((x == 0) | (y == 1))
