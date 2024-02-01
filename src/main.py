from voxel import *
from puzzle import puzzle

import random

block = puzzle['l']

block.world.plot()
# block.rotate(Axis.X, 1)
# block.rotate(Axis.Y, 1)
# block.rotate(Axis.Z, 1)
# block.rotate(Axis.Y, 2)
block.rotate(Axis.Y, -1)

print(block.ltow((0, 1, 0)))
block.world.plot()
