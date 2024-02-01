from voxel import *
from puzzle import puzzle

import random

block = puzzle['l']

block.world.plot()

block.rotate(Axis.X, 1)
block.rotate(Axis.Y, 1)
block.rotate(Axis.Z, 1)
block.rotate(Axis.Y, 2)
block.translate((2, 1, 0))
block.world.plot()

block.pivot(Axis.Y, -1, (2, 1, 0))
block.world.plot()
