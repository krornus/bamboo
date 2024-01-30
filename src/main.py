from voxel import *
from puzzle import puzzle

import random

a = puzzle['l']
b = puzzle['s1']

fa = random.choice(tuple(a.faces()))
fb = random.choice(tuple(b.faces()))

c = a.translate((1, 0, 1))

c.combine(b).plot()
