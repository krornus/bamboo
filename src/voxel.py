from functools import cached_property
from enum import Enum, IntEnum

import matplotlib.pyplot as plt
import numpy as np

# https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)

class Axis(IntEnum):
    X = 0
    Y = 1
    Z = 2

class Face(IntEnum):
    Bottom = 0
    Top = 1
    Left = 2
    Right = 3
    Front = 4
    Back = 5


class IntersectionError(ValueError):
    pass


class Voxels:
    # rotation axes, as index by an axis
    rota = (
        (Axis.Y, Axis.Z),
        (Axis.X, Axis.Z),
        (Axis.X, Axis.Y),
    )

    iface = (
        Face.Top,
        Face.Bottom,
        Face.Right,
        Face.Left,
        Face.Back,
        Face.Front,
    )

    # rotate axes, face -> face mapping
    rotf = (
        # bottom
        (
            (Axis.X,  0),
            (Axis.X,  2),
            (Axis.Y,  1),
            (Axis.Y, -1),
            (Axis.X,  1),
            (Axis.X, -1),
        ),
        # top
        (
            (Axis.X,  2),
            (Axis.X,  0),
            (Axis.Y, -1),
            (Axis.Y,  1),
            (Axis.X, -1),
            (Axis.X,  1),
        ),
        # left
        (
            (Axis.Y, -1),
            (Axis.Y,  1),
            (Axis.Z,  0),
            (Axis.Z,  2),
            (Axis.Z, -1),
            (Axis.Z,  1),
        ),
        # right
        (
            (Axis.Y,  1),
            (Axis.Y, -1),
            (Axis.Z,  2),
            (Axis.Z,  0),
            (Axis.Z,  1),
            (Axis.Z, -1),
        ),
        # front
        (
            (Axis.X, -1),
            (Axis.X,  1),
            (Axis.Z,  1),
            (Axis.Z, -1),
            (Axis.X,  0),
            (Axis.X,  2),
        ),
        # back
        (
            (Axis.X,  1),
            (Axis.X, -1),
            (Axis.Z, -1),
            (Axis.Z,  1),
            (Axis.X,  2),
            (Axis.X,  0),
        ),
    )

    def __init__(self, array, dof=6):
        self._voxels = array
        self._dof = dof

    def plot(self, color='blue'):
        colors = np.empty(self._voxels.shape, dtype=object)
        colors[self._voxels] = color

        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(self._voxels, facecolors=colors, edgecolor='k')

        plt.show()

    def faces(self):
        """get all exposed faces of the block"""
        xm, ym, zm = self._voxels.shape

        for x, y, z in np.ndindex(self._voxels.shape):
            if not self._voxels[x, y, z]:
                continue

            if z == 0 or not self._voxels[x, y, z - 1]:
                yield x,y,z,Face.Bottom

            if z + 1 >= zm or not self._voxels[x, y, z + 1]:
                yield x,y,z,Face.Top

            if x == 0 or not self._voxels[x - 1, y, z]:
                yield x,y,z,Face.Left

            if x + 1 >= xm or not self._voxels[x + 1, y, z]:
                yield x,y,z,Face.Right

            if y == 0 or not self._voxels[x, y - 1, z]:
                yield x,y,z,Face.Front

            if y + 1 >= ym or not self._voxels[x, y + 1, z]:
                yield x,y,z,Face.Back

    def rotate(self, axis, k):
        '''rotate the voxel by 90 degrees along an axis

        reduce down to 2D (one axis is constant) and apply standard rule
        90 cw -> x,y = y,-x, etc...
        '''
        k = k % 4
        a, a2 = self.rota[axis]

        if k == 0:
            b = self._voxels.copy()
        if k == 1:
            b = np.flip(self._voxels, a)
            b = np.swapaxes(b, a, a2)
        elif k == 2:
            b = np.flip(self._voxels, (a, a2))
        elif k == 3:
            b = np.flip(self._voxels, a2)
            b = np.swapaxes(b, a, a2)

        return Voxels(b)

    def translate(self, offset):
        '''create a new array with this block translated by x,y,z'''
        x, y, z = offset
        a, b, c = self._voxels.shape

        b = np.zeros((x + a, y + b, z + c), dtype=bool)
        b[x:, y:, z:] = self._voxels[:,:,:]

        return Voxels(b)

    def face(self, face, target):
        '''rotate block so that face turns against target face direction'''
        target = self.iface[target]
        return self.rotate(*self.rotf[face][target])

    def combine(self, other):
        '''add two shapes together'''

        # assert not self.intersects(other), "intersecting pieces"

        ax, ay, az = self._voxels.shape
        bx, by, bz = other._voxels.shape
        shape = max(ax, bx), max(ay, by), max(az, bz)

        b = np.zeros(shape, dtype=bool)

        b[0:ax, 0:ay, 0:az] = self._voxels[:,:,:]

        if (b[0:bx, 0:by, 0:bz] & other._voxels[:,:,:]).any():
            raise IntersectionError(other)

        b[0:bx, 0:by, 0:bz] |= other._voxels[:,:,:]

        return Voxels(b)

def neg(x):
    """dumb function to flip the direction of an axis for Block.rotate()"""
    return x[0], -x[1]

class Block:
    """a set of voxels with a local and global transform"""
    def __init__(self, ary):
        self._voxels = ary
        self._translate = [0, 0, 0]
        self._rotate = [(Axis.X, 1), (Axis.Y, 1), (Axis.Z, 1)]

    """designates which two axes are modified for a 90 degree rotation on a given axis"""
    _rotation_axes = (
        (Axis.Y, Axis.Z),
        (Axis.X, Axis.Z),
        (Axis.X, Axis.Y),
    )

    def rotate(self, axis, k):
        # modulus number of 90 degree steps to be [0..3]
        k = k % 4

        if k == 0:
            return

        # get the two axes that are being swapped. because this is a 90 degree
        # turn on a single axis, we reduce it to two dimensions. there are
        # simple rules for how to do this, depending on the number of steps
        a, b = self._rotation_axes[axis]

        if k == 1:
            self._rotate[a], self._rotate[b] = self._rotate[b], neg(self._rotate[a])
        elif k == 2:
            self._rotate[a], self._rotate[b] = neg(self._rotate[a]), neg(self._rotate[b])
        elif k == 3:
            self._rotate[a], self._rotate[b] = neg(self._rotate[b]), self._rotate[a]

        self.__dict__.pop('world', None)

    def translate(self, vector):
        self._translate[Axis.X] += vector[Axis.X]
        self._translate[Axis.Y] += vector[Axis.Y]
        self._translate[Axis.Z] += vector[Axis.Z]

        self.__dict__.pop('world', None)

    @cached_property
    def local(self):
        """return Voxels in local coordinates"""
        return Voxels(self._voxels.copy())

    @cached_property
    def world(self):
        """return Voxels in world coordinates, applying rotate and translate"""
        (x, xdir), (y, ydir), (z, zdir) = self._rotate

        # first rotate the voxels --
        # step 1: axis inversions
        flips = []

        if xdir < 0:
            flips.append(x)

        if ydir < 0:
            flips.append(y)

        if zdir < 0:
            flips.append(z)

        if flips:
            rot = np.flip(self._voxels, flips)
        else:
            rot = self._voxels.copy()

        # step 2: axis swaps
        if x == Axis.Y:
            rot = rot.swapaxes(Axis.X, Axis.Y)
        elif x == Axis.Z:
            rot = rot.swapaxes(Axis.X, Axis.Z)

        if y == Axis.Z:
            rot = rot.swapaxes(Axis.Y, Axis.Z)

        # finally, translate the voxels
        x, y, z = self._translate

        if not x and not y and not z:
            return Voxels(rot)

        a, b, c = rot.shape

        v = np.zeros((x + a, y + b, z + c), dtype=bool)
        v[x:, y:, z:] = rot[:,:,:]

        return Voxels(v)

    def ltow(self, vector):
        """local to world vector transform"""

        # we do something similar to the world() function, in that we flip and
        # swap values before translating, but flipping in this case is a little
        # different. x, y = y, -x in an example:
        #
        # x, y = 0, 2, given shape = 3,3 produces
        # x, y = 2, 2 because negation places the coordinate on the opposite
        # side of the given array.
        shape = self._voxels.shape
        (x, xdir), (y, ydir), (z, zdir) = self._rotate

        if xdir < 0:
            outx = shape[x] - 1 - vector[x]
        else:
            outx = vector[x]

        if ydir < 0:
            outy = shape[y] - 1 - vector[y]
        else:
            outy = vector[y]

        if zdir < 0:
            outz = shape[z] - 1 - vector[z]
        else:
            outz = vector[z]

        x, y, z = self._translate

        return outx + x, outy + y, outz + z

    def pivot(self, axis, k, local):
        """rotate, keeping local point in the same global location

        NOTE: point is in local space, NOT global
        """
        ax, ay, az = self.ltow(local)
        self.rotate(axis, k)
        bx, by, bz = self.ltow(local)

        self.translate((ax - bx, ay - by, az - bz))

# class Blocks:
#     def __init__(self, block):
#         self._space = block._voxels
#         self._blocks = [block]

#     def add(self, block):
#         self._blocks.append(block)

#     def plot(self):
#         colors = np.empty(self._space, dtype=object)

#         cmap = get_cmap(len(self._blocks))
#         for i, b in enumerate(self._blocks):
#             colors[b] = cmap(i)

#         ax = plt.figure().add_subplot(projection='3d')
#         ax.voxels(self._space, facecolors=colors, edgecolor='k')

#         plt.show()
