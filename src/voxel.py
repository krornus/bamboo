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


class Block:
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
        self._block = array
        self._dof = dof

    def plot(self, color='blue'):
        colors = np.empty(self._block.shape, dtype=object)
        colors[self._block] = color

        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(self._block, facecolors=colors, edgecolor='k')

        plt.show()

    def faces(self):
        """get all exposed faces of the block"""
        xm, ym, zm = self._block.shape

        for x, y, z in np.ndindex(self._block.shape):
            if not self._block[x, y, z]:
                continue

            if z == 0 or not self._block[x, y, z - 1]:
                yield x,y,z,Face.Bottom

            if z + 1 >= zm or not self._block[x, y, z + 1]:
                yield x,y,z,Face.Top

            if x == 0 or not self._block[x - 1, y, z]:
                yield x,y,z,Face.Left

            if x + 1 >= xm or not self._block[x + 1, y, z]:
                yield x,y,z,Face.Right

            if y == 0 or not self._block[x, y - 1, z]:
                yield x,y,z,Face.Front

            if y + 1 >= ym or not self._block[x, y + 1, z]:
                yield x,y,z,Face.Back

    def rotate(self, axis, k):
        '''rotate the voxel by 90 degrees along an axis

        reduce down to 2D (one axis is constant) and apply standard rule
        90 cw -> x,y = y,-x, etc...
        '''
        k = k % 4
        a, a2 = self.rota[axis]

        if k == 0:
            b = self._block.copy()
        if k == 1:
            b = np.flip(self._block, a)
            b = np.swapaxes(b, a, a2)
        elif k == 2:
            b = np.flip(self._block, (a, a2))
        elif k == 3:
            b = np.flip(self._block, a2)
            b = np.swapaxes(b, a, a2)

        return Block(b)

    def translate(self, offset):
        '''create a new array with this block translated by x,y,z'''
        x, y, z = offset
        a, b, c = self._block.shape

        b = np.zeros((x + a, y + b, z + c), dtype=bool)
        b[x:, y:, z:] = self._block[:,:,:]

        return Block(b)

    def face(self, face, target):
        '''rotate block so that face turns against target face direction'''
        target = self.iface[target]
        return self.rotate(*self.rotf[face][target])

    def combine(self, other):
        '''add two shapes together'''

        # assert not self.intersects(other), "intersecting pieces"

        ax, ay, az = self._block.shape
        bx, by, bz = other._block.shape
        shape = max(ax, bx), max(ay, by), max(az, bz)

        b = np.zeros(shape, dtype=bool)

        b[0:ax, 0:ay, 0:az] = self._block[:,:,:]

        if (b[0:bx, 0:by, 0:bz] & other._block[:,:,:]).any():
            raise IntersectionError(other)

        b[0:bx, 0:by, 0:bz] |= other._block[:,:,:]

        return Block(b)


class Blocks:
    def __init__(self, block):
        self._space = block._block
        self._blocks = [block]

    def add(self, block):
        self._blocks.append(block)

    def plot(self):
        colors = np.empty(self._space, dtype=object)

        cmap = get_cmap(len(self._blocks))
        for i, b in enumerate(self._blocks):
            colors[b] = cmap(i)

        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(self._space, facecolors=colors, edgecolor='k')

        plt.show()
