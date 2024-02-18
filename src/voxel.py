from collections import namedtuple
from functools import cached_property, lru_cache
from enum import Enum, IntEnum

import matplotlib.pyplot as plt
import numpy as np

# https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


Point = namedtuple("Point", ("x", "y", "z", "face"))

class Axis(IntEnum):
    X = 0
    Y = 1
    Z = 2

# dumb kludge but i dont wanna deal with anything annoying and bloaty here
# just dont change the enum values
_inverted_face = (1, 0, 3, 2, 5, 4)
_normals = (Axis.Z, Axis.Z, Axis.X, Axis.X, Axis.Y, Axis.Y)

class Face(IntEnum):
    Bottom = 0
    Top = 1
    Left = 2
    Right = 3
    Front = 4
    Back = 5

    def invert(self):
        return type(self)(_inverted_face[self])

    def normal(self):
        return _normals[self]


class IntersectionError(ValueError):
    pass


def plot(ary, color='blue'):
    if isinstance(ary, Block):
        ary = ary.world()

    colors = np.empty(ary.shape, dtype=object)
    colors[ary] = color

    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(ary, facecolors=colors, edgecolor='k')

    plt.show()


def neg(x):
    """dumb function to flip the direction of an axis for Block.rotate()"""
    return x[0], -x[1]

class Block:
    """designates which two axes are modified for a 90 degree rotation on a given axis"""
    _rotation_axes = (
        (Axis.Y, Axis.Z),
        (Axis.X, Axis.Z),
        (Axis.X, Axis.Y),
    )

    """maps (current face direction, target face direction) -> rotation arguments"""
    _rotation_faces = (
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

    _translate_faces = (
        (0, 0, -1),
        (0, 0, +1),
        (-1, 0, 0),
        (+1, 0, 0),
        (0, -1, 0),
        (0, +1, 0),
    )

    """a set of voxels with a local and global transform"""
    def __init__(self, ary):
        self._voxels = ary
        self._translate = [0, 0, 0]
        self._rotate = [(Axis.X, +1), (Axis.Y, +1), (Axis.Z, +1)]

    def clear_translation(self):
        translate = [0, 0, 0]

        if translate == self._translate:
            return

        self._translate = [0, 0, 0]

    def clear_rotation(self):
        rotate = [0, 0, 0]

        if rotate == self._rotate:
            return

        self._rotate = rotate

    def rotation(self):
       def s(x):
           if x[1] > 0:
               sign = "+"
           else:
               sign = "-"

           return f"{sign}{Axis(x[0]).name.lower()}"

       return f"({s(self._rotate[0])}, {s(self._rotate[1])}, {s(self._rotate[2])})"

    def rotate(self, axis, k):
        """
        The base concept for rotation here is as follows:

        90°  clockwise (+x, +y) ➔ (+y, -x)
        180° clockwise (+x, +y) ➔ (-x, -y)
        270° clockwise (+x, +y) ➔ (-y, +x)
        360°           (+x, +y) ➔ (+x, +y)

        Extending this to 3 dimensions is easy because we are only working in
        90° intervals. We can simply ignore on axis.

        The complication comes with rotation being non-commutative. We cannot
        simply accumulate rotations, instead we have to store a vector which
        contains a reconstructable set of rotations. This is generally done
        with euler angles which are overkill for this type of orthoganal space

        A rotation of (y, 90) would result in axes (+z, +y, -x)
        A subsequent rotation of (x, 90) would produce (+z, -x, -y)
        A thrid rotation of (z, 90) would produce (-x, -z, -y)

        Drawn out below

        y, 90 ➔ +x, +z = +z, -x ➔ (+z, +y, -x)
        x, 90 ➔ +y, +z = +z, -y ➔ (+z, -x, -y)
        (+z, -x, -y)

        z, 90 ➔ +x, +y = +y, -x ➔ (-x, -z, -y)
        (-x, -z, -y)
        """

        k = k % 4
        if k == 0:
            return

        x, y = self._rotation_axes[axis]

        if k == 1:
            self._rotate[x], self._rotate[y] = self._rotate[y], neg(self._rotate[x])
        elif k == 2:
            self._rotate[x], self._rotate[y] = neg(self._rotate[x]), neg(self._rotate[y])
        elif k == 3:
            self._rotate[x], self._rotate[y] = neg(self._rotate[y]), self._rotate[x]

    def translate(self, vector):
        self._translate[Axis.X] += vector[Axis.X]
        self._translate[Axis.Y] += vector[Axis.Y]
        self._translate[Axis.Z] += vector[Axis.Z]

        assert self._translate[Axis.X] >= 0, "negative X translation"
        assert self._translate[Axis.Y] >= 0, "negative Y translation"
        assert self._translate[Axis.Z] >= 0, "negative Z translation"

    def face(self, src, dest):
        """rotate block so that src faces toward dest"""
        return self.rotate(*self._rotation_faces[src][dest])

    @cached_property
    def local(self):
        """return Voxels in local coordinates"""
        return self._voxels.copy()

    # @cached_property
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

        # step 2: axis swaps. there are 6 cases
        # xyz - no swaps
        # xzy - zy swap
        # yxz - xy swap
        # yzx - xz, yz swap
        # zxy - xy, yz swap
        # zyx - zx, yz swap
        order = (Axis.X, Axis.Y, Axis.Z)

        if (x, z, y) == order:
            rot = rot.swapaxes(Axis.Z, Axis.Y)
        elif (y, x, z) == order:
            rot = rot.swapaxes(Axis.X, Axis.Y)
        elif (y, z, x) == order:
            rot = rot.swapaxes(Axis.X, Axis.Z)
            rot = rot.swapaxes(Axis.Y, Axis.Z)
        elif (z, x, y) == order:
            rot = rot.swapaxes(Axis.X, Axis.Y)
            rot = rot.swapaxes(Axis.Y, Axis.Z)
        elif (z, y, x) == order:
            rot = rot.swapaxes(Axis.Z, Axis.X)
            rot = rot.swapaxes(Axis.Y, Axis.Z)

        # finally, translate the voxels
        x, y, z = self._translate

        if not x and not y and not z:
            return rot

        a, b, c = rot.shape

        v = np.zeros((x + a, y + b, z + c), dtype=bool)
        v[x:, y:, z:] = rot[:,:,:]

        return v

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

    def faces(self):
        """get all exposed faces of the block"""
        xm, ym, zm = self._voxels.shape

        for x, y, z in np.ndindex(self._voxels.shape):
            if not self._voxels[x, y, z]:
                continue

            if z == 0 or not self._voxels[x, y, z - 1]:
                yield Point(x, y, z, Face.Bottom)

            if z + 1 >= zm or not self._voxels[x, y, z + 1]:
                yield Point(x, y, z, Face.Top)

            if x == 0 or not self._voxels[x - 1, y, z]:
                yield Point(x, y, z, Face.Left)

            if x + 1 >= xm or not self._voxels[x + 1, y, z]:
                yield Point(x, y, z, Face.Right)

            if y == 0 or not self._voxels[x, y - 1, z]:
                yield Point(x, y, z, Face.Front)

            if y + 1 >= ym or not self._voxels[x, y + 1, z]:
                yield Point(x, y, z, Face.Back)

    def adjacencies(self):
        """get points adjacent to a block"""
        for x, y, z, f in self.faces():
            a, b, c = self._translate_faces[f]
            yield Point(x + a, y + b, z + c, f)

    def combine(self, other):
        """add two shapes together"""
        va = self.world()
        vb = other.world()

        ax, ay, az = va.shape
        bx, by, bz = vb.shape
        shape = max(ax, bx), max(ay, by), max(az, bz)

        b = np.zeros(shape, dtype=bool)

        b[0:ax, 0:ay, 0:az] = va[:,:,:]

        if (b[0:bx, 0:by, 0:bz] & vb[:,:,:]).any():
            raise IntersectionError(other)

        b[0:bx, 0:by, 0:bz] |= vb[:,:,:]

        return b


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
