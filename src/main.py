from voxel import *
from puzzle import puzzle

import random

def r(i):
    if isinstance(i, dict):
        i = i.values()
    return random.choice(tuple(i))

def move(base, target, block, face):
    """we cannot have a sum negative translation"""
    ax, ay, az = block.ltow(face)
    bx, by, bz = base.ltow(target)

    x, y, z = (bx - ax, by - ay, bz - az)

    print(f"move:({x}, {y}, {z})")

    if x < 0:
        ax = 0
        bx = -x
    else:
        ax = x
        bx = 0

    if y < 0:
        ay = 0
        by = -y
    else:
        ay = y
        by = 0

    if z < 0:
        az = 0
        bz = -z
    else:
        az = z
        bz = 0

    print(f"move block:({ax}, {ay}, {az})")
    print(f"move target:({bx}, {by}, {bz})")

    block.translate((ax, ay, az))
    base.translate((bx, by, bz))

def connect(base, target, block, face):
    print(f"adjacency:({target.x}, {target.y}, {target.z}), {target.face!r}")
    print(f"face:({face.x}, {face.y}, {face.z}), {face.face!r}")

    plot(base.local)
    plot(block.local)


    print(f"rotate:{face.face.name}")
    # rotate input block so that face points toward target location
    block.face(face.face, target.face.invert())

    plot(block.world())

    # we want to try all 4 possible rotations along target face normal
    axis = target.face.normal()

    print(f"normal:{axis.name}")

    for i in range(4):
        block.rotate(axis, 1)
        plot(block.world())

        move(base, target, block, face)

        try:
            yield base.combine(block)
        except IntersectionError:
            print("intersecting")
            pass

b1 = puzzle['t']
b2 = puzzle['l']

adjacency = Point(2, 0, 1, Face.Top)
face = Point(0, 0, 0, Face.Front)

b2.face(face.face, adjacency.face.invert())
plot(b2)
b2.rotate(Axis.Z, 1)
plot(b2)

# for block in connect(b1, adjacency, b2, face):
#     plot(block)
