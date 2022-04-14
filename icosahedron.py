# import bpy
from math import sqrt
import numpy as np
"""
https://sinestesia.co/blog/tutorials/python-icospheres/
"""


class icosahedron:
    def __init__(self):
        self.scale = 1
        self.subdiv = 4
        self.middle_point_cache = {}

        # Golden ratio
        PHI = (1 + sqrt(5)) / 2

        self.verts = [
            self.vertex(-1, PHI, 0),
            self.vertex(1, PHI, 0),
            self.vertex(-1, -PHI, 0),
            self.vertex(1, -PHI, 0),

            self.vertex(0, -1, PHI),
            self.vertex(0, 1, PHI),
            self.vertex(0, -1, -PHI),
            self.vertex(0, 1, -PHI),

            self.vertex(PHI, 0, -1),
            self.vertex(PHI, 0, 1),
            self.vertex(-PHI, 0, -1),
            self.vertex(-PHI, 0, 1),
        ]

        self.faces = [
            # 5 faces around point 0
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],

            # Adjacent faces
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],

            # 5 faces around 3
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],

            # Adjacent faces
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ]

        # Subdivisions

        for i in range(self.subdiv):
            self.faces_subdiv = []

            for tri in self.faces:
                v1 = self.middle_point(tri[0], tri[1])
                v2 = self.middle_point(tri[1], tri[2])
                v3 = self.middle_point(tri[2], tri[0])

                self.faces_subdiv.append([tri[0], v1, v3])
                self.faces_subdiv.append([tri[1], v2, v1])
                self.faces_subdiv.append([tri[2], v3, v2])
                self.faces_subdiv.append([v1, v2, v3])

            self.faces = self.faces_subdiv

    def vertex(self, x, y, z):
        """ Return vertex coordinates fixed to the unit sphere """

        length = sqrt(x ** 2 + y ** 2 + z ** 2)

        return [(i * self.scale) / length for i in (x, y, z)]

    def middle_point(self, point_1, point_2):
        """ Find a middle point and project to the unit sphere """

        # We check if we have already cut this edge first
        # to avoid duplicated verts
        smaller_index = min(point_1, point_2)
        greater_index = max(point_1, point_2)

        key = '{0}-{1}'.format(smaller_index, greater_index)

        if key in self.middle_point_cache:
            return self.middle_point_cache[key]

        # If it's not in cache, then we can cut it
        vert_1 = self.verts[point_1]
        vert_2 = self.verts[point_2]
        middle = [sum(i) / 2 for i in zip(vert_1, vert_2)]

        self.verts.append(self.vertex(*middle))

        index = len(self.verts) - 1
        self.middle_point_cache[key] = index

        return index

    def getVertices(self):
        self.verts = np.asarray(self.verts)
        self.verts = self.verts[self.verts[:,2] >= 0]
        return list(self.verts)

