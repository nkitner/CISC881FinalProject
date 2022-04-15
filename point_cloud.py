"""
This class stores the point clouds, which are the non zero points of the 3D U-Nets prediction
"""

import numpy as np


class PointCloud:
    def __init__(self, points=None):
        if points is None:
            self.points = []
        else:
            self.points = points

        self.shift = 0

    def addPoints(self, points):
        """
        Adds points to instance of class
        :param points: list or array
        """
        if len(points) == 0:
            self.points = points
        else:
            if isinstance(self.points, list):
                self.points.extend([points])
            else:
                self.points = self.points.tolist()
                self.points.extend([points])

    def getMinMax3D(self):
        """
        Calculates the minimum and maximum point of the point cloud
        :return: array of min and max point
        """
        min_point = [0, 0, 0]
        max_point = [0, 0, 0]
        if len(self.points) > 0:
            for i in range(len(self.points)):
                if min_point[0] > self.points[i][0]: min_point[0] = self.points[i][0]
                if min_point[1] > self.points[i][1]: min_point[1] = self.points[i][1]
                if min_point[2] > self.points[i][2]: min_point[2] = self.points[i][2]

                if max_point[0] < self.points[i][0]: max_point[0] = self.points[i][0]
                if max_point[1] < self.points[i][1]: max_point[1] = self.points[i][1]
                if max_point[2] < self.points[i][2]: max_point[2] = self.points[i][2]
        return np.asarray(min_point), np.asarray(max_point)

    def pointsClosestToLine(self, a, b, dx, pc):
        """
        Calculates the points in the pointcloud, pc, closest to the line, which is given by parameters a and b.
        :param a: point on line
        :param b: direction of line
        :param dx: maximum distance to line
        :param pc: PointCloud
        """
        pc.clear_points()
        for i in range(len(self.points)):
            # Equation 7
            t = (b * (self.points[i] - a))
            d = (self.points[i] - (a + (t * b)))
            if np.linalg.norm(d, np.inf) <= dx:
                pc.addPoints(self.points[i])
        pc.points = np.asarray(pc.points)

    def removePoints(self, pc):
        """
        Remove specified points, pc, from pointcloud
        :param pc: PointCloud
        """
        if len(pc.points) == 0: return
        j = 0
        k = 0
        new_points = []

        for i in range(len(self.points)):
            if j < len(pc.points):
                comparison = self.points[i] == pc.points[j]
                if comparison.all():
                    j += 1
                else:
                    new_points.append(self.points[i])
                k = i
            else:
                break

        for i in range(k, len(self.points)):
            new_points.append(self.points[i])

        self.points = np.asarray(new_points)

    def clear_points(self):
        """
        Clears all points
        """
        self.points = []
