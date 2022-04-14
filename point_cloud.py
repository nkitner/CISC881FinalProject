import numpy as np


class PointCloud:
    def __init__(self, points=None):
        if points is None:
            self.points = []
        else:
            self.points = points

        self.shift = 0

    def addPoints(self, points):
        if len(points) == 0:
            self.points = points
        else:
            if isinstance(self.points, list):
                self.points.extend([points])
            else:
                self.points = self.points.tolist()
                self.points.extend([points])

    def shiftToOrigin(self):
        p1, p2 = self.getMinMax3D()
        newshift = (p1 + p2) / 2.0
        for i in range(len(self.points)):
            self.points[i] = self.points[i] - newshift
        self.shift = self.shift + newshift

    def meanValue(self):
        """
        calculates mean value of all points, aka center of gravity
        :return:
        """
        return_vec = [0, 0, 0]
        for i in range(len(self.points)):
            return_vec = return_vec + self.points[i]
        if len(self.points) > 0:
            return return_vec / (len(self.points))
        else:
            return return_vec

    def getMinMax3D(self):
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

    def pointsClostToLine(self, a, b, dx, pc):
        pc.clear_points()
        for i in range(len(self.points)):
            # Equation 7
            t = (b * (self.points[i] - a))
            d = (self.points[i] - (a + (t * b)))
            if np.linalg.norm(d, np.inf) <= dx:
                pc.addPoints(self.points[i])
        pc.points = np.asarray(pc.points)

    def removePoints(self, pc):
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

    def getPoints(self):
        return self.points

    def clear_points(self):
        self.points = []
