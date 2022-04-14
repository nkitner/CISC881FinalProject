import numpy as np
from CISC881FinalProject import icosahedron as icosahedron
from CISC881FinalProject.point_cloud import PointCloud
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

"""
source: https://github.com/cdalitz/hough-3d-lines
"""


def getDirectionalVectors():
    ico = icosahedron.icosahedron()
    return ico.getVertices()


def getPointCloud(arr):
    point_cloud = []
    for x in range(128):
        for y in range(128):
            for z in range(128):
                if arr[x, y, z] == 1:
                    point_cloud.append([x, y, z])
    return np.asarray(point_cloud)


def median(x):
    m, n = x.shape
    middle = np.arange((m - 1) >> 1, (m >> 1) + 1)
    x = np.partition(x, middle, axis=0)
    return x[middle].mean(axis=0)


def remove_outliers(data, thresh=2.0):
    m = median(data)
    s = np.abs(data - m)
    return data[(s < median(s) * thresh).all(axis=1)]


class Hough:
    def __init__(self, minP, maxP, var_dx, dir_arr):
        self.num_b = len(dir_arr)
        self.direction_arr = dir_arr
        d_max = np.linalg.norm(maxP, 1)
        d_min = np.linalg.norm(minP, 1)
        self.max_x = max(d_max, d_min)
        self.range_x = 2 * self.max_x
        self.dx = var_dx
        self.num_x = np.round(self.range_x / self.dx)
        self.votingSpace = np.zeros(int(self.num_x * self.num_x * self.num_b))

    def update_point_cloud(self):
        self.addPointCloud(self.pointCloud)

    def addPointCloud(self, pc):
        self.votingSpace = np.zeros(int(self.num_x * self.num_x * self.num_b))
        self.pointCloud = pc
        i = 0
        for point in pc.points:
            self.pointVote(point, True)
            i += 1
        print(np.unique(self.votingSpace))

    def pointVote(self, point, add):

        for j in range(self.num_b):
            b = self.direction_arr[j]
            beta = 1 / (1 + b[2])

            x_new = ((1 - (beta * (b[0] * b[0]))) * point[0]) - ((beta * (b[0] * b[1])) * point[1]) - (b[0] * point[2])

            y_new = ((-beta * (b[0] * b[1])) * point[0]) + ((1 - (beta * (b[1] * b[1]))) * point[1]) - (
                    b[1] * point[2])
            x_i = np.round((x_new + self.max_x) / self.dx)
            y_i = np.round((y_new + self.max_x) / self.dx)

            index = int((x_i * self.num_x * self.num_b) + (y_i * self.num_b) + j)

            if index < len(self.votingSpace):
                if add:
                    self.votingSpace[index] += 1
                else:
                    self.votingSpace[index] -= 1

    def removePointCloud(self, pc):
        i = 0
        for point in pc.points:
            self.pointVote(point, False)
            i += 1

    def getLines(self):
        a = []

        index = np.argmax(self.votingSpace)
        votes = self.votingSpace[index]

        x = int(index / (self.num_x * self.num_b))
        index -= int(x * self.num_x * self.num_b)
        x = x * self.dx - self.max_x

        y = int(index / self.num_b)
        index -= int(y * self.num_b)
        y = y * self.dx - self.max_x

        b = self.direction_arr[index]

        a.append((1 - ((b[0] * b[0]) / (1 + b[2]))) - y * ((b[0] * b[1]) / (1 + b[2])))
        a.append(x * (-((b[0] * b[1]) / (1 + b[2]))) + y * (1 - ((b[1] * b[1]) / (1 + b[2]))))
        a.append(- x * b[0] - y * b[1])

        return votes, a, b


def getPolynomial(pc, num_slices):
    new_pc = remove_outliers(pc, thresh=5)
    if len(new_pc) > 50:
        x, y, z = new_pc.T
    else:
        x, y, z = pc.T

    fity = np.poly1d(np.polyfit(x, y, 2))
    fitz = np.poly1d(np.polyfit(x, z, 2))

    x = np.arange(num_slices)
    new_y = fity(x)
    new_z = fitz(x)

    return np.asarray([x, new_y, new_z]), [fity.coefficients[0], fitz.coefficients[0]]


def performHough(pred_arr, opt_nlines, threshold=0.5):
    pred_arr[pred_arr > threshold] = 1
    pred_arr[pred_arr <= threshold] = 0
    pred_arr = skeletonize(pred_arr)
    pred_arr = pred_arr / 255

    # Get directional vectors using icosahedron method described in paper
    dir_vecs = np.asarray(getDirectionalVectors())
    new_dir_vecs = []
    for i in dir_vecs:
        if (0.1 >= i[1] >= -0.1) and (0.1 >= i[2] >= -0.1):
            new_dir_vecs.append(i)

    dir_vecs = np.asarray(new_dir_vecs)
    opt_dx = 10

    # Initialize point cloud
    x_pt, y_pt, z_pt = np.where(pred_arr == 1)
    point_cloud = np.array([x_pt, y_pt, z_pt]).T
    pointCloud = PointCloud(point_cloud)

    # Plot 3D U-Net predictions
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    point_cloud = point_cloud.T
    ax.scatter(point_cloud[0], point_cloud[1], point_cloud[2], 'r')
    ax.view_init(60, 0)
    plt.show(block=False)

    min_pt_shifted, max_pt_shifted = pointCloud.getMinMax3D()

    # First Hough
    hough = Hough(min_pt_shifted, max_pt_shifted, opt_dx, dir_vecs)
    hough.addPointCloud(pointCloud)

    # Iterative hough transform
    y = PointCloud()
    nvotes = 3
    nlines = 0

    curr_points = []
    curr_dirs = []

    pts_tracking = []
    while len(hough.pointCloud.points) > 1 and (opt_nlines == 0 or opt_nlines > nlines) and (nvotes > 2):
        print("Lines: ", nlines)
        hough.pointCloud.removePoints(y)
        if len(y.points) > 0:
            hough.update_point_cloud()

        nvotes, a, b = hough.getLines()
        a[0] = 0
        print("nvotes: ", nvotes)
        hough.pointCloud.pointsClostToLine(a, b, opt_dx, y)

        nlines += 1

        curr_points.append(a)
        curr_dirs.append(b)

        x_points = y.points[:, 0]
        range_x = np.max(x_points) - np.min(x_points)
        pts, slopes = getPolynomial(y.points, range_x)
        pts_tracking.append(pts)

    # Append all points together to plot
    x_new, y_new, z_new = zip(*pts_tracking)
    x_new = np.concatenate(list(x_new)).ravel()
    y_new = np.concatenate(list(y_new)).ravel()
    z_new = np.concatenate(list(z_new)).ravel()

    # Plot the curved catheters and the model predictions
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_new, y_new, z_new, 'b')
    ax.scatter(x_pt, y_pt, z_pt, 'r')
    ax.view_init(60, 0)
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    plt.show(block=False)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_new, y_new, z_new, 'b')
    ax.view_init(60, 0)
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    plt.show(block=False)

    return pts_tracking


pred_arr = np.load("./pred_array.npy")
pred_arr = pred_arr[:, :, :]
results = performHough(pred_arr, 16, threshold=0.25)
plt.show()