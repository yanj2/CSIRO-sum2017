import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
fig = plt.figure()
ax = fig.gca(projection='3d')

X = np.arange(-5, 5, 0.2)
Y = np.arange(-5, 5, 0.2)
X, Y = np.meshgrid(X,Y)
first = 1.5 - X + X * Y
second = 2.25 - X + X * Y ** 2
third = 2.625 - X + X * Y ** 3
Z = (first ** 2 + second ** 2 + third ** 2)

surf = ax.plot_surface(X,Y,Z, cmap=cm.pink)
ax.plot([3],[0.05],'wo')

plt.savefig("surface.jpg")
