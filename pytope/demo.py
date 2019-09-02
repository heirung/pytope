import numpy as np

from pytope import Polytope

import matplotlib.pyplot as plt


# Create a polytope in R^2 with -1 <= x1 <= 4, -2 <= x2 <= 3
lower_bound1 = (-1, -2)  # [-1, -2]' <= x
upper_bound1 = (4, 3)  # x <= [4, 3]'
P1 = Polytope(lb=lower_bound1, ub=upper_bound1)
# Print the halfspace representation A*x <= b and H = [A b]
print('P1: ', repr(P1))
print('A =\n', P1.A)
print('b =\n', P1.b)
print('H =\n', P1.H)

# Create a square polytope in R^2 from specifying the four vertices
V2 = np.array([[1, 0], [0, -1], [-1, 0], [0, 1]])
P2 = Polytope(V2)
# Print the array of vertices:
print('P2: ', repr(P2))
print('V =\n', P2.V)

# Create a triangle in R^2 from specifying three half spaces (inequalities)
A3 = [[1, 0], [0, 1], [-1, -1]]
b3 = (3, 2, -1.5)
P3 = Polytope(A3, b3)
# Print the halfspace representation A*x <= b and H = [A b]
print('P3: ', repr(P3))
print('A =\n', P3.A)
print('b =\n', P3.b)
print('H =\n', P3.H)
# Determine and print the vertices:
print('V =\n', P3.V)

# Plot all of the polytopes.
# See the matplotlib.patches.Polygon documentation for a list of valid kwargs
fig, ax = plt.subplots()
plt.grid()
plt.axis([-1.5, 4.5, -2.5, 3.5])
P1.plot(ax, fill=False, edgecolor='r', linewidth=2)
P2.plot(ax, facecolor='g', edgecolor='k', linewidth=2, alpha=0.5)
P3.plot(ax, facecolor='b', edgecolor='k', linewidth=2, alpha=0.5)
