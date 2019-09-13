import numpy as np

from pytope import Polytope

import matplotlib.pyplot as plt

np.random.seed(1)

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
b3 = (2, 1, -1.5)
P3 = Polytope(A3, b3)
# Print the halfspace representation A*x <= b and H = [A b]
print('P3: ', repr(P3))
print('A =\n', P3.A)
print('b =\n', P3.b)
print('H =\n', P3.H)
# Determine and print the vertices:
print('V =\n', P3.V)

# P4: P3 shifted by a point p4
p4 = (1.4, 0.7)
P4 = P3 + p4

# P5: P4 shifted by a point p5 (in negative direction)
p5 = [0.4, 2]
P5 = P4 - p5

# P6: P2 scaled by s6 and shifted by p6
s6 = 0.2
p6 = -np.array([[0.4], [1.6]])
P6 = s6 * P2 + p6

# P7: P2 rotated 20 degrees
rot7 = np.pi / 9.0
rot_mat7 = np.array([[np.cos(rot7), -np.sin(rot7)],
                     [np.sin(rot7), np.cos(rot7)]])
P7 = rot_mat7 * P2

# P8: -P6
P8 = -P6

# P9: The convex hull of a set of 30 random points in [1, 2]' <= x [2, 3]'
V9 = np.random.uniform((1, 2), (2, 3), (30, 2))
P9 = Polytope(V9)
P9.minimize_V_rep()

# P10: the Minkowski sum of two squares (one large and one rotated and smaller)
P10_1 = Polytope(lb=(-0.6, -0.6), ub=(0.6, 0.6))
P10_2 = rot_mat7 * Polytope(lb=(-0.3, -0.3), ub=(0.3, 0.3))
P10 = P10_1 + P10_2

# Plot all of the polytopes.
# See the matplotlib.patches.Polygon documentation for a list of valid kwargs
fig1, ax1 = plt.subplots(num=1)
plt.grid()
plt.axis([-1.5, 4.5, -2.5, 3.5])
P1.plot(ax1, fill=False, edgecolor='r', linewidth=2)
P2.plot(ax1, facecolor='g', edgecolor=(0, 0, 0), linewidth=1)
P3.plot(ax1, facecolor='b', edgecolor='k', linewidth=2, alpha=0.5)
P4.plot(ax1, facecolor='lightsalmon')
plt.scatter(P4.V[:, 0], P4.V[:, 1], c='k', marker='x')  # the vertices of P4
# Polytope implements an additional keyword edgealpha:
P5.plot(ax1, fill=False, edgecolor='b', linewidth=8, edgealpha=0.2)
plt.plot(P5.centroid[0], P5.centroid[1], 'o')  # the centroid of P5
P6.plot(ax1, facecolor='g', edgecolor=(0, 0, 0), linewidth=1)
P7.plot(ax1, facecolor='g', edgecolor=(0, 0, 0), alpha=0.3,
        linewidth=1, edgealpha=0.3)
P8.plot(ax1, facecolor='g', edgecolor=(0, 0, 0), alpha=0.3,
        linewidth=1, edgealpha=0.3)
P9.plot(ax1, facecolor='gray', alpha=0.6, edgecolor='k')
plt.plot(V9[:, 0], V9[:, 1], 'or', marker='o', markersize=2) # random points
plt.plot(P9.V[:, 0], P9.V[:, 1], 'og', marker='o', markersize=1) # P9's vertices
plt.title('Demonstration of various polytope operations')

# Plot the Minkowski sum of two squares
fig2, ax2 = plt.subplots(num=2)
plt.grid()
plt.axis([-2.5, 2.5, -2.5, 2.5])
P10_1.plot(ax2, fill=False, edgecolor=(1, 0, 0))
P10_2.plot(ax2, fill=False, edgecolor=(0, 0, 1))
P10.plot(ax2, fill=False,
         edgecolor=(1, 0, 1), linestyle='--', linewidth=2)
for p in P10_1.V: # the smaller square + each of the vertices of the larger one
  (P10_2 + p).plot(ax2, facecolor='grey', alpha=0.4,
                   edgecolor='k', linewidth=0.5)
ax2.legend((r'$P$', r'$Q$', r'$P \oplus Q$'))
plt.title('Minkowski sum of two polytopes')

# Plot two rotated rectangles and their intersection
rot1 = -np.pi / 18.0
rot_mat1 = np.array([[np.cos(rot1), -np.sin(rot1)],
                     [np.sin(rot1), np.cos(rot1)]])
rot2 = np.pi / 18.0
rot_mat2 = np.array([[np.cos(rot2), -np.sin(rot2)],
                     [np.sin(rot2), np.cos(rot2)]])
P_i1 = rot_mat1 * Polytope(lb=(-2,-1),ub=(1,1))
P_i2 = rot_mat2 * Polytope(lb=(0,0),ub=(2,2))
P_i = P_i1 & P_i2  # intersection
fig3, ax3 = plt.subplots(num=3)
plt.grid()
plt.axis([-3.5, 3.5, -3.5, 3.5])
P_i1.plot(fill=False, edgecolor=(1, 0, 0), linestyle='--')
P_i2.plot(fill=False, edgecolor=(0, 0, 1), linestyle='--')
P_i.plot(fill=False,
         edgecolor=(1, 0, 1), linestyle='-', linewidth=2)
ax3.legend((r'$P$', r'$Q$', r'$P \cap Q$'))
plt.title('Intersection of two polytopes')

# Plot two polytopes and their Pontryagin difference
P_m1 = Polytope(lb=(-3, -3), ub=(3, 3))
P_m2 = Polytope([[1, 0], [0, -1], [-1, 0], [0, 1]])
P_diff = P_m1 - P_m2
fig4, ax4 = plt.subplots(num=4)
plt.grid()
plt.axis([-3.5, 3.5, -3.5, 3.5])
P_m1.plot(fill=False, edgecolor=(1, 0, 0))
P_m2.plot(fill=False, edgecolor=(0, 0, 1))
P_diff.plot(fill=False,
            edgecolor=(1, 0, 1), linestyle='--', linewidth=2)
ax4.legend((r'$P$', r'$Q$', r'$P \ominus Q$'))
plt.title('Pontryagin difference of two polytopes')

plt.setp([ax1, ax2, ax3, ax4], xlabel=r'$x_1$', ylabel=r'$x_2$')
