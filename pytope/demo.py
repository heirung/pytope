import numpy as np

from pytope import Polytope


# Create a polytope in R^2 with 1 <= x1 <= 3, 3 <= x2 4
lower_bound1 = (1, 2) # [1, 2]' <= x
upper_bound1 = (3, 4) # x <= [3, 4]'
P1 = Polytope(lb=lower_bound1, ub=upper_bound1)
# Print the halfspace representation A*x <= b and H = [A b]
print('P1: ', repr(P1))
print('A =\n', P1.A)
print('b =\n', P1.b)
print('H =\n', P1.H)

# Create a square polytope in R^2 from specifying the four vertices
V2 = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
P2 = Polytope(V2)
# Print the halfspace representation A*x <= b and H = [A b]
print('P2: ', repr(P2))
print('V =\n', P2.V)
