import numpy as np
import cdd # pycddlib -- for vertex enumeration from H-representation
from matplotlib.patches import Polygon

class Polytope:

  def __init__(self, *args, **kwargs):

    self.n = 0
    self.in_H_rep = False
    self.in_V_rep = False
    self._A = np.empty((0, self.n))
    self._b = np.empty((0, 1))
    self._V = np.empty((0, self.n))

    # Check how the constructor was called. TODO: account for V=None, V=[],
    # or similar
    V_or_R_passed = len(args) == 1 or any(kw in kwargs for kw in ('V', 'R'))
    A_and_b_passed = len(args) == 2 or all(k in kwargs for k in ('A', 'b'))
    lb_or_ub_passed = any(kw in kwargs for kw in ('lb', 'ub'))

    if V_or_R_passed and A_and_b_passed:
      raise ValueError('Cannot specify V in addition to A and b.')

    if (V_or_R_passed or A_and_b_passed) and lb_or_ub_passed:
      raise ValueError('Cannot specify bounds in addition to V, R, A, or b.')

    if ('A' in kwargs) ^ ('b' in kwargs): # XOR
      raise ValueError('Cannot pass just one of A and b as keywords.')

    # Parse V if passed as the only positional argument or as a keyword
    V = kwargs.get('V')  # None if not
    if len(args) == 1:  # V is the only positional argument
      # Catch the case of passing different vertex lists to the constructor, as
      # in P = Polytope(V_list1, V=V_list2):
      if V and np.any(V != args[0]):
        raise ValueError(('V passed as first argument and as a keyword, but '
                          'with different values!'))
      V = args[0]  # (overwrites the kwarg if both were passed)
    # Parse R if passed as keyword
    if 'R' in kwargs:
      raise ValueError('Rays are not implemented.')

    if V_or_R_passed:
      self._set_V(V)

    if A_and_b_passed:
      A = kwargs.get('A')  # None if not
      b = kwargs.get('b')  # None if not
      #  Catch the case of passing different A or b as arg and kwarg:
      if len(args) == 2:
        if (A and A != args[0]) or (b and b != args[1]):
          raise ValueError(('A (or b) passed as first (or second) argument and '
                            'as a keyword, but with different values'))
        A, b = args[:2]  # (overwrites the kwarg if both were passed)
      self._set_Ab(A, b)

    if lb_or_ub_passed:
      # Parse lower and upper bounds. Defaults to [], rather than None, if key is
      # not in kwargs (cleaner below).
      lb = np.atleast_1d(np.squeeze(kwargs.get('lb', [])))
      ub = np.atleast_1d(np.squeeze(kwargs.get('ub', [])))
      self._set_Ab_from_bounds(lb, ub)  # sets A, b, n, and in_H_rep (to True)

  @property
  def A(self):
    return self._get_A()

  @A.setter
  def A(self, A):
    return self._set_A(A)

  def _get_A(self):
    # if not self.in_H_rep and self.in_V_rep:
    #   self.set_Ab_from_V()
    return self._A

  def _set_A(self, A):  # careful if setting A independent of b
    self._A = np.array(A, ndmin=2, dtype=float)  # prevents shape (n,) when m = 1

  @property
  def b(self):
    return self._get_b()

  @b.setter
  def b(self, b):
    return self._set_b(b)

  def _get_b(self):
    # if not self.in_H_rep and self.in_V_rep:
    #   self.set_Ab_from_V()
    return self._b

  def _set_b(self, b):  # Careful when setting b indep of A (use for scaling P)
    # Ensure np.hstack((A, b)) works by having b be a column vector
    self._b = np.array(np.squeeze(b), dtype=float)[:, np.newaxis]

  @property
  def H(self):  # the matrix [A b]
    return self._get_H_mat()

  def _get_H_mat(self):
    return np.hstack((self.A, self.b))

  def get_H_rep(self):  # the pair -- or tuple -- (A, b)
    return (self.A, self.b)

  def _set_Ab(self, A, b):
    A = np.array(A, ndmin=2)  # ndmin=2 prevents shape (n,) when m = 1
    b = np.squeeze(b)[:, np.newaxis]  # overlaps with _set_b(b)
    m, n = A.shape
    if b.shape[0] != m:
      raise ValueError(f'A has {m} rows; b has {b.shape[0]} rows!')
    # For rows with b = +- inf: set a_i = 0' and b_i = +- 1 (indicates which
    # direction the constraint was unbounded -- not important)
    inf_rows = np.squeeze(np.isinf(b))  # True also for -np.inf
    if inf_rows.any():
      A[inf_rows, :] = 0
      b[inf_rows] = 1 * np.sign(b[inf_rows])
    self._set_A(A)
    self._set_b(b)
    self.n = n
    self.in_H_rep = True

  def _set_Ab_from_bounds(self, lb, ub):
    A_bound = []
    b_bound = []
    n = np.max((lb.size, ub.size))  # 0 for bound not specified
    # Test size of lb instead of "if lb" since the bool value of
    # np.array(None) is False whereas np.array([]) is ambiguous. Since
    # np.array(None).size is 1 and np.array([]).size is 0 (default, set
    # above), lb.size > 0 implies lb was a kwarg.
    if lb.size > 0:
      if not lb.size == n:
        raise ValueError((f'Dimension of lower bound lb is {lb.size}; '
                          f'should be {n}.'))
      A_bound.extend(-np.eye(n))
      b_bound.extend(-lb)
    if ub.size > 0:
      if not ub.size == n:
        raise ValueError((f'Dimension of upper bound ub is f{ub.size}; '
                          f'should be {n}.'))
      A_bound.extend(np.eye(n))
      b_bound.extend(ub)
      self._set_Ab(A_bound, b_bound)  # sets n and in_H_rep to True

  @property
  def V(self):
    return self._get_V()

  @V.setter
  def V(self, V):
    return self._set_V(V)

  def _get_V(self):
    if not self.in_V_rep:
      if self.in_H_rep:
        self.determine_V_rep()
      else:
        raise ValueError('Polytope in neither H nor V representation')
    return self._V

  def _set_V(self, V):
    self._V = np.asarray(V, dtype=float)
    nV, n = self._V.shape
    self.nV = nV
    self.n = n
    self.in_V_rep = True

  @property
  def centroid(self):
    return np.sum(self.V, axis=0) / self.nV

  def V_sorted(self):
    # Sort vertices (increasing angle: the point (x1, x2) = (1, 0) has angle 0).
    # np.arctan2(y, x) returns angles in the range [-pi, pi], so vertices are
    # sorted clockwise from 9:00 (angle pi). Note that the first argument is y.
    # Mainly for plotting and not implemented for n != 2.
    if self.n != 2:
      raise  ValueError('V_sorted() not implemented for n != 2')
    c = self.centroid
    order = np.argsort(np.arctan2(self.V[:, 1] - c[1], self.V[:, 0] - c[0]))
    return self.V[order, :]

  def __repr__(self):
    r = ['Polytope ']
    r += ['(empty)' if self.n == 0 else f'in R^{self.n}']
    if self.in_H_rep:
      ineq_spl = 'inequalities' if self.A.shape[0] > 1 else 'inequality'
      r += [f'\n\tIn H-rep: {self.A.shape[0]} {ineq_spl}']
    if self.in_V_rep:
      vert_spl = 'vertices' if self.V.shape[0] > 1 else 'vertex'
      r += [f'\n\tIn V-rep: {self.nV} {vert_spl}']
    return ''.join(r)

  def __str__(self):
    return f'Polytope in R^{self.n}'

  def __add__(self, other):
    if isinstance(other, Polytope):
      raise ValueError('Minkowski sum of two polytopes not implemented')
    else:
      return P_plus_p(self, other)

  def __radd__(self, other):
    return P_plus_p(self, other)

  def determine_V_rep(self):  # also sets rays R (not implemented)
    # Vertex enumeration from halfspace representation using cddlib.
    # TODO: shift the polytope to the center?
    #        print('b: ', self.b)
    #        print(self.b.shape, ' h-stacked with ', self.A.shape)
    # cdd uses the halfspace representation [b, -A] | b - Ax >= 0
    b_mA = np.hstack((self.b, -self.A))  # [b, -A]
    H = cdd.Matrix(b_mA, number_type='float')
    H.rep_type = cdd.RepType.INEQUALITY
    H_P = cdd.Polyhedron(H)
    # From the get_generators() documentation: For a polyhedron described as
    #   P = conv(v_1, ..., v_n) + nonneg(r_1, ..., r_s),
    # the V-representation matrix is [t V] where t is the column vector with
    # n ones followed by s zeroes, and V is the stacked matrix of n vertex
    # row vectors on top of s ray row vectors.
    P_tV = H_P.get_generators()  # type(P_tV):  <class 'cdd.Matrix'>
    tV = np.array(P_tV[:])
    V_rows = tV[:, 0] == 1  # bool array of which rows contain vertices
    R_rows = tV[:, 0] == 0  # and which contain rays (~ V_rows)
    V = tV[V_rows, 1:]  # array of vertices (one per row)
    R = tV[R_rows, 1:]  # and of rays
    self._set_V(V)
    if R_rows.any():
      raise ValueError('Support for rays not implemented')

  def plot(self, ax, **kwargs):
    h_patch = ax.add_patch(Polygon(self.V_sorted(), **kwargs))
    return h_patch # handle to the patch


def P_plus_p(P, point):
  # Polytope + point: The sum of a polytope in R^n and an n-vector
  p = np.array(np.squeeze(point), dtype=float)[:, np.newaxis]
  if p.size != P.n or p.shape[1] != 1 or p.ndim != 2:  # ensure p is n x 1
    raise ValueError(f'The point must be a {P.n}x1 vector')
  # V-rep: The sum is all vertices of P shifted by p.
  # ﻿Not necessary to tile/repeat since p.T broadcasts, but could do something
  # like np.tile(q.T, (P.nV, 1)) or np.repeat(p.T, P.nV, axis=0).
  if P.in_V_rep:
    P_plus_p_V = P.V + p.T
  elif P.in_H_rep:
    raise ValueError('Sum of Polytope and point not implemented for H-rep')
  return Polytope(P_plus_p_V)