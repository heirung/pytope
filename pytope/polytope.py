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
      raise ValueError('Cannot specify V in addition to A and b')

    if (V_or_R_passed or A_and_b_passed) and lb_or_ub_passed:
      raise ValueError('Cannot specify bounds in addition to V, R, A, or b')

    if ('A' in kwargs) ^ ('b' in kwargs): # XOR
      raise ValueError('Cannot pass just one of A and b as keywords')

    # Parse V if passed as the only positional argument or as a keyword
    if V_or_R_passed:
      V = kwargs.get('V')  # None if not
      if len(args) == 1:  # V is the only positional argument
        # Prevent passing vertex lists as first positional argument and as
        # keyword, as in P = Polytope(V_list1, V=V_list2):
        if V is not None:
          raise ValueError('V cannot be passed as the first argument and as a ' 
                           'keyword')
        V = args[0]  # The first positional argument is V
      # Parse R if passed as keyword
      if 'R' in kwargs:
        raise ValueError('Support for rays currently not implemented')
      self._set_V(V)

    # Parse A and b if passed as first two positional arguments or as keywords
    if A_and_b_passed:
      A = kwargs.get('A')  # None if not
      b = kwargs.get('b')  # None if not
      # Prevent passing A or b in both args and kwargs:
      if len(args) == 2:  # A and b passed in args
        if A is not None or b is not None:  # A or b passed in kwargs
          raise ValueError(('A (or b) cannot be passed as the first (or second)'
                            ' argument and as a keyword'))
        A, b = args[:2]  # The first two positional arguments are A and b
      self._set_Ab(A, b)

    if lb_or_ub_passed:
      # Parse lower and upper bounds. Defaults to [], rather than None,
      # if key is not in kwargs (cleaner below).
      lb = np.atleast_1d(np.squeeze(kwargs.get('lb', [])))
      ub = np.atleast_1d(np.squeeze(kwargs.get('ub', [])))
      if (lb > ub).any():
        raise ValueError('No lower bound can be greater than an upper bound')
      self._set_Ab_from_bounds(lb, ub)  # sets A, b, n, and in_H_rep (to True)

  # To enable linear mapping (a numpy ndarray M multiplies a polytope P: M * P)
  # with the * operator, set __array_ufunc__ = None so that the result is not
  # M.__mul__(P), which is an ndarray of scalar multiples of P: {m_ij * P}. With
  # __array_ufunc__ set to None here, the result of M * P is P.__rmul__(M),
  # which calls P.linear_map(M). See
  # https://docs.scipy.org/doc/numpy-1.13.0/neps/ufunc-overrides.html
  # An alternative is to set __array_priority__ = 1000 or some other high value.
  __array_ufunc__ = None

  @property
  def A(self):
    return self._get_A()

  @A.setter
  def A(self, A):
    return self._set_A(A)

  def _get_A(self):
    if not self.in_H_rep and self.in_V_rep:
      # self.set_Ab_from_V()
      if self._A.shape[0] == 0: # A is empty, ensure it has correct dimension
        self._A = np.empty((0, self.n))
    return self._A

  def _set_A(self, A):  # careful if setting A independent of b
    self._A = np.array(A, ndmin=2, dtype=float)  # prevents shape (n,) for m = 1

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
  def H(self):  # the matrix H = [A b]
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
    if A.shape[0] and b.shape[0]:  # in_H_rep if A and b are not empty arrays
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
                          f'should be {n}'))
      A_bound.extend(-np.eye(n))
      b_bound.extend(-lb)
    if ub.size > 0:
      if not ub.size == n:
        raise ValueError((f'Dimension of upper bound ub is f{ub.size}; '
                          f'should be {n}'))
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
    if nV:  # in_V_rep if V is not an empty array
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

  def __repr__(self):  # TODO: does not print nicely for list of Polytopes
    r = ['Polytope ']
    r += ['(empty)' if self.n == 0 else f'in R^{self.n}']
    if self.in_H_rep:
      ineq_spl = 'inequalities' if self.A.shape[0] != 1 else 'inequality'
      r += [f'\n\tIn H-rep: {self.A.shape[0]} {ineq_spl}']
    if self.in_V_rep:
      vert_spl = 'vertices' if self.V.shape[0] != 1 else 'vertex'
      r += [f'\n\tIn V-rep: {self.nV} {vert_spl}']
    return ''.join(r)

  def __str__(self):
    return f'Polytope in R^{self.n}'

  def __bool__(self): # return true if the polytope is not empty
    return self.in_V_rep or self.in_H_rep

  def __neg__(self):
    neg_P = Polytope()
    if self.in_V_rep:
      neg_P.V = -self.V
    if self.in_H_rep:
      neg_P._set_Ab(-self.A, self.b)
    return neg_P

  def __add__(self, other):
    if isinstance(other, Polytope):
      raise ValueError('Minkowski sum of two polytopes not implemented')
    else:
      return P_plus_p(self, other)

  def __radd__(self, other):
    return P_plus_p(self, other)

  def __sub__(self, other):
    if isinstance(other, Polytope):
      raise ValueError('Pontryagin difference of two polytopes not implemented')
    else:
      return P_plus_p(self, other, subtract_p=True)

  # def __rsub__(self, other):  # p - P -- raise TypeError (not well defined)

  def __mul__(self, other):
    # self * other: scaling if other is scalar, inverse linear map if other is a
    # matrix (inverse=True has no effect on scaling).
    return self.multiply(other, inverse=True)

  def __rmul__(self, other):
    # other * self: scaling if other is scalar, linear map if other is a matrix.
    return self.multiply(other)

  def multiply(self, other, inverse=False):
    # scale: s * P or P * s with s a scalar and P a polytope
    # linear map: M * P with M a matrix
    # inverse linear map: P * M with M a matrix
    if isinstance(other, Polytope):
      raise ValueError('Product of two polytopes not implemented')
    # TODO: now assuming a numeric type that can be squeezed -- fix
    # other can be a scalar (ndim=0) or a matrix (ndim=2)
    factor = np.squeeze(other)
    if factor.ndim == 0:
      return scale(self, other)
    elif factor.ndim == 2:
      if inverse:
        raise ValueError('Inverse linear map P * M not implemeted')
      else:
        return linear_map(other, self)
    else:
      raise ValueError('Mulitplication with numeric type other than scalar and '
                       'matrix not implemented')

  def determine_V_rep(self):  # also sets rays R (not implemented)
    # Vertex enumeration from halfspace representation using cddlib.
    # TODO: shift the polytope to the center? (centroid? Chebyshev center?)
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
    if tV.any(): # tV == [] if the Polytope is empty
      V_rows = tV[:, 0] == 1  # bool array of which rows contain vertices
      R_rows = tV[:, 0] == 0  # and which contain rays (~ V_rows)
      V = tV[V_rows, 1:]  # array of vertices (one per row)
      R = tV[R_rows, 1:]  # and of rays
      if R_rows.any():
        raise ValueError('Support for rays not implemented')
    else:
      V = np.empty((0, self.n))
    self._set_V(V)

  def plot(self, ax, **kwargs):
    # Plot Polytope. Add separate patches for the fill and the edge, so that
    # the fill is below the gridlines (at zorder 0.4) and the edge edge is
    # above (at zorder 2, same as regular plot). Gridlines are at zorder 1.5
    # by default and at  (1 is default), which is below gridlines, which are
    # at zorder 1.5 by default 0.5 if setting ax.set_axisbelow(True),
    # so plotting the fill at 0.4 ensures the fill is always below the grid.
    h_patch = [] # handle, return as tuple
    V_sorted = self.V_sorted()
    # Check for edgecolor. Default is (0, 0, 0, 0), with the fourth 0 being
    # alpha (0 is fully transparent). Passingedgecolor='r', e.g., later
    # translates to (1.0, 0.0, 0.0, 1).
    edgecolor_default = (0, 0, 0, 0)
    edgecolor = kwargs.pop('edgecolor', edgecolor_default)
    # Handle specific cases: edge transparent or set explicitly to None
    if ((type(edgecolor) is tuple and len(edgecolor) == 4 and edgecolor[3] == 0)
        or edgecolor is None):
      edgecolor = edgecolor_default
    # Added keyword for edge transparency: edgealpha (default: 1.0)
    edgealpha_default = 1.0
    edgealpha = kwargs.pop('edgealpha', edgealpha_default)
    # Check for fill and facecolor. fill is default True. The Polygon
    # documentation lists facecolor=None as valid but it results in blue
    # filling (preserving this behavior).
    fill = kwargs.pop('fill', True)
    facecolor = kwargs.pop('facecolor', None)
    # test for any non-empty string and rbg tuple, and handle black as rbg
    if any(edgecolor) or edgecolor == (0, 0, 0):
      # Plot edge:
      temp_dict = {**kwargs,
                   **{'fill': False, 'edgecolor': edgecolor, 'zorder': 2,
                      'alpha': edgealpha}}
      h_patch.append(ax.add_patch(Polygon(V_sorted, **temp_dict)))
    if fill or facecolor:
      # Plot fill:
      temp_dict = {**kwargs,
                   **{'edgecolor': None, 'facecolor': facecolor, 'zorder': 0.4}}
      h_patch.append(ax.add_patch(Polygon(V_sorted, **temp_dict)))
    return tuple(h_patch)  # handle(s) to the patch(es)

  def plot_basic(self, ax, **kwargs):
    h_patch = ax.add_patch(Polygon(self.V_sorted(), **kwargs))
    return h_patch # handle to the patch


def P_plus_p(P, point, subtract_p=False):
  # Polytope + point: The sum of a polytope in R^n and an n-vector -- P + p
  # If subtract_p == True, compute P - p instead. This implementation allows
  # writing Polytope(...) - (1.0, 0.5) or similar by overloading __sub__ with
  # this function.
  p = np.array(np.squeeze(point), dtype=float)[:, np.newaxis]
  if p.size != P.n or p.shape[1] != 1 or p.ndim != 2:  # ensure p is n x 1
    raise ValueError(f'The point must be a {P.n}x1 vector')
  if subtract_p: p = -p # add -p if 'sub'
  P_shifted = Polytope()
  # V-rep: The sum is all vertices of P shifted by p.
  # Not necessary to tile/repeat since p.T broadcasts, but could do something
  # like np.tile(q.T, (P.nV, 1)) or np.repeat(p.T, P.nV, axis=0).
  if P.in_V_rep:
    P_shifted.V = P.V + p.T
  # H-rep: A * x <= b shifted by p means that any point q = x + p in the shifted
  # polytope satisfies A * q <= b + A * p, since
  #   A * x <= b  ==>  A * (q - p) <= b  ==>  A * q <= b + A * p
  # That is, the shifted polytope has H-rep (A, b + A * p).
  if P.in_H_rep:
    P_shifted._set_Ab(P.A, P.b + P.A @ p)  # TODO: redesign this
  return P_shifted

def scale(P, s):
  # TODO: handle s == 0 specifically
  P_scaled = Polytope()
  if P.in_H_rep:
    P_scaled._set_Ab(P.A, P.b * s)  # TODO: redesign this
  if P.in_V_rep:
    P_scaled.V = P.V * s
  return P_scaled

def linear_map(M, P):
  # Compute the linear map M * P through the vertex representation of V. If P
  # does not have a vertex representation, determine the vertex representation
  # first. In this case, the H-representation of M * P is NOT determined before
  # returning M * P.
  n = M.shape[1]
  if P.n != n:
    raise ValueError('Dimension of M and P do not agree in linear map M * P')
  # TODO: implement linear map in H-rep for invertible M
  # TODO: M_P = Polytope(P.V @ M.T), if P.in_H_rep: M_P.determine_H_rep()?
  return Polytope(P.V @ M.T)
