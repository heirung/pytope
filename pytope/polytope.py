

import numpy as np


class Polytope:

  def __init__(self, *args, **kwargs):

    self.n = 0
    self.in_H_rep = False
    self.in_V_rep = False
    self.A = np.empty((0, self.n))
    self.b = np.empty((0, 0))

    # Check how the constructor was called:
    lb_or_ub_passed = any(k in kwargs for k in ('lb', 'ub'))

    # Parse lower and upper bounds. Defaults to [], rather than None, if key is
    # not in kwargs (cleaner below).
    lb = np.atleast_1d(np.squeeze(kwargs.get('lb', [])))
    ub = np.atleast_1d(np.squeeze(kwargs.get('ub', [])))

    if lb_or_ub_passed:
       self._set_Ab_from_bounds(lb, ub) # sets A, b, n, and in_H_rep (to True)

  @property
  def A(self):
    return self._get_A()

  @A.setter
  def A(self, A):
    return self._set_A(A)

  def _get_A(self):
    if not self.in_H_rep and self.in_V_rep:
      self.determineHRep()
    return self._A

  def _set_A(self, A):  # maybe never allow setting/changing A indep of b?
    self._A = np.array(A, ndmin=2)  # prevents shape (n,) when m = 1

  @property
  def b(self):
    return self._get_b()

  @b.setter
  def b(self, b):
    return self._set_b(b)

  def _get_b(self):
    if not self.in_H_rep and self.in_V_rep:
      self.determineHRep()
    return self._b

  def _set_b(self, b):  # risky to allow setting/changing b independent of A
    self._b = np.array(b, ndmin=2).T  # have to ensure np.hstack((A, b)) works

  @property
  def H(self): # the matrix [A b]
    return self._get_H_mat()

  def _get_H_mat(self):
    return np.hstack((self.A, self.b))

  def get_H_rep(self): # the pair -- or tuple -- (A, b)
    return (self.A, self.b)

  def _set_Ab(self, A, b):
    A = np.array(A, ndmin=2)  # ndmin=2 prevents shape (n,) when m = 1
    b = np.array(np.squeeze(b), ndmin=2).T  # IMPROVE (squeeze handles correct b)
    m, n = A.shape
    if b.shape[0] != m:
      raise ValueError(f'A has {m} rows; b has {b.shape[0]} rows!')
    # For rows with b = +- inf: set a_i = 0' and b_i = +- 1 (indicates which
    # direction the constraint was unbounded -- not important)
    inf_rows = np.squeeze(np.isinf(b))  # True also for -np.inf
    if inf_rows.any():
      A[inf_rows, :] = 0
      b[inf_rows] = 1 * np.sign(b[inf_rows])
    self._A = A
    self._b = b
    self.n = n
    self.in_H_rep = True
  #        if not self.in_V_rep:
  #            self.V = np.empty((0,n))
  #            self.R = np.empty((0,n))

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
        raise ValueError(('Dimension of lower bound lb is '
                          f'{lb.size}; should be {n}'))
      A_bound.extend(-np.eye(n))
      b_bound.extend(-lb)
    if ub.size > 0:
      if not ub.size == n:
        raise ValueError(('Dimension of upper bound ub is '
                          f'{ub.size}; should be {n}'))
      A_bound.extend(np.eye(n))
      b_bound.extend(ub)
      self._set_Ab(A_bound, b_bound) # sets n and in_H_rep to True

  def __repr__(self):
    r = ['Polytope ']
    r += ['(empty)' if self.n == 0 else f'in R^{self.n}']
    if self.in_H_rep:
      ineq_spl = 'inequalities' if self.A.shape[0] > 1 else 'inequality'
      r += [f'\n\tHas H-rep with {self.A.shape[0]} {ineq_spl}']
    # if self.hasVRep:
    #     vert_spl = 'vertices' if self.V.shape[0] > 1 else 'vertex'
    #     r += [f'\n\tHas V-rep with {self.nV} {vert_spl}']
    return ''.join(r)

  def __str__(self):
    return f'Polytope in R^{self.n}'
