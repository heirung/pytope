# from unittest import TestCase
import unittest

import numpy as np

from pytope import Polytope


class TestPolytope(unittest.TestCase):
  def test___init__(self):

    # Create an R^2 Polytope in H-representation from upper and lower bounds.
    # Check that dimension n and the matrices A, b, and H = [A b] are all
    # set correctly.
    lb = (1, -4)
    ub = (3, -2)

    n = len(ub)
    A = np.vstack((-np.eye(n), np.eye(n)))
    b = np.concatenate((-np.asarray(lb), np.asarray(ub)))[:, None]

    P = Polytope(lb=lb, ub=ub)

    self.assertTrue(P.in_H_rep)
    self.assertEqual(P.n, n)
    self.assertTrue(np.all(P.A == A))
    self.assertTrue(np.all(P.b == b))
    self.assertTrue(np.all(P.H == np.hstack((A, b))))


if __name__ == '__main__':
  unittest.main()
