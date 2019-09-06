
import unittest

import numpy as np

from pytope import Polytope


class TestPolytope(unittest.TestCase):
  def test___init__(self):

    # Create an R^2 Polytope in H-representation from upper and lower bounds.
    # Check that dimension n and the matrices A, b, and H = [A b] are all
    # set correctly.
    lb1 = (1, -4)
    ub1 = (3, -2)

    n1 = len(ub1)
    A1 = np.vstack((-np.eye(n1), np.eye(n1)))
    b1 = np.concatenate((-np.asarray(lb1), np.asarray(ub1)))[:, np.newaxis]
    V1 = [[1, -4], [1, -2], [3, -4], [3, -2]]

    P1 = Polytope(lb=lb1, ub=ub1)

    self.assertTrue(P1.in_H_rep)
    self.assertFalse(P1.in_V_rep)
    self.assertEqual(P1.n, n1)
    self.assertTrue(np.all(P1.A == A1))
    self.assertTrue(np.all(P1.b == b1))
    self.assertTrue(np.all(P1.H == np.hstack((A1, b1))))
    self.assertTrue(all(v in P1.V.tolist() for v in V1))
    self.assertTrue(P1.in_V_rep)
    self.assertTrue(np.issubdtype(P1.A.dtype, np.float))
    self.assertTrue(np.issubdtype(P1.b.dtype, np.float))
    self.assertTrue(np.issubdtype(P1.H.dtype, np.float))
    self.assertTrue(np.issubdtype(P1.V.dtype, np.float))

    # Create an R^2 Polytope in V-representation from a list of four vertices
    # Check that dimension n and vertex list V are set correctly
    V2 = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    n2 = V2.shape[1]

    P2 = Polytope(V2)

    self.assertTrue(P2.in_V_rep)
    self.assertFalse(P2.in_H_rep)
    self.assertEqual(P2.n, n2)
    self.assertTrue(all(v in P2.V.tolist() for v in V2.tolist()))
    self.assertTrue(np.issubdtype(P2.V.dtype, np.float))

    # Create an R^2 Polytope in H-representation by specifying A and b in
    # Ax <= b. Check that dimension n and vertex list V are set correctly
    A3 = [[-1, 0], [0, -1], [1, 1]]
    b3 = (0, 0, 2)
    n3 = 2
    H3 = np.hstack((A3, np.asarray(b3, dtype=float)[:, np.newaxis]))
    V3 = [[0, 0], [0, 2], [2, 0]]

    P3 = Polytope(A3, b3)

    self.assertTrue(P3.in_H_rep)
    self.assertFalse(P3.in_V_rep)
    self.assertEqual(P3.n, n3)
    self.assertTrue(np.all(P3.A == np.asarray(A3, dtype=float)))
    self.assertTrue(np.all(P3.b == np.asarray(b3, dtype=float)[:, np.newaxis]))
    self.assertTrue(np.all(P3.H == H3))
    self.assertTrue(all(v in P3.V.tolist() for v in V3))
    self.assertTrue(P3.in_V_rep)
    self.assertTrue(np.issubdtype(P3.A.dtype, np.float))
    self.assertTrue(np.issubdtype(P3.b.dtype, np.float))
    self.assertTrue(np.issubdtype(P3.H.dtype, np.float))
    self.assertTrue(np.issubdtype(P3.V.dtype, np.float))

    # Ensure illegal use of the constructor raises an error.
    with self.assertRaises(ValueError):
      Polytope(V2, A=A3, b=b3)
    with self.assertRaises(ValueError):
      Polytope(A3, b3, V=V2)
    with self.assertRaises(ValueError):
      Polytope(A=A3)
    with self.assertRaises(ValueError):
      Polytope(b=b3)
    with self.assertRaises(ValueError):
      Polytope(V2, lb=lb1, ub=ub1)
    with self.assertRaises(ValueError):
      Polytope(A3, b3, lb=lb1, ub=ub1)

  def test_P_plus_p(self):
    # Test that a 2D Polytope plus a few different 2D vectors give the
    # correct shift when using the + and - operators. Test by manually
    # computing the shifted vertices and comparing the result to the vertices
    # of the Polytope that results from the addition using the +/- operators.
    # Test that the vector can be in a variety of formats, including tuple,
    # list, list of lists, 1D numpy array, and 2D numpy row vector (array). Test
    # by constructing the Polytope both from the vertex list V and from a half-
    # space representation (A, b)
    V = np.array([[-1, 0], [1, 0], [0, 1]])
    PV = Polytope(V)
    A = [[-1, 0], [0, -1], [1, 1]]
    b = [-2, -3, 8]
    PH = Polytope(A, b)
    PH_V = PH.V
    points = [(1, 1),
              [-1, 2],
              [[1.5], [-0.5]],
              np.array([-2, -0.1]),
              np.array([[-2], [-0.1]])]
    p_columns = [np.array(np.squeeze(p), dtype=float)[:, np.newaxis]
                 for p in points]
    PV_plus_p_results = [PV + p for p in points]
    PV_minus_p_results = [PV - p for p in points]
    PH_plus_p_results = [PH + p for p in points]
    PH_minus_p_results = [PH - p for p in points]
    self.assertTrue(all([(PVpp.V == V + p.T).all()
                         for PVpp, p in zip(PV_plus_p_results, p_columns)]))
    self.assertTrue(all([(PVmp.V == V - p.T).all()
                         for PVmp, p in zip(PV_minus_p_results, p_columns)]))
    self.assertTrue(all([(PHpp.V == PH_V + p.T).all()
                         for PHpp, p in zip(PH_plus_p_results, p_columns)]))
    self.assertTrue(all([(PHmp.V == PH_V - p.T).all()
                         for PHmp, p in zip(PH_minus_p_results, p_columns)]))

if __name__ == '__main__':
  unittest.main()
