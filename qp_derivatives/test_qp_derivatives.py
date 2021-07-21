import unittest

import numpy as np
from pydrake.all import OsqpSolver, GurobiSolver

from .qp_derivatives import (QpDerivativesKkt, QpDerivativesNumerical,
                             build_qp_and_solve)


class TestQpDerivatives(unittest.TestCase):
    def setUp(self):
        """
        The first problem in this test case is
        min. 0.5 * ((x0 - 1)^2 + (x1 + 1)^2) - 1
        s.t. x0 >= 0, x1 >= 0
        
        Solution is x_star = [1, 0]
        """
        n_z = 2
        self.Q_list = [np.eye(n_z)]
        self.b_list = [-np.array([1., -1])]
        self.G_list = [-np.eye(n_z)]
        self.e_list = [np.zeros(n_z)]

        n_z = 2
        n_lambda = 3
        np.random.seed(2024)
        L = np.random.rand(n_z, n_z) - 1
        self.Q_list.append(L.T.dot(L))
        self.b_list.append(np.random.rand(n_z) - 1)
        self.G_list.append(np.random.rand(n_lambda, n_z))
        self.e_list.append(np.array([0, 0, 10.]))

        gurobi_solver = GurobiSolver()
        if gurobi_solver.available():
            self.solver = GurobiSolver()
        else:
            self.solver = OsqpSolver()

        self.dqp_kkt = QpDerivativesKkt()
        self.dqp_numerical = QpDerivativesNumerical(solver=self.solver)

    def test_derivatives(self):
        for Q, b, G, e in zip(self.Q_list,
                              self.b_list, self.G_list, self.e_list):
            z_star, lambda_star = build_qp_and_solve(Q, b, G, e, self.solver)
            self.dqp_kkt.update_problem(
                Q=Q, b=b, G=G, e=e, z_star=z_star, lambda_star=lambda_star)
            self.dqp_numerical.update_problem(Q=Q, b=b, G=G, e=e)

            DzDe_kkt = self.dqp_kkt.calc_DzDe()
            DzDe_numerical = self.dqp_numerical.calc_DzDe(epsilon=1e-4)
            # print('kkt\n', DzDe_kkt)
            # print('numerical\n', DzDe_numerical)

            DzDb_kkt = self.dqp_kkt.calc_DzDb()
            DzDb_numerical = self.dqp_numerical.calc_DzDb(epsilon=1e-4)
            # print('kkt\n', DzDb_kkt)
            # print('numerical\n', DzDb_numerical)

            DzDG_vec = self.dqp_kkt.calc_DzDG_vec()
            DzDG_vec_numerical = self.dqp_numerical.calc_DzDG_vec(
                epsilon=1e-4)

            self.assertTrue(np.allclose(DzDe_kkt, DzDe_numerical, atol=1e-5))
            self.assertTrue(np.allclose(DzDb_kkt, DzDb_numerical, atol=1e-5))
            self.assertTrue(np.allclose(DzDG_vec, DzDG_vec_numerical,
                                        atol = 1e-4))
