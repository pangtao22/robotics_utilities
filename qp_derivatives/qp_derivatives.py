import numpy as np
import pydrake.solvers.mathematicalprogram as mp

"""
QP:
min. 1 / 2 * z.dot(Q).dot(z) + b.dot(z)
s.t. G.dot(z) <= e
"""


class QpDerivativesKkt:
    def __init__(self):
        self.n_z = 0  # number of decision variables.
        self.n_lambda = 0  # number constraints / lagrange multipliers.
        self.A11 = np.array([])
        self.A12 = np.array([])
        self.z_star = np.array([])
        self.lambda_star = np.array([])

    def update_problem(self, Q: np.ndarray, b: np.ndarray, G: np.ndarray,
                       e: np.ndarray, z_star: np.ndarray,
                       lambda_star: np.ndarray):
        n_z = len(z_star)
        n_lambda = len(lambda_star)
        assert Q.shape[1] == Q.shape[0] == n_z
        assert b.size == n_z
        assert G.shape[0] == n_lambda
        assert G.shape[1] == n_z
        assert e.size == n_lambda

        self.n_z = n_z
        self.n_lambda = n_lambda

        A_inverse = np.eye(n_z + n_lambda)
        A_inverse[:n_z, :n_z] = Q
        A_inverse[:n_z, n_z:] = G.T
        A_inverse[n_z:, :n_z] = np.diag(lambda_star) @ G
        A_inverse[n_z:, n_z:] = np.diag(G @ z_star - e)
        A = np.linalg.pinv(A_inverse)
        self.A11 = A[:n_z, :n_z]
        self.A12 = A[:n_z, n_z:]
        self.z_star = z_star
        self.lambda_star = lambda_star

    def calc_DzDe(self):
        return self.A12 @ np.diag(self.lambda_star)

    def calc_DzDb(self):
        return -self.A11

    def calc_DzDG_vec(self):
        """
        Derivatives of z w.r.t the column-major vectorized version of G,
            i.e. vec(G) using the notations from "Matrix Differential
            Calculus with Applications in Statics and Econometrics" by Magnus
            and Neudecker.
        Operations involving taking derivatives w.r.t vec(G) can be found in
            the book too.
        """
        dzdG_vec = -np.kron(self.A11, self.lambda_star)
        dzdG_vec -= np.kron(self.z_star, self.A12 * self.lambda_star)
        return dzdG_vec


def build_qp_and_solve(
        Q: np.ndarray, b: np.ndarray, G: np.ndarray, e: np.ndarray, solver):
    n_z = Q.shape[0]
    n_lambda = G.shape[0]

    prog = mp.MathematicalProgram()
    z = prog.NewContinuousVariables(n_z, 'z')
    prog.AddQuadraticCost(Q, b, z)
    inequalities = prog.AddLinearConstraint(
        A=G,
        lb=-np.full(n_lambda, np.inf),
        ub=e,
        vars=z)
    results = solver.Solve(prog)
    assert results.is_success()

    return results.GetSolution(z), -results.GetDualSolution(inequalities)


class QpDerivativesNumerical:
    def __init__(self, solver):
        self.Q = np.array([])
        self.b = np.array([])
        self.G = np.array([])
        self.e = np.array([])
        self.n_z = 0
        self.n_lambda = 0
        self.solver = solver

    def update_problem(self, Q: np.ndarray, b: np.ndarray, G: np.ndarray,
                       e: np.ndarray):
        self.Q = Q
        self.b = b
        self.G = G
        self.e = e
        self.n_z = Q.shape[0]
        self.n_lambda = G.shape[0]

    def calc_DzDe(self, epsilon=1e-3):
        DzDe = np.zeros((self.n_z, self.n_lambda))

        for i in range(self.n_lambda):
            z_star_values = np.zeros((2, self.n_z))
            for ki, k in enumerate([-1, 1]):
                e_new = self.e.copy()
                e_new[i] += epsilon * k

                z_star_values[ki], _ = build_qp_and_solve(
                    self.Q, self.b, self.G, e_new, self.solver)

            DzDe[:, i] = (z_star_values[1] - z_star_values[0]) / epsilon / 2

        return DzDe

    def calc_DzDb(self, epsilon=1e-3):
        DzDb = np.zeros((self.n_z, self.n_z))

        for i in range(self.n_z):
            z_star_values = np.zeros((2, self.n_z))
            for ki, k in enumerate([-1, 1]):
                b_new = self.b.copy()
                b_new[i] += epsilon * k

                z_star_values[ki], _ = build_qp_and_solve(
                    self.Q, b_new, self.G, self.e, self.solver)

            DzDb[:, i] = (z_star_values[1] - z_star_values[0]) / epsilon / 2

        return DzDb

    def calc_DzDG_vec(self, epsilon=1e-3):
        DzDG_vec = np.zeros((self.n_z, self.n_z * self.n_lambda))

        for j in range(self.n_z):  # G has n_z columns.
            for i in range(self.n_lambda):  # G has n_lambda rows.
                # index into the column-major vectorized version of G.
                idx = j * self.n_lambda + i
                z_star_values = np.zeros((2, self.n_z))
                for ki, k in enumerate([-1, 1]):
                    G_new = self.G.copy()
                    G_new[i, j] += epsilon * k

                    z_star_values[ki], _ = build_qp_and_solve(
                        self.Q, self.b, G_new, self.e, self.solver)

                DzDG_vec[:, idx] = ((z_star_values[1] - z_star_values[0])
                                    / epsilon / 2)
        return DzDG_vec
