import unittest

from .sim_setup import *
from pydrake.multibody.tree import JacobianWrtVariable


class TestIiwaStaticLoad(unittest.TestCase):
    def test_static_load(self):
        # coordinate of point C expressed in frame L7.
        p_L7oC_L7 = np.zeros(3)
        # force at C.
        f_C_W = np.array([0, 0, -20])
        # Stiffness matrix of the robot.
        Kq_a = np.array([800., 600, 600, 600, 400, 200, 200])
        gravity = np.array([0, 0, -10.])

        # robot trajectory (hold q0).
        q0 = np.array([0, 0, 0, -1.70, 0, 1.0, 0])
        q_iiwa_knots = np.zeros((2, 7))
        q_iiwa_knots[0] = q0
        q_iiwa_knots[1] = q0

        # run simulation for 1s.
        qa_traj = PiecewisePolynomial.FirstOrderHold([0, 1], q_iiwa_knots.T)
        iiwa_log, controller_iiwa = run_sim(qa_traj, Kq_a, gravity, f_C_W,
                                            time_step=1e-5)

        # check simulation results.
        q_iiwa = iiwa_log.data().T[:, :7]
        v_iiwa = iiwa_log.data().T[:, 7:]
        plant = controller_iiwa.plant
        context = plant.CreateDefaultContext()

        plant.SetPositions(context, q_iiwa[-1])
        J_WB1 = plant.CalcJacobianTranslationalVelocity(
            context=context,
            with_respect_to=JacobianWrtVariable.kQDot,
            frame_B=plant.GetFrameByName("iiwa_link_7"),
            p_BoBi_B=p_L7oC_L7,
            frame_A=plant.world_frame(),
            frame_E=plant.world_frame())

        delta_q = q_iiwa[-1] - q_iiwa[0]
        tau_dq = delta_q * Kq_a
        tau_1 = J_WB1.T.dot(f_C_W)
        # elastic force balances external load.
        self.assertLessEqual(np.linalg.norm(tau_dq - tau_1), 1e-5)
        # the robot is stationary at the end of the simulation.
        self.assertLessEqual(np.linalg.norm(v_iiwa[-1]), 1e-6)


if __name__ == '__main__':
    unittest.main()
