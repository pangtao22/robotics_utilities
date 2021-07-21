import numpy as np

from pydrake.systems.framework import (LeafSystem, BasicVector, PortDataType)
from pydrake.multibody.tree import (MultibodyForces)

from ..primitives.low_pass_filter import LowPassFilter


class RobotInternalController(LeafSystem):
    def __init__(self, plant_robot, joint_stiffness,
                 controller_mode="impedance", name="robot_internal_controller"):
        """
        Impedance controller implements the controller in Ott2008 for IIWA.
        Inverse dynamics controller makes use of drake's
            InverseDynamicsController.
        :param plant_robot:
        :param joint_stiffness: (nq,) numpy array which defines the stiffness
            of all joints.
        """
        LeafSystem.__init__(self)
        self.set_name(name)
        self.plant = plant_robot
        self.context = plant_robot.CreateDefaultContext()

        self.nq = plant_robot.num_positions()
        self.nv = plant_robot.num_velocities()
        self.robot_state_input_port = \
            self.DeclareInputPort(
                "robot_state", PortDataType.kVectorValued, self.nq + self.nv)
        self.tau_feedforward_input_port = \
            self.DeclareInputPort(
                "tau_feedforward", PortDataType.kVectorValued, self.nq)
        self.joint_angle_commanded_input_port = \
            self.DeclareInputPort(
                "q_robot_commanded", PortDataType.kVectorValued, self.nq)
        self.joint_torque_output_port = \
            self.DeclareVectorOutputPort(
                "joint_torques", BasicVector(self.nv), self.CalcJointTorques)

        # control rate
        self.control_period = 2e-4  # 5000Hz.
        self.DeclareDiscreteState(self.nv)
        self.DeclarePeriodicDiscreteUpdate(period_sec=self.control_period)

        # joint velocity estimator
        self.q_prev = None
        self.w_cutoff = 2 * np.pi * 400
        self.velocity_estimator = LowPassFilter(
            self.nv, self.control_period, self.w_cutoff)

        # damping coefficient filter
        self.Kv_filter = LowPassFilter(
            self.nq, self.control_period, 2 * np.pi * 1)

        # controller gains
        assert len(joint_stiffness) == plant_robot.num_positions()
        self.Kp = joint_stiffness
        self.damping_ratio = 1.0
        self.Kv = 2 * self.damping_ratio * np.sqrt(self.Kp)
        self.controller_mode = controller_mode

        # logs
        self.Kv_log = []
        self.tau_stiffness_log = []
        self.tau_damping_log = []
        self.sample_times = []

    def DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        LeafSystem.DoCalcDiscreteVariableUpdates(
            self, context, events, discrete_state)

        # read input ports
        x = self.robot_state_input_port.Eval(context)
        q_cmd = self.joint_angle_commanded_input_port.Eval(context)
        tau_ff = self.tau_feedforward_input_port.Eval(context)
        q = x[:self.nq]
        v = x[self.nq:]

        # estimate velocity
        if self.q_prev is None:
            self.velocity_estimator.update(np.zeros(self.nv))
        else:
        # low pass filter velocity.
            v_diff = (q - self.q_prev) / self.control_period
            self.velocity_estimator.update(v_diff)

        self.q_prev = q
        v_est = self.velocity_estimator.get_current_state()

        # log the P and D parts of desired acceleration
        self.sample_times.append(context.get_time())

        # update plant context
        self.plant.SetPositions(self.context, q)
        # self.plant.SetVelocities(self.context, v_est)

        # gravity compenstation
        tau_g = self.plant.CalcGravityGeneralizedForces(self.context)
        tau = -tau_g

        if self.controller_mode == "impedance":
            M = self.plant.CalcMassMatrixViaInverseDynamics(self.context)

            # m = np.sort(np.linalg.eig(M)[0])[::-1]
            m = M.diagonal()
            Kv = 2 * self.damping_ratio * np.sqrt(self.Kp * m)
            self.Kv_filter.update(Kv)
            Kv = self.Kv_filter.get_current_state()
            # Kv = np.array([100, 100, 100, 100., 1., 1, 1])
            tau_stiffness = self.Kp * (q_cmd - q)
            tau_damping = -Kv * v
            tau += tau_damping + tau_stiffness

            self.Kv_log.append(Kv)
            self.tau_stiffness_log.append(tau_stiffness.copy())
            self.tau_damping_log.append(tau_damping.copy())

        elif self.controller_mode == "inverse_dynamics":
            # compute desired acceleration
            qDDt_d = self.Kp * (q_cmd - q) + self.Kv * (-v_est)
            tau += self.plant.CalcInverseDynamics(
                context=self.context,
                known_vdot=qDDt_d,
                external_forces=MultibodyForces(self.plant))

        output = discrete_state.get_mutable_vector().get_mutable_value()
        output[:] = tau + tau_ff

    def CalcJointTorques(self, context, y_data):
        state = context.get_discrete_state_vector().get_value()
        y = y_data.get_mutable_value()
        y[:] = state

