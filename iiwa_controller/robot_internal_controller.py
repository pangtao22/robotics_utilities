import numpy as np

from pydrake.systems.framework import LeafSystem, BasicVector, PortDataType
from pydrake.multibody.tree import MultibodyForces
from pydrake.all import MultibodyPlant
from primitives import low_pass_filter


class RobotInternalController(LeafSystem):
    def __init__(
        self,
        plant_robot: MultibodyPlant,
        joint_stiffness: np.ndarray,
        joint_damping: np.ndarray = None,
        controller_mode="impedance",
        name="robot_internal_controller",
    ):
        """
        Impedance controller implements the controller in Ott2008 for IIWA.
        Inverse dynamics controller makes use of drake's
            InverseDynamicsController.
        :param plant_robot:
        :param joint_stiffness: (nq,) numpy array which defines the stiffness
            of all joints.
        :param joint_damping: (nq,) numpy array which defines the damping
            of all joints.
            If None, a fixed critically-damped value is used.
            In impedance mode, damping changes with the diagonal of the mass
            matrix.
        """
        LeafSystem.__init__(self)
        self.set_name(name)
        self.plant = plant_robot
        self.context = plant_robot.CreateDefaultContext()

        self.nq = plant_robot.num_positions()
        self.nv = plant_robot.num_velocities()
        self.robot_state_input_port = self.DeclareInputPort(
            "robot_state", PortDataType.kVectorValued, self.nq + self.nv
        )
        self.tau_feedforward_input_port = self.DeclareInputPort(
            "tau_feedforward", PortDataType.kVectorValued, self.nq
        )
        self.joint_angle_commanded_input_port = self.DeclareInputPort(
            "q_robot_commanded", PortDataType.kVectorValued, self.nq
        )
        self.joint_torque_output_port = self.DeclareVectorOutputPort(
            "joint_torques", BasicVector(self.nv), self.CalcJointTorques
        )

        # control rate
        self.control_period = 2e-4  # 5000Hz.
        self.DeclareDiscreteState(self.nv)
        self.DeclarePeriodicDiscreteUpdateNoHandler(
            period_sec=self.control_period, 
            # offset_sec=0.0001*self.control_period,
        )

        # joint velocity estimator
        self.q_prev = None
        self.w_cutoff = 2 * np.pi * 400
        self.velocity_estimator = low_pass_filter.LowPassFilter(
            self.nv, self.control_period, self.w_cutoff
        )

        self.Kv = joint_damping
        # damping coefficient LPF (if adaptive critical damping is used.)
        self.Kv_filter = low_pass_filter.LowPassFilter(
            self.nq, self.control_period, 2 * np.pi * 1
        )

        # controller gains
        assert len(joint_stiffness) == plant_robot.num_positions()
        self.Kp = joint_stiffness
        self.controller_mode = controller_mode

        # logs
        self.Kv_log = []
        self.tau_stiffness_log = []
        self.tau_damping_log = []
        self.sample_times = []

    def DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        LeafSystem.DoCalcDiscreteVariableUpdates(
            self, context, events, discrete_state
        )

        # read input ports
        x = self.robot_state_input_port.Eval(context)
        q_cmd = self.joint_angle_commanded_input_port.Eval(context)
        tau_ff = self.tau_feedforward_input_port.Eval(context)
        # print("DoCalcDiscreteVariableUpdates robot_state = ", x)
        print("robot internal controller q_cmd = ", q_cmd)
        # print("DoCalcDiscreteVariableUpdates tau_ff = ", tau_ff)
        # print("Called RobotInternalController")
        q = x[: self.nq]
        v = x[self.nq :]

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
        self.plant.SetVelocities(self.context, v_est)

        # gravity compenstation
        tau_g = self.plant.CalcGravityGeneralizedForces(self.context)
        tau = -tau_g

        if self.controller_mode == "impedance":
            Kv = self.CalcDamping(damping_ratio=1.0)
            tau_stiffness = self.Kp * (q_cmd - q)
            tau_damping = -Kv * v
            tau += tau_damping + tau_stiffness

            self.Kv_log.append(Kv)
            self.tau_stiffness_log.append(tau_stiffness.copy())
            self.tau_damping_log.append(tau_damping.copy())

        elif self.controller_mode == "inverse_dynamics":
            # compute desired acceleration
            Kv = self.CalcDamping(damping_ratio=1.0)
            qDDt_d = self.Kp * (q_cmd - q) + Kv * (-v_est)
            tau += self.plant.CalcInverseDynamics(
                context=self.context,
                known_vdot=qDDt_d,
                external_forces=MultibodyForces(self.plant),
            )

        output = discrete_state.get_mutable_vector().get_mutable_value()
        output[:] = tau + tau_ff
        # print("DoCalcDiscreteVariableUpdates output = ", output)

    def CalcJointTorques(self, context, y_data):
        state = context.get_discrete_state_vector().get_value()
        y = y_data.get_mutable_value()
        # print("CalcJointTorques state = ", state)
        y[:] = state

    def CalcDamping(self, damping_ratio: float):
        if self.controller_mode == "inverse_dynamics":
            return 2 * damping_ratio * np.sqrt(self.Kp)

        assert self.controller_mode == "impedance"
        if self.Kv is not None:
            return self.Kv

        M = self.plant.CalcMassMatrixViaInverseDynamics(self.context)
        m = M.diagonal()
        Kv = 2 * damping_ratio * np.sqrt(self.Kp * m)
        self.Kv_filter.update(Kv)
        return self.Kv_filter.get_current_state()
