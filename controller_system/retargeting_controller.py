import numpy as np

from pydrake.all import (
    LeafSystem,
    BasicVector,
    PortDataType,
)


class RetargetingControllerParams:
    def __init__(self, 
                 desired_joint_stiffness: np.array, 
                 joint_stiffness: np.array,
                 control_period: float,
                 nq: int,
                 nv: int,
                 nu: int,
                 ):
        assert np.all(desired_joint_stiffness <= joint_stiffness)
        self.desired_joint_stiffness = desired_joint_stiffness
        self.joint_stiffness = joint_stiffness
        self.control_period = control_period    
        self.nq = nq
        self.nv = nv
        self.nu = nu


class RetargetingController:
    def __init__(self, q_nominal, u_nominal, controller_params):
        self.q_nominal = q_nominal # for now this is a single configuration not a trajectory
        self.u_nominal = u_nominal # for now this is a single control not a trajectory
        self.controller_params = controller_params
        return None

    def calc_u(
        self,
        q_nominal: np.ndarray, # nominal configuration trajectory, not used 
        u_nominal: np.ndarray, # nominal control trajectory
        robot_state: np.ndarray, # current configuration and velocity
        q_goal: np.ndarray, # goal configuration, not used 
        u_goal: np.ndarray, # goal control, not used
    ):
        # Predictive sampling outputs a control plan (a sequence of target 
        # configurations that MJ tracks using a very soft PD controller).
        # Predictive sampling also outputs a configuration plan that results 
        # from applying the control plan with low joint stiffness. 
        # We care about the control plan, we want to replicate the behavior 
        # seen in MJ by applying a retargeted control plan on the 
        # {IIWA controller + IIWA robot} plant. 
        # Controls are configuration targets for the IIWA internal controller 
        # to track with very high stiffness.

        nq = self.controller_params.nq
        q = robot_state[:nq]
        # v = robot_state[nq:]

        # retargeted configuration
        desired_joint_stiffness = self.controller_params.desired_joint_stiffness
        joint_stiffness = self.controller_params.joint_stiffness
        q_retargeted = q + desired_joint_stiffness / joint_stiffness * (q_goal - q)
        return q_retargeted 

    def find_closest_on_nominal_path(self, q):
        t_value = 0.0 # not used anyway
        indices_closest = 0 # not used anyway
        return self.q_nominal, self.u_nominal, t_value, indices_closest


class RetargetingControllerSystem(LeafSystem):
    def __init__(
        self,
        q_nominal: np.ndarray,
        u_nominal: np.ndarray,
        controller_params: RetargetingControllerParams,
    ):
        super().__init__()

        self.set_name("retargeting_controller")
        # Periodic state update
        self.control_period = controller_params.control_period
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=self.control_period,
            offset_sec=0.0,
            update=self.Update
        )

        # The object configuration is declared as part of the state, but not
        # used, so that indexing becomes easier.
        self.DeclareDiscreteState(BasicVector(controller_params.nq))
        self.controller = RetargetingController(
            q_nominal=q_nominal,
            u_nominal=u_nominal,
            controller_params=controller_params,
        )

        self.robot_state_input_port = self.DeclareInputPort(
            "robot_state", 
            PortDataType.kVectorValued, 
            controller_params.nq + controller_params.nv,
        )

        self.q_ref_input_port = self.DeclareInputPort(
            "q_ref",
            PortDataType.kVectorValued,
            controller_params.nq,
        )

        self.u_ref_input_port = self.DeclareInputPort(
            "u_ref",
            PortDataType.kVectorValued,
            controller_params.nu,
        )

        self.position_cmd_output_ports = {}

        def calc_output(context, output):
            output.SetFromVector(
                context.get_discrete_state().value()
                )

        self.position_cmd_output_ports = self.DeclareVectorOutputPort(
            "q_retargeted", 
            BasicVector(controller_params.nq), 
            calc_output)

    def Update(self, context, discrete_state):
        q_goal = self.q_ref_input_port.Eval(context)
        u_goal = self.u_ref_input_port.Eval(context)
        robot_state = self.robot_state_input_port.Eval(context)
        
        nq = self.controller.controller_params.nq
        q = robot_state[:nq]
        (
            q_nominal,
            u_nominal,
            t_value,
            indices_closest,
        ) = self.controller.find_closest_on_nominal_path(q)

        q_retargeted = self.controller.calc_u(
            q_nominal=q_nominal,
            u_nominal=u_nominal,
            robot_state=robot_state,
            q_goal=q_goal,
            u_goal=u_goal,
        )
        discrete_state.set_value(q_retargeted) 
