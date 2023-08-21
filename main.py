# %%
import numpy as np
from typing import Tuple
from pydrake.all import (
    PiecewisePolynomial, 
    StartMeshcat,
    LeafSystem,
    BasicVector,
    PortDataType,
    DiagramBuilder,
    TrajectorySource,
    MultibodyPlant,
    AddMultibodyPlantSceneGraph,
    Parser,
    LogVectorOutput,
    MeshcatVisualizer,
    ContactVisualizer,
    Simulator,
    ExternallyAppliedSpatialForce,
    SpatialForce,
)
from iiwa_controller.utils import *
from iiwa_controller.robot_internal_controller import RobotInternalController
from iiwa_controller.examples.test_iiwa_static_load import TestIiwaStaticLoad

kQTrjSrcName = "QTrajectorySource"
kUTrjSrcName = "UTrajectorySource"

# %%
# visualization
meshcat = StartMeshcat()



# %% 
# TODO
# follow a synthetic trajectory in joint space
# use the retargeting trick to make the robot "soft"
    # fix diagram
    # big cleanup
    # think about expansions:
        # controller plant vs mock plant
        # add end-effector, ground, multiple robots, objects
        # record and display data
    # fix issue with the single target q vs trajectory of target qs
    # integrate the retargeting controller with the system {IIWA controller + IIWA robot}
# load a trajectory form MJPC 
# test the trajectory rollout 
# add a second arm to the scene
# add floor to the scene
# add end-effectors
# add an object in the scene
# test the interaction between the soft PD and the object
# test the MJPC trajectory single arm with the object and compare 
# record the positions, velocities, torques
# show joint limits
#
# interface work
    # clean trajectory rollout interface
    # clean retargeting interface (can make the Kp softer not harder)
    # clean controller rollout interface

# %%

def sine_trajectory(horizon, N=100):
    q0 = np.array([+0.0, -0.0, 0, -1.70, 0, -1.0, 0])
    q1 = np.array([+np.pi, +0.0, 0, -1.70, 0, -1.0, 0])
    q_iiwa_knots = np.zeros((N, 7))
    for i in range(N):
        q_iiwa_knots[i] = q0 + (q1 - q0) * (-np.cos(2*np.pi * i / (N-1)) + 1) / 2
    t_iiwa_knots = np.array([horizon * i / (N-1) for i in range(N)])
    return q_iiwa_knots, t_iiwa_knots

def linear_trajectory(horizon, N=100):
    q0 = np.array([+0.0, +0.0,   0, -1.70, 0, -1.0, 0])
    q1 = np.array([+0.0, +np.pi, 0, -1.70, 0, -1.0, 0])
    q_iiwa_knots = np.zeros((N, 7))
    for i in range(N):
        q_iiwa_knots[i] = q0 + (q1 - q0) * i / (N-1)
    t_iiwa_knots = np.array([horizon * i / (N-1) for i in range(N)])
    return q_iiwa_knots, t_iiwa_knots

def render_system_with_graphviz(system, output_file="system_view.gz"):
    """Renders the Drake system (presumably a diagram,
    otherwise this graph will be fairly trivial) using
    graphviz to a specified file."""
    from graphviz import Source

    string = system.GetGraphvizString()
    src = Source(string)
    src.render(output_file, view=False)




# %%
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

# %%
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
        q: np.ndarray, # current configuration
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

        # retargeted configuration
        desired_joint_stiffness = self.controller_params.desired_joint_stiffness
        joint_stiffness = self.controller_params.joint_stiffness
        u_retargeted = q + desired_joint_stiffness / joint_stiffness * (u_nominal - q)
        return u_retargeted

    def find_closest_on_nominal_path(self, q):
        t_value = 0.0 # not used anyway
        indices_closest = 0 # not used anyway
        return self.q_nominal, self.u_nominal, t_value, indices_closest

#%%
class RetargetingControllerSystem(LeafSystem):
    def __init__(
        self,
        q_nominal: np.ndarray,
        u_nominal: np.ndarray,
        controller_params: RetargetingControllerParams,
    ):
        super().__init__()
        # self.plant = self.q_sim.get_plant()

        self.set_name("retargeting_controller")
        # Periodic state update
        self.control_period = controller_params.control_period
        self.DeclarePeriodicDiscreteUpdateNoHandler(self.control_period)

        # The object configuration is declared as part of the state, but not
        # used, so that indexing becomes easier.
        self.DeclareDiscreteState(BasicVector(controller_params.nq))
        self.controller = RetargetingController(
            q_nominal=q_nominal,
            u_nominal=u_nominal,
            # q_sim=q_sim_q_control,
            controller_params=controller_params,
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

        self.robot_state_input_port = self.DeclareInputPort(
            "robot_state", PortDataType.kVectorValued, 
            controller_params.nq + controller_params.nv,
        )

        self.position_cmd_output_ports = {}

        def calc_output(context, output):
            output.SetFromVector(
                context.get_discrete_state().value()
                )

        self.position_cmd_output_ports = self.DeclareVectorOutputPort(
            "$$$_cmd", BasicVector(controller_params.nq), calc_output)

    def DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        super().DoCalcDiscreteVariableUpdates(context, events, discrete_state)
        q_goal = self.q_ref_input_port.Eval(context)
        u_goal = self.u_ref_input_port.Eval(context)

        robot_state = self.robot_state_input_port.Eval(context)
    
        (
            q_nominal,
            u_nominal,
            t_value,
            indices_closest,
        ) = self.controller.find_closest_on_nominal_path(q)

        u = self.controller.calc_u(
            q_nominal=q_nominal,
            u_nominal=u_nominal,
            robot_state=robot_state,
            q_goal=q_goal,
            u_goal=u_goal,
        )

        q_nominal = u
        discrete_state.set_value(q_nominal)

#%%
def add_controller_system_to_diagram(
    builder: DiagramBuilder,
    controller_iiwa,
    iiwa_model,
    t_knots: np.ndarray,
    u_knots_ref: np.ndarray,
    q_knots_ref: np.ndarray,
    controller_params: RetargetingControllerParams,
) -> Tuple[RetargetingControllerSystem, PiecewisePolynomial, PiecewisePolynomial]:
    """
    Adds the following three system to the diagram, and makes the following
     two connections.
    |trj_src_q| ---> |                  |
                     | ControllerSystem |
    |trj_src_u| ---> |                  |
    """
    # Create trajectory sources.
    if t_knots is None:
        q_ref_trj = PiecewisePolynomial(q_knots_ref)
        u_ref_trj = PiecewisePolynomial(u_knots_ref)
    else:
        u_ref_trj = PiecewisePolynomial.FirstOrderHold(t_knots, u_knots_ref.T)
        q_ref_trj = PiecewisePolynomial.FirstOrderHold(t_knots, q_knots_ref.T)

    trj_src_u = TrajectorySource(u_ref_trj)
    trj_src_q = TrajectorySource(q_ref_trj)
    trj_src_u.set_name(kUTrjSrcName)
    trj_src_q.set_name(kQTrjSrcName)

    # controller system.
    q_controller = RetargetingControllerSystem(
        q_nominal=q_knots_ref,
        u_nominal=u_knots_ref,
        controller_params=controller_params,
    )

    builder.AddSystem(trj_src_u)
    builder.AddSystem(trj_src_q)
    builder.AddSystem(q_controller)

    # Make connections.
    builder.Connect(trj_src_q.get_output_port(), q_controller.q_ref_input_port)
    builder.Connect(trj_src_u.get_output_port(), q_controller.u_ref_input_port)
    builder.Connect(
        plant.get_state_output_port(iiwa_model),
        q_controller.robot_state_input_port)


    # cmd_v2l = CommandVec2LcmSystem(q_sim)
    # builder.AddSystem(cmd_v2l)
    builder.Connect(
        q_controller.position_cmd_output_ports,
        controller_iiwa.joint_angle_commanded_input_port,
    )
    return q_controller, q_ref_trj, u_ref_trj




# %%

def build_sim(
    q_traj_iiwa: PiecewisePolynomial,
    Kp_iiwa: np.array,
    gravity: np.array,
    f_C_W,
    time_step,
    add_schunk: bool,
    is_visualizing=False,
    meshcat=None,
):
    # Build diagram.
    builder = DiagramBuilder()

    # MultibodyPlant
    plant = MultibodyPlant(time_step)
    plant.mutable_gravity_field().set_gravity_vector(gravity)

    _, scene_graph = AddMultibodyPlantSceneGraph(builder, plant=plant)
    parser = Parser(plant=plant, scene_graph=scene_graph)
    add_package_paths(parser)

    if add_schunk:
        ProcessModelDirectives(
            LoadModelDirectives(
                os.path.join(models_dir, "iiwa_and_schunk.yml")
            ),
            plant,
            parser,
        )
        schunk_model = plant.GetModelInstanceByName("schunk")
    else:
        ProcessModelDirectives(
            LoadModelDirectives(os.path.join(models_dir, "iiwa.yml")),
            plant,
            parser,
        )

    iiwa_model = plant.GetModelInstanceByName("iiwa")

    plant.Finalize()


    # IIWA controller
    plant_robot, _ = create_iiwa_controller_plant(
        gravity, add_schunk_inertia=add_schunk
    )
    controller_iiwa = RobotInternalController(
        plant_robot=plant_robot,
        joint_stiffness=Kp_iiwa,
        controller_mode="impedance",
    )
    builder.AddSystem(controller_iiwa)
    builder.Connect(
        controller_iiwa.GetOutputPort("joint_torques"),
        plant.get_actuation_input_port(iiwa_model),
    )
    builder.Connect(
        plant.get_state_output_port(iiwa_model),
        controller_iiwa.robot_state_input_port,
    )

    # IIWA Trajectory source
    traj_source_iiwa = TrajectorySource(q_traj_iiwa)
    builder.AddSystem(traj_source_iiwa)
    print("disconnected controller_iiwa input port")
    # builder.Connect(
    #     traj_source_iiwa.get_output_port(0),
    #     controller_iiwa.joint_angle_commanded_input_port,
    # )

    # Logs
    iiwa_log_sink = LogVectorOutput(
        plant.get_state_output_port(iiwa_model), builder, publish_period=0.001
    )

    # visualizer
    if is_visualizing:
        if meshcat is None:
            meshcat = StartMeshcat()
        meshcat_vis = MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, meshcat
        )
        ContactVisualizer.AddToBuilder(
            builder,
            plant,
            meshcat,
        )

    return builder, controller_iiwa, iiwa_model, iiwa_log_sink, plant, meshcat_vis



def run_sim(
    builder, 
    controller_iiwa, 
    iiwa_model,
    iiwa_log_sink,
    plant, 
    meshcat_vis,
    q_traj_iiwa: PiecewisePolynomial,
    f_C_W,
    add_schunk: bool,
    is_visualizing=False,
    ):

    # diagram = builder.Build()

    # Run simulation.
    sim = Simulator(diagram)
    context = sim.get_context()
    context_controller = diagram.GetSubsystemContext(controller_iiwa, context)
    context_plant = diagram.GetSubsystemContext(plant, context)

    controller_iiwa.tau_feedforward_input_port.FixValue(
        context_controller, np.zeros(7)
    )

    # robot initial configuration.
    q_iiwa_0 = q_traj_iiwa.value(0).squeeze()
    t_final = q_traj_iiwa.end_time()
    plant.SetPositions(context_plant, iiwa_model, q_iiwa_0)
    if add_schunk:
        plant.get_actuation_input_port(schunk_model).FixValue(
            context_plant, np.zeros(2)
        )

    # constant force on link 7.
    easf = ExternallyAppliedSpatialForce()
    easf.F_Bq_W = SpatialForce([0, 0, 0], f_C_W)
    easf.body_index = plant.GetBodyByName("iiwa_link_7").index()
    plant.get_applied_spatial_force_input_port().FixValue(context_plant, [easf])

    # meshcat visualizer
    if is_visualizing:
        meshcat_vis.DeleteRecording()
        meshcat_vis.StartRecording()

    sim.Initialize()
    sim.set_target_realtime_rate(0)
    sim.AdvanceTo(t_final)

    # meshcat visualizer
    if is_visualizing:
        meshcat_vis.PublishRecording()

    iiwa_log = iiwa_log_sink.FindLog(context)
    return iiwa_log, controller_iiwa


# %%
tester = TestIiwaStaticLoad()

# coordinate of point C expressed in frame L7.
tester.p_L7oC_L7 = np.zeros(3)
# force at C.
tester.f_C_W = np.array([0, 0, -10.0])
# Stiffness matrix of the robot.
tester.Kp_iiwa = np.array([800.0, 600, 600, 600, 400, 200, 200])
tester.gravity = np.array([0, 0, -0.0])

# robot trajectory (hold q0).
N = 100
horizon = 5.0
q_iiwa_knots, t_iiwa_knots = linear_trajectory(horizon, N=N)
u_knots_ref, t_knots = linear_trajectory(horizon, N=N)
q_knots_ref, t_knots = linear_trajectory(horizon, N=N)


# run simulation for horizon.
tester.qa_traj = PiecewisePolynomial.FirstOrderHold(
    t_iiwa_knots, q_iiwa_knots.T
)

# %%
builder, controller_iiwa, iiwa_model, iiwa_log_sink, plant, meshcat_vis = build_sim(
    q_traj_iiwa=tester.qa_traj,
    Kp_iiwa=tester.Kp_iiwa,
    gravity=tester.gravity,
    f_C_W=tester.f_C_W,
    time_step=1e-4,
    add_schunk=False,
    is_visualizing=True,
    meshcat=meshcat,
    )

#%%
desired_joint_stiffness = np.array([1,1,1,1,1,1,1.0])
joint_stiffness = np.array([800.0, 600, 600, 600, 400, 200, 200])
control_period = 0.001
nq = 7
nv = 7
nu = 7

q_nominal = np.array([1,1,1,1,1,1,1.0])
u_nominal = np.array([1,1,1,1,1,1,1.0])
q_goal = np.array([1,1,1,1,1,1,1.0])
u_goal = np.array([1,1,1,1,1,1,1.0])
q = np.array([1,1,1,1,1,1,1.0])
 
controller_params = RetargetingControllerParams(
    desired_joint_stiffness, joint_stiffness, control_period, nq, nv, nu)
controller = RetargetingController(q_nominal, u_nominal, controller_params)
controller_system = RetargetingControllerSystem(q_nominal, u_nominal, controller_params)


u_retargeted = controller.calc_u(q_nominal, u_nominal, q, q_goal, u_goal)
controller.find_closest_on_nominal_path(q)

q_controller, q_ref_trj, u_ref_trj = add_controller_system_to_diagram(
    builder,
    controller_iiwa,
    iiwa_model,
    t_knots,
    u_knots_ref,
    q_knots_ref,
    controller_params,)


# %%

diagram = builder.Build()
render_system_with_graphviz(diagram, "controller_hardware.gz")

# %%





iiwa_log, controller_iiwa = run_sim(
    builder, 
    controller_iiwa, 
    iiwa_model, 
    iiwa_log_sink,
    plant, 
    meshcat_vis,
    q_traj_iiwa=tester.qa_traj,
    f_C_W=tester.f_C_W,
    add_schunk=False,
    is_visualizing=True,
    )

# # Run simulation.
# sim = Simulator(diagram)
# context = sim.get_context()
# context_controller = diagram.GetSubsystemContext(controller_iiwa, context)
# context_plant = diagram.GetSubsystemContext(plant, context)

# controller_iiwa.tau_feedforward_input_port.FixValue(
#     context_controller, np.zeros(7)
# )

# # robot initial configuration.
# q_iiwa_0 = q_traj_iiwa.value(0).squeeze()
# t_final = q_traj_iiwa.end_time()
# plant.SetPositions(context_plant, iiwa_model, q_iiwa_0)
# if add_schunk:
#     plant.get_actuation_input_port(schunk_model).FixValue(
#         context_plant, np.zeros(2)
#     )

# # constant force on link 7.
# easf = ExternallyAppliedSpatialForce()
# easf.F_Bq_W = SpatialForce([0, 0, 0], f_C_W)
# easf.body_index = plant.GetBodyByName("iiwa_link_7").index()
# plant.get_applied_spatial_force_input_port().FixValue(context_plant, [easf])

# # meshcat visualizer
# if is_visualizing:
#     meshcat_vis.DeleteRecording()
#     meshcat_vis.StartRecording()

# sim.Initialize()
# sim.set_target_realtime_rate(0)
# sim.AdvanceTo(t_final)

# # meshcat visualizer
# if is_visualizing:
#     meshcat_vis.PublishRecording()

# iiwa_log = iiwa_log_sink.FindLog(context)
# return iiwa_log, controller_iiwa











# # Run simulator.
# simulator = Simulator(diagram)
# simulator.set_target_realtime_rate(1.0)
# simulator.set_publish_every_time_step(False)

# # Make sure that the first status message read my the sliders is the real
# # status of the hand.
# context = simulator.get_context()
# context_allegro_sub = allegro_status_sub.GetMyContextFromRoot(context)
# context_allegro_sub.SetAbstractState(0, allegro_status_msg)
# # context_ctrller = ctrller_allegro.GetMyContextFromRoot(context)
# # ctrller_allegro.q_input_port.FixValue(context_ctrller, q0)

# q_scope_msg = wait_for_msg(
#     kQEstimatedChannelName, lcmt_scope, lambda msg: msg.size > 0
# )
# context_q_sub = q_sub.GetMyContextFromRoot(context)
# context_q_sub.SetAbstractState(0, q_scope_msg)


# print("Running!")
# simulator.AdvanceTo(t_knots[-1] + 5)
# print("Done!")
















# %%
