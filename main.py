# %%
import numpy as np
import os

from typing import Tuple
from pydrake.all import (
    PiecewisePolynomial, 
    StartMeshcat,
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

from controller_system.utils import *
from controller_system.retargeting_controller import *

# %%
# visualization
meshcat = StartMeshcat()


# %% 
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
def build_simulation(
    Kp_iiwa: np.array,
    gravity: np.array,
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
    # feed joint_torques into iiwa_robot
    builder.Connect(
        controller_iiwa.GetOutputPort("joint_torques"),
        plant.get_actuation_input_port(iiwa_model),
    )
    # feed robot_state into controller_iiwa
    builder.Connect(
        plant.get_state_output_port(iiwa_model),
        controller_iiwa.robot_state_input_port,
    )

    # Logging
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


#%%
def add_controller_system_to_diagram(
    builder: DiagramBuilder,
    controller_iiwa,
    iiwa_model,
    t_knots: np.ndarray,
    q_knots_ref: np.ndarray,
    u_knots_ref: np.ndarray,
    controller_params: RetargetingControllerParams,
) -> Tuple[RetargetingControllerSystem, PiecewisePolynomial, PiecewisePolynomial]:
    """
    Adds the following three system to the diagram, and makes the following
     two connections.
    |trj_src_q| ---> |                  |
                     | ControllerSystem | ---> | IIWA |
    |trj_src_u| ---> |                  |
    """
    # Create trajectory sources.
    if t_knots is None:
        q_ref_trj = PiecewisePolynomial(q_knots_ref)
        u_ref_trj = PiecewisePolynomial(u_knots_ref)
    else:
        q_ref_trj = PiecewisePolynomial.FirstOrderHold(t_knots, q_knots_ref.T)
        u_ref_trj = PiecewisePolynomial.FirstOrderHold(t_knots, u_knots_ref.T)

    trj_src_q = TrajectorySource(q_ref_trj)
    trj_src_u = TrajectorySource(u_ref_trj)
    trj_src_q.set_name("q_trajectory_source")
    trj_src_u.set_name("u_trajectory_source")

    # controller system.
    retargeting_controller = RetargetingControllerSystem(
        q_nominal=q_knots_ref,
        u_nominal=u_knots_ref,
        controller_params=controller_params,
    )

    builder.AddSystem(trj_src_q)
    builder.AddSystem(trj_src_u)
    builder.AddSystem(retargeting_controller)

    # Make connections.
    builder.Connect(trj_src_q.get_output_port(), retargeting_controller.q_ref_input_port)
    builder.Connect(trj_src_u.get_output_port(), retargeting_controller.u_ref_input_port)
    builder.Connect(
        plant.get_state_output_port(iiwa_model),
        retargeting_controller.robot_state_input_port)


    # cmd_v2l = CommandVec2LcmSystem(q_sim)
    # builder.AddSystem(cmd_v2l)
    builder.Connect(
        retargeting_controller.position_cmd_output_ports,
        controller_iiwa.joint_angle_commanded_input_port,
    )
    return retargeting_controller, q_ref_trj, u_ref_trj



def run_simulation(
    iiwa_model,
    iiwa_log_sink,
    plant, 
    meshcat_vis,
    q_traj_iiwa: PiecewisePolynomial,
    f_C_W,
    add_schunk: bool,
    is_visualizing=False,
    ):

    # Run simulation.
    sim = Simulator(diagram)
    context = sim.get_context()
    controller_iiwa = diagram.GetSubsystemByName("robot_internal_controller")
    context_controller = diagram.GetSubsystemContext(controller_iiwa, context)
    context_plant = diagram.GetSubsystemContext(plant, context)
    
    controller_iiwa.tau_feedforward_input_port.FixValue(
        context_controller, np.zeros(7)
    )

    q_iiwa_0 = q_traj_iiwa.value(0).squeeze()

    retargeting_controller = diagram.GetSubsystemByName(
        "retargeting_controller")
    context_retargeting = retargeting_controller.GetMyContextFromRoot(context)
    context_retargeting.SetDiscreteState(q_iiwa_0)

    # robot initial configuration.
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
    sim.set_target_realtime_rate(0.0)
    sim.AdvanceTo(t_final)

    # meshcat visualizer
    if is_visualizing:
        meshcat_vis.PublishRecording()

    iiwa_log = iiwa_log_sink.FindLog(context)
    return iiwa_log, controller_iiwa





# %%
# robot simulation period
iiwa_period = 1e-4
# force at C.
f_C_W = np.array([0, 0, -0.0])
# Stiffness matrix of the robot.
Kp_iiwa = np.array([800.0, 600, 600, 600, 400, 200, 200])
# Kp_iiwa_desired = Kp_iiwa
Kp_iiwa_desired = 100 * np.array([1,1,1,1,1,1,1.0])
# Gravity vector
gravity = np.array([0, 0, -10.0])
# retargeting_controller period
control_period = 1e-4
# dimensions
nq = 7 # configuration
nv = 7 # velocity
nu = 7 # control


# robot trajectory (hold q0).
N = 100
horizon = 1.5
u_knots_ref, t_knots = sine_trajectory(horizon, N=N)
q_knots_ref, t_knots = sine_trajectory(horizon, N=N)
q_traj_ref = PiecewisePolynomial.FirstOrderHold(
    t_knots, q_knots_ref.T
)

# %%
builder, controller_iiwa, iiwa_model, iiwa_log_sink, plant, meshcat_vis = build_simulation(
    Kp_iiwa=Kp_iiwa,
    gravity=gravity,
    time_step=iiwa_period,
    add_schunk=False,
    is_visualizing=True,
    meshcat=meshcat,
    )

#%%

q_nominal = np.array([+0.0, +0.0,   0, -1.70, 0, -1.0, 0])
u_nominal = np.array([+0.0, +0.0,   0, -1.70, 0, -1.0, 0])
q_goal = np.array([+0.0, +0.0,   0, -1.70, 0, -1.0, 0])
u_goal = np.array([+0.0, +0.0,   0, -1.70, 0, -1.0, 0])
robot_state = np.array([+0.0, +0.0,   0, -1.70, 0, -1.0, 0,   0,0,0,0,0,0,0])

#%%
controller_params = RetargetingControllerParams(
    Kp_iiwa_desired, Kp_iiwa, control_period, nq, nv, nu)
controller_system = RetargetingControllerSystem(q_nominal, u_nominal, controller_params)

retargeting_controller, q_ref_trj, u_ref_trj = add_controller_system_to_diagram(
    builder,
    controller_iiwa,
    iiwa_model,
    t_knots,
    q_knots_ref,
    u_knots_ref,
    controller_params,)


# %%

diagram = builder.Build()
render_system_with_graphviz(diagram, "controller_hardware.gz")

#%%
controller_iiwa = diagram.GetSubsystemByName("retargeting_controller")


# %%

iiwa_log, controller_iiwa = run_simulation(
    iiwa_model, 
    iiwa_log_sink,
    plant, 
    meshcat_vis,
    q_traj_iiwa=q_traj_ref,
    f_C_W=f_C_W,
    add_schunk=False,
    is_visualizing=True,
    )

# %%
