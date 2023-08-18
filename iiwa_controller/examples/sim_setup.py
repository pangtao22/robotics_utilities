import numpy as np

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Simulator,
    SpatialForce,
    RigidTransform,
    VectorLogSink,
    LogVectorOutput,
    StartMeshcat,
    MeshcatVisualizer,
    ContactVisualizer,
)
from pydrake.trajectories import PiecewisePolynomial
from pydrake.systems.framework import DiagramBuilder
from pydrake.multibody.plant import ExternallyAppliedSpatialForce
from pydrake.systems.primitives import TrajectorySource

from ..utils import *
from ..robot_internal_controller import RobotInternalController


def run_sim(
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
    builder.Connect(
        traj_source_iiwa.get_output_port(0),
        controller_iiwa.joint_angle_commanded_input_port,
    )


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

    diagram = builder.Build()

    # %% Run simulation.
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
    return iiwa_log, controller_iiwa, builder
