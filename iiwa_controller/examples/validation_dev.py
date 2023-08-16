# %%
import numpy as np
from robotics_utilities.iiwa_controller.examples.sim_setup import run_sim
from robotics_utilities.iiwa_controller.examples.test_iiwa_static_load import TestIiwaStaticLoad
from pydrake.all import (
    PiecewisePolynomial, 
    VectorLog, 
    JacobianWrtVariable,
    StartMeshcat,
    MeshcatVisualizer,
)


# %%
tester = TestIiwaStaticLoad()
# tester.setUp()

# coordinate of point C expressed in frame L7.
tester.p_L7oC_L7 = np.zeros(3)
# force at C.
tester.f_C_W = np.array([0, 0, -20])
# Stiffness matrix of the robot.
tester.Kp_iiwa = np.array([800.0, 600, 600, 600, 400, 200, 200])
tester.gravity = np.array([0, 0, -10.0])

# robot trajectory (hold q0).
q0 = np.array([0, 0, 0, -1.70, 0, 1.0, 0])
q_iiwa_knots = np.zeros((2, 7))
q_iiwa_knots[0] = q0
q_iiwa_knots[1] = q0

# run simulation for 1s.
tester.qa_traj = PiecewisePolynomial.FirstOrderHold(
    [0, 1], q_iiwa_knots.T
)

# %%
# visualization.
# Start the visualizer.
meshcat = StartMeshcat()
meshcat_vis = MeshcatVisualizer.AddToBuilder(
    builder, scene_graph, meshcat
)
ContactVisualizer.AddToBuilder(
    builder,
    plant,
    meshcat,
)


iiwa_log, controller_iiwa = run_sim(
    q_traj_iiwa=tester.qa_traj,
    Kp_iiwa=tester.Kp_iiwa,
    gravity=tester.gravity,
    f_C_W=tester.f_C_W,
    time_step=1e-5,
    add_schunk=False,
    is_visualizing=True,
    )


meshcat_vis.DeleteRecording()
meshcat_vis.StartRecording()
meshcat_vis.PublishRecording()



# %%
# Stiffness matrix of the robot.
Kp_iiwa = np.array([800.0, 600, 600, 600, 400, 200, 200])
gravity = np.array([0, 0, -10.0])

f_C_W = np.array([0, 0, -20])

time_step = 0.001
add_schunk = False
is_visualizing = False

# robot trajectory (hold q0).
q0 = np.array([0, 0, 0, -1.70, 0, 1.0, 0])
q_iiwa_knots = np.zeros((2, 7))
q_iiwa_knots[0] = q0
q_iiwa_knots[1] = q0


# %%
# run simulation for 1s.
q_traj_iiwa = PiecewisePolynomial.FirstOrderHold(
    [0, 1], q_iiwa_knots.T
)


# %%
run_sim(
    q_traj_iiwa,
    Kp_iiwa,
    gravity,
    f_C_W,
    time_step,
    add_schunk,
    is_visualizing=False,
)


