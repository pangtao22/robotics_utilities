#%% 
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
from iiwa_controller.examples.sim_setup import run_sim

kQTrjSrcName = "QTrajectorySource"
kUTrjSrcName = "UTrajectorySource"

# %%
# visualization
meshcat = StartMeshcat()


#%%
def linear_trajectory(horizon, N=100):
    q0 = np.array([+0.0, +0*np.pi, 0, -1.70, 0, -1.0, 0])
    q1 = np.array([+0.0, +0*np.pi, 0, -1.70, 0, -1.0, 0])
    q_iiwa_knots = np.zeros((N, 7))
    for i in range(N):
        q_iiwa_knots[i] = q0 + (q1 - q0) * i / (N-1)
    t_iiwa_knots = np.array([horizon * i / (N-1) for i in range(N)])
    return q_iiwa_knots, t_iiwa_knots


#%%
joint_stiffness = np.array([800.0, 600, 600, 600, 400, 200, 200])
# force at C.
f_C_W = np.array([0, 0, -0.0])
# Stiffness matrix of the robot.
gravity = np.array([0, 0, -0.0])

N = 100
horizon = 5.0
q_knots_ref, t_knots = linear_trajectory(horizon, N=N)
q_traj_ref = PiecewisePolynomial.FirstOrderHold(
    t_knots, q_knots_ref.T
)


run_sim(
    q_traj_iiwa=q_traj_ref,
    Kp_iiwa=joint_stiffness,
    gravity=gravity,
    f_C_W=f_C_W,
    time_step=1e-4,
    add_schunk=False,
    is_visualizing=True,
    meshcat=meshcat,
)

# %%
