import numpy as np

from pydrake.all import (
    RotationMatrix,
    RigidTransform,
    CoulombFriction,
    HalfSpace,
)

def sine_trajectory(horizon, 
    N=100, 
    q0=np.array([+0.0, +0.00*np.pi, 0, -1.70, 0, -1.0, 0]),
    q1=np.array([+0.0, +0.35*np.pi, 0, -1.70, 0, -1.0, 0]),
):
    q_iiwa_knots = np.zeros((N, 7))
    for i in range(N):
        q_iiwa_knots[i] = q0 + (q1 - q0) * (1 - np.cos(2*np.pi * i / (N-1))) / 2
    t_iiwa_knots = np.array([horizon * i / (N-1) for i in range(N)])
    return q_iiwa_knots, t_iiwa_knots

def linear_trajectory(horizon, 
    N=100, 
    q0=np.array([+0.0, +0.0*np.pi, 0, -1.70, 0, -1.0, 0]),
    q1=np.array([+0.0, +0.5*np.pi, 0, -1.70, 0, -1.0, 0]),
):
    q_iiwa_knots = np.zeros((N, 7))
    for i in range(N):
        q_iiwa_knots[i] = q0 + (q1 - q0) * i / (N-1)
    t_iiwa_knots = np.array([horizon * i / (N-1) for i in range(N)])
    return q_iiwa_knots, t_iiwa_knots

def render_system_with_graphviz(system, output_file="system_view.gz"):
    """ 
    Renders the Drake system (presumably a diagram,
    otherwise this graph will be fairly trivial) using
    graphviz to a specified file. 
    """
    
    from graphviz import Source

    string = system.GetGraphvizString()
    src = Source(string)
    src.render(output_file, view=False)
    return 


def AddGround(plant):
    """
    Add a flat ground with friction
    """

    # Constants
    nontransparent_color = np.array([1,0.5,0.5,0.1])

    p_GroundOrigin = [0, 0.0, -1.0]
    R_GroundOrigin = RotationMatrix.MakeXRotation(0.0)
    X_GroundOrigin = RigidTransform(R_GroundOrigin, p_GroundOrigin)

    # Set Up Ground on Plant

    surface_friction = CoulombFriction(
            static_friction = 0.7,
            dynamic_friction = 0.5)
    plant.RegisterCollisionGeometry(
            plant.world_body(),
            X_GroundOrigin,
            HalfSpace(),
            "ground_collision",
            surface_friction)
    plant.RegisterVisualGeometry(
            plant.world_body(),
            X_GroundOrigin,
            HalfSpace(),
            "ground_visual",
            nontransparent_color)  # transparent
    return