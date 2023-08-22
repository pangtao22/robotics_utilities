import numpy as np

def sine_trajectory(horizon, 
    N=100, 
    q0=np.array([+0.0, +0.0*np.pi, 0, -1.70, 0, -1.0, 0]),
    q1=np.array([+0.0, +0.5*np.pi, 0, -1.70, 0, -1.0, 0]),
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
    """Renders the Drake system (presumably a diagram,
    otherwise this graph will be fairly trivial) using
    graphviz to a specified file."""
    from graphviz import Source

    string = system.GetGraphvizString()
    src = Source(string)
    src.render(output_file, view=False)

