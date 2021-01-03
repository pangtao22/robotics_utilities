import numpy as np

from pydrake.all import (FindResourceOrThrow, Parser, MultibodyPlant,
    Joint, SpatialInertia, RigidTransform)
from pydrake.math import RollPitchYaw

iiwa_sdf_path_drake = FindResourceOrThrow(
    "drake/manipulation/models/iiwa_description/iiwa7/iiwa7_no_collision.sdf")

schunk_sdf_path_drake = FindResourceOrThrow(
    "drake/manipulation/models/wsg_50_description/sdf"
    "/schunk_wsg_50_ball_contact.sdf")

X_L7E = RigidTransform(
    RollPitchYaw(np.pi / 2, 0, np.pi / 2), np.array([0, 0, 0.114]))


def create_iiwa_controller_plant(gravity, add_schunk_inertia=False):
    """
    Creates plant that includes only the robot, used for controllers.
    :param add_schunk_inertia:
    :param gravity:
    :return:
    """
    plant = MultibodyPlant(1e-3)
    parser = Parser(plant=plant)
    iiwa_model = parser.AddModelFromFile(iiwa_sdf_path_drake)
    plant.WeldFrames(A=plant.world_frame(),
                           B=plant.GetFrameByName("iiwa_link_0"))
    plant.mutable_gravity_field().set_gravity_vector(gravity)

    if add_schunk_inertia:
        wsg_equivalent = plant.AddRigidBody(
            "wsg_equivalent", iiwa_model, calc_schunk_inertia())
        plant.WeldFrames(
            A=plant.GetFrameByName("iiwa_link_7", iiwa_model),
            B=wsg_equivalent.body_frame(),
            X_AB=X_L7E)

    plant.Finalize()

    link_frame_indices = []
    for i in range(8):
        link_frame_indices.append(
            plant.GetFrameByName("iiwa_link_" + str(i)).index())

    return plant, link_frame_indices


def calc_schunk_inertia():
    """
    Verbatim translation from a function in drake's ManipulationStation.
    :return:
    """
    def calc_finger_pose_in_gripper_frame(slider: Joint):
        # Pose of the joint's parent frame P (attached on gripper body G) in the
        #  frame of the gripper G.
        X_GP = slider.frame_on_parent().GetFixedPoseInBodyFrame()
        # Pose of the joint's child frame C (attached on the slider's finger
        #  body) in the frame of the slider's finger F.
        X_FC = slider.frame_on_child().GetFixedPoseInBodyFrame()
        # When the slider's translational dof is zero, then P coincides with C.
        # Therefore:
        X_GF = X_GP.multiply(X_FC.inverse())
        return X_GF

    def calc_finger_spatial_inertia_in_gripper_frame(
            M_FFo_F: SpatialInertia, X_GF: RigidTransform):
        """
        Helper to compute the spatial inertia of a finger F in about the
            gripper's origin Go, expressed in G.
        """
        M_FFo_G = M_FFo_F.ReExpress(X_GF.rotation())
        p_FoGo_G = -X_GF.translation()
        M_FGo_G = M_FFo_G.Shift(p_FoGo_G)
        return M_FGo_G

    plant = MultibodyPlant(1e-3)
    parser = Parser(plant)
    parser.AddModelFromFile(schunk_sdf_path_drake)
    plant.Finalize()

    gripper_body = plant.GetBodyByName("body")
    left_finger_body = plant.GetBodyByName("left_finger")
    right_finger_body = plant.GetBodyByName("right_finger")

    M_GGo_G = gripper_body.default_spatial_inertia()
    M_LLo_L = left_finger_body.default_spatial_inertia()
    M_RRo_R = right_finger_body.default_spatial_inertia()
    left_slider = plant.GetJointByName("left_finger_sliding_joint")
    right_slider = plant.GetJointByName("right_finger_sliding_joint")

    X_GL = calc_finger_pose_in_gripper_frame(left_slider)
    X_GR = calc_finger_pose_in_gripper_frame(right_slider)

    M_LGo_G = calc_finger_spatial_inertia_in_gripper_frame(M_LLo_L, X_GL)
    M_RGo_G = calc_finger_spatial_inertia_in_gripper_frame(M_RRo_R, X_GR)

    M_CGo_G = M_GGo_G
    M_CGo_G += M_LGo_G
    M_CGo_G += M_RGo_G

    return M_CGo_G
