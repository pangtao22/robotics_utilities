from pydrake.all import FindResourceOrThrow, Parser, MultibodyPlant

iiwa_sdf_path_drake = FindResourceOrThrow(
    "drake/manipulation/models/iiwa_description/iiwa7/iiwa7_no_collision.sdf")


def create_iiwa_controller_plant(gravity):
    # creates plant that includes only the robot, used for controllers.
    plant_robot = MultibodyPlant(1e-3)
    parser = Parser(plant=plant_robot)
    parser.AddModelFromFile(iiwa_sdf_path_drake)
    plant_robot.WeldFrames(A=plant_robot.world_frame(),
                           B=plant_robot.GetFrameByName("iiwa_link_0"))
    plant_robot.mutable_gravity_field().set_gravity_vector(gravity)
    plant_robot.Finalize()

    link_frame_indices = []
    for i in range(8):
        link_frame_indices.append(
            plant_robot.GetFrameByName("iiwa_link_" + str(i)).index())

    return plant_robot, link_frame_indices
