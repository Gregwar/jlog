import meshcat
import pinocchio as pin
import numpy as np
import meshcat.geometry as g
import meshcat.transformations as tf
from pinocchio.robot_wrapper import RobotWrapper


viewer = None
robot_names: dict = {}


def get_viewer() -> meshcat.Visualizer:
    """
    Gets the meshcat viewer, if it doesn't exist, create it
    """
    global viewer

    if viewer is None:
        viewer = meshcat.Visualizer()
        print(f"Viewer URL: {viewer.url()}")

    return viewer


def robot_viz(robot: RobotWrapper, name: str = "robot") -> pin.visualize.MeshcatVisualizer:
    """
    Builds an instance of pinocchio MeshcatVisualizer, which allows to push the model to the meshcat
    visualizer passed as parameter

    The robot can further be displayed using:

    > viz.display(q)
    """
    global robot_names

    robot_names[robot] = name
    viz = pin.visualize.MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer(viewer=get_viewer())
    viz.loadViewerModel(name)

    return viz


cylinders: dict = {}


def frame_viz(name: str, T: np.ndarray, opacity: float = 1.0) -> None:
    """
    Visualizes a given frame
    """
    global cylinders
    vis = get_viewer()
    axises = {
        "x": (0xFF0000, (-np.pi / 2, [0, 0, 1])),
        "y": (0x00FF00, (0, [0, 1, 0])),
        "z": (0x0000FF, (np.pi / 2, [1, 0, 0])),
    }

    for axis_name in axises:
        color, rotate = axises[axis_name]
        node_name = f"frames/{name}/{axis_name}"

        if node_name not in cylinders:
            cylinders[node_name] = vis[node_name]
            cylinders[node_name].set_object(
                g.Cylinder(0.1, 0.005),
                g.MeshLambertMaterial(color=color, opacity=opacity),
            )
        obj = cylinders[node_name]

        obj.set_transform(T @ tf.rotation_matrix(*rotate) @ tf.translation_matrix([0, 0.05, 0]))


def point_viz(name: str, point: np.ndarray, radius: float = 0.01, color: float = 0xFF0000, opacity: float = 1.0) -> None:
    """
    Prints a point (sphere)
    """
    vis = get_viewer()
    vis["points"][name].set_object(g.Sphere(radius), g.MeshPhongMaterial(color=color, opacity=opacity))
    vis["points"][name].set_transform(tf.translation_matrix(point))
