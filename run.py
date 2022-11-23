import time
import meshcat
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import numpy as np
from visualization import frame_viz, robot_viz

robots = [
    {"name": "j", "wrapper": RobotWrapper.BuildFromURDF("6axis/robot.urdf", "6axis/"), "use_jlog": False},
    {"name": "jlog", "wrapper": RobotWrapper.BuildFromURDF("6axis/robot.urdf", "6axis/"), "use_jlog": True},
]

for robot in robots:
    robot["viz"] = robot_viz(robot["wrapper"], robot["name"])
    robot["q"] = robot["wrapper"].q0
    robot["frame_id"] = robot["wrapper"].model.getFrameId("effector")


def compute_target():
    r = pin.exp6(np.array([0.0, 0.0, 0.0, *np.random.uniform([-1.0] * 3, [1.0] * 3)]))
    t = pin.exp6(np.array([*np.random.uniform([1.0, -1.0, 0.5], [1.5, 1.0, 1.2]), 0.0, 0.0, 0.0]))
    return t * r


T_world_target = compute_target()

xs = []
t: float = 0
dt: float = 0.01
vmax: float = 0.01


def update_q(robot: RobotWrapper, q: list, frame_id: int, use_jlog: bool = False):
    T_world_effector = robot.data.oMf[frame_id]

    error = -pin.log6(pin.SE3(np.linalg.inv(T_world_target) @ T_world_effector))
    J = robot.computeFrameJacobian(q, frame_id)

    if use_jlog:
        Jlog = pin.Jlog6(pin.SE3(np.linalg.inv(T_world_target) @ T_world_effector))
        J = Jlog @ J

    delta = np.linalg.pinv(J) @ np.array(error)

    m = max(abs(delta).flatten())
    if m > vmax:
        delta = delta * vmax / m

    return delta


while True:
    all_reached: bool = True

    for robot in robots:
        robot["wrapper"].framePlacement(robot["q"], robot["frame_id"], True)
        robot["viz"].display(robot["q"])

        delta = update_q(robot["wrapper"], robot["q"], robot["frame_id"], robot["use_jlog"])

        robot["q"] += delta

        # Dirty check if we reached
        if np.linalg.norm(delta) > 1e-4:
            all_reached = False

        frame_viz(robot["name"] + "_effector", robot["wrapper"].data.oMf[robot["frame_id"]])

    if all_reached:
        T_world_target = compute_target()

    # Vizualizing the frames
    frame_viz("target", T_world_target, 0.25)

    t += dt
    time.sleep(dt)
