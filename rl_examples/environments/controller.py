import numpy as np


class Robot:
    def __init__(self, x=1.0, y=1.0, theta=0.3):
        self.pose = np.array([x, y, theta], dtype=float)

    def step(self, v, omega, dt=0.1):
        x, y, theta = self.pose
        # Midpoint integration for unicycle/differential-drive kinematics.
        next_theta = theta + omega * dt
        x += v * dt * np.cos(theta + omega * dt / 2)
        y += v * dt * np.sin(theta + omega * dt / 2)
        self.pose = np.array([x, y, (next_theta + np.pi) % (2 * np.pi) - np.pi])
        return self.pose.copy()


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def waypoint_controller(
    pose, waypoint, k_distance=0.9, k_heading=2.5, max_speed=0.8, max_turn_rate=1.5
):
    x, y, theta = pose
    dx, dy = np.asarray(waypoint) - np.array([x, y])
    distance = np.hypot(dx, dy)
    desired_heading = np.arctan2(dy, dx)
    heading_error = wrap_angle(desired_heading - theta)

    v = min(max_speed, k_distance * distance) * max(0.0, np.cos(heading_error))
    omega = np.clip(k_heading * heading_error, -max_turn_rate, max_turn_rate)
    return float(v), float(omega)
