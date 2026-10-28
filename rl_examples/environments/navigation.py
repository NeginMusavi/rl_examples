"""Continuous 2D waypoint navigation; physical geometry is unchanged by margin."""

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from rl_examples.configs.settings import NavigationConfig
from .controller import Robot, wrap_angle, waypoint_controller
from .geometry import WORLD_SIZE, ROBOT_RADIUS, LIDAR_RANGE, N_BEAMS
from .geometry import world_collision, world_scan, reachable

HEADING_NOISE = 0.01  # Standard deviation (rad) of the heading estimate.


class RandomizedNavigationEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config=NavigationConfig(), render_mode=None):
        super().__init__()
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        self.render_mode = render_mode
        if not np.isfinite(config.lidar_margin) or config.lidar_margin < 0:
            raise ValueError("lidar_margin must be finite and nonnegative")
        if config.max_actions < 1 or min(config.n_circles, config.n_rectangles) < 0:
            raise ValueError("Invalid episode length or obstacle count")
        if min(config.lidar_noise, config.pose_noise) < 0:
            raise ValueError("Noise standard deviations must be nonnegative")
        self.config = config
        self.n_circles = int(config.n_circles)
        self.n_rectangles = int(config.n_rectangles)
        self.lidar_noise = float(config.lidar_noise)
        self.pose_noise = float(config.pose_noise)
        self.max_actions = int(config.max_actions)
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-1, 1, shape=(40,), dtype=np.float32)
        self.circles = []
        self.rectangles = []
        self.start = np.array([1.0, 1.0, 0.0])
        self.goal = np.array([9.0, 9.0])
        self.robot = Robot(*self.start)
        self.path = [self.robot.pose.copy()]
        self.last_v = self.last_omega = 0.0
        self.action_count = 0
        # First world is sampled by reset(), after the caller supplies its seed.

    def _generate_world(self):
        rng = self.np_random
        for _ in range(250):
            circles = []
            rectangles = []
            for _ in range(self.n_circles):
                radius = float(rng.uniform(0.35, 0.7))
                circles.append(
                    (
                        float(rng.uniform(1 + radius, 9 - radius)),
                        float(rng.uniform(1 + radius, 9 - radius)),
                        radius,
                    )
                )
            for _ in range(self.n_rectangles):
                w, h = rng.uniform(0.55, 1.3, size=2)
                rectangles.append(
                    (
                        float(rng.uniform(0.7, 9.3 - w)),
                        float(rng.uniform(0.7, 9.3 - h)),
                        float(w),
                        float(h),
                    )
                )
            for _ in range(120):
                s = rng.uniform(0.6, 9.4, size=2)
                g = rng.uniform(0.6, 9.4, size=2)
                if np.linalg.norm(s - g) < 4:
                    continue
                if world_collision([*s, 0], circles, rectangles):
                    continue
                if world_collision([*g, 0], circles, rectangles):
                    continue
                if reachable(s, g, circles, rectangles):
                    self.circles = circles
                    self.rectangles = rectangles
                    self.start = np.array([*s, float(rng.uniform(-np.pi, np.pi))])
                    self.goal = np.asarray(g)
                    return
        raise RuntimeError("Could not sample reachable map; reduce obstacle count.")

    def _observation(self):
        ranges = world_scan(
            self.robot.pose,
            self.circles,
            self.rectangles,
            self.np_random,
            self.lidar_noise,
            margin=self.config.lidar_margin,
        )
        estimated_xy = self.robot.pose[:2] + self.np_random.normal(
            0, self.pose_noise, 2
        )
        estimated_theta = wrap_angle(
            self.robot.pose[2] + self.np_random.normal(0, HEADING_NOISE)
        )
        delta = self.goal - estimated_xy
        dist = np.linalg.norm(delta)
        bearing = wrap_angle(np.arctan2(delta[1], delta[0]) - estimated_theta)
        return np.concatenate(
            (
                ranges / LIDAR_RANGE,
                [
                    np.clip(dist / 15, 0, 1),
                    bearing / np.pi,
                    np.clip(self.last_v / 0.8, -1, 1),
                    np.clip(self.last_omega / 1.5, -1, 1),
                ],
            )
        ).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._generate_world()
        self.robot = Robot(*self.start)
        self.last_v = self.last_omega = 0.0
        self.action_count = 0
        self.path = [self.robot.pose.copy()]
        return self._observation(), {
            "distance_to_goal": float(np.linalg.norm(self.goal - self.robot.pose[:2]))
        }

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError("Action must be a finite vector of shape (2,)")
        action = np.clip(action, -1, 1)
        theta = self.robot.pose[2]
        rot = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        waypoint = self.robot.pose[:2] + rot @ (1.5 * action)
        before = np.linalg.norm(self.goal - self.robot.pose[:2])
        collision = False
        reached = False
        v = omega = 0.0
        # 0.5 s per RL action; 10 controller ticks; each tick split into
        # small discrete motion increments (<= 2 cm translation and <= 0.04 rad).
        for _ in range(10):
            v, omega = waypoint_controller(self.robot.pose, waypoint)
            n = max(
                1, int(np.ceil(max(abs(v) * 0.05 / 0.02, abs(omega) * 0.05 / 0.04)))
            )
            for _ in range(n):
                old = self.robot.pose.copy()
                self.robot.step(v, omega, 0.05 / n)
                if world_collision(self.robot.pose, self.circles, self.rectangles):
                    self.robot.pose = old
                    collision = True
                    break
                self.path.append(self.robot.pose.copy())
                if np.linalg.norm(self.goal - self.robot.pose[:2]) < 0.35:
                    reached = True
                    break
            if collision or reached:
                break
        self.last_v, self.last_omega = v, omega
        self.action_count += 1
        distance = np.linalg.norm(self.goal - self.robot.pose[:2])
        reward = 10 * (before - distance) - 0.05 - 20 * collision + 50 * reached
        terminated = bool(collision or reached)
        truncated = bool(self.action_count >= self.max_actions and not terminated)
        info = {
            "collision": collision,
            "success": reached,
            "distance_to_goal": float(distance),
        }
        return self._observation(), float(reward), terminated, truncated, info

    def render(self):
        from rl_examples.visualization.plots import plot_navigation

        plot_navigation(self)
        plt.show()

    def close(self):
        pass
