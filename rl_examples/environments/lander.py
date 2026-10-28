"""Normalized engine-command lag, not a physical thrust-lag model."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class EngineCommandLag(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """First-order command filter, exposing its memory in the observation."""

    def __init__(self, env, tau_seconds=0.0):
        gym.utils.RecordConstructorArgs.__init__(self, tau_seconds=tau_seconds)
        gym.Wrapper.__init__(self, env)
        if not np.isfinite(tau_seconds) or tau_seconds < 0:
            raise ValueError("tau_seconds must be finite and nonnegative")
        self.tau_seconds = float(tau_seconds)
        self.dt = 1.0 / self.unwrapped.metadata["render_fps"]
        self.beta = 1.0 if tau_seconds == 0 else -np.expm1(-self.dt / tau_seconds)
        self.u = np.array([-1.0, 0.0], dtype=np.float32)
        low = np.concatenate([env.observation_space.low, [-1.0, -1.0]])
        high = np.concatenate([env.observation_space.high, [1.0, 1.0]])
        self.observation_space = spaces.Box(
            low.astype(np.float32), high.astype(np.float32)
        )

    def _augment(self, obs):
        return np.concatenate([obs, self.u]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.u = np.array([-1.0, 0.0], dtype=np.float32)
        return self._augment(obs), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Expected a finite action with shape (2,)")
        requested = np.clip(action, -1, 1)
        # Direct assignment preserves exact baseline behavior at beta=1.
        self.u = (
            requested.copy()
            if self.beta == 1.0
            else np.clip(self.u + self.beta * (requested - self.u), -1, 1).astype(
                np.float32
            )
        )
        obs, reward, terminated, truncated, info = self.env.step(self.u)
        main = float((self.u[0] + 1) / 2) if self.u[0] > 0 else 0.0
        side = float(abs(self.u[1])) if abs(self.u[1]) > 0.5 else 0.0
        outcome = "running"
        # The base implementation tests failure first, then asleep, so follow
        # the final native terminal reward to respect its exact precedence.
        if terminated:
            outcome = "landed" if reward == 100.0 else "failed"
        elif truncated:
            outcome = "timeout"
        info = dict(
            info,
            outcome=outcome,
            requested_action=requested.copy(),
            applied_command=self.u.copy(),
            main_power=main,
            side_power=side,
        )
        return self._augment(obs), reward, terminated, truncated, info
