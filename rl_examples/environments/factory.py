import gymnasium as gym
from rl_examples.configs.settings import NavigationConfig, LanderConfig
from .navigation import RandomizedNavigationEnv
from .lander import EngineCommandLag


def make_env(config, render_mode=None):
    """Create a fresh, unseeded environment; reset supplies the scenario seed."""
    if isinstance(config, NavigationConfig):
        return RandomizedNavigationEnv(config, render_mode=render_mode)
    if not isinstance(config, LanderConfig):
        raise TypeError("Expected NavigationConfig or LanderConfig")
    if config.max_steps < 1:
        raise ValueError("max_steps must be positive")
    env = gym.make(
        "LunarLander-v3",
        continuous=True,
        enable_wind=False,
        gravity=config.gravity,
        max_episode_steps=config.max_steps,
        render_mode=render_mode,
    )
    return EngineCommandLag(env, config.tau_seconds)
