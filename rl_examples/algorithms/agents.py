"""Create and load Stable-Baselines3 SAC and PPO agents."""

from dataclasses import asdict
from stable_baselines3 import SAC, PPO
from rl_examples.configs.settings import SACConfig, PPOConfig
from rl_examples.models.networks import policy_kwargs


def algorithm_name(config):
    if isinstance(config, SACConfig):
        return "sac"
    if isinstance(config, PPOConfig):
        return "ppo"
    raise TypeError("Expected SACConfig or PPOConfig")


def create_agent(env, config, seed):
    name = algorithm_name(config)
    options = asdict(config)
    options.pop("hidden_sizes")
    options["policy_kwargs"] = policy_kwargs(config, name)
    model = SAC if name == "sac" else PPO
    return model("MlpPolicy", env, seed=seed, verbose=0, **options)


def load_agent(path, algorithm, device="cpu"):
    if algorithm not in ("sac", "ppo"):
        raise ValueError("algorithm must be 'sac' or 'ppo'")
    model = SAC if algorithm == "sac" else PPO
    return model.load(path, device=device)
