"""Matched trajectory examples and optional videos from saved checkpoints."""

from pathlib import Path
import math
from rl_examples.algorithms.agents import load_agent
from rl_examples.configs.settings import NavigationConfig, LanderConfig
from rl_examples.utils.files import read_json, save_json
from rl_examples.utils.labels import condition_label
from .rollout import simulate


def simulate_runs(runs, seeds=(21000, 21001, 21002), video_dir=None):
    seeds = tuple(seeds)
    if not runs or not seeds:
        raise ValueError("Provide runs and at least one scenario seed")
    results = {}
    for condition, folder in runs.items():
        folder = Path(folder)
        saved = read_json(folder / "config.json")
        cls = (
            NavigationConfig if "lidar_margin" in saved["environment"] else LanderConfig
        )
        config = cls(**saved["environment"])
        agent = load_agent(folder / "best.zip", saved["algorithm"])
        results[condition] = {}
        for seed in seeds:
            video = (
                Path(video_dir) / f"{condition}_seed_{seed}.mp4" if video_dir else None
            )
            result = simulate(agent, config, seed=seed, video_path=video)
            result.update(label=condition_label(config), video_path=video)
            if video:
                save_json(
                    video.with_suffix(".json"),
                    dict(
                        algorithm=saved["algorithm"],
                        environment=saved["environment"],
                        training_seed=saved["training"]["seed"],
                        checkpoint="best",
                        checkpoint_steps=agent.num_timesteps,
                        metrics={
                            key: (
                                None
                                if isinstance(value, float) and not math.isfinite(value)
                                else value
                            )
                            for key, value in result["metrics"].items()
                        },
                        frames_per_second=2 if cls is NavigationConfig else 50,
                    ),
                )
            results[condition][seed] = result
    return results
