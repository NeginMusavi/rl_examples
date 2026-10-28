from pathlib import Path
import pandas as pd
from rl_examples.algorithms.agents import load_agent
from rl_examples.configs.settings import NavigationConfig, LanderConfig
from rl_examples.utils.files import read_json
from .metrics import evaluate, summarize


def compare_runs(runs, seeds, split="validation", checkpoints=("best", "final")):
    """Read the saved environment settings so evaluation matches each run."""
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test")
    if split == "test" and checkpoints != ("best",):
        raise ValueError(
            "Freeze the selected model before test; use checkpoints=('best',)"
        )
    summaries, frames = [], []
    for condition, folder in runs.items():
        folder = Path(folder)
        saved = read_json(folder / "config.json")
        if split == "test":
            used = set(
                saved["training"]["validation_a"] + saved["training"]["validation_b"]
            )
            if used.intersection(seeds):
                raise ValueError("Test seeds overlap validation")
        cls = (
            NavigationConfig if "lidar_margin" in saved["environment"] else LanderConfig
        )
        env_config = cls(**saved["environment"])
        for checkpoint in checkpoints:
            if checkpoint not in ("best", "final", "latest"):
                raise ValueError("Unknown checkpoint")
            model = load_agent(folder / f"{checkpoint}.zip", saved["algorithm"])
            data = evaluate(model, env_config, seeds)
            data["condition"], data["checkpoint"] = condition, checkpoint
            data["training_seed"] = saved["training"]["seed"]
            data.to_csv(folder / f"{split}_{checkpoint}.csv", index=False)
            frames.append(data)
            summaries.append(
                dict(
                    condition=condition,
                    checkpoint=checkpoint,
                    checkpoint_steps=model.num_timesteps,
                    **summarize(data),
                )
            )
    return pd.DataFrame(summaries), pd.concat(frames, ignore_index=True)
