"""Run from the project root after installation: python scripts/smoke_test.py."""

from tempfile import TemporaryDirectory
from pathlib import Path
from dataclasses import replace
from rl_examples.configs.settings import (
    NavigationConfig,
    LanderConfig,
    SACConfig,
    PPOConfig,
    TrainingConfig,
)
from rl_examples.configs.smoke import smoke_settings
from rl_examples.training.runner import run_experiment
from rl_examples.evaluation.comparison import compare_runs
from rl_examples.algorithms.agents import load_agent, algorithm_name
from rl_examples.simulation.rollout import simulate


def main():
    for env in (
        NavigationConfig(n_circles=1, n_rectangles=1),
        LanderConfig(tau_seconds=0.1),
    ):
        for agent in (SACConfig(), PPOConfig()):
            conditions, agent, config, _ = smoke_settings(
                {"check": env}, agent, TrainingConfig()
            )
            with TemporaryDirectory() as folder:
                runs = run_experiment(conditions, agent, config, Path(folder) / "run")
                run_experiment(
                    conditions, agent, config, Path(folder) / "run", mode="load"
                )
                summary, _ = compare_runs(runs, config.validation_b)
                model = load_agent(runs["check"] / "best.zip", algorithm_name(agent))
                result = simulate(model, conditions["check"])
                assert len(summary) == 2 and result["metrics"]["steps"] > 0
                print(type(env).__name__, algorithm_name(agent), "passed")


if __name__ == "__main__":
    main()
