from dataclasses import replace
import pandas as pd
import pytest
from rl_examples.configs.settings import (
    NavigationConfig,
    SACConfig,
    PPOConfig,
    TrainingConfig,
)
from rl_examples.configs.smoke import smoke_settings
from rl_examples.training.runner import run_experiment, validate_settings
from rl_examples.evaluation.comparison import compare_runs
from rl_examples.evaluation.metrics import paired_efficiency
from rl_examples.utils.files import read_json


def test_budget_guard():
    with pytest.raises(ValueError, match="multiples"):
        validate_settings(
            PPOConfig(), TrainingConfig(eval_every=1500), [NavigationConfig()]
        )
    with pytest.raises(ValueError, match="disjoint"):
        validate_settings(
            SACConfig(),
            TrainingConfig(validation_a=(1,), validation_b=(1,)),
            [NavigationConfig()],
        )


def test_paired_success_only():
    data = pd.DataFrame(
        dict(
            condition=["a", "a", "b", "b"],
            scenario_seed=[1, 2, 1, 2],
            success=[True, True, True, False],
            steps=[10, 20, 30, 1],
        )
    )
    result = paired_efficiency(data)
    assert result.common_successes.tolist() == [1, 1]
    assert result.mean_steps.tolist() == [10, 30]


def test_curriculum_checkpoints_and_load(tmp_path):
    env = NavigationConfig(n_circles=0, n_rectangles=0)
    conditions, agent, config, _ = smoke_settings(
        {"check": env}, SACConfig(), TrainingConfig()
    )
    stages = {"check": [conditions["check"], replace(conditions["check"], n_circles=1)]}
    runs = run_experiment(conditions, agent, config, tmp_path, stages=stages)
    folder = runs["check"]
    assert read_json(folder / "completed.json")["steps"] == 128
    history = pd.read_csv(folder / "learning_curves.csv")
    summary, _ = compare_runs(runs, config.validation_b)
    best = summary[summary.checkpoint == "best"].iloc[0]
    row = history[history.steps == best.checkpoint_steps].iloc[0]
    assert best.mean_return == pytest.approx(row.b_mean_return)
    run_experiment(conditions, agent, config, tmp_path, stages=stages, mode="load")
    with pytest.raises(FileExistsError):
        run_experiment(conditions, agent, config, tmp_path, stages=stages)
    with pytest.raises(ValueError, match="Settings differ"):
        run_experiment(
            conditions,
            agent,
            replace(config, seed=42, patience=9),
            tmp_path,
            stages=stages,
            mode="load",
        )
    with pytest.raises(ValueError, match="overlap"):
        compare_runs(runs, config.validation_b, split="test", checkpoints=("best",))


def test_patience_stops_after_post_update_validation(tmp_path, monkeypatch):
    import rl_examples.training.runner as runner

    conditions, agent, config, _ = smoke_settings(
        {"check": NavigationConfig(n_circles=0, n_rectangles=0)},
        SACConfig(),
        TrainingConfig(),
    )
    config = replace(
        config, steps_per_stage=128, early_stopping=True, min_steps=0, patience=1
    )
    fixed = pd.DataFrame(
        dict(
            scenario_seed=[21000],
            episode_return=[1.0],
            steps=[8],
            success=[False],
            outcome=["timeout"],
        )
    )
    monkeypatch.setattr(runner, "evaluate", lambda *args: fixed)
    runs = runner.run_experiment(conditions, agent, config, tmp_path)
    completed = read_json(runs["check"] / "completed.json")
    assert completed == dict(steps=64, best_steps=32, stop_reason="validation_patience")
