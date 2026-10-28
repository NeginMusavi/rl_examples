"""Train in chunks so validation and saved weights are always post-update."""

import json
from dataclasses import asdict
from pathlib import Path
import pandas as pd
from stable_baselines3.common.monitor import Monitor
from rl_examples.algorithms.agents import create_agent, algorithm_name
from rl_examples.configs.settings import NavigationConfig, PPOConfig
from rl_examples.environments.factory import make_env
from rl_examples.evaluation.metrics import evaluate, summarize
from rl_examples.utils.files import save_json, read_json, package_versions
from rl_examples.utils.labels import condition_label


def validate_settings(agent_config, config, stages):
    if config.steps_per_stage < 1 or config.eval_every < 1 or not stages:
        raise ValueError("Training needs positive budgets and at least one stage")
    if config.patience < 1 or config.min_steps < 0 or config.min_delta < 0:
        raise ValueError("Invalid early-stopping settings")
    panels = [
        set(config.validation_a),
        set(config.validation_b),
        set(config.test_seeds),
    ]
    if not all(panels) or any(
        panels[i] & panels[j] for i in range(3) for j in range(i)
    ):
        raise ValueError(
            "Validation A, validation B, and test seeds must be nonempty and disjoint"
        )
    if isinstance(agent_config, PPOConfig):
        n = agent_config.n_steps
        if n < 2 or agent_config.batch_size < 2 or n % agent_config.batch_size:
            raise ValueError("PPO batch_size must be >=2 and divide n_steps")
        if config.steps_per_stage % n or config.eval_every % n:
            raise ValueError(
                "PPO stage budget and eval_every must be multiples of n_steps"
            )


def train(env_config, agent_config, config, output_dir, stages=None):
    """Train one condition. Optional stages are environment configs for a curriculum.

    Validation B selects best.zip; A is diagnostic. Patience is counted only in
    the final stage. Navigation ranks success first, then return; lander ranks
    return. min_delta applies to return when the success score is unchanged.
    Existing folders are never overwritten. Checkpoints support inference,
    not exact training resumption after a Colab disconnect.
    """
    stages = [env_config] if stages is None else list(stages)
    validate_settings(agent_config, config, stages)
    if any(type(stage) is not type(env_config) for stage in stages):
        raise ValueError("Training and evaluation environments must have the same type")
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=False)
    name = algorithm_name(agent_config)
    save_json(
        folder / "config.json",
        dict(
            algorithm=name,
            environment=asdict(env_config),
            agent=asdict(agent_config),
            training=asdict(config),
            stages=[asdict(s) for s in stages],
            packages=package_versions(),
        ),
    )
    history = []
    best = reference = (-float("inf"), -float("inf"))
    stale = 0
    best_steps = 0
    stop_reason = "budget"
    env = None
    try:
        agent = None
        for index, stage in enumerate(stages):
            if env is not None:
                env.close()
            env = Monitor(
                make_env(stage),
                filename=str(folder / f"train_stage_{index+1}.monitor.csv"),
            )
            if agent is None:
                agent = create_agent(env, agent_config, config.seed)
            else:
                agent.set_env(env)
            agent.get_env().seed(config.seed + 1000 * index)
            remaining = config.steps_per_stage
            while remaining:
                chunk = min(config.eval_every, remaining)
                before = agent.num_timesteps
                agent.learn(total_timesteps=chunk, reset_num_timesteps=False)
                if agent.num_timesteps - before != chunk:
                    raise RuntimeError(
                        "Training exceeded the requested interaction budget"
                    )
                remaining -= chunk
                row = dict(steps=agent.num_timesteps, stage=index + 1)
                for panel, seeds in (
                    ("a", config.validation_a),
                    ("b", config.validation_b),
                ):
                    episodes = evaluate(agent, env_config, seeds)
                    episodes.to_csv(
                        folder / f"validation_{panel}_{agent.num_timesteps}.csv",
                        index=False,
                    )
                    row.update(
                        {
                            f"{panel}_{key}": value
                            for key, value in summarize(episodes).items()
                        }
                    )
                history.append(row)
                pd.DataFrame(history).to_csv(
                    folder / "learning_curves.csv", index=False
                )
                primary = (
                    row["b_success_rate"]
                    if isinstance(env_config, NavigationConfig)
                    else 0.0
                )
                score = (primary, row["b_mean_return"])
                if score > best:
                    best, best_steps = score, agent.num_timesteps
                    agent.save(folder / "best")
                    save_json(folder / "best_checkpoint.json", row)
                agent.save(folder / "latest")
                eligible = (
                    index == len(stages) - 1 and agent.num_timesteps >= config.min_steps
                )
                improved = score[0] > reference[0] or (
                    score[0] == reference[0]
                    and score[1] > reference[1] + config.min_delta
                )
                if not eligible or improved:
                    reference, stale = score, 0
                else:
                    stale += 1
                print(
                    f"{name.upper()} | {condition_label(env_config)} | {agent.num_timesteps:,} steps | "
                    f"B success={row['b_success_rate']:.0%} | return={row['b_mean_return']:.1f}",
                    flush=True,
                )
                if config.early_stopping and eligible and stale >= config.patience:
                    stop_reason = "validation_patience"
                    break
            agent.save(folder / f"stage_{index+1}")
        agent.save(folder / "final")
        save_json(
            folder / "completed.json",
            dict(
                steps=agent.num_timesteps,
                best_steps=best_steps,
                stop_reason=stop_reason,
            ),
        )
    finally:
        if env is not None:
            env.close()
    return folder


def run_experiment(
    conditions, agent_config, config, output_dir, stages=None, mode="train"
):
    """Run independent conditions. stages is an optional dict of curriculum lists."""
    if mode not in ("train", "load") or not conditions:
        raise ValueError("Use mode='train' or 'load' and provide conditions")
    root = Path(output_dir)
    folders = {name: root / name / f"seed_{config.seed}" for name in conditions}
    for name, folder in folders.items():
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Condition names must be simple folder names")
        expected_stages = stages[name] if stages is not None else [conditions[name]]
        validate_settings(agent_config, config, expected_stages)
        if mode == "train" and folder.exists():
            raise FileExistsError(
                f"Run exists: {folder}. Use load mode or a new RUN_NAME."
            )
        if mode == "load":
            if not (folder / "completed.json").exists():
                raise FileNotFoundError(f"No completed run: {folder}")
            saved = read_json(folder / "config.json")
            expected = dict(
                environment=asdict(conditions[name]),
                agent=asdict(agent_config),
                training=asdict(config),
                stages=[asdict(s) for s in expected_stages],
            )
            if any(
                saved[key] != value
                for key, value in json.loads(json.dumps(expected)).items()
            ):
                raise ValueError(f"Settings differ from saved run: {folder}")
    if mode == "train":
        for name, env_config in conditions.items():
            train(
                env_config,
                agent_config,
                config,
                folders[name],
                stages[name] if stages is not None else None,
            )
    return folders
