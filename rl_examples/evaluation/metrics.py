import pandas as pd
from rl_examples.simulation.rollout import simulate


def evaluate(agent, config, seeds):
    """No learning or checkpoint selection. Reuse for validation or held-out test."""
    if not len(seeds):
        raise ValueError("Provide at least one scenario seed")
    return pd.DataFrame(
        [simulate(agent, config, seed, record=False)["metrics"] for seed in seeds]
    )


def summarize(episodes):
    result = dict(
        episodes=len(episodes),
        mean_return=episodes.episode_return.mean(),
        success_rate=episodes.success.mean(),
    )
    navigation = "path_length_m" in episodes
    for outcome in ("collision" if navigation else "failed", "timeout"):
        result[outcome + "_rate"] = (episodes.outcome == outcome).mean()
    successful = episodes[episodes.success]
    for metric in (
        "steps",
        "path_length_m",
        "main_power_seconds",
        "side_power_seconds",
    ):
        if metric in episodes:
            result["successful_mean_" + metric] = successful[metric].mean()
    return result


def paired_efficiency(episodes):
    """Compare conditions only on seeds where every condition succeeded.

    Expects one episode per condition and scenario seed (a single checkpoint).
    """
    if episodes.duplicated(["condition", "scenario_seed"]).any():
        raise ValueError(
            "Expected one episode per condition and seed; select one checkpoint first"
        )
    wide = episodes.pivot(index="scenario_seed", columns="condition", values="success")
    common = wide.index[wide.fillna(False).all(axis=1)]
    rows = []
    for condition, group in episodes.groupby("condition"):
        selected = group[group.scenario_seed.isin(common)]
        row = dict(condition=condition, common_successes=len(selected))
        for metric in (
            "steps",
            "path_length_m",
            "main_power_seconds",
            "side_power_seconds",
        ):
            if metric in selected:
                row["mean_" + metric] = selected[metric].mean()
        rows.append(row)
    return pd.DataFrame(rows)
