from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.ticker import MaxNLocator
from rl_examples.utils.labels import run_labels
from rl_examples.environments.geometry import (
    WORLD_SIZE,
    ROBOT_RADIUS,
    inflate_obstacles,
)


def plot_navigation(env, ax=None, title=None, show_margin=True):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    for x, y, r in env.circles:
        ax.add_patch(Circle((x, y), r, color="0.6"))
    for x, y, w, h in env.rectangles:
        ax.add_patch(Rectangle((x, y), w, h, color="0.6"))
    if show_margin and env.config.lidar_margin > 0:
        circles, rectangles = inflate_obstacles(
            env.circles, env.rectangles, env.config.lidar_margin
        )
        for x, y, r in circles:
            ax.add_patch(
                Circle((x, y), r, fill=False, edgecolor="darkorange", linestyle="--")
            )
        for x, y, w, h in rectangles:
            ax.add_patch(
                Rectangle(
                    (x, y), w, h, fill=False, edgecolor="darkorange", linestyle="--"
                )
            )
    path = np.asarray(env.path)
    ax.plot(path[:, 0], path[:, 1], color="royalblue", label="Trajectory")
    ax.scatter(*env.start[:2], color="blue", label="Start")
    ax.scatter(*env.goal, color="green", marker="*", s=150, label="Goal")
    ax.add_patch(Circle(env.robot.pose[:2], ROBOT_RADIUS, fill=False, edgecolor="blue"))
    ax.set(
        xlim=(0, WORLD_SIZE),
        ylim=(0, WORLD_SIZE),
        xlabel="x (m)",
        ylabel="y (m)",
        title=title or f"margin = {env.config.lidar_margin:g} m",
    )
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    return ax


def plot_simulation(result, title=None):
    scene = result["scene"]
    outcome = result["metrics"]["outcome"]
    if scene is not None:
        env = SimpleNamespace(
            **{k: scene[k] for k in ("path", "start", "goal", "circles", "rectangles")},
            robot=SimpleNamespace(pose=scene["pose"]),
            config=SimpleNamespace(lidar_margin=scene["margin"]),
        )
        ax = plot_navigation(env, title=title or outcome)
        return ax.figure
    data = result["trajectory"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(data.x, data.y)
    axes[0].set(xlabel="Normalized x", ylabel="Normalized y", title=title or outcome)
    for ax, engine in zip(axes[1:], ("main", "side")):
        ax.plot(data.step / 50, data[engine + "_requested"], ":", label="Requested")
        ax.plot(data.step / 50, data[engine + "_applied"], label="Applied")
        ax.set(xlabel="Time (s)", ylabel="Engine command", title=engine)
        ax.legend()
    fig.tight_layout()
    return fig


def plot_learning_curves(runs):
    labels = run_labels(runs)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for color, (condition, folder) in enumerate(runs.items()):
        folder = Path(folder)
        curves = pd.read_csv(folder / "learning_curves.csv")
        logs = sorted(
            folder.glob("train_stage_*.monitor.csv"),
            key=lambda p: int(p.name.split("_")[2].split(".")[0]),
        )
        # Offset each stage by the interactions of earlier stages, including
        # episodes left unfinished at stage boundaries.
        offset, labelled = 0, False
        for stage, log in enumerate(logs, 1):
            frame = pd.read_csv(log, comment="#")
            if len(frame):
                axes[0].plot(
                    offset + frame.l.cumsum(),
                    frame.r.rolling(20, min_periods=1).mean(),
                    color=f"C{color}",
                    label=None if labelled else labels[condition],
                )
                labelled = True
            offset = curves.loc[curves.stage == stage, "steps"].max()
        for ax, metric in zip(axes[1:], ("mean_return", "success_rate")):
            for panel, style in (("a", "--"), ("b", "-")):
                ax.plot(
                    curves.steps,
                    curves[f"{panel}_{metric}"],
                    style,
                    color=f"C{color}",
                    label=f"{labels[condition]}, validation {panel.upper()}",
                )
        boundaries = curves.groupby("stage").steps.max().iloc[:-1]
        for ax in axes:
            for boundary in boundaries:
                ax.axvline(boundary, color="gray", linestyle=":", alpha=0.3)
    for ax, title in zip(
        axes,
        (
            "Training return (rolling 20)",
            "Validation return",
            "Validation success / landing",
        ),
    ):
        ax.set(title=title, xlabel="Training interactions")
        ax.grid(alpha=0.2)
        if ax.lines:
            ax.legend(fontsize=7)
    axes[2].set_ylim(-0.03, 1.03)
    fig.tight_layout()
    return fig


def plot_trajectories(rollouts):
    """Rows are matched scenario seeds; columns are experimental conditions."""
    conditions = list(rollouts)
    seeds = list(rollouts[conditions[0]])
    fig, axes = plt.subplots(
        len(seeds),
        len(conditions),
        figsize=(5 * len(conditions), 4.5 * len(seeds)),
        squeeze=False,
    )
    for column, condition in enumerate(conditions):
        for row, seed in enumerate(seeds):
            result = rollouts[condition][seed]
            ax = axes[row, column]
            title = f"{result['label']} | seed = {seed}\n{result['metrics']['outcome']}"
            scene = result["scene"]
            if scene is not None:
                env = SimpleNamespace(
                    **{
                        key: scene[key]
                        for key in ("path", "start", "goal", "circles", "rectangles")
                    },
                    robot=SimpleNamespace(pose=scene["pose"]),
                    config=SimpleNamespace(lidar_margin=scene["margin"]),
                )
                plot_navigation(env, ax=ax, title=title)
            else:
                data = result["trajectory"]
                ax.plot(data.x, data.y, label="Trajectory")
                ax.scatter(
                    data.x.iloc[0], data.y.iloc[0], label="First recorded position"
                )
                ax.scatter(
                    data.x.iloc[-1], data.y.iloc[-1], marker="x", label="Final position"
                )
                ax.set(xlabel="Normalized x", ylabel="Normalized y", title=title)
                ax.xaxis.set_major_locator(MaxNLocator(4))
                ax.yaxis.set_major_locator(MaxNLocator(5))
                ax.legend(fontsize=8)
                ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig
