import json
import numpy as np
import pandas as pd
import imageio.v2 as imageio
import pytest
import matplotlib.pyplot as plt
from rl_examples.configs.settings import NavigationConfig, LanderConfig
from rl_examples.simulation.rollout import simulate
from rl_examples.utils.labels import condition_label, run_labels, format_results
from rl_examples.utils.files import save_json
from rl_examples.visualization.plots import plot_trajectories


class FixedPolicy:
    def predict(self, observation, deterministic=True):
        return np.array([0.7, 0.2], dtype=np.float32), None


@pytest.mark.parametrize(
    "config", [NavigationConfig(max_actions=8), LanderConfig(max_steps=8)]
)
def test_video_preserves_rollout_and_encodes(tmp_path, config):
    expected = simulate(FixedPolicy(), config, seed=21000)
    path = tmp_path / "episode.mp4"
    recorded = simulate(FixedPolicy(), config, seed=21000, video_path=path)
    pd.testing.assert_frame_equal(expected["trajectory"], recorded["trajectory"])
    assert (
        recorded["metrics"]["episode_return"] == expected["metrics"]["episode_return"]
    )
    assert recorded["metrics"]["outcome"] == expected["metrics"]["outcome"]
    json.dumps(recorded["metrics"])
    with imageio.get_reader(path) as reader:
        assert reader.count_frames() == recorded["metrics"]["steps"] + 1
        assert reader.get_data(0).shape[-1] == 3
        assert reader.get_meta_data()["fps"] == (
            2 if isinstance(config, NavigationConfig) else 50
        )
    assert imageio.imread(path.with_suffix(".png")).shape[-1] == 3


def test_labels_and_matched_trajectory_grid(tmp_path):
    from dataclasses import asdict

    runs, rollouts = {}, {}
    for name, margin in [("margin_0", 0.0), ("margin_010", 0.1)]:
        folder = tmp_path / name
        folder.mkdir()
        config = NavigationConfig(max_actions=8, lidar_margin=margin)
        save_json(folder / "config.json", {"environment": asdict(config)})
        runs[name] = folder
        rollouts[name] = {}
        for seed in [21000, 21001]:
            result = simulate(FixedPolicy(), config, seed=seed)
            result["label"] = condition_label(config)
            rollouts[name][seed] = result
    labels = run_labels(runs)
    assert labels["margin_010"] == "margin = 0.1 m"
    table = format_results(pd.DataFrame({"condition": ["margin_010"]}), labels)
    assert table.condition.iloc[0] == "margin = 0.1 m"
    fig = plot_trajectories(rollouts)
    assert len(fig.axes) == 4
    assert "margin = 0.1 m | seed = 21001" in fig.axes[-1].get_title()
    plt.close(fig)
