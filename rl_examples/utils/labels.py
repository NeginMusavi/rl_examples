"""Presentation labels, independent of directory names."""

from dataclasses import asdict
from pathlib import Path
from .files import read_json


def condition_label(config):
    values = config if isinstance(config, dict) else asdict(config)
    if "lidar_margin" in values:
        return f"margin = {values['lidar_margin']:g} m"
    return f"engine lag = {values['tau_seconds']:g} s"


def run_labels(runs):
    return {
        name: condition_label(read_json(Path(folder) / "config.json")["environment"])
        for name, folder in runs.items()
    }


def format_results(table, labels):
    return table.assign(condition=table.condition.replace(labels)).round(3)
