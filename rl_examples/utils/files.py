import json
from importlib.metadata import version
from pathlib import Path


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2))


def read_json(path):
    return json.loads(Path(path).read_text())


def package_versions():
    return {
        name: version(name)
        for name in ("numpy", "torch", "gymnasium", "stable-baselines3", "pandas")
    }
