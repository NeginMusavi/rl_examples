from dataclasses import dataclass


@dataclass(frozen=True)
class NavigationConfig:
    lidar_margin: float = 0.0
    n_circles: int = 3
    n_rectangles: int = 2
    lidar_noise: float = 0.02
    pose_noise: float = 0.01
    max_actions: int = 160


@dataclass(frozen=True)
class LanderConfig:
    tau_seconds: float = 0.0
    max_steps: int = 1000
    gravity: float = -10.0


@dataclass(frozen=True)
class SACConfig:
    learning_rate: float = 3e-4
    buffer_size: int = 100_000
    learning_starts: int = 1000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    hidden_sizes: tuple = (256, 256)
    device: str = "cpu"


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    n_steps: int = 1000
    batch_size: int = 250
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_sizes: tuple = (256, 256)
    device: str = "cpu"


@dataclass(frozen=True)
class TrainingConfig:
    steps_per_stage: int = 30_000
    eval_every: int = 2000
    seed: int = 42
    validation_a: tuple = tuple(range(20000, 20025))
    validation_b: tuple = tuple(range(21000, 21025))
    test_seeds: tuple = tuple(range(30000, 30100))
    early_stopping: bool = False
    min_steps: int = 10_000
    patience: int = 5
    min_delta: float = 0.0
