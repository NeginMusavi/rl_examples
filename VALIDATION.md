# Software Validation

Verification date: 2026-10-28. Environment: Linux, Python 3.12.14.

## Automated checks

| Check | Result |
|---|---|
| Regression suite | 9 tests passed |
| SAC/PPO notebook execution | All four notebook bodies passed in smoke mode |
| Notebook structure | Valid schema; description above every code cell; no function/class definitions |
| Trajectory comparisons | Three matched scenarios per condition; figures visually inspected |
| Video export | Navigation and LunarLander MP4s, previews, and JSON metadata generated |
| Recording consistency | Recorded and unrecorded rollouts produce identical trajectories and returns |

Regression coverage includes environment geometry, margin invariance, engine-lag
behavior, budget and split guards, checkpoint selection/loading, patience stopping,
paired-success metrics, presentation labels, and encoded frame counts.

## Scope

Smoke runs verify execution, not convergence or policy quality. Full-budget
training and performance comparisons are outside this verification. The notebook
bodies were executed locally; the Google Drive authentication and mount
cells were not exercised in this environment. Box2D deprecation warnings and the
Gymnasium wrapped-environment checker warning did not prevent test completion.

## Dependency versions

| Package | Version |
|---|---|
| gymnasium | 1.2.3 |
| stable-baselines3 | 2.7.1 |
| torch | 2.15.0+cpu |
| numpy | 2.3.5 |
| pandas | 2.2.3 |
| matplotlib | 3.10.8 |
| Box2D | 2.3.10 |

Reproduce the checks from the project root with `python -m pytest -q` and
`python scripts/smoke_test.py` after installing the test dependencies.
