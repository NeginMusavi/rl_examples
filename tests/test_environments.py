import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from dataclasses import replace
from rl_examples.configs.settings import NavigationConfig, LanderConfig
from rl_examples.environments.factory import make_env
from rl_examples.environments.geometry import world_collision, reachable


def test_navigation_margin_and_physics():
    plain = make_env(NavigationConfig())
    padded = make_env(NavigationConfig(lidar_margin=0.1))
    check_env(plain, skip_render_check=True)
    a, _ = plain.reset(seed=321)
    b, _ = padded.reset(seed=321)
    assert plain.circles == padded.circles and plain.rectangles == padded.rectangles
    np.testing.assert_array_equal(plain.goal, padded.goal)
    np.testing.assert_array_equal(a[36:], b[36:])
    assert (b[:36] <= a[:36] + 1e-7).all()
    assert plain.step([0.4, 0.2])[1:] == padded.step([0.4, 0.2])[1:]
    np.testing.assert_array_equal(plain.robot.pose, padded.robot.pose)
    assert all(
        not world_collision(p, plain.circles, plain.rectangles) for p in plain.path
    )
    assert reachable([1, 1], [9, 9], [], [])
    assert not reachable([1, 5], [9, 5], [], [(4.8, 0, 0.4, 10)])
    plain.close()
    padded.close()


def test_lander_zero_lag_and_filter():
    base = gym.make(
        "LunarLander-v3", continuous=True, enable_wind=False, max_episode_steps=8
    )
    wrapped = make_env(LanderConfig(max_steps=8))
    lagged = make_env(LanderConfig(tau_seconds=0.1, max_steps=8))
    try:
        check_env(lagged, skip_render_check=True)
        a, _ = base.reset(seed=12)
        b, _ = wrapped.reset(seed=12)
        np.testing.assert_array_equal(a, b[:8])
        for _ in range(8):
            a, r, te, tr, _ = base.step(np.array([0.7, 0.2], dtype=np.float32))
            b, rr, tte, ttr, _ = wrapped.step([0.7, 0.2])
            np.testing.assert_array_equal(a, b[:8])
            assert (r, te, tr) == (rr, tte, ttr)
            if te or tr:
                break
        lagged.reset(seed=12)
        obs, *_ = lagged.step([1, 1])
        np.testing.assert_allclose(
            obs[-2:], [-1 + 2 * lagged.beta, lagged.beta], atol=1e-7
        )
    finally:
        base.close()
        wrapped.close()
        lagged.close()
