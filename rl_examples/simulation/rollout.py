"""One frozen-policy episode; the same function powers evaluation and plots."""

import numpy as np
import pandas as pd
from rl_examples.configs.settings import NavigationConfig
from rl_examples.environments.factory import make_env


def simulate(agent, config, seed=21000, record=True, video_path=None):
    navigation = isinstance(config, NavigationConfig)
    env = make_env(
        config, render_mode="rgb_array" if video_path and not navigation else None
    )
    writer = None
    history = []
    total = main_use = side_use = 0.0
    contact_speed = contact_angle = np.nan
    try:
        obs, info = env.reset(seed=int(seed))
        if video_path:
            from .video import VideoRecorder

            writer = VideoRecorder(env, config, video_path)
            writer.capture(0)
        limit = config.max_actions if navigation else config.max_steps
        for step in range(1, limit + 1):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if navigation:
                x, y = env.robot.pose[:2]
            else:
                x, y = obs[:2]
                main_use += info["main_power"] * env.dt
                side_use += info["side_power"] * env.dt
                if np.isnan(contact_speed) and (obs[6] or obs[7]):
                    body = env.unwrapped.lander
                    contact_speed = float(np.hypot(*body.linearVelocity))
                    contact_angle = abs(float(body.angle))
            if record:
                row = dict(step=step, x=float(x), y=float(y), reward=reward)
                if not navigation:
                    row.update(
                        main_requested=float(action[0]),
                        side_requested=float(action[1]),
                        main_applied=float(obs[-2]),
                        side_applied=float(obs[-1]),
                    )
                history.append(row)
            if writer:
                state = (
                    (
                        "success"
                        if info["success"]
                        else (
                            "collision"
                            if info["collision"]
                            else "timeout" if truncated else "running"
                        )
                    )
                    if navigation
                    else info["outcome"]
                )
                writer.capture(step, state)
            if terminated or truncated:
                break
        outcome = (
            (
                "success"
                if info["success"]
                else "collision" if info["collision"] else "timeout"
            )
            if navigation
            else info["outcome"]
        )
        metrics = dict(
            scenario_seed=int(seed),
            episode_return=total,
            steps=step,
            success=outcome in ("success", "landed"),
            outcome=outcome,
        )
        scene = None
        if navigation:
            path = np.asarray(env.path)
            metrics.update(
                path_length_m=float(
                    np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum()
                ),
                distance_to_goal_m=info["distance_to_goal"],
            )
            if record:
                scene = dict(
                    path=path,
                    start=env.start,
                    goal=env.goal,
                    circles=env.circles,
                    rectangles=env.rectangles,
                    pose=env.robot.pose.copy(),
                    margin=config.lidar_margin,
                )
        else:
            metrics.update(
                main_power_seconds=main_use,
                side_power_seconds=side_use,
                first_contact_speed_m_s=contact_speed,
                first_contact_abs_angle_rad=contact_angle,
                return_ge_200=bool(total >= 200),
            )
        return dict(metrics=metrics, trajectory=pd.DataFrame(history), scene=scene)
    finally:
        try:
            if writer:
                writer.close()
        finally:
            env.close()
