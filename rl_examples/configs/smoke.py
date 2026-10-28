"""Tiny pipeline checks, deliberately too short to learn a useful policy."""

from dataclasses import replace
from .settings import SACConfig, NavigationConfig


def smoke_settings(conditions, agent, training, stages=None):
    conditions = {
        name: replace(
            env,
            **(
                {"max_actions": 8}
                if isinstance(env, NavigationConfig)
                else {"max_steps": 8}
            )
        )
        for name, env in conditions.items()
    }
    if isinstance(agent, SACConfig):
        agent = replace(
            agent,
            learning_starts=16,
            batch_size=16,
            buffer_size=256,
            hidden_sizes=(32, 32),
        )
    else:
        agent = replace(
            agent, n_steps=32, batch_size=16, n_epochs=2, hidden_sizes=(32, 32)
        )
    training = replace(
        training,
        steps_per_stage=64,
        eval_every=32,
        early_stopping=False,
        validation_a=(20000,),
        validation_b=(21000,),
        test_seeds=(30000,),
    )
    if stages is not None:
        stages = {
            name: [replace(stage, max_actions=8) for stage in sequence]
            for name, sequence in stages.items()
        }
    return conditions, agent, training, stages
