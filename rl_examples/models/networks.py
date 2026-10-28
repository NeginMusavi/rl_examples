"""Network settings; SB3 constructs and trains the actual PyTorch networks."""

from torch import nn


def policy_kwargs(config, algorithm):
    widths = list(config.hidden_sizes)
    architecture = dict(pi=widths, vf=widths) if algorithm == "ppo" else widths
    return dict(net_arch=architecture, activation_fn=nn.ReLU)
