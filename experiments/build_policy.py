import gymnasium as gym
from tianshou.algorithm.modelfree.c51 import C51Policy
from tianshou.algorithm.modelfree.rainbow import RainbowDQN
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.utils.net.common import Net


def _normalize_state_shape(state_shape):
    if isinstance(state_shape, int):
        normalized = (int(state_shape),)
    else:
        try:
            normalized = tuple(int(dim) for dim in state_shape)
        except TypeError as exc:
            raise TypeError("state_shape 必须为整数或整数序列") from exc
    if not normalized or any(dim <= 0 for dim in normalized):
        raise ValueError("state_shape 必须全部为正整数且不能为空")
    return normalized


def _normalize_action_shape(action_shape: int) -> int:
    if isinstance(action_shape, int):
        normalized = int(action_shape)
    else:
        try:
            dimensions = [int(dim) for dim in action_shape]
        except TypeError as exc:
            raise TypeError("action_shape 必须为正整数或仅包含正整数的序列") from exc
        if not dimensions or any(dim <= 0 for dim in dimensions):
            raise ValueError("action_shape 必须为正整数")
        normalized = 1
        for dim in dimensions:
            normalized *= dim
    if normalized <= 0:
        raise ValueError("action_shape 必须为正整数")
    return normalized


def build_rainbow_components(
    state_shape,
    action_shape: int,
    hidden_size: int,
    device: str,
):
    normalized_state_shape = _normalize_state_shape(state_shape)
    normalized_action_shape = _normalize_action_shape(action_shape)
    normalized_hidden_size = max(1, int(hidden_size))
    net = Net(
        state_shape=normalized_state_shape,
        action_shape=normalized_action_shape,
        hidden_sizes=[normalized_hidden_size, normalized_hidden_size],
        num_atoms=51,
        dueling_param=({}, {}),
    ).to(device)
    policy = C51Policy(
        model=net,
        action_space=gym.spaces.Discrete(normalized_action_shape),
        num_atoms=51,
        v_min=-10,
        v_max=10,
    ).to(device)
    optim_factory = AdamOptimizerFactory(lr=1e-4)
    algorithm = RainbowDQN(
        policy=policy,
        optim=optim_factory,
        gamma=0.99,
        n_step_return_horizon=3,
        target_update_freq=100,
    )
    return algorithm, policy
