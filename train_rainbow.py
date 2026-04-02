import os
import sys
import torch
import gymnasium as gym

# 注入路径
PROJECT_ROOT = "/home/lidaeus/TianShouDSSAT"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from maize_wrapper import MaizeEnvWrapper
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.algorithm.modelfree.c51 import C51Policy
from tianshou.algorithm.modelfree.rainbow import RainbowDQN
from tianshou.trainer import OffPolicyTrainer, OffPolicyTrainerParams
from tianshou.utils.net.common import Net
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter

DEFAULT_RUN_DSSAT_LOCATION = "/opt/dssat_env/inst/run_dssat"
DEFAULT_POLICY_PATH = os.path.join(PROJECT_ROOT, "policy.pth")


def make_maize_env(run_dssat_location=DEFAULT_RUN_DSSAT_LOCATION, history_len=14, forecast_len=7):
    return MaizeEnvWrapper(
        run_dssat_location=run_dssat_location,
        history_len=history_len,
        forecast_len=forecast_len,
    )


def build_env_factory(run_dssat_location=DEFAULT_RUN_DSSAT_LOCATION, history_len=14, forecast_len=7):
    def _factory():
        return make_maize_env(
            run_dssat_location=run_dssat_location,
            history_len=history_len,
            forecast_len=forecast_len,
        )

    return _factory


def collect_env_specs(run_dssat_location=DEFAULT_RUN_DSSAT_LOCATION, history_len=14, forecast_len=7):
    example_env = make_maize_env(
        run_dssat_location=run_dssat_location,
        history_len=history_len,
        forecast_len=forecast_len,
    )
    try:
        return example_env.observation_space.shape, example_env.action_space.n
    finally:
        example_env.close()


def train(
    run_dssat_location=DEFAULT_RUN_DSSAT_LOCATION,
    history_len=14,
    forecast_len=7,
    train_env_num=1,
    test_env_num=1,
    max_epochs=5,
    epoch_num_steps=200,
    test_step_num_episodes=2,
    collection_step_num_env_steps=10,
    batch_size=64,
    log_path=None,
    policy_path=DEFAULT_POLICY_PATH,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- 天授 2.0 训练 (Device: {device}) ---")

    log_path = log_path or os.path.join(PROJECT_ROOT, "logs", "rainbow_maize")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)
    train_envs = None
    test_envs = None

    try:
        train_factory = build_env_factory(
            run_dssat_location=run_dssat_location,
            history_len=history_len,
            forecast_len=forecast_len,
        )
        test_factory = build_env_factory(
            run_dssat_location=run_dssat_location,
            history_len=history_len,
            forecast_len=forecast_len,
        )
        train_envs = DummyVectorEnv([train_factory for _ in range(train_env_num)])
        test_envs = DummyVectorEnv([test_factory for _ in range(test_env_num)])

        state_shape, action_shape = collect_env_specs(
            run_dssat_location=run_dssat_location,
            history_len=history_len,
            forecast_len=forecast_len,
        )

        net = Net(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_sizes=[128, 128],
            num_atoms=51,
            dueling_param=({}, {}),
        )

        policy = C51Policy(
            model=net,
            action_space=gym.spaces.Discrete(action_shape),
            num_atoms=51,
            v_min=-10,
            v_max=10,
        ).to(device)

        optim_factory = AdamOptimizerFactory(lr=1e-4)
        algo = RainbowDQN(
            policy=policy,
            optim=optim_factory,
            gamma=0.99,
            n_step_return_horizon=3,
            target_update_freq=100,
        )

        train_collector = Collector(policy, train_envs, VectorReplayBuffer(2000, len(train_envs)))
        test_collector = Collector(policy, test_envs)

        params = OffPolicyTrainerParams(
            max_epochs=max_epochs,
            epoch_num_steps=epoch_num_steps,
            training_collector=train_collector,
            test_collector=test_collector,
            test_step_num_episodes=test_step_num_episodes,
            batch_size=batch_size,
            collection_step_num_env_steps=collection_step_num_env_steps,
            logger=logger,
        )

        print("--- 启动 OffPolicyTrainer ---")
        trainer = OffPolicyTrainer(algorithm=algo, params=params)
        result = trainer.run()

        print(f"\n训练结束: {result}")
        torch.save(policy.state_dict(), policy_path)
        return result
    finally:
        if train_envs is not None:
            train_envs.close()
        if test_envs is not None:
            test_envs.close()
        writer.close()

if __name__ == "__main__":
    train()
