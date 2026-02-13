import os
import sys
import torch
import numpy as np
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

def make_maize_env():
    return MaizeEnvWrapper(run_dssat_location='/opt/dssat_env/inst/run_dssat')

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- 天授 2.0 训练 (Device: {device}) ---")
    
    log_path = os.path.join(PROJECT_ROOT, "logs", "rainbow_maize")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    train_envs = DummyVectorEnv([make_maize_env for _ in range(1)])
    test_envs = DummyVectorEnv([make_maize_env for _ in range(1)])
    
    example_env = make_maize_env()
    state_shape = example_env.observation_space.shape
    action_shape = example_env.action_space.n
    example_env.close()
    
    # 1. 网络
    net = Net(state_shape=state_shape, 
              action_shape=action_shape,
              hidden_sizes=[128, 128],
              num_atoms=51,
              dueling_param=({}, {}))
    
    # 2. 策略
    policy = C51Policy(
        model=net,
        action_space=gym.spaces.Discrete(action_shape),
        num_atoms=51,
        v_min=-10,
        v_max=10
    ).to(device)

    # 3. 算法 (使用 Factory)
    optim_factory = AdamOptimizerFactory(lr=1e-4)
    algo = RainbowDQN(
        policy=policy,
        optim=optim_factory,
        gamma=0.99,
        n_step_return_horizon=3,
        target_update_freq=100
    )

    # 4. 收集器
    train_collector = Collector(policy, train_envs, VectorReplayBuffer(2000, len(train_envs)))
    test_collector = Collector(policy, test_envs)

    # 5. 训练参数对象
    params = OffPolicyTrainerParams(
        max_epochs=5,
        epoch_num_steps=200,
        training_collector=train_collector,
        test_collector=test_collector,
        test_step_num_episodes=2,
        batch_size=64,
        collection_step_num_env_steps=10,
        logger=logger
    )

    # 6. 训练器
    print("--- 启动 OffPolicyTrainer ---")
    trainer = OffPolicyTrainer(algorithm=algo, params=params)
    result = trainer.run()
    
    print(f"\n训练结束: {result}")
    torch.save(policy.state_dict(), "policy.pth")

if __name__ == "__main__":
    train()
