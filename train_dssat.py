import os
import sys
import torch
import numpy as np
import gymnasium as gym
import argparse
import datetime

# 注入路径
PROJECT_ROOT = "/home/lidaeus/TianShouDSSAT"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dssat_generic_wrapper import DssatGenericWrapper
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.algorithm.modelfree.c51 import C51Policy
from tianshou.algorithm.modelfree.rainbow import RainbowDQN
from tianshou.trainer import OffPolicyTrainer, OffPolicyTrainerParams
from tianshou.utils.net.common import Net
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter

def get_args():
    parser = argparse.ArgumentParser(description="Train Rainbow DQN on DSSAT")
    parser.add_argument("--crop", type=str, default="maize", choices=["maize", "tomato"], help="Crop to train on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden size for network")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory to save logs")
    return parser.parse_args()

def make_env_factory(crop_name, seed):
    """
    Returns a function that creates the environment.
    This is needed for DummyVectorEnv to pickle correctly if we used lambda.
    """
    def _make():
        return DssatGenericWrapper(
            crop_name=crop_name, 
            seed=seed,
            history_len=5
        )
    return _make

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"rainbow_{args.crop}_{timestamp}"
    log_path = os.path.join(PROJECT_ROOT, args.log_dir, log_name)
    
    print(f"--- 启动训练: {args.crop.upper()} (Device: {device}) ---")
    print(f"Log path: {log_path}")

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Environments
    env_factory = make_env_factory(args.crop, args.seed)
    train_envs = DummyVectorEnv([env_factory for _ in range(1)])
    test_envs = DummyVectorEnv([env_factory for _ in range(1)])
    
    # Get shape from a dummy instance
    example_env = env_factory()
    state_shape = example_env.observation_space.shape
    action_shape = example_env.action_space.n
    print(f"Obs Shape: {state_shape}, Action Space: Discrete({action_shape})")
    example_env.close()
    
    # 1. 网络
    net = Net(state_shape=state_shape, 
              action_shape=action_shape,
              hidden_sizes=[args.hidden_size, args.hidden_size],
              num_atoms=51,
              dueling_param=({}, {})).to(device)
    
    # 2. 策略
    policy = C51Policy(
        model=net,
        action_space=gym.spaces.Discrete(action_shape),
        num_atoms=51,
        v_min=-10,
        v_max=10
    ).to(device)

    # 3. 算法
    optim_factory = AdamOptimizerFactory(lr=1e-4)
    algo = RainbowDQN(
        policy=policy,
        optim=optim_factory,
        gamma=0.99,
        n_step_return_horizon=3,
        target_update_freq=100
    )

    # 4. 收集器
    # Buffer size: 2000 is small, good for testing. Production usually 10k-100k.
    train_collector = Collector(policy, train_envs, VectorReplayBuffer(10000, len(train_envs)))
    test_collector = Collector(policy, test_envs)

    # 5. Logger
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    # 6. 训练参数
    params = OffPolicyTrainerParams(
        max_epochs=args.epochs,
        epoch_num_steps=500, # Steps per epoch
        training_collector=train_collector,
        test_collector=test_collector,
        test_step_num_episodes=1, # Test 1 episode per epoch
        batch_size=args.batch_size,
        collection_step_num_env_steps=10,
        logger=logger,
        save_best_fn=lambda policy: torch.save(policy.state_dict(), os.path.join(log_path, "policy_best.pth"))
    )

    # 7. 启动
    trainer = OffPolicyTrainer(algorithm=algo, params=params)
    result = trainer.run()
    
    print(f"\n训练结束: {result}")
    torch.save(policy.state_dict(), os.path.join(log_path, "policy_final.pth"))

if __name__ == "__main__":
    args = get_args()
    train(args)
