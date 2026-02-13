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
    parser = argparse.ArgumentParser(description="在 DSSAT 环境下训练 Rainbow DQN")
    parser.add_argument("--crop", type=str, default="maize", choices=["maize", "tomato"], help="训练作物类型")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="批次大小")
    parser.add_argument("--hidden-size", type=int, default=128, help="网络隐藏层大小")
    parser.add_argument("--log-dir", type=str, default="logs", help="日志保存目录")
    return parser.parse_args()

def make_env_factory(crop_name, seed):
    """
    返回环境创建函数。
    用于 DummyVectorEnv 以确保正确的序列化。
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
    
    print(f"--- 启动训练: {args.crop.upper()} (设备: {device}) ---")
    print(f"日志路径: {log_path}")

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 环境初始化
    env_factory = make_env_factory(args.crop, args.seed)
    train_envs = DummyVectorEnv([env_factory for _ in range(1)])
    test_envs = DummyVectorEnv([env_factory for _ in range(1)])
    
    # 从虚拟实例获取观察和动作空间维度
    example_env = env_factory()
    state_shape = example_env.observation_space.shape
    action_shape = example_env.action_space.n
    print(f"观察空间维度: {state_shape}, 动作空间: Discrete({action_shape})")
    example_env.close()
    
    # 1. 构建网络
    net = Net(state_shape=state_shape, 
              action_shape=action_shape,
              hidden_sizes=[args.hidden_size, args.hidden_size],
              num_atoms=51,
              dueling_param=({}, {})).to(device)
    
    # 2. 定义策略
    policy = C51Policy(
        model=net,
        action_space=gym.spaces.Discrete(action_shape),
        num_atoms=51,
        v_min=-10,
        v_max=10
    ).to(device)

    # 3. 配置算法
    optim_factory = AdamOptimizerFactory(lr=1e-4)
    algo = RainbowDQN(
        policy=policy,
        optim=optim_factory,
        gamma=0.99,
        n_step_return_horizon=3,
        target_update_freq=100
    )

    # 4. 收集器
    train_collector = Collector(policy, train_envs, VectorReplayBuffer(10000, len(train_envs)))
    test_collector = Collector(policy, test_envs)

    # 5. 日志记录器
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    # 6. 训练参数设置
    params = OffPolicyTrainerParams(
        max_epochs=args.epochs,
        epoch_num_steps=500, # 每轮步数
        training_collector=train_collector,
        test_collector=test_collector,
        test_step_num_episodes=1, # 每轮测试 1 个回合
        batch_size=args.batch_size,
        collection_step_num_env_steps=10,
        logger=logger,
        save_best_fn=lambda policy: torch.save(policy.state_dict(), os.path.join(log_path, "policy_best.pth"))
    )

    # 7. 启动训练器
    trainer = OffPolicyTrainer(algorithm=algo, params=params)
    result = trainer.run()
    
    print(f"\n训练结束: {result}")
    torch.save(policy.state_dict(), os.path.join(log_path, "policy_final.pth"))

if __name__ == "__main__":
    args = get_args()
    train(args)
