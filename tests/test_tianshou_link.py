import os
import sys
import signal
import numpy as np
import pytest
import gymnasium as gym

# 1. 强制超时设置 (60秒)
def timeout_handler(signum, frame):
    print("\n[TIMEOUT] 脚本执行超过 60 秒，强制退出以防阻塞！")
    sys.exit(1)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(60)

# 2. 注入路径
PROJECT_ROOT = "/home/lidaeus/TianShouDSSAT"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from maize_wrapper import MaizeEnvWrapper
import experiments.build_policy as build_policy_module
import train_rainbow


class DummyWrapperEnv(gym.Env):
    metadata = {}

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)
        self.closed = False

    def reset(self, **kwargs):
        return np.zeros(6, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(6, dtype=np.float32), 0.0, True, False, {}

    def close(self):
        self.closed = True


class DummyVectorEnv:
    def __init__(self, factories):
        self.factories = factories
        self.closed = False

    def __len__(self):
        return len(self.factories)

    def close(self):
        self.closed = True


class DummyPolicy:
    def __init__(self, model, action_space, num_atoms, v_min, v_max):
        self.model = model
        self.action_space = action_space
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def state_dict(self):
        return {"weights": 1}


class DummyTrainer:
    def __init__(self, algorithm, params):
        self.algorithm = algorithm
        self.params = params

    def run(self):
        return {"status": "ok"}


class DummyNetModule:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.device = None

    def to(self, device):
        self.device = device
        return self


class DummyWriter:
    def __init__(self, path):
        self.path = path
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def patch_train_rainbow(monkeypatch):
    created_envs = []
    created_vector_envs = []
    saved_models = []
    created_writers = []

    def fake_wrapper(**kwargs):
        env = DummyWrapperEnv(**kwargs)
        created_envs.append(env)
        return env

    def fake_vector_env(factories):
        env = DummyVectorEnv(factories)
        created_vector_envs.append(env)
        return env

    def fake_save(state_dict, path):
        saved_models.append((state_dict, path))

    def fake_writer(path):
        writer = DummyWriter(path)
        created_writers.append(writer)
        return writer

    monkeypatch.setattr(train_rainbow, "MaizeEnvWrapper", fake_wrapper)
    monkeypatch.setattr(train_rainbow, "DummyVectorEnv", fake_vector_env)
    monkeypatch.setattr(train_rainbow, "Net", lambda **kwargs: {"net_kwargs": kwargs})
    monkeypatch.setattr(train_rainbow, "C51Policy", DummyPolicy)
    monkeypatch.setattr(train_rainbow, "AdamOptimizerFactory", lambda **kwargs: {"optim_kwargs": kwargs})
    monkeypatch.setattr(train_rainbow, "RainbowDQN", lambda **kwargs: {"algo_kwargs": kwargs})
    monkeypatch.setattr(train_rainbow, "Collector", lambda *args: {"collector_args": args})
    monkeypatch.setattr(train_rainbow, "VectorReplayBuffer", lambda size, env_num: {"size": size, "env_num": env_num})
    monkeypatch.setattr(train_rainbow, "OffPolicyTrainerParams", lambda **kwargs: {"params_kwargs": kwargs})
    monkeypatch.setattr(train_rainbow, "OffPolicyTrainer", DummyTrainer)
    monkeypatch.setattr(train_rainbow, "TensorboardLogger", lambda writer: {"writer": writer})
    monkeypatch.setattr(train_rainbow, "SummaryWriter", fake_writer)
    monkeypatch.setattr(train_rainbow.torch, "save", fake_save)
    monkeypatch.setattr(train_rainbow.torch.cuda, "is_available", lambda: False)
    return {
        "created_envs": created_envs,
        "created_vector_envs": created_vector_envs,
        "saved_models": saved_models,
        "created_writers": created_writers,
    }


def test_make_maize_env_forwards_runtime_configuration(patch_train_rainbow):
    env = train_rainbow.make_maize_env(
        run_dssat_location="/tmp/run_dssat",
        history_len=5,
        forecast_len=3,
    )

    assert env.kwargs["run_dssat_location"] == "/tmp/run_dssat"
    assert env.kwargs["history_len"] == 5
    assert env.kwargs["forecast_len"] == 3


def test_collect_env_specs_closes_example_env(patch_train_rainbow):
    state_shape, action_shape = train_rainbow.collect_env_specs(
        run_dssat_location="/tmp/run_dssat",
        history_len=2,
        forecast_len=4,
    )

    assert state_shape == (6,)
    assert action_shape == 4
    assert patch_train_rainbow["created_envs"][-1].closed is True


def test_train_uses_configured_factories_and_closes_resources(tmp_path, patch_train_rainbow):
    result = train_rainbow.train(
        run_dssat_location="/tmp/run_dssat",
        history_len=4,
        forecast_len=2,
        train_env_num=2,
        test_env_num=3,
        max_epochs=1,
        epoch_num_steps=5,
        test_step_num_episodes=1,
        collection_step_num_env_steps=2,
        batch_size=8,
        log_path=str(tmp_path / "logs"),
        policy_path=str(tmp_path / "policy.pth"),
    )

    assert result == {"status": "ok"}
    assert len(patch_train_rainbow["created_vector_envs"]) == 2
    assert len(patch_train_rainbow["created_vector_envs"][0].factories) == 2
    assert len(patch_train_rainbow["created_vector_envs"][1].factories) == 3
    assert all(env.closed for env in patch_train_rainbow["created_vector_envs"])
    assert patch_train_rainbow["created_writers"][0].closed is True
    assert patch_train_rainbow["saved_models"] == [({"weights": 1}, str(tmp_path / "policy.pth"))]


def test_build_policy_normalizes_shapes_and_hidden_size(monkeypatch):
    captured = {}

    def fake_net(**kwargs):
        net = DummyNetModule(**kwargs)
        captured["net"] = net
        return net

    monkeypatch.setattr(build_policy_module, "Net", fake_net)
    monkeypatch.setattr(build_policy_module, "C51Policy", DummyPolicy)
    monkeypatch.setattr(build_policy_module, "AdamOptimizerFactory", lambda **kwargs: {"optim_kwargs": kwargs})
    monkeypatch.setattr(build_policy_module, "RainbowDQN", lambda **kwargs: {"algo_kwargs": kwargs})

    algorithm, policy = build_policy_module.build_rainbow_components(
        state_shape=[6],
        action_shape=(2, 3),
        hidden_size=0,
        device="cpu",
    )

    assert captured["net"].kwargs["state_shape"] == (6,)
    assert captured["net"].kwargs["action_shape"] == 6
    assert captured["net"].kwargs["hidden_sizes"] == [1, 1]
    assert captured["net"].device == "cpu"
    assert policy.action_space.n == 6
    assert policy.device == "cpu"
    assert algorithm["algo_kwargs"]["policy"] is policy


def test_build_policy_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="state_shape"):
        build_policy_module.build_rainbow_components(
            state_shape=(0,),
            action_shape=4,
            hidden_size=128,
            device="cpu",
        )

    with pytest.raises(ValueError, match="action_shape"):
        build_policy_module.build_rainbow_components(
            state_shape=(6,),
            action_shape=0,
            hidden_size=128,
            device="cpu",
        )

def test_single_env():
    print("--- 启动单环境天授链路测试 (带 60s 强退) ---")
    env = None
    try:
        env = MaizeEnvWrapper(run_dssat_location='/opt/dssat_env/inst/run_dssat')
        
        print("开始首次 Reset...")
        obs, info = env.reset()
        print(f"首次 Reset 成功, Obs shape: {obs.shape}")
        
        for i in range(3):
            action = env.action_space.sample()
            obs, rew, done, trunc, info = env.step(action)
            print(f"Step {i}: Action={action}, Reward={rew}, Done={done}")
            if done:
                break
    except Exception as e:
        print(f"测试过程中发生异常: {e}")
    finally:
        if env:
            env.close()
        print("--- 单环境测试清理完成 ---")

if __name__ == "__main__":
    test_single_env()
    # 正常结束则关闭闹钟
    signal.alarm(0)
