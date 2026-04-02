import os
import sys
import numpy as np
import pytest
import gymnasium as gym

PROJECT_ROOT = "/home/lidaeus/TianShouDSSAT"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import maize_wrapper
from maize_wrapper import MaizeEnvWrapper


class DummyLegacyVendorEnv(gym.Env):
    metadata = {}

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def reset(self, **kwargs):
        return {"nstres": 1.0, "swfac": 0.5, "xlai": 2.0, "dap": 10}

    def step(self, action):
        return {"nstres": 0.8, "swfac": 0.4, "xlai": 2.5}, [1.0, 2.0], True, {"source": "legacy"}

    def close(self):
        return None


class DummyGymnasiumVendorEnv(gym.Env):
    metadata = {}

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def reset(self, **kwargs):
        return {"nstres": 0.2, "swfac": 0.9, "xlai": 1.5}, {"source": "gymnasium", "seed": kwargs.get("seed")}

    def step(self, action):
        return {"nstres": 0.1, "swfac": 0.7, "xlai": 2.1}, 4.5, False, True, {"source": "gymnasium"}

    def close(self):
        return None


@pytest.fixture
def patch_vendor_env(monkeypatch):
    def _apply(vendor_env_cls):
        monkeypatch.setattr(maize_wrapper, "DssatPdi", vendor_env_cls)
        return vendor_env_cls

    return _apply


def test_wrapper_normalizes_legacy_reset_and_step_results(patch_vendor_env):
    patch_vendor_env(DummyLegacyVendorEnv)
    env = MaizeEnvWrapper(run_dssat_location="/tmp/run_dssat", history_len=2, forecast_len=1)
    obs, info = env.reset(seed=5)
    next_obs, reward, terminated, truncated, step_info = env.step(3)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert info == {}
    assert isinstance(next_obs, np.ndarray)
    assert reward == 3.0
    assert terminated is True
    assert truncated is False
    assert step_info["applied_action"] == {"anfer": 60.0}
    assert step_info["action_index"] == 3
    assert step_info["native_reward"] == 3.0
    env.close()


def test_wrapper_preserves_gymnasium_metadata_and_truncation(patch_vendor_env):
    patch_vendor_env(DummyGymnasiumVendorEnv)
    env = MaizeEnvWrapper(run_dssat_location="/tmp/run_dssat", history_len=1, forecast_len=2)
    obs, info = env.reset(seed=11, options={"season": "test"})
    next_obs, reward, terminated, truncated, step_info = env.step(1)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert info["source"] == "gymnasium"
    assert info["seed"] == 11
    assert isinstance(next_obs, np.ndarray)
    assert reward == 4.5
    assert terminated is False
    assert truncated is True
    assert step_info["source"] == "gymnasium"
    assert step_info["applied_action"] == {"anfer": 20.0}
    env.close()

def test_wrapper():
    print("--- 开始测试 MaizeEnvWrapper ---")
    run_dssat = "/opt/dssat_env/inst/run_dssat"
    
    # 清理残留进程
    os.system("pkill -9 dscsm048")
    
    try:
        env = MaizeEnvWrapper(run_dssat_location=run_dssat)
        obs, info = env.reset()
        print(f"Reset 成功，Obs shape: {obs.shape}")
        
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            print(f"Step {i+1} 成功，Reward: {reward}, Done: {done}")
            if done:
                break
        
        env.close()
        print("测试完成，环境已关闭。")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_maize_wrapper = test_wrapper()
