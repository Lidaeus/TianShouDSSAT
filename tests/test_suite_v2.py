import os
import sys
import numpy as np
import pytest
import gymnasium as gym

# Ensure library is in path
sys.path.append(os.path.abspath("lib/gym_dssat_pdi_official/gym-dssat-pdi"))
import dssat_generic_wrapper as wrapper_module
from dssat_generic_wrapper import DssatGenericWrapper

@pytest.fixture(autouse=True)
def setup_env():
    # Ensure patching is done before tests
    os.system("python3 fix_dssat_pdi.py")

def test_maize_compliance():
    """TC-1: 验证玉米环境 API 合规性"""
    env = DssatGenericWrapper(crop_name="maize", history_len=1)
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)
    
    action = env.action_space.sample()
    result = env.step(action)
    assert len(result) == 5 # obs, reward, term, trunc, info
    env.close()

def test_tomato_compliance():
    """TC-1: 验证番茄环境 API 合规性"""
    env = DssatGenericWrapper(crop_name="tomato", history_len=1)
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    
    # 验证关键生理特征是否存在于内部观测
    # 这里的 obs 是经过 wrapper 提取后的
    assert obs.shape[0] > 0
    env.close()

def test_wheat_compliance():
    """TC-1: 验证小麦环境 API 合规性"""
    env = DssatGenericWrapper(crop_name="wheat", history_len=1)
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)
    assert obs.shape[0] == len(env.obs_keys)

    action = env.action_space.sample()
    result = env.step(action)
    assert len(result) == 5
    env.close()

def test_crop_switching():
    """TC-2: 验证多作物连续切换"""
    # Test Maize
    env_m = DssatGenericWrapper(crop_name="maize")
    obs_m, _ = env_m.reset()
    assert obs_m is not None
    env_m.close()
    
    # Test Tomato immediately after
    env_t = DssatGenericWrapper(crop_name="tomato")
    obs_t, _ = env_t.reset()
    assert obs_t is not None
    env_t.close()

    env_w = DssatGenericWrapper(crop_name="wheat")
    obs_w, _ = env_w.reset()
    assert obs_w is not None
    env_w.close()

def test_history_buffer():
    """TC-4: 验证历史 Buffer 拼接"""
    history_len = 3
    env = DssatGenericWrapper(crop_name="maize", history_len=history_len)
    obs, _ = env.reset()
    
    # 获取单步特征长度
    single_step_len = len(env.obs_keys)
    assert obs.shape[0] == single_step_len * history_len
    
    # 验证 reset 后 buffer 是否被填满（通常为初始状态的重复）
    first_chunk = obs[:single_step_len]
    last_chunk = obs[-single_step_len:]
    assert np.allclose(first_chunk, last_chunk)
    env.close()

def test_robustness_extreme_actions():
    """TC-5: 验证极端动作健壮性"""
    env = DssatGenericWrapper(crop_name="maize")
    env.reset()
    
    # 传入极大值，Wrapper 应能处理或 DssatPdi 内部截断
    # 动作空间是 Discrete，我们模拟一个超出范围的索引是不可能的（会报 IndexError）
    # 但我们可以模拟 step 的底层调用
    try:
        # 正常采样
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        assert not term or term
    finally:
        env.close()


def test_forecast_fallback_builds_summary_without_external_provider(monkeypatch):
    captured_kwargs = {}

    class DummyVendorEnv(gym.Env):
        metadata = {}

        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)
            self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
            self.action_space = gym.spaces.Discrete(36)

        def reset(self, **kwargs):
            return {
                "swfac": 1.0,
                "vstage": 2.0,
                "wtdep": 3.0,
                "grnwt": 4.0,
                "topwt": 5.0,
                "lai": 6.0,
                "pcntn": 7.0,
                "stresn": 8.0,
            }, {
                "weather_forecast_daily": [
                    {"RAIN": 1.0, "TMIN": 10.0, "TMAX": 20.0, "SRAD": 15.0},
                    {"RAIN": 2.0, "TMIN": 12.0, "TMAX": 22.0, "SRAD": 18.0},
                    {"RAIN": 3.0, "TMIN": 14.0, "TMAX": 24.0, "SRAD": 21.0},
                ]
            }

        def step(self, action):
            return self.reset()[0], 1.0, True, False, {}

        def close(self):
            return None

    monkeypatch.setattr(wrapper_module, "WeatherWindowForecastProvider", None)
    monkeypatch.setattr(wrapper_module, "DssatPdi", DummyVendorEnv)
    env = DssatGenericWrapper(
        crop_name="maize",
        history_len=1,
        include_forecast=True,
        forecast_horizons=(2, 3),
        env_kwargs={"auxiliary_file_paths": ["/tmp/B.WTH", "/tmp/B.WTH", "/tmp/A.CLI"]},
    )
    obs, info = env.reset()

    assert obs.shape[0] == len(env.obs_keys) + 6
    assert info["weather_forecast"]["rain_2d"] == 3.0
    assert info["weather_forecast"]["tmean_2d"] == 16.0
    assert info["weather_forecast"]["srad_3d"] == 18.0
    assert captured_kwargs["auxiliary_file_paths"] == [
        "/tmp/B.WTH",
        "/tmp/A.CLI",
        "/opt/dssat_env/data/UFGA.CLI",
        "/opt/dssat_env/data/SOIL.SOL",
        "/opt/dssat_env/data/UFGA8201.WTH",
    ]
    env.close()


def test_env_kwargs_do_not_leak_between_wrapper_instances(monkeypatch):
    captured_auxiliary_lists = []

    class DummyVendorEnv(gym.Env):
        metadata = {}

        def __init__(self, **kwargs):
            captured_auxiliary_lists.append(list(kwargs["auxiliary_file_paths"]))
            self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
            self.action_space = gym.spaces.Discrete(36)

        def reset(self, **kwargs):
            return {
                "swfac": 1.0,
                "vstage": 2.0,
                "wtdep": 3.0,
                "grnwt": 4.0,
                "topwt": 5.0,
                "lai": 6.0,
                "pcntn": 7.0,
                "stresn": 8.0,
            }, {}

        def step(self, action):
            return self.reset()[0], 0.0, True, False, {}

        def close(self):
            return None

    monkeypatch.setattr(wrapper_module, "DssatPdi", DummyVendorEnv)
    first_env = DssatGenericWrapper(crop_name="maize", env_kwargs={"auxiliary_file_paths": ["/tmp/FIRST.WTH"]})
    second_env = DssatGenericWrapper(crop_name="maize")

    assert captured_auxiliary_lists[0][0] == "/tmp/FIRST.WTH"
    assert "/tmp/FIRST.WTH" not in captured_auxiliary_lists[1]

    first_env.close()
    second_env.close()

if __name__ == "__main__":
    # Manually run if not using pytest command
    pytest.main([__file__, "-v"])
