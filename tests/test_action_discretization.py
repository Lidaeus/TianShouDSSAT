import os
import sys

import gymnasium as gym
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath("lib/gym_dssat_pdi_official/gym-dssat-pdi"))

import dssat_generic_wrapper as wrapper_module
from dssat_generic_wrapper import DssatGenericWrapper
from experiments.build_evaluator import (
    EpisodeEvaluator,
    EvaluationResult,
    MultiSeedEvaluationResult,
    build_actor_leaderboard,
    serialize_multiseed_evaluation_suite,
)
from tasks.forecast import WeatherWindowForecastProvider
from tasks.schemas import (
    AgronomicObservationSchema,
    TaskEnv,
    WeeklyDiscreteActionSchema,
    YieldCostRewardSchema,
)

@pytest.fixture(autouse=True)
def setup_path():
    sys.path.insert(0, os.path.abspath("lib/gym_dssat_pdi_official/gym-dssat-pdi"))

def test_action_space_is_discrete():
    """Verify that the wrapper exposes a Discrete action space."""
    env = DssatGenericWrapper(crop_name="maize")
    try:
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 36
    finally:
        env.close()

def test_step_accepts_integer():
    """Verify that step() accepts an integer action."""
    env = DssatGenericWrapper(crop_name="maize")
    env.reset()
    try:
        # Action 0 -> Anfer=0, Amir=0
        obs, reward, term, trunc, info = env.step(0)
        assert obs is not None
        assert not term, "Environment terminated immediately on first step!"
        
        # Action 35 -> Max Anfer, Max Amir
        obs, reward, term, trunc, info = env.step(35)
        assert obs is not None
    finally:
        env.close()

def test_map_action_logic():
    """Verify the mapping logic directly (if method is exposed or via subclassing)."""
    env = DssatGenericWrapper(crop_name="maize")
    try:
        # Check if internal helper exists and works
        if hasattr(env, "_map_action"):
            # 0 -> 0, 0
            act_dict = env._map_action(0)
            assert act_dict['anfer'] == 0
            assert act_dict['amir'] == 0
            
            # 6 -> Anfer index 1 (40), Amir index 0 (0) OR vice versa depending on implementation
            # Let's assume Anfer is outer loop (rows), Amir is inner (cols)
            # idx = anfer_idx * len(amir) + amir_idx
            # 6 = 1 * 6 + 0 -> Anfer=40, Amir=0
            
            # 35 -> Anfer index 5 (200), Amir index 5 (50)
            act_dict_max = env._map_action(35)
            # Check values roughly
            assert act_dict_max['anfer'] > 150
            assert act_dict_max['amir'] > 40
    finally:
        env.close()


def test_forecast_summary_logic():
    daily_weather = [
        {"RAIN": 0.0, "TMIN": 10.0, "TMAX": 20.0, "SRAD": 18.0},
        {"RAIN": 1.5, "TMIN": 8.0, "TMAX": 18.0, "SRAD": 16.0},
        {"RAIN": 2.0, "TMIN": 12.0, "TMAX": 24.0, "SRAD": 20.0},
        {"RAIN": 3.0, "TMIN": 11.0, "TMAX": 19.0, "SRAD": 14.0},
    ]
    summary = DssatGenericWrapper.summarize_forecast(daily_weather, horizons=(3, 7))
    assert summary["rain_3d"] == pytest.approx(3.5)
    assert summary["tmean_3d"] == pytest.approx((15.0 + 13.0 + 18.0) / 3.0)
    assert summary["srad_3d"] == pytest.approx((18.0 + 16.0 + 20.0) / 3.0)
    assert summary["rain_7d"] == pytest.approx(6.5)
    assert summary["tmean_7d"] == pytest.approx((15.0 + 13.0 + 18.0 + 15.0) / 4.0)


def test_forecast_provider_filters_history_rows_from_daily_forecast():
    provider = WeatherWindowForecastProvider(horizons=(3, 7))
    metadata = provider.enrich_metadata(
        raw_state={"yrdoy": 2024002},
        info={
            "daily_forecast": [
                {"DATE": 2024001, "RAIN": 9.0, "TMIN": 9.0, "TMAX": 19.0, "SRAD": 10.0},
                {"DATE": 2024002, "RAIN": 8.0, "TMIN": 10.0, "TMAX": 20.0, "SRAD": 11.0},
                {"DATE": 2024003, "RAIN": 1.0, "TMIN": 11.0, "TMAX": 21.0, "SRAD": 12.0},
                {"DATE": 2024004, "RAIN": 2.0, "TMIN": 12.0, "TMAX": 22.0, "SRAD": 13.0},
            ],
        },
    )
    assert metadata["weather_forecast"]["rain_3d"] == pytest.approx(3.0)
    assert metadata["weather_forecast"]["tmean_3d"] == pytest.approx((16.0 + 17.0) / 2.0)
    assert metadata["weather_forecast_available_days"] == 2
    assert metadata["weather_forecast_source"] == "daily_forecast"
    assert len(metadata["weather_forecast_raw"]) == 2


def test_forecast_provider_accepts_iso_dates():
    provider = WeatherWindowForecastProvider(horizons=(2,))
    metadata = provider.enrich_metadata(
        raw_state={"yrdoy": 2024001},
        info={
            "weather_forecast_daily": [
                {"date": "2024-01-01", "rain": 6.0, "tmin": 8.0, "tmax": 16.0, "srad": 9.0},
                {"date": "2024-01-02", "rain": 1.5, "tmin": 9.0, "tmax": 19.0, "srad": 12.0},
                {"date": "2024-01-03", "rain": 2.5, "tmin": 10.0, "tmax": 20.0, "srad": 15.0},
            ],
        },
    )
    assert metadata["weather_forecast"]["rain_2d"] == pytest.approx(4.0)
    assert metadata["weather_forecast"]["tmean_2d"] == pytest.approx((14.0 + 15.0) / 2.0)
    assert metadata["weather_forecast"]["srad_2d"] == pytest.approx((12.0 + 15.0) / 2.0)


def test_step_standardizes_info_with_action_weather_and_native_reward(monkeypatch):
    class DummyVendorEnv(gym.Env):
        metadata = {}

        def __init__(self, **kwargs):
            self.kwargs = kwargs
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
            return {
                "swfac": 0.8,
                "vstage": 3.0,
                "wtdep": 4.0,
                "grnwt": 6.0,
                "topwt": 7.0,
                "lai": 8.0,
                "pcntn": 9.0,
                "stresn": 10.0,
            }, [1.25, 2.75], False, True, {"vendor_flag": True}

        def close(self):
            return None

    monkeypatch.setattr(wrapper_module, "DssatPdi", DummyVendorEnv)
    env = DssatGenericWrapper(
        crop_name="maize",
        history_len=1,
        env_kwargs={"auxiliary_file_paths": ["/tmp/PRIMARY.WTH", "/tmp/SECONDARY.WTH", "/tmp/SOIL.SOL"]},
    )
    env.reset()
    observation, reward, terminated, truncated, info = env.step(7)

    assert isinstance(observation, np.ndarray)
    assert reward == pytest.approx(4.0)
    assert terminated is False
    assert truncated is True
    assert info["vendor_flag"] is True
    assert info["applied_action"] == {"anfer": 40.0, "amir": 10.0}
    assert info["action_index"] == 7
    assert info["primary_weather_path"] == "/tmp/PRIMARY.WTH"
    assert info["weather_file_paths"][:2] == ["/tmp/PRIMARY.WTH", "/tmp/SECONDARY.WTH"]
    assert info["native_reward"] == pytest.approx(4.0)
    env.close()


def test_step_normalizes_none_reward_and_empty_info(monkeypatch):
    class DummyVendorEnv(gym.Env):
        metadata = {}

        def __init__(self, **kwargs):
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
            return {
                "swfac": 0.8,
                "vstage": 3.0,
                "wtdep": 4.0,
                "grnwt": 6.0,
                "topwt": 7.0,
                "lai": 8.0,
                "pcntn": 9.0,
                "stresn": 10.0,
            }, None, True, False, None

        def close(self):
            return None

    monkeypatch.setattr(wrapper_module, "DssatPdi", DummyVendorEnv)
    env = DssatGenericWrapper(
        crop_name="maize",
        history_len=1,
        env_kwargs={"auxiliary_file_paths": ["/tmp/ONLY.WTH"]},
    )
    env.reset()
    _, reward, terminated, truncated, info = env.step(35)

    assert reward == 0.0
    assert terminated is True
    assert truncated is False
    assert info["applied_action"] == {"anfer": 200.0, "amir": 50.0}
    assert info["action_index"] == 35
    assert info["primary_weather_path"] == "/tmp/ONLY.WTH"
    assert info["native_reward"] == 0.0
    env.close()


def test_observation_schema_describe_exposes_feature_groups():
    schema = AgronomicObservationSchema(crop_name="wheat", include_forecast=True, forecast_horizons=(3, 7))
    description = schema.describe()
    assert description["type"] == "agronomic"
    assert description["crop_name"] == "wheat"
    assert description["include_forecast"] is True
    assert description["forecast_horizons"] == (3, 7)
    assert "crop_state" in description["feature_groups"]
    assert "forecast_weather" in description["feature_groups"]
    assert "forecast_rain_3d_norm" in description["feature_groups"]["forecast_weather"]
    assert description["feature_names"][0] == "istage_norm"


def test_task_env_exposes_observation_and_reward_schema_metadata():
    class DummyBaseEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            self.action_space = gym.spaces.Dict(
                {
                    "anfer": gym.spaces.Box(low=0.0, high=100.0, shape=(), dtype=np.float32),
                    "amir": gym.spaces.Box(low=0.0, high=100.0, shape=(), dtype=np.float32),
                }
            )
            self.observation_space = gym.spaces.Dict({})
            self._step = 0

        def reset(self, *, seed=None, options=None):
            self._step = 0
            return {
                "istage": 1,
                "vstage": 2,
                "xlai": 1.0,
                "grnwt": 1000.0,
                "topwt": 2000.0,
                "rtdep": 40.0,
                "swfac": 0.8,
                "nstres": 0.2,
                "wtdep": 150.0,
                "trnu": 0.8,
                "tleachd": 0.1,
                "tnoxd": 0.05,
                "rain": 3.0,
                "tmin": 10.0,
                "tmax": 20.0,
                "srad": 15.0,
                "sw": [0.22, 0.24],
                "ll": [0.12, 0.13],
                "dul": [0.30, 0.31],
                "dlayr": [15.0, 20.0],
            }, {"crop_name": "wheat", "context": {}}

        def step(self, action):
            self._step += 1
            return {
                "istage": 2,
                "vstage": 3,
                "xlai": 1.5,
                "grnwt": 1200.0,
                "topwt": 2300.0,
                "rtdep": 50.0,
                "swfac": 0.7,
                "nstres": 0.1,
                "wtdep": 160.0,
                "trnu": 1.0,
                "tleachd": 0.2,
                "tnoxd": 0.08,
                "rain": 2.0,
                "tmin": 11.0,
                "tmax": 21.0,
                "srad": 16.0,
                "sw": [0.21, 0.23],
                "ll": [0.12, 0.13],
                "dul": [0.30, 0.31],
                "dlayr": [15.0, 20.0],
            }, 0.5, True, False, {"crop_name": "wheat", "context": {}}

        def close(self):
            return None

    env = TaskEnv(
        base_env=DummyBaseEnv(),
        observation_schema=AgronomicObservationSchema(crop_name="wheat", include_forecast=False),
        action_schema=WeeklyDiscreteActionSchema(),
        reward_schema=YieldCostRewardSchema(),
    )
    observation, info = env.reset()
    assert isinstance(observation, np.ndarray)
    assert info["observation_schema"]["type"] == "agronomic"
    assert info["reward_schema"]["type"] == "yield_cost"
    assert info["action_schema"]["type"] == "weekly_discrete"
    observation, reward, terminated, truncated, info = env.step(0)
    assert isinstance(observation, np.ndarray)
    assert isinstance(reward, float)
    assert terminated is True
    assert truncated is False
    assert info["observation_schema"]["feature_groups"]["management_memory"][-1] == "remaining_irrigation_ratio"
    assert info["reward_schema"]["fertilizer_cost_weight"] == pytest.approx(0.02)
    env.close()


def test_episode_evaluator_deduplicates_duplicate_seed_runs():
    class CountingEnv(gym.Env):
        metadata = {"render_modes": []}
        created_seeds = []

        def __init__(self):
            self.action_space = gym.spaces.Discrete(1)
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            self._seed_used = None

        def reset(self, *, seed=None, options=None):
            self._seed_used = seed
            CountingEnv.created_seeds.append(seed)
            return np.array([0.0], dtype=np.float32), {}

        def step(self, action):
            return np.array([1.0], dtype=np.float32), 0.0, True, False, {
                "episode_report": {
                    "crop_name": "mock",
                    "episode_steps": 1,
                    "task_return": 1.0,
                    "native_return": 1.0,
                    "total_fertilizer": 0.0,
                    "total_irrigation": 0.0,
                    "final_grain_weight": 1000.0,
                    "final_biomass": 2000.0,
                    "final_leaching": 0.0,
                    "final_denitrification": 0.0,
                    "profit": 10.0,
                }
            }

        def close(self):
            return None

    evaluator = EpisodeEvaluator(env_factory=CountingEnv)
    result = evaluator.evaluate_over_seeds(seeds=[5, 3, 5, None, 3], num_episodes=1)

    assert list(result.by_seed.keys()) == [5, 3, None]
    assert CountingEnv.created_seeds == [5, 3, None]
    assert result.aggregate["seed_count"] == 3
    assert result.aggregate["seeds"] == [None, 3, 5]


def test_multiseed_serialization_is_stable_and_sanitizes_non_finite_values():
    result = MultiSeedEvaluationResult(
        by_seed={
            5: EvaluationResult(
                reports=[{"task_return": np.inf}],
                aggregate={
                    "episodes": 1,
                    "mean_task_return": np.inf,
                    "mean_final_grain_weight": 1800.0,
                    "mean_total_fertilizer": 0.0,
                    "mean_total_irrigation": 0.0,
                    "mean_nitrogen_use_efficiency": 0.0,
                    "mean_water_use_efficiency": 0.0,
                    "mean_profit": np.nan,
                },
            ),
            None: EvaluationResult(
                reports=[{"task_return": 1.0}],
                aggregate={
                    "episodes": 1,
                    "mean_task_return": 1.0,
                    "mean_final_grain_weight": 1500.0,
                    "mean_total_fertilizer": 0.0,
                    "mean_total_irrigation": 0.0,
                    "mean_nitrogen_use_efficiency": 0.0,
                    "mean_water_use_efficiency": 0.0,
                    "mean_profit": 20.0,
                },
            ),
            3: EvaluationResult(
                reports=[{"task_return": 2.0}],
                aggregate={
                    "episodes": 1,
                    "mean_task_return": 2.0,
                    "mean_final_grain_weight": 2100.0,
                    "mean_total_fertilizer": 0.0,
                    "mean_total_irrigation": 0.0,
                    "mean_nitrogen_use_efficiency": 0.0,
                    "mean_water_use_efficiency": 0.0,
                    "mean_profit": 30.0,
                },
            ),
        },
        aggregate={},
    )
    payload = serialize_multiseed_evaluation_suite({"actor_a": result})
    leaderboard = build_actor_leaderboard({"actor_a": result})

    assert list(payload["actors"]["actor_a"]["by_seed"].keys()) == ["default", "3", "5"]
    assert payload["tables"]["actor_summary"][0]["actor"] == "actor_a"
    assert [row["seed_label"] for row in payload["tables"]["seed_summary"]] == ["default", "3", "5"]
    assert payload["actors"]["actor_a"]["aggregate"]["metrics"]["mean_task_return"]["mean"] == pytest.approx(1.0)
    assert payload["leaderboard"][0]["mean_profit"] == pytest.approx(50.0 / 3.0)
    assert payload["tables"]["episode_reports"][0]["seed_label"] == "default"
    assert leaderboard[0]["mean_task_return"] == pytest.approx(1.0)
