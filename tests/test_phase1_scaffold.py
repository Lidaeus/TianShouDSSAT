import json
import gymnasium as gym
import numpy as np
import pytest
from argparse import Namespace

from envs.dssat_base_env import DssatBaseEnv
from evaluation.episode_report import EpisodeReporter
from experiments.experiment_config import ExperimentConfig, Phase1TaskConfig, build_phase1_benchmark_config
from experiments.build_env import resolve_phase1_task_config
from experiments.build_evaluator import (
    EpisodeEvaluator,
    EvaluationResult,
    MultiSeedEvaluationResult,
    _resolve_action,
    build_baseline_actor,
    build_actor_leaderboard,
    build_evaluation_tables,
    serialize_evaluation_suite,
    serialize_multiseed_evaluation_suite,
    write_benchmark_artifacts,
)
from tasks.forecast import WeatherWindowForecastProvider
from tasks.schemas import (
    AgronomicObservationSchema,
    TaskEnv,
    WeeklyDiscreteActionSchema,
    YieldCostRewardSchema,
)
from train_dssat import _normalize_cli_args, evaluate_actor_suite, train


def make_mock_state():
    return {
        "istage": 3,
        "vstage": 6,
        "xlai": 3.2,
        "grnwt": 4200.0,
        "topwt": 9600.0,
        "rtdep": 85.0,
        "swfac": 0.7,
        "nstres": 0.2,
        "wtdep": 160.0,
        "trnu": 2.5,
        "tleachd": 1.2,
        "tnoxd": 0.1,
        "rain": 8.0,
        "tmin": 12.0,
        "tmax": 24.0,
        "srad": 18.0,
    }


def make_mock_context():
    return {
        "sw": [0.22, 0.24, 0.26],
        "ll": [0.12, 0.13, 0.14],
        "dul": [0.28, 0.30, 0.31],
        "dlayr": [15.0, 20.0, 25.0],
    }


def test_agronomic_observation_schema_shape():
    schema = AgronomicObservationSchema(crop_name="wheat", include_forecast=True, forecast_horizons=(3, 7))
    management = WeeklyDiscreteActionSchema().initial_management_state()
    obs = schema.encode(
        raw_state=make_mock_state(),
        context=make_mock_context(),
        management_state=management,
        metadata={
            "weather_forecast": {
                "rain_3d": 15.0,
                "tmean_3d": 18.0,
                "srad_3d": 20.0,
                "rain_7d": 40.0,
                "tmean_7d": 17.0,
                "srad_7d": 19.0,
            }
        },
    )
    assert isinstance(obs, np.ndarray)
    assert obs.shape == schema.observation_space.shape
    assert len(schema.feature_names) == schema.observation_space.shape[0]


def test_forecast_provider_summarizes_daily_weather():
    provider = WeatherWindowForecastProvider(horizons=(3, 7))
    enriched = provider.enrich_metadata(
        raw_state=make_mock_state(),
        info={
            "weather_forecast_daily": [
                {"rain": 2.0, "tmin": 10.0, "tmax": 20.0, "srad": 15.0},
                {"rain": 3.0, "tmin": 11.0, "tmax": 21.0, "srad": 16.0},
                {"rain": 4.0, "tmin": 12.0, "tmax": 22.0, "srad": 17.0},
                {"rain": 5.0, "tmin": 13.0, "tmax": 23.0, "srad": 18.0},
            ]
        },
    )
    summary = enriched["weather_forecast"]
    assert summary["rain_3d"] == 9.0
    assert summary["rain_7d"] == 14.0
    assert summary["tmean_3d"] == 16.0
    assert summary["srad_7d"] == 16.5


def test_forecast_provider_reads_weather_file(tmp_path):
    weather_file = tmp_path / "TEST.WTH"
    weather_file.write_text(
        "*WEATHER:TEST\n"
        "@DATE  SRAD  TMAX  TMIN  RAIN\n"
        "24001  10.0  20.0  10.0   1.0\n"
        "24002  11.0  21.0  11.0   2.0\n"
        "24003  12.0  22.0  12.0   3.0\n"
        "24004  13.0  23.0  13.0   4.0\n"
        "24005  14.0  24.0  14.0   5.0\n",
        encoding="utf-8",
    )
    raw_state = make_mock_state() | {"yrdoy": 24001}
    provider = WeatherWindowForecastProvider(horizons=(3, 7))
    enriched = provider.enrich_metadata(
        raw_state=raw_state,
        info={"weather_file_paths": [str(weather_file)]},
    )
    summary = enriched["weather_forecast"]
    assert summary["rain_3d"] == 9.0
    assert summary["rain_7d"] == 14.0
    assert summary["tmean_3d"] == 17.0
    assert enriched["weather_forecast_source"] == str(weather_file)


def test_weekly_action_schema_interval_and_budget():
    schema = WeeklyDiscreteActionSchema(
        nitrogen_levels=(0.0, 30.0, 60.0),
        irrigation_levels=(0.0, 10.0),
        decision_interval_days=7,
        fertilizer_budget=40.0,
        irrigation_budget=15.0,
    )
    management = schema.initial_management_state()
    action = schema.decode(action=5, raw_state={"istage": 2}, management_state=management)
    assert action == {"anfer": 40.0, "amir": 10.0}
    management = schema.update_management(management, action)
    blocked_action = schema.decode(action=5, raw_state={"istage": 2}, management_state=management)
    assert blocked_action == {"anfer": 0.0, "amir": 0.0}
    description = schema.describe()
    assert description["type"] == "weekly_discrete"
    assert description["decision_interval_days"] == 7
    assert description["supports_action_mask"] is True


def test_weekly_action_schema_action_mask_respects_stage_budget_and_interval():
    schema = WeeklyDiscreteActionSchema(
        nitrogen_levels=(0.0, 30.0, 60.0),
        irrigation_levels=(0.0, 10.0),
        decision_interval_days=7,
        fertilizer_budget=40.0,
        irrigation_budget=15.0,
        fertilizer_stage_cutoff=6.0,
    )
    initial_management = schema.initial_management_state()
    initial_mask = schema.valid_action_mask(raw_state={"istage": 2.0}, management_state=initial_management)
    assert initial_mask.tolist() == [1, 1, 1, 1, 0, 0]

    blocked_management = schema.update_management(initial_management, {"anfer": 30.0, "amir": 10.0})
    blocked_mask = schema.valid_action_mask(raw_state={"istage": 2.0}, management_state=blocked_management)
    assert blocked_mask.tolist() == [1, 0, 0, 0, 0, 0]

    mature_management = WeeklyDiscreteActionSchema(
        nitrogen_levels=(0.0, 30.0, 60.0),
        irrigation_levels=(0.0, 10.0),
        fertilizer_budget=40.0,
        irrigation_budget=15.0,
        fertilizer_stage_cutoff=6.0,
    ).initial_management_state()
    mature_mask = schema.valid_action_mask(raw_state={"istage": 7.0}, management_state=mature_management)
    assert mature_mask.tolist() == [1, 1, 0, 0, 0, 0]


def test_reward_schema_and_reporter():
    reward_schema = YieldCostRewardSchema()
    management = WeeklyDiscreteActionSchema().initial_management_state()
    reward, components = reward_schema.compute(
        previous_state={"grnwt": 1000.0},
        next_state={"grnwt": 1500.0, "tleachd": 0.5, "tnoxd": 0.1},
        management_state=management,
        applied_action={"anfer": 30.0, "amir": 10.0},
        terminated=True,
    )
    assert isinstance(reward, float)
    assert "terminal_yield_bonus" in components

    reporter = EpisodeReporter(grain_price=0.3, fertilizer_unit_cost=1.0, irrigation_unit_cost=0.2)
    reporter.reset(crop_name="wheat", metadata={"split_name": "validation", "weather_id": "WTH001"})
    reporter.record_transition(
        raw_state={"grnwt": 1500.0, "topwt": 4000.0, "tleachd": 0.5, "tnoxd": 0.1},
        applied_action={"anfer": 30.0, "amir": 10.0},
        task_reward=reward,
        native_reward=1.0,
        terminated=True,
    )
    summary = reporter.finalize()
    assert summary["crop_name"] == "wheat"
    assert summary["split_name"] == "validation"
    assert summary["weather_id"] == "WTH001"
    assert summary["final_grain_weight"] == 1500.0
    assert summary["nitrogen_use_efficiency"] == 50.0
    assert summary["water_use_efficiency"] == 150.0
    assert summary["gross_revenue"] == 450.0
    assert summary["total_input_cost"] == 32.0
    assert summary["profit"] == 418.0


def test_reactive_baseline_actor_uses_task_metadata():
    actor = build_baseline_actor("reactive", nitrogen_dose=30.0, irrigation_dose=20.0)
    action = actor.compute_action(
        observation=np.array(
            [
                3.0 / 8.0,
                6.0 / 30.0,
                0.3,
                0.15,
                0.2,
                85.0 / 300.0,
                0.4,
                0.5,
                0.35,
            ],
            dtype=np.float32,
        ),
        info={
            "observation_features": [
                "istage_norm",
                "vstage_norm",
                "canopy_norm",
                "grnwt_norm",
                "topwt_norm",
                "rtdep_norm",
                "swfac_norm",
                "nstres_norm",
                "root_zone_sw_ratio",
            ],
            "action_schema": {
                "type": "weekly_discrete",
                "nitrogen_levels": (0.0, 30.0, 60.0, 90.0),
                "irrigation_levels": (0.0, 10.0, 20.0, 30.0),
                "decision_interval_days": 7,
                "fertilizer_stage_cutoff": 6.0,
            },
            "management_state": WeeklyDiscreteActionSchema().initial_management_state(),
        },
        action_space=gym.spaces.Discrete(16),
    )
    assert action == 6


def test_calendar_baseline_actor_uses_calendar_schedule():
    actor = build_baseline_actor(
        "calendar",
        fertilizer_days=(0, 14),
        irrigation_days=(7, 14),
        fertilizer_dose=30.0,
        irrigation_dose=20.0,
    )
    management = WeeklyDiscreteActionSchema().initial_management_state()
    action_day0 = actor.compute_action(
        observation=np.array([0.0], dtype=np.float32),
        info={
            "action_schema": {
                "type": "weekly_discrete",
                "nitrogen_levels": (0.0, 30.0, 60.0, 90.0),
                "irrigation_levels": (0.0, 10.0, 20.0, 30.0),
            },
            "management_state": management,
        },
        action_space=gym.spaces.Discrete(16),
    )
    assert action_day0 == 4

    management_day7 = management
    for _ in range(7):
        management_day7 = WeeklyDiscreteActionSchema().update_management(
            management_day7,
            {"anfer": 0.0, "amir": 0.0},
        )
    action_day7 = actor.compute_action(
        observation=np.array([0.0], dtype=np.float32),
        info={
            "action_schema": {
                "type": "weekly_discrete",
                "nitrogen_levels": (0.0, 30.0, 60.0, 90.0),
                "irrigation_levels": (0.0, 10.0, 20.0, 30.0),
            },
            "management_state": management_day7,
        },
        action_space=gym.spaces.Discrete(16),
    )
    assert action_day7 == 2


def test_rule_based_baseline_respects_cooldown_and_stage_gates():
    actor = build_baseline_actor(
        "rule_based",
        nitrogen_stress_trigger=0.45,
        irrigation_trigger=0.5,
        root_zone_trigger=0.4,
        min_crop_stage=1.0,
        min_fertilizer_gap_days=10.0,
        min_irrigation_gap_days=7.0,
        nitrogen_dose=30.0,
        irrigation_dose=20.0,
    )
    management = WeeklyDiscreteActionSchema().initial_management_state()
    action_blocked = actor.compute_action(
        observation=np.array(
            [
                3.0 / 8.0,
                0.45,
                0.4,
                0.35,
            ],
            dtype=np.float32,
        ),
        info={
            "observation_features": [
                "istage_norm",
                "nstres_norm",
                "swfac_norm",
                "root_zone_sw_ratio",
            ],
            "action_schema": {
                "type": "weekly_discrete",
                "nitrogen_levels": (0.0, 30.0, 60.0, 90.0),
                "irrigation_levels": (0.0, 10.0, 20.0, 30.0),
                "decision_interval_days": 7,
                "fertilizer_stage_cutoff": 6.0,
            },
            "management_state": management,
        },
        action_space=gym.spaces.Discrete(16),
    )
    assert action_blocked == 0

    management_ready = management
    for _ in range(14):
        management_ready = WeeklyDiscreteActionSchema().update_management(
            management_ready,
            {"anfer": 0.0, "amir": 0.0},
        )
    action_ready = actor.compute_action(
        observation=np.array(
            [
                3.0 / 8.0,
                0.5,
                0.4,
                0.35,
            ],
            dtype=np.float32,
        ),
        info={
            "observation_features": [
                "istage_norm",
                "nstres_norm",
                "swfac_norm",
                "root_zone_sw_ratio",
            ],
            "action_schema": {
                "type": "weekly_discrete",
                "nitrogen_levels": (0.0, 30.0, 60.0, 90.0),
                "irrigation_levels": (0.0, 10.0, 20.0, 30.0),
                "decision_interval_days": 7,
                "fertilizer_stage_cutoff": 6.0,
            },
            "management_state": management_ready,
        },
        action_space=gym.spaces.Discrete(16),
    )
    assert action_ready == 6


class MockEvaluationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self._step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        return np.array([0.0], dtype=np.float32), {"seed_used": seed}

    def step(self, action):
        self._step += 1
        terminated = self._step >= 2
        info = {}
        if terminated:
            info["episode_report"] = {
                "crop_name": "mock",
                "episode_steps": 2,
                "task_return": 1.5,
                "native_return": 2.0,
                "total_fertilizer": 30.0,
                "total_irrigation": 10.0,
                "final_grain_weight": 2000.0,
                "final_biomass": 5000.0,
                "final_leaching": 0.2,
                "final_denitrification": 0.05,
                "nitrogen_use_efficiency": 2000.0 / 30.0,
                "water_use_efficiency": 200.0,
            }
        return np.array([float(self._step)], dtype=np.float32), 0.5, terminated, False, info


def test_episode_evaluator_aggregates_reports():
    evaluator = EpisodeEvaluator(env_factory=MockEvaluationEnv, evaluation_seed_stride=7)

    class FixedActor:
        def compute_action(self, observation, info, action_space):
            return 0

    result = evaluator.evaluate(actor=FixedActor(), num_episodes=2, seed=11)
    assert len(result.reports) == 2
    assert result.aggregate["episodes"] == 2
    assert result.aggregate["mean_task_return"] == 1.5
    assert result.aggregate["mean_final_grain_weight"] == 2000.0
    assert result.aggregate["mean_nitrogen_use_efficiency"] == 2000.0 / 30.0
    assert result.aggregate["mean_profit"] == 0.0


def test_episode_evaluator_supports_actor_suite():
    evaluator = EpisodeEvaluator(env_factory=MockEvaluationEnv)
    results = evaluator.evaluate_actor_suite(
        actors={
            "zero": build_baseline_actor("zero"),
            "reactive": build_baseline_actor("reactive"),
        },
        num_episodes=1,
        seed=5,
    )
    assert set(results.keys()) == {"zero", "reactive"}
    assert results["zero"].aggregate["episodes"] == 1
    assert results["reactive"].aggregate["mean_water_use_efficiency"] == 200.0


def test_episode_evaluator_supports_actor_suite_over_seeds():
    class MockSeededEvaluationEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            self.action_space = gym.spaces.Discrete(16)
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            self._step = 0
            self._seed_used = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._step = 0
            self._seed_used = 0 if seed is None else int(seed)
            return np.array([0.0], dtype=np.float32), {
                "seed_used": seed,
                "action_schema": {
                    "type": "weekly_discrete",
                    "nitrogen_levels": (0.0, 30.0, 60.0, 90.0),
                    "irrigation_levels": (0.0, 10.0, 20.0, 30.0),
                },
                "management_state": WeeklyDiscreteActionSchema().initial_management_state(),
            }

        def step(self, action):
            self._step += 1
            terminated = self._step >= 1
            info = {}
            if terminated:
                info["episode_report"] = {
                    "crop_name": "mock",
                    "episode_steps": 1,
                    "task_return": float(self._seed_used) / 10.0,
                    "native_return": float(self._seed_used) / 5.0,
                    "total_fertilizer": 30.0,
                    "total_irrigation": 10.0,
                    "final_grain_weight": 1000.0 + float(self._seed_used),
                    "final_biomass": 3000.0 + float(self._seed_used),
                    "final_leaching": 0.1,
                    "final_denitrification": 0.02,
                    "nitrogen_use_efficiency": (1000.0 + float(self._seed_used)) / 30.0,
                    "water_use_efficiency": (1000.0 + float(self._seed_used)) / 10.0,
                }
            return np.array([1.0], dtype=np.float32), 0.0, terminated, False, info

    evaluator = EpisodeEvaluator(env_factory=MockSeededEvaluationEnv)
    results = evaluator.evaluate_actor_suite_over_seeds(
        actors={
            "zero": build_baseline_actor("zero"),
            "calendar": build_baseline_actor("calendar"),
        },
        seeds=[3, 5],
        num_episodes=1,
    )
    assert set(results.keys()) == {"zero", "calendar"}
    assert set(results["calendar"].by_seed.keys()) == {3, 5}
    assert results["calendar"].aggregate["seed_count"] == 2
    assert results["calendar"].aggregate["seeds"] == [3, 5]
    assert results["calendar"].aggregate["metrics"]["mean_final_grain_weight"]["mean"] == 1004.0
    assert np.isclose(results["calendar"].aggregate["metrics"]["mean_task_return"]["std"], 0.1)
    assert np.isclose(results["calendar"].aggregate["metrics"]["mean_task_return"]["stderr"], 0.1 / np.sqrt(2.0))
    assert results["calendar"].aggregate["metrics"]["mean_task_return"]["median"] == 0.4


def test_task_env_exposes_action_mask_in_reset_and_step():
    class DummyBaseEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            self.action_space = gym.spaces.Discrete(6)
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            self._step = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._step = 0
            return {
                "istage": 2.0,
                "vstage": 1.0,
                "grnwt": 0.0,
                "topwt": 0.0,
                "rtdep": 10.0,
                "swfac": 1.0,
                "nstres": 0.0,
                "sw": [0.2],
                "ll": [0.1],
                "dul": [0.3],
                "dlayr": [20.0],
            }, {"crop_name": "wheat", "context": {}}

        def step(self, action):
            self._step += 1
            return {
                "istage": 2.0,
                "vstage": 1.0,
                "grnwt": 1000.0,
                "topwt": 2000.0,
                "rtdep": 20.0,
                "swfac": 0.8,
                "nstres": 0.1,
                "tleachd": 0.0,
                "tnoxd": 0.0,
                "sw": [0.21],
                "ll": [0.1],
                "dul": [0.3],
                "dlayr": [20.0],
            }, 0.0, True, False, {"crop_name": "wheat", "context": {}}

        def close(self):
            return None

    env = TaskEnv(
        base_env=DummyBaseEnv(),
        observation_schema=AgronomicObservationSchema(crop_name="wheat", include_forecast=False),
        action_schema=WeeklyDiscreteActionSchema(
            nitrogen_levels=(0.0, 30.0, 60.0),
            irrigation_levels=(0.0, 10.0),
            fertilizer_budget=40.0,
            irrigation_budget=15.0,
        ),
        reward_schema=YieldCostRewardSchema(),
    )
    _, reset_info = env.reset()
    assert reset_info["action_mask"].tolist() == [1, 1, 1, 1, 0, 0]
    _, _, terminated, truncated, step_info = env.step(3)
    assert terminated is True
    assert truncated is False
    assert step_info["applied_action"] == {"anfer": 30.0, "amir": 10.0}
    assert step_info["action_mask"].tolist() == [1, 0, 0, 0, 0, 0]
    env.close()


def test_task_env_standardizes_state_metadata_in_reset_and_step():
    class DummyBaseEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            self.action_space = gym.spaces.Discrete(6)
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return {
                "istage": 2.0,
                "vstage": 3.0,
                "grnwt": 100.0,
                "topwt": 200.0,
                "rtdep": 20.0,
                "swfac": 0.7,
                "nstres": 0.2,
                "rain": 5.0,
                "tmin": 10.0,
                "tmax": 20.0,
                "srad": 15.0,
                "sw": [0.22, 0.24],
                "ll": [0.12, 0.13],
                "dul": [0.30, 0.31],
                "dlayr": [15.0, 20.0],
            }, {"crop_name": "wheat", "context": {}}

        def step(self, action):
            return {
                "istage": 4.0,
                "vstage": 5.0,
                "grnwt": 300.0,
                "topwt": 500.0,
                "rtdep": 25.0,
                "swfac": 0.6,
                "nstres": 0.3,
                "rain": 1.0,
                "tmin": 11.0,
                "tmax": 22.0,
                "srad": 16.0,
                "sw": [0.23, 0.25],
                "ll": [0.12, 0.13],
                "dul": [0.30, 0.31],
                "dlayr": [15.0, 20.0],
            }, 0.5, True, False, {"crop_name": "wheat", "context": {}}

        def close(self):
            return None

    env = TaskEnv(
        base_env=DummyBaseEnv(),
        observation_schema=AgronomicObservationSchema(crop_name="wheat", include_forecast=False),
        action_schema=WeeklyDiscreteActionSchema(nitrogen_levels=(0.0, 30.0, 60.0), irrigation_levels=(0.0, 10.0)),
        reward_schema=YieldCostRewardSchema(),
    )
    _, reset_info = env.reset()
    assert reset_info["istage"] == pytest.approx(2.0)
    assert reset_info["swfac"] == pytest.approx(0.7)
    assert reset_info["rain"] == pytest.approx(5.0)
    assert reset_info["root_zone_sw_ratio"] == pytest.approx(0.5694444, rel=1e-4)
    assert reset_info["profile_sw_ratio"] == pytest.approx(0.5833333, rel=1e-4)

    _, _, terminated, truncated, step_info = env.step(0)
    assert terminated is True
    assert truncated is False
    assert step_info["istage"] == pytest.approx(4.0)
    assert step_info["nstres"] == pytest.approx(0.3)
    assert step_info["root_zone_sw_ratio"] == pytest.approx(0.6333333, rel=1e-4)
    assert step_info["profile_sw_ratio"] == pytest.approx(0.6388889, rel=1e-4)
    env.close()


@pytest.mark.parametrize(
    ("native_reward_payload", "expected_native_reward"),
    [
        ([1.0, 2.5], 3.5),
        (None, 0.0),
    ],
)
def test_task_env_normalizes_native_reward_payloads(native_reward_payload, expected_native_reward):
    class DummyBaseEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            self.action_space = gym.spaces.Discrete(1)
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return {
                "istage": 1.0,
                "vstage": 1.0,
                "grnwt": 100.0,
                "topwt": 200.0,
                "rtdep": 10.0,
                "swfac": 1.0,
                "nstres": 0.0,
                "sw": [0.2],
                "ll": [0.1],
                "dul": [0.3],
                "dlayr": [20.0],
            }, {"crop_name": "wheat", "context": {}}

        def step(self, action):
            return {
                "istage": 2.0,
                "vstage": 2.0,
                "grnwt": 150.0,
                "topwt": 250.0,
                "rtdep": 15.0,
                "swfac": 0.9,
                "nstres": 0.1,
                "tleachd": 0.0,
                "tnoxd": 0.0,
                "sw": [0.2],
                "ll": [0.1],
                "dul": [0.3],
                "dlayr": [20.0],
            }, native_reward_payload, True, False, {"crop_name": "wheat", "context": {}}

        def close(self):
            return None

    reporter = EpisodeReporter(grain_price=0.3, fertilizer_unit_cost=1.0, irrigation_unit_cost=0.2)
    env = TaskEnv(
        base_env=DummyBaseEnv(),
        observation_schema=AgronomicObservationSchema(crop_name="wheat", include_forecast=False),
        action_schema=WeeklyDiscreteActionSchema(nitrogen_levels=(0.0,), irrigation_levels=(0.0,)),
        reward_schema=YieldCostRewardSchema(),
        episode_reporter=reporter,
    )
    env.reset()
    _, _, terminated, truncated, info = env.step(0)
    assert terminated is True
    assert truncated is False
    assert info["native_reward"] == pytest.approx(expected_native_reward)
    assert info["episode_report"]["native_return"] == pytest.approx(expected_native_reward)
    env.close()


def test_episode_evaluator_is_reproducible_for_same_seed():
    class DeterministicEvaluationEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            self.action_space = gym.spaces.Discrete(1)
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            self._seed_used = 0
            self._step = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._seed_used = 0 if seed is None else int(seed)
            self._step = 0
            return np.array([0.0], dtype=np.float32), {"seed_used": self._seed_used}

        def step(self, action):
            self._step += 1
            terminated = self._step >= 1
            info = {}
            if terminated:
                grain = 1500.0 + float(self._seed_used)
                info["episode_report"] = {
                    "crop_name": "mock",
                    "split_name": "test",
                    "weather_id": f"YEAR_{self._seed_used}",
                    "episode_steps": 1,
                    "task_return": grain / 1000.0,
                    "native_return": grain / 800.0,
                    "total_fertilizer": 30.0,
                    "total_irrigation": 10.0,
                    "final_grain_weight": grain,
                    "final_biomass": grain * 2.5,
                    "final_leaching": 0.1,
                    "final_denitrification": 0.02,
                    "nitrogen_use_efficiency": grain / 30.0,
                    "water_use_efficiency": grain / 10.0,
                    "gross_revenue": grain * 0.25,
                    "fertilizer_input_cost": 36.0,
                    "irrigation_input_cost": 0.5,
                    "total_input_cost": 36.5,
                    "profit": grain * 0.25 - 36.5,
                }
            return np.array([1.0], dtype=np.float32), 0.0, terminated, False, info

        def close(self):
            return None

    evaluator = EpisodeEvaluator(env_factory=DeterministicEvaluationEnv, evaluation_seed_stride=5)
    first = evaluator.evaluate(actor=build_baseline_actor("zero"), num_episodes=2, seed=7)
    second = evaluator.evaluate(actor=build_baseline_actor("zero"), num_episodes=2, seed=7)
    assert first.reports == second.reports
    assert first.aggregate == second.aggregate


def test_episode_evaluator_tracks_multi_weather_coverage():
    class WeatherRotationEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            self.action_space = gym.spaces.Discrete(1)
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            self._seed_used = 0
            self._step = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._seed_used = 0 if seed is None else int(seed)
            self._step = 0
            return np.array([0.0], dtype=np.float32), {"seed_used": self._seed_used}

        def step(self, action):
            self._step += 1
            terminated = self._step >= 1
            info = {}
            if terminated:
                info["episode_report"] = {
                    "crop_name": "mock",
                    "split_name": "validation",
                    "weather_id": f"YEAR_{self._seed_used}",
                    "episode_steps": 1,
                    "task_return": 1.0,
                    "native_return": 1.5,
                    "total_fertilizer": 0.0,
                    "total_irrigation": 0.0,
                    "final_grain_weight": 1000.0 + float(self._seed_used),
                    "final_biomass": 2500.0 + float(self._seed_used),
                    "final_leaching": 0.0,
                    "final_denitrification": 0.0,
                    "nitrogen_use_efficiency": 0.0,
                    "water_use_efficiency": 0.0,
                    "gross_revenue": 250.0,
                    "fertilizer_input_cost": 0.0,
                    "irrigation_input_cost": 0.0,
                    "total_input_cost": 0.0,
                    "profit": 250.0,
                }
            return np.array([1.0], dtype=np.float32), 0.0, terminated, False, info

        def close(self):
            return None

    evaluator = EpisodeEvaluator(env_factory=WeatherRotationEnv, evaluation_seed_stride=7)
    result = evaluator.evaluate(actor=build_baseline_actor("zero"), num_episodes=3, seed=2)
    assert result.aggregate["unique_weather_count"] == 3
    assert result.aggregate["weather_ids"] == ["YEAR_2", "YEAR_9", "YEAR_16"]
    assert result.aggregate["unique_weather_ids"] == ["YEAR_16", "YEAR_2", "YEAR_9"]
    assert result.aggregate["mean_profit"] == 250.0


def test_serialize_evaluation_suite_includes_leaderboard():
    results = {
        "zero": EvaluationResult(
            reports=[{"task_return": 1.0, "crop_name": "wheat", "split_name": "test", "weather_id": "A"}],
            aggregate={
                "episodes": 1,
                "mean_task_return": 1.0,
                "mean_final_grain_weight": 1800.0,
                "mean_total_fertilizer": 0.0,
                "mean_total_irrigation": 0.0,
                "mean_nitrogen_use_efficiency": 0.0,
                "mean_water_use_efficiency": 0.0,
                "mean_profit": 90.0,
                "unique_weather_count": 1,
            },
        ),
        "reactive": EvaluationResult(
            reports=[{"task_return": 1.5, "crop_name": "wheat", "split_name": "test", "weather_id": "B"}],
            aggregate={
                "episodes": 1,
                "mean_task_return": 1.5,
                "mean_final_grain_weight": 2200.0,
                "mean_total_fertilizer": 30.0,
                "mean_total_irrigation": 10.0,
                "mean_nitrogen_use_efficiency": 73.3,
                "mean_water_use_efficiency": 220.0,
                "mean_profit": 120.0,
                "unique_weather_count": 1,
            },
        ),
    }
    payload = serialize_evaluation_suite(results)
    assert set(payload.keys()) == {"actors", "leaderboard", "tables"}
    assert set(payload["actors"].keys()) == {"zero", "reactive"}
    assert payload["leaderboard"][0]["actor"] == "reactive"
    assert payload["leaderboard"][0]["mean_final_grain_weight"] == 2200.0
    assert payload["tables"]["actor_summary"][0]["actor"] == "reactive"
    assert payload["tables"]["seed_summary"][0]["seed_label"] == "default"
    assert payload["tables"]["episode_reports"][1]["weather_id"] == "B"


def test_serialize_multiseed_evaluation_suite_includes_seed_reports_and_leaderboard():
    zero_seed_result = EvaluationResult(
        reports=[{"task_return": 1.0, "crop_name": "wheat", "split_name": "test", "weather_id": "A"}],
        aggregate={
            "episodes": 1,
            "mean_task_return": 1.0,
            "mean_final_grain_weight": 1800.0,
            "mean_total_fertilizer": 0.0,
            "mean_total_irrigation": 0.0,
            "mean_nitrogen_use_efficiency": 0.0,
            "mean_water_use_efficiency": 0.0,
            "mean_profit": 90.0,
            "unique_weather_count": 1,
        },
    )
    reactive_seed_result = EvaluationResult(
        reports=[{"task_return": 1.5, "crop_name": "wheat", "split_name": "test", "weather_id": "B"}],
        aggregate={
            "episodes": 1,
            "mean_task_return": 1.5,
            "mean_final_grain_weight": 2200.0,
            "mean_total_fertilizer": 30.0,
            "mean_total_irrigation": 10.0,
            "mean_nitrogen_use_efficiency": 73.3,
            "mean_water_use_efficiency": 220.0,
            "mean_profit": 120.0,
            "unique_weather_count": 1,
        },
    )
    results = {
        "zero": MultiSeedEvaluationResult(
            by_seed={3: zero_seed_result},
            aggregate={
                "seed_count": 1,
                "metrics": {
                    "mean_task_return": {"mean": 1.0},
                    "mean_final_grain_weight": {"mean": 1800.0},
                    "mean_total_fertilizer": {"mean": 0.0},
                    "mean_total_irrigation": {"mean": 0.0},
                    "mean_nitrogen_use_efficiency": {"mean": 0.0},
                    "mean_water_use_efficiency": {"mean": 0.0},
                    "episodes": {"mean": 1.0},
                },
            },
        ),
        "reactive": MultiSeedEvaluationResult(
            by_seed={3: reactive_seed_result, 5: reactive_seed_result},
            aggregate={
                "seed_count": 2,
                "metrics": {
                    "mean_task_return": {"mean": 1.5},
                    "mean_final_grain_weight": {"mean": 2200.0},
                    "mean_total_fertilizer": {"mean": 30.0},
                    "mean_total_irrigation": {"mean": 10.0},
                    "mean_nitrogen_use_efficiency": {"mean": 73.3},
                    "mean_water_use_efficiency": {"mean": 220.0},
                    "episodes": {"mean": 1.0},
                },
            },
        ),
    }
    payload = serialize_multiseed_evaluation_suite(results)
    assert set(payload.keys()) == {"actors", "leaderboard", "tables"}
    assert set(payload["actors"]["reactive"]["by_seed"].keys()) == {"3", "5"}
    assert payload["actors"]["reactive"]["aggregate"]["metrics"]["mean_task_return"]["median"] == 1.5
    assert payload["leaderboard"][0]["actor"] == "reactive"
    assert payload["leaderboard"][0]["mean_task_return"] == 1.5
    assert payload["tables"]["actor_summary"][0]["seed_count"] == 2
    assert payload["tables"]["seed_summary"][0]["seed_label"] == "3"
    assert payload["tables"]["episode_reports"][2]["seed_label"] == "5"


def test_build_evaluation_tables_flattens_single_and_multiseed_results():
    single_result = EvaluationResult(
        reports=[
            {"task_return": 1.0, "crop_name": "wheat", "split_name": "test", "weather_id": "W1"},
            {"task_return": 2.0, "crop_name": "wheat", "split_name": "test", "weather_id": "W2"},
        ],
        aggregate={
            "episodes": 2,
            "mean_task_return": 1.5,
            "mean_final_grain_weight": 1900.0,
            "mean_total_fertilizer": 10.0,
            "mean_total_irrigation": 5.0,
            "mean_nitrogen_use_efficiency": 50.0,
            "mean_water_use_efficiency": 200.0,
            "mean_profit": 100.0,
            "unique_weather_count": 2,
        },
    )
    multiseed_result = MultiSeedEvaluationResult(
        by_seed={
            7: EvaluationResult(
                reports=[{"task_return": 3.0, "crop_name": "wheat", "split_name": "test", "weather_id": "W3"}],
                aggregate={
                    "episodes": 1,
                    "mean_task_return": 3.0,
                    "mean_final_grain_weight": 2300.0,
                    "mean_total_fertilizer": 20.0,
                    "mean_total_irrigation": 8.0,
                    "mean_nitrogen_use_efficiency": 70.0,
                    "mean_water_use_efficiency": 250.0,
                    "mean_profit": 130.0,
                    "unique_weather_count": 1,
                },
            )
        },
        aggregate={"seed_count": 1, "metrics": {"mean_task_return": {"mean": 3.0}}},
    )

    tables = build_evaluation_tables({"single": single_result, "multi": multiseed_result})

    assert [row["actor"] for row in tables["actor_summary"]] == ["multi", "single"]
    assert len(tables["seed_summary"]) == 2
    assert tables["seed_summary"][0]["seed_label"] == "default"
    assert tables["seed_summary"][1]["seed_label"] == "7"
    assert len(tables["episode_reports"]) == 3
    assert tables["episode_reports"][2]["actor"] == "multi"
    assert tables["episode_reports"][2]["seed"] == 7


def test_write_benchmark_artifacts_writes_summary_manifest_and_csv_files(tmp_path):
    payload = {
        "config": {
            "algorithm_name": "rainbow",
            "benchmark_preset": "phase1_single_crop",
            "crop": "wheat",
            "seed": 42,
            "evaluation_base_seeds": [11, 13],
            "baseline_actors": ["zero", "calendar"],
            "protocol": {
                "split_name": "train",
                "evaluation_split_name": "test",
            },
        },
        "leaderboard": [{"actor": "rainbow", "mean_profit": 120.0}],
        "tables": {
            "actor_summary": [{"actor": "rainbow", "rank": 1, "seed_count": 3}],
            "seed_summary": [{"actor": "rainbow", "seed": 11, "seed_label": "11"}],
            "episode_reports": [{"actor": "rainbow", "episode_index": 1, "weather_id": "W1"}],
        },
    }

    artifact_outputs = write_benchmark_artifacts(payload, tmp_path)

    manifest = json.loads((tmp_path / "benchmark_artifacts_manifest.json").read_text(encoding="utf-8"))
    batch_index = json.loads((tmp_path.parent / "benchmark_runs_index.json").read_text(encoding="utf-8"))
    cross_run_summary = json.loads((tmp_path.parent / "benchmark_run_summary.json").read_text(encoding="utf-8"))
    summary_payload = json.loads((tmp_path / "evaluation_summary.json").read_text(encoding="utf-8"))
    actor_summary_csv = (tmp_path / "actor_summary.csv").read_text(encoding="utf-8").splitlines()
    cross_run_csv = (tmp_path.parent / "benchmark_run_summary.csv").read_text(encoding="utf-8").splitlines()

    assert artifact_outputs["summary_path"].endswith("evaluation_summary.json")
    assert artifact_outputs["manifest_path"].endswith("benchmark_artifacts_manifest.json")
    assert artifact_outputs["batch_index_path"].endswith("benchmark_runs_index.json")
    assert artifact_outputs["cross_run_summary_json_path"].endswith("benchmark_run_summary.json")
    assert artifact_outputs["cross_run_summary_csv_path"].endswith("benchmark_run_summary.csv")
    assert summary_payload["leaderboard"][0]["actor"] == "rainbow"
    assert manifest["algorithm_name"] == "rainbow"
    assert manifest["crop"] == "wheat"
    assert manifest["evaluation_split_name"] == "test"
    assert manifest["tables"]["actor_summary"]["row_count"] == 1
    assert manifest["tables"]["actor_summary"]["csv"] == "actor_summary.csv"
    assert batch_index["runs"][0]["top_actor"] == "rainbow"
    assert batch_index["runs"][0]["top_mean_profit"] == pytest.approx(120.0)
    assert batch_index["runs"][0]["artifact_dir"] == str(tmp_path)
    assert cross_run_summary[0]["run_name"] == tmp_path.name
    assert cross_run_summary[0]["top_actor"] == "rainbow"
    assert actor_summary_csv[0] == "actor,rank,seed_count"
    assert actor_summary_csv[1] == "rainbow,1,3"
    assert "run_name" in cross_run_csv[0]
    assert "top_actor" in cross_run_csv[0]


def test_build_actor_leaderboard_uses_yield_then_return():
    leaderboard = build_actor_leaderboard(
        {
            "actor_a": EvaluationResult(
                reports=[],
                aggregate={
                    "episodes": 1,
                    "mean_task_return": 1.0,
                    "mean_final_grain_weight": 2000.0,
                    "mean_total_fertilizer": 0.0,
                    "mean_total_irrigation": 0.0,
                    "mean_nitrogen_use_efficiency": 0.0,
                    "mean_water_use_efficiency": 0.0,
                },
            ),
            "actor_b": EvaluationResult(
                reports=[],
                aggregate={
                    "episodes": 1,
                    "mean_task_return": 1.2,
                    "mean_final_grain_weight": 2000.0,
                    "mean_total_fertilizer": 0.0,
                    "mean_total_irrigation": 0.0,
                    "mean_nitrogen_use_efficiency": 0.0,
                    "mean_water_use_efficiency": 0.0,
                },
            ),
        }
    )
    assert [row["actor"] for row in leaderboard] == ["actor_b", "actor_a"]


def test_resolve_action_accepts_policy_style_compute_action():
    class PolicyStyleActor:
        def compute_action(self, obs, info=None, state=None):
            assert isinstance(obs, np.ndarray)
            assert info == {"seed_used": 3}
            return 1

    action = _resolve_action(
        actor=PolicyStyleActor(),
        observation=np.array([0.0], dtype=np.float32),
        info={"seed_used": 3},
        action_space=gym.spaces.Discrete(2),
    )
    assert action == 1


def test_weekly_action_schema_monotonic_inputs():
    schema = WeeklyDiscreteActionSchema(
        nitrogen_levels=(0.0, 30.0, 60.0),
        irrigation_levels=(0.0, 10.0, 20.0),
        fertilizer_budget=120.0,
        irrigation_budget=90.0,
    )
    management = schema.initial_management_state()
    low_input = schema.decode(action=4, raw_state={"istage": 2.0}, management_state=management)
    high_input = schema.decode(action=8, raw_state={"istage": 2.0}, management_state=management)
    low_state = schema.update_management(management, low_input)
    high_state = schema.update_management(management, high_input)
    assert low_input["anfer"] == 30.0
    assert low_input["amir"] == 10.0
    assert high_input["anfer"] == 60.0
    assert high_input["amir"] == 20.0
    assert low_state.cumulative_fertilizer < high_state.cumulative_fertilizer
    assert low_state.cumulative_irrigation < high_state.cumulative_irrigation
    assert low_state.remaining_fertilizer > high_state.remaining_fertilizer
    assert low_state.remaining_irrigation > high_state.remaining_irrigation


def test_resolve_phase1_task_config_accepts_nested_mapping():
    resolved = resolve_phase1_task_config(
        {
            "crop_name": "wheat",
            "seed": 99,
            "observation": {"include_forecast": True, "forecast_horizons": [7, 3, 7]},
            "action": {"fertilizer_budget": 120.0, "decision_interval_days": 5},
            "reward": {"terminal_yield_weight": 0.08},
            "economics": {"grain_price": 0.3, "fertilizer_unit_cost": 1.1, "irrigation_unit_cost": 0.07},
        }
    )
    assert resolved["crop_name"] == "wheat"
    assert resolved["seed"] == 99
    assert resolved["observation"]["include_forecast"] is True
    assert resolved["observation"]["forecast_horizons"] == (3, 7)
    assert resolved["action"]["fertilizer_budget"] == 120.0
    assert resolved["action"]["decision_interval_days"] == 5
    assert resolved["reward"]["terminal_yield_weight"] == 0.08
    assert resolved["economics"]["grain_price"] == 0.3
    assert resolved["economics"]["fertilizer_unit_cost"] == 1.1
    assert resolved["economics"]["irrigation_unit_cost"] == 0.07


def test_phase1_task_config_normalizes_nested_mapping_inputs():
    config = Phase1TaskConfig(
        crop_name=" wheat ",
        seed="7",
        observation={"include_forecast": 1, "forecast_horizons": [7, 0, 3, 7]},
        action={
            "nitrogen_levels": [0, 60, 30],
            "irrigation_levels": [20, 0],
            "decision_interval_days": 0,
        },
        protocol={
            "split_name": "VALIDATION",
            "weather_files": ["/tmp/A.WTH", " ", "/tmp/A.WTH", "/tmp/B.WTH"],
            "split_ratios": {"train": 2, "validation": 1, "test": 1},
            "split_seed": "9",
        },
    )
    payload = config.to_dict()

    assert config.crop_name == "wheat"
    assert config.seed == 7
    assert config.observation.forecast_horizons == (3, 7)
    assert config.action.nitrogen_levels == (0.0, 60.0, 30.0)
    assert config.action.irrigation_levels == (20.0, 0.0)
    assert config.action.decision_interval_days == 1
    assert config.protocol.split_name == "validation"
    assert config.protocol.weather_files == ("/tmp/A.WTH", "/tmp/B.WTH")
    assert config.protocol.split_ratios == pytest.approx((0.5, 0.25, 0.25))
    assert payload["protocol"]["split_seed"] == 9


def test_experiment_config_normalizes_suite_fields_and_serialization():
    config = ExperimentConfig(
        task={"crop_name": "maize", "protocol": {"split_name": "TEST"}},
        train_env_num=0,
        test_env_num="2",
        evaluation_episodes=0,
        evaluation_seed_stride=0,
        evaluation_split_name=" VALIDATION ",
        baseline_actors=["zero", "calendar", "zero", " "],
        evaluation_base_seeds=[5, "7", 5],
        log_dir="  ",
    )
    payload = config.to_dict()

    assert config.task.crop_name == "maize"
    assert config.task.protocol.split_name == "test"
    assert config.train_env_num == 1
    assert config.test_env_num == 2
    assert config.evaluation_episodes == 1
    assert config.evaluation_seed_stride == 1
    assert config.evaluation_split_name == "validation"
    assert config.baseline_actors == ("zero", "calendar")
    assert config.evaluation_base_seeds == (5, 7)
    assert config.log_dir == "logs"
    assert payload["evaluation_split_name"] == "validation"
    assert payload["baseline_actors"] == ("zero", "calendar")
    assert payload["evaluation_base_seeds"] == (5, 7)


def test_build_phase1_benchmark_config_sets_single_crop_protocol_defaults():
    config = build_phase1_benchmark_config(
        crop_name="maize",
        seed=9,
        train_weather_files=["/tmp/TRAIN.WTH"],
        validation_weather_files=["/tmp/VAL.WTH"],
        test_weather_files=["/tmp/TEST.WTH"],
    )
    payload = config.to_dict()

    assert config.task.crop_name == "maize"
    assert config.task.seed == 9
    assert config.task.observation.include_forecast is True
    assert config.task.observation.forecast_horizons == (3, 7)
    assert config.task.protocol.split_name == "train"
    assert config.evaluation_split_name == "test"
    assert config.evaluation_base_seeds == (11, 23, 37)
    assert config.baseline_actors == ("zero", "calendar", "rule_based")
    assert config.log_dir == "logs/benchmarks/phase1_single_crop"
    assert payload["task"]["protocol"]["test_weather_files"] == ("/tmp/TEST.WTH",)


def test_train_cli_args_are_normalized_and_validated():
    args = Namespace(
        forecast_horizons=[7, 3, 7, 0],
        baseline_actors=["zero", "calendar", "zero", " "],
        train_env_num=1,
        test_env_num=1,
        epochs=1,
        batch_size=64,
        hidden_size=128,
        replay_buffer_size=128,
        epoch_num_steps=10,
        collection_step_num_env_steps=5,
        test_step_num_episodes=1,
        eval_episodes=1,
        evaluation_seed_stride=1000,
        weather_split_ratios=[0.7, 0.15, 0.15],
        evaluation_base_seeds=[5, 7],
    )
    normalized = _normalize_cli_args(args)
    assert normalized.forecast_horizons == [3, 7]
    assert normalized.baseline_actors == ["zero", "calendar"]
    assert normalized.evaluation_base_seeds == [5, 7]

    invalid_args = Namespace(
        forecast_horizons=[3],
        baseline_actors=["zero"],
        train_env_num=0,
        test_env_num=1,
        epochs=1,
        batch_size=64,
        hidden_size=128,
        replay_buffer_size=128,
        epoch_num_steps=10,
        collection_step_num_env_steps=5,
        test_step_num_episodes=1,
        eval_episodes=1,
        evaluation_seed_stride=1000,
        weather_split_ratios=[0.7, 0.15, 0.15],
        evaluation_base_seeds=None,
    )
    with pytest.raises(ValueError, match="train_env_num"):
        _normalize_cli_args(invalid_args)


def test_train_cli_args_apply_phase1_benchmark_preset():
    args = Namespace(
        crop="wheat",
        seed=42,
        include_forecast=False,
        forecast_horizons=[3, 7],
        baseline_actors=["zero", "calendar", "rule_based"],
        benchmark_preset="phase1_single_crop",
        evaluation_split_name=None,
        train_env_num=1,
        test_env_num=1,
        epochs=1,
        batch_size=64,
        hidden_size=128,
        replay_buffer_size=128,
        epoch_num_steps=10,
        collection_step_num_env_steps=5,
        test_step_num_episodes=1,
        eval_episodes=2,
        evaluation_seed_stride=1000,
        weather_split_ratios=[0.7, 0.15, 0.15],
        evaluation_base_seeds=None,
        split_name="train",
        weather_files=None,
        train_weather_files=None,
        validation_weather_files=None,
        test_weather_files=None,
        log_dir="logs",
    )

    normalized = _normalize_cli_args(args)

    assert normalized.benchmark_preset == "phase1_single_crop"
    assert normalized.include_forecast is True
    assert normalized.evaluation_split_name == "test"
    assert normalized.evaluation_base_seeds == [11, 23, 37]
    assert normalized.eval_episodes == 3
    assert normalized.log_dir == "logs/benchmarks/phase1_single_crop"


def test_evaluate_actor_suite_uses_evaluation_split_name(monkeypatch, tmp_path):
    captured = {}

    class DummyPolicy:
        def eval(self):
            captured["policy_eval"] = True

    class FakeEvaluator:
        def evaluate_actor_suite(self, actor_suite, num_episodes, seed):
            captured["actor_names"] = tuple(actor_suite.keys())
            captured["num_episodes"] = num_episodes
            captured["seed"] = seed
            return {
                "rainbow": Namespace(
                    aggregate={"mean_final_grain_weight": 1.0, "mean_task_return": 0.5},
                    reports=[],
                )
            }

        def evaluate_actor_suite_over_seeds(self, actors, num_episodes, seeds):
            captured["actor_names"] = tuple(actors.keys())
            captured["num_episodes"] = num_episodes
            captured["seed_suite"] = tuple(seeds)
            return {
                "rainbow": Namespace(
                    aggregate={"mean_final_grain_weight": 1.0, "mean_task_return": 0.5},
                    by_seed={},
                )
            }

    def fake_build_phase1_evaluator(**kwargs):
        captured["overrides"] = kwargs
        return FakeEvaluator()

    monkeypatch.setattr("train_dssat.build_phase1_evaluator", fake_build_phase1_evaluator)
    monkeypatch.setattr("train_dssat.build_baseline_actor", lambda name: f"baseline:{name}")
    monkeypatch.setattr(
        "train_dssat.serialize_evaluation_suite",
        lambda results: {"actors": list(results.keys()), "leaderboard": []},
    )
    monkeypatch.setattr(
        "train_dssat.serialize_multiseed_evaluation_suite",
        lambda results: {
            "actors": list(results.keys()),
            "leaderboard": [],
            "tables": {"actor_summary": [{"actor": "rainbow", "rank": 1}]},
        },
    )

    args = Namespace(
        crop="wheat",
        seed=42,
        include_forecast=False,
        forecast_horizons=[3, 7],
        split_name="train",
        evaluation_split_name="test",
        weather_files=None,
        train_weather_files=None,
        validation_weather_files=None,
        test_weather_files=None,
        weather_split_ratios=[0.7, 0.15, 0.15],
        split_seed=42,
        evaluation_seed_stride=1000,
        grain_price=0.25,
        fertilizer_unit_cost=1.2,
        irrigation_unit_cost=0.05,
        baseline_actors=["zero"],
        evaluation_base_seeds=None,
        eval_episodes=2,
        benchmark_preset="phase1_single_crop",
        log_dir=str(tmp_path),
        train_env_num=1,
        test_env_num=1,
        epochs=1,
        batch_size=64,
        hidden_size=128,
        replay_buffer_size=128,
        epoch_num_steps=10,
        collection_step_num_env_steps=5,
        test_step_num_episodes=1,
    )

    evaluate_actor_suite(DummyPolicy(), args, str(tmp_path))

    assert captured["policy_eval"] is True
    assert captured["overrides"]["split_name"] == "test"
    assert captured["actor_names"] == ("rainbow", "zero")

    payload = json.loads((tmp_path / "evaluation_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "benchmark_artifacts_manifest.json").read_text(encoding="utf-8"))
    batch_index = json.loads((tmp_path.parent / "benchmark_runs_index.json").read_text(encoding="utf-8"))
    cross_run_summary_csv = (tmp_path.parent / "benchmark_run_summary.csv").read_text(encoding="utf-8")
    assert payload["config"]["benchmark_preset"] == "phase1_single_crop"
    assert payload["config"]["algorithm_name"] == "rainbow"
    assert payload["config"]["protocol"]["split_name"] == "train"
    assert payload["config"]["protocol"]["evaluation_split_name"] == "test"
    assert manifest["algorithm_name"] == "rainbow"
    assert manifest["tables"]["actor_summary"]["csv"] == "actor_summary.csv"
    assert any(run["run_name"] == tmp_path.name for run in batch_index["runs"])
    assert "run_name" in cross_run_summary_csv


def test_train_closes_envs_when_setup_fails(monkeypatch, tmp_path):
    close_log = []

    class FakeEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            self.action_space = gym.spaces.Discrete(2)
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.array([0.0, 0.0, 0.0], dtype=np.float32), {}

        def step(self, action):
            return np.array([0.0, 0.0, 0.0], dtype=np.float32), 0.0, True, False, {}

        def close(self):
            close_log.append(id(self))

    monkeypatch.setattr("train_dssat.build_phase1_task_env", lambda **kwargs: FakeEnv())
    monkeypatch.setattr("train_dssat.build_rainbow_components", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    args = Namespace(
        crop="wheat",
        seed=42,
        epochs=1,
        batch_size=64,
        hidden_size=128,
        log_dir=str(tmp_path),
        train_env_num=2,
        test_env_num=2,
        include_forecast=False,
        forecast_horizons=[3, 7],
        split_name="train",
        weather_files=None,
        train_weather_files=None,
        validation_weather_files=None,
        test_weather_files=None,
        weather_split_ratios=[0.7, 0.15, 0.15],
        split_seed=42,
        run_dssat_location=None,
        replay_buffer_size=128,
        epoch_num_steps=10,
        collection_step_num_env_steps=5,
        test_step_num_episodes=1,
        eval_episodes=1,
        evaluation_seed_stride=1000,
        grain_price=0.25,
        fertilizer_unit_cost=1.2,
        irrigation_unit_cost=0.05,
        baseline_actors=["zero"],
        evaluation_base_seeds=None,
    )

    with pytest.raises(RuntimeError, match="boom"):
        train(args)
    assert len(close_log) == 5


def test_dssat_base_env_normalizes_tuple_native_reward(monkeypatch):
    class DummyVendorEnv:
        def __init__(self, **kwargs):
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            self.action_space = gym.spaces.Discrete(1)

        def reset(self, **kwargs):
            return {"grnwt": 0.0}, {"context": {}}

        def step(self, action):
            return {"grnwt": 1200.0}, (1.25, 0.75), True, False, {"context": {}}

        def close(self):
            return None

    class DummyResources:
        run_dssat_location = "/tmp/run_dssat"
        filex_template_path = "/tmp/test.WHX"
        pdi_template_path = "/tmp/test.jinja2"
        auxiliary_file_paths = ["/tmp/SOIL.SOL", "/tmp/CUSTOM_TEST.WTH"]
        extra_env_kwargs = {}

    monkeypatch.setattr("envs.dssat_base_env.DssatPdi", DummyVendorEnv)
    monkeypatch.setattr("envs.dssat_base_env.resolve_crop_resources", lambda **kwargs: DummyResources())

    env = DssatBaseEnv(crop_name="wheat", seed=5, env_kwargs={"split_name": "test"})
    env.reset()
    _, reward, terminated, truncated, info = env.step({"anfer": 30.0, "amir": 10.0})

    assert terminated is True
    assert truncated is False
    assert reward == pytest.approx(2.0)
    assert info["native_reward"] == pytest.approx(2.0)
    assert info["episode_summary"]["native_return"] == pytest.approx(2.0)
    env.close()


def test_dssat_base_env_rejects_blank_auxiliary_file_paths(monkeypatch):
    monkeypatch.setattr(
        "envs.dssat_base_env.resolve_crop_resources",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not resolve resources")),
    )

    with pytest.raises(ValueError, match="auxiliary_file_paths 中包含空路径"):
        DssatBaseEnv(
            crop_name="wheat",
            env_kwargs={"auxiliary_file_paths": ["   ", "/tmp/VALID.WTH"]},
        )
