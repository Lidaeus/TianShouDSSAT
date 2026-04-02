import gymnasium as gym
import numpy as np

from envs.dssat_base_env import DssatBaseEnv
from envs.resource_resolver import ResolvedCropResources, resolve_crop_resources
from experiments.build_evaluator import EpisodeEvaluator, _resolve_action
from experiments.experiment_config import EconomicsConfig, ExperimentConfig, Phase1TaskConfig, ProtocolConfig
from experiments.build_env import build_phase1_task_env, resolve_phase1_task_config


def test_resolve_phase1_task_config_supports_protocol_mapping():
    resolved = resolve_phase1_task_config(
        {
            "crop_name": "wheat",
            "protocol": {
                "split_name": "test",
                "train_weather_files": ["/tmp/TRAIN_A.WTH"],
                "validation_weather_files": ["/tmp/VAL_A.WTH"],
                "test_weather_files": ["/tmp/TEST_A.WTH", "/tmp/TEST_A.WTH"],
            },
        }
    )
    assert resolved["protocol"]["split_name"] == "test"
    assert resolved["protocol"]["train_weather_files"] == ("/tmp/TRAIN_A.WTH",)
    assert resolved["protocol"]["validation_weather_files"] == ("/tmp/VAL_A.WTH",)
    assert resolved["protocol"]["test_weather_files"] == ("/tmp/TEST_A.WTH",)


def test_resolve_phase1_task_config_partitions_weather_files_deterministically():
    resolved = resolve_phase1_task_config(
        {
            "crop_name": "wheat",
            "protocol": {
                "split_name": "validation",
                "weather_files": [
                    "/tmp/YEAR_2001.WTH",
                    "/tmp/YEAR_2002.WTH",
                    "/tmp/YEAR_2003.WTH",
                    "/tmp/YEAR_2004.WTH",
                    "/tmp/YEAR_2005.WTH",
                ],
                "split_ratios": (0.4, 0.2, 0.4),
                "split_seed": 11,
            },
        }
    )
    assert resolved["protocol"]["split_name"] == "validation"
    assert len(resolved["protocol"]["train_weather_files"]) == 2
    assert len(resolved["protocol"]["validation_weather_files"]) == 1
    assert len(resolved["protocol"]["test_weather_files"]) == 2
    assert resolved["protocol"]["selected_weather_files"] == resolved["protocol"]["validation_weather_files"]
    assert resolved["protocol"]["weather_file_counts"] == {"train": 2, "validation": 1, "test": 2}
    assert resolved["protocol"]["selected_weather_count"] == 1
    rerun = resolve_phase1_task_config(
        {
            "crop_name": "wheat",
            "protocol": {
                "split_name": "validation",
                "weather_files": [
                    "/tmp/YEAR_2001.WTH",
                    "/tmp/YEAR_2002.WTH",
                    "/tmp/YEAR_2003.WTH",
                    "/tmp/YEAR_2004.WTH",
                    "/tmp/YEAR_2005.WTH",
                ],
                "split_ratios": (0.4, 0.2, 0.4),
                "split_seed": 11,
            },
        }
    )
    assert rerun["protocol"]["train_weather_files"] == resolved["protocol"]["train_weather_files"]
    assert rerun["protocol"]["validation_weather_files"] == resolved["protocol"]["validation_weather_files"]
    assert rerun["protocol"]["test_weather_files"] == resolved["protocol"]["test_weather_files"]


def test_resolve_phase1_task_config_rejects_overlapping_weather_splits():
    try:
        resolve_phase1_task_config(
            {
                "crop_name": "wheat",
                "protocol": {
                    "split_name": "validation",
                    "train_weather_files": ["/tmp/SHARED.WTH"],
                    "validation_weather_files": ["/tmp/SHARED.WTH"],
                },
            }
        )
    except ValueError as exc:
        assert "重叠文件" in str(exc)
    else:
        raise AssertionError("expected overlap validation to fail")


def test_resolve_phase1_task_config_rejects_empty_selected_split():
    try:
        resolve_phase1_task_config(
            {
                "crop_name": "wheat",
                "protocol": {
                    "split_name": "validation",
                    "weather_files": ["/tmp/YEAR_2001.WTH"],
                    "split_ratios": (1.0, 0.0, 0.0),
                    "split_seed": 3,
                },
            }
        )
    except ValueError as exc:
        assert "未分配到任何天气文件" in str(exc)
    else:
        raise AssertionError("expected empty selected split to fail")


def test_resolve_phase1_task_config_backfills_missing_splits_from_shared_weather_pool():
    resolved = resolve_phase1_task_config(
        {
            "crop_name": "wheat",
            "protocol": {
                "split_name": "validation",
                "weather_files": [
                    "/tmp/YEAR_2001.WTH",
                    "/tmp/YEAR_2002.WTH",
                    "/tmp/YEAR_2003.WTH",
                    "/tmp/YEAR_2004.WTH",
                ],
                "train_weather_files": ["/tmp/PINNED_TRAIN.WTH"],
                "split_ratios": (0.5, 0.25, 0.25),
                "split_seed": 9,
            },
        }
    )
    assert resolved["protocol"]["train_weather_files"] == ("/tmp/PINNED_TRAIN.WTH",)
    assert len(resolved["protocol"]["validation_weather_files"]) == 2
    assert len(resolved["protocol"]["test_weather_files"]) == 2
    assert resolved["protocol"]["selected_weather_files"] == resolved["protocol"]["validation_weather_files"]
    assert "/tmp/PINNED_TRAIN.WTH" not in resolved["protocol"]["selected_weather_files"]
    assert resolved["protocol"]["selected_weather_count"] == 2


def test_resolve_crop_resources_replaces_default_weather_files(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for filename in ("UFGA.CLI", "SOIL.SOL", "UFGA8201.WTH", "CUSTOM_TEST.WTH"):
        (data_dir / filename).write_text(filename, encoding="utf-8")
    run_dssat = tmp_path / "run_dssat"
    run_dssat.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    run_dssat.chmod(0o755)
    monkeypatch.setenv("DSSAT_DATA_DIR", str(data_dir))
    monkeypatch.setenv("DSSAT_RUN_PATH", str(run_dssat))
    resolved = resolve_crop_resources(
        crop_name="maize",
        extra_auxiliary_files=["CUSTOM_TEST.WTH"],
    )
    assert str(data_dir / "CUSTOM_TEST.WTH") in resolved.auxiliary_file_paths
    assert not any(
        path.endswith("UFGA8201.WTH")
        for path in resolved.auxiliary_file_paths
    )
    assert resolved.auxiliary_file_paths == [
        str(data_dir / "UFGA.CLI"),
        str(data_dir / "SOIL.SOL"),
        str(data_dir / "CUSTOM_TEST.WTH"),
    ]
    assert any(path.endswith("SOIL.SOL") for path in resolved.auxiliary_file_paths)
    assert resolved.run_dssat_location == str(run_dssat)


def test_resolve_crop_resources_deduplicates_extra_files_without_reordering_non_weather_inputs(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for filename in ("UFGA.CLI", "SOIL.SOL", "CUSTOM_A.WTH", "CUSTOM_B.WTH"):
        (data_dir / filename).write_text(filename, encoding="utf-8")
    monkeypatch.setenv("DSSAT_DATA_DIR", str(data_dir))
    resolved = resolve_crop_resources(
        crop_name="maize",
        extra_auxiliary_files=["SOIL.SOL", "CUSTOM_B.WTH", "CUSTOM_B.WTH"],
    )
    assert resolved.auxiliary_file_paths == [
        str(data_dir / "UFGA.CLI"),
        str(data_dir / "SOIL.SOL"),
        str(data_dir / "CUSTOM_B.WTH"),
    ]


def test_dssat_base_env_keeps_split_metadata_outside_vendor_kwargs(monkeypatch):
    captured_kwargs = {}

    class DummyVendorEnv:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)
            self.observation_space = "obs-space"
            self.action_space = "action-space"

        def reset(self, **kwargs):
            return {"grnwt": 10.0}, {"context": {}}

        def step(self, action):
            return {"grnwt": 12.0}, 1.0, True, False, {"context": {}}

        def close(self):
            return None

    def fake_resolve_crop_resources(crop_name, run_dssat_location, extra_auxiliary_files):
        return ResolvedCropResources(
            crop_name=crop_name,
            run_dssat_location=run_dssat_location,
            filex_template_path="/tmp/test.WHX",
            pdi_template_path="/tmp/test.jinja2",
            auxiliary_file_paths=["/tmp/SOIL.SOL", "/tmp/CUSTOM_TEST.WTH"],
            extra_env_kwargs={},
        )

    monkeypatch.setattr("envs.dssat_base_env.DssatPdi", DummyVendorEnv)
    monkeypatch.setattr("envs.dssat_base_env.resolve_crop_resources", fake_resolve_crop_resources)

    env = DssatBaseEnv(
        crop_name="wheat",
        seed=7,
        env_kwargs={
            "split_name": "validation",
            "auxiliary_file_paths": ["/tmp/CUSTOM_TEST.WTH"],
        },
    )
    observation, info = env.reset(seed=9)

    assert observation == {"grnwt": 10.0}
    assert captured_kwargs["seed"] == 7
    assert "split_name" not in captured_kwargs
    assert info["split_name"] == "validation"
    assert info["weather_id"] == "CUSTOM_TEST"
    assert info["episode_summary"]["split_name"] == "validation"

    env.close()


def test_build_phase1_task_env_passes_selected_split_weather_file(monkeypatch):
    captured_env_kwargs = {}

    class DummyBaseEnv:
        def __init__(self, crop_name, mode, seed, env_kwargs=None):
            captured_env_kwargs.update(env_kwargs or {})
            self.crop_name = crop_name
            self.mode = mode
            self.seed = seed
            self.action_space = type("DummyActionSpace", (), {"n": 16})()
            self.observation_space = type("DummyObservationSpace", (), {"shape": (4,)})()

        def reset(self, **kwargs):
            return {"grnwt": 0.0}, {"context": {}, "crop_name": "wheat"}

        def step(self, action):
            return {"grnwt": 1.0}, 0.0, True, False, {"context": {}, "crop_name": "wheat"}

        def close(self):
            return None

    monkeypatch.setattr("experiments.build_env.DssatBaseEnv", DummyBaseEnv)

    env = build_phase1_task_env(
        {
            "crop_name": "wheat",
            "protocol": {
                "split_name": "test",
                "test_weather_files": ["/tmp/SPLIT_TEST.WTH"],
            },
        }
    )
    observation, info = env.reset()

    assert captured_env_kwargs["split_name"] == "test"
    assert captured_env_kwargs["auxiliary_file_paths"] == ["/tmp/SPLIT_TEST.WTH"]
    assert isinstance(observation, np.ndarray)
    assert info["action_schema"]["type"] == "weekly_discrete"

    env.close()


def test_build_phase1_task_env_can_use_partitioned_weather_files(monkeypatch):
    captured_env_kwargs = {}

    class DummyBaseEnv:
        def __init__(self, crop_name, mode, seed, env_kwargs=None):
            captured_env_kwargs.update(env_kwargs or {})
            self.action_space = type("DummyActionSpace", (), {"n": 16})()
            self.observation_space = type("DummyObservationSpace", (), {"shape": (4,)})()

        def reset(self, **kwargs):
            return {"grnwt": 0.0}, {"context": {}, "crop_name": "wheat"}

        def step(self, action):
            return {"grnwt": 1.0}, 0.0, True, False, {"context": {}, "crop_name": "wheat"}

        def close(self):
            return None

    monkeypatch.setattr("experiments.build_env.DssatBaseEnv", DummyBaseEnv)

    env = build_phase1_task_env(
        crop_name="wheat",
        split_name="train",
        weather_files=(
            "/tmp/YEAR_2001.WTH",
            "/tmp/YEAR_2002.WTH",
            "/tmp/YEAR_2003.WTH",
            "/tmp/YEAR_2004.WTH",
        ),
        split_ratios=(0.5, 0.25, 0.25),
        split_seed=3,
        run_dssat_location="/tmp/fake_run_dssat",
    )
    env.reset()

    assert captured_env_kwargs["split_name"] == "train"
    assert len(captured_env_kwargs["auxiliary_file_paths"]) == 2
    assert captured_env_kwargs["run_dssat_location"] == "/tmp/fake_run_dssat"

    env.close()


def test_build_phase1_task_env_uses_shared_weather_pool_to_fill_unset_selected_split(monkeypatch):
    captured_env_kwargs = {}

    class DummyBaseEnv:
        def __init__(self, crop_name, mode, seed, env_kwargs=None):
            captured_env_kwargs.update(env_kwargs or {})
            self.action_space = type("DummyActionSpace", (), {"n": 16})()
            self.observation_space = type("DummyObservationSpace", (), {"shape": (4,)})()

        def reset(self, **kwargs):
            return {"grnwt": 0.0}, {"context": {}, "crop_name": "wheat"}

        def step(self, action):
            return {"grnwt": 1.0}, 0.0, True, False, {"context": {}, "crop_name": "wheat"}

        def close(self):
            return None

    monkeypatch.setattr("experiments.build_env.DssatBaseEnv", DummyBaseEnv)

    env = build_phase1_task_env(
        crop_name="wheat",
        split_name="validation",
        weather_files=(
            "/tmp/YEAR_2001.WTH",
            "/tmp/YEAR_2002.WTH",
            "/tmp/YEAR_2003.WTH",
            "/tmp/YEAR_2004.WTH",
        ),
        train_weather_files=("/tmp/PINNED_TRAIN.WTH",),
        split_ratios=(0.5, 0.25, 0.25),
        split_seed=9,
    )
    env.reset()

    assert captured_env_kwargs["split_name"] == "validation"
    assert len(captured_env_kwargs["auxiliary_file_paths"]) == 2
    assert "/tmp/PINNED_TRAIN.WTH" not in captured_env_kwargs["auxiliary_file_paths"]

    env.close()


def test_resolve_phase1_task_config_accepts_dataclass_config_with_economics():
    config = Phase1TaskConfig(
        crop_name="wheat",
        seed=123,
        economics=EconomicsConfig(
            grain_price=0.32,
            fertilizer_unit_cost=1.05,
            irrigation_unit_cost=0.08,
        ),
        protocol=ProtocolConfig(
            split_name="validation",
            validation_weather_files=("/tmp/VAL_2003.WTH",),
        ),
    )
    resolved = resolve_phase1_task_config(config)
    assert resolved["crop_name"] == "wheat"
    assert resolved["seed"] == 123
    assert resolved["economics"]["grain_price"] == 0.32
    assert resolved["economics"]["fertilizer_unit_cost"] == 1.05
    assert resolved["economics"]["irrigation_unit_cost"] == 0.08
    assert resolved["protocol"]["split_name"] == "validation"
    assert resolved["protocol"]["selected_weather_files"] == ("/tmp/VAL_2003.WTH",)


def test_experiment_config_to_dict_preserves_nested_task_fields():
    config = ExperimentConfig(
        task=Phase1TaskConfig(
            crop_name="wheat",
            economics=EconomicsConfig(grain_price=0.4),
            protocol=ProtocolConfig(split_name="test", test_weather_files=("/tmp/TEST_2001.WTH",)),
        ),
        baseline_actors=("zero", "rule_based"),
        evaluation_base_seeds=(3, 5),
    )
    payload = config.to_dict()
    assert payload["task"]["crop_name"] == "wheat"
    assert payload["task"]["economics"]["grain_price"] == 0.4
    assert payload["task"]["protocol"]["split_name"] == "test"
    assert payload["task"]["protocol"]["test_weather_files"] == ("/tmp/TEST_2001.WTH",)
    assert payload["baseline_actors"] == ("zero", "rule_based")
    assert payload["evaluation_base_seeds"] == (3, 5)


def test_resolve_action_respects_action_mask_for_invalid_actor_output():
    class InvalidActor:
        def compute_action(self, observation, info, action_space):
            return 5

    action = _resolve_action(
        actor=InvalidActor(),
        observation=np.array([0.0], dtype=np.float32),
        info={"action_mask": np.array([1, 0, 0, 1], dtype=np.int8)},
        action_space=gym.spaces.Discrete(4),
    )
    assert action == 0


def test_episode_evaluator_synthesizes_report_when_env_omits_episode_report():
    class MinimalEvaluationEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            self.action_space = gym.spaces.Discrete(4)
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.array([0.0], dtype=np.float32), {
                "crop_name": "wheat",
                "split_name": "validation",
                "weather_id": "WTH_SYNTH",
                "action_mask": np.array([1, 0, 0, 0], dtype=np.int8),
            }

        def step(self, action):
            info = {
                "crop_name": "wheat",
                "split_name": "validation",
                "weather_id": "WTH_SYNTH",
                "grnwt": 1800.0,
                "topwt": 4200.0,
                "tleachd": 0.3,
                "tnoxd": 0.05,
                "task_reward": 2.5,
                "native_reward": [1.0, 0.5],
                "applied_action": {"anfer": 30.0, "amir": 10.0},
            }
            return np.array([1.0], dtype=np.float32), 0.0, True, False, info

        def close(self):
            return None

    evaluator = EpisodeEvaluator(env_factory=MinimalEvaluationEnv, evaluation_seed_stride=7)
    result = evaluator.evaluate(actor=lambda obs, info, action_space: 3, num_episodes=1, seed=4)

    assert len(result.reports) == 1
    report = result.reports[0]
    assert report["crop_name"] == "wheat"
    assert report["split_name"] == "validation"
    assert report["weather_id"] == "WTH_SYNTH"
    assert report["task_return"] == 2.5
    assert report["native_return"] == 1.5
    assert report["total_fertilizer"] == 30.0
    assert report["total_irrigation"] == 10.0
    assert report["final_grain_weight"] == 1800.0
    assert result.aggregate["unique_weather_count"] == 1
    assert result.aggregate["mean_final_grain_weight"] == 1800.0
