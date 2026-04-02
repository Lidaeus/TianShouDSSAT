from __future__ import annotations

from collections.abc import Mapping
import math
import random
from typing import Any

from envs.dssat_base_env import DssatBaseEnv
from evaluation.episode_report import EpisodeReporter
from tasks.forecast import NullForecastProvider, WeatherWindowForecastProvider
from tasks.schemas import (
    AgronomicObservationSchema,
    TaskEnv,
    WeeklyDiscreteActionSchema,
    YieldCostRewardSchema,
)


def _read_value(container: Any, key: str, default: Any) -> Any:
    if container is None:
        return default
    if isinstance(container, Mapping):
        return container.get(key, default)
    return getattr(container, key, default)


def _normalize_horizons(value: Any) -> tuple[int, ...]:
    if value is None:
        return (3, 7)
    horizons = value if isinstance(value, (list, tuple, set)) else [value]
    normalized = sorted({int(horizon) for horizon in horizons if int(horizon) > 0})
    return tuple(normalized) or (3, 7)


def _normalize_paths(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    paths = value if isinstance(value, (list, tuple, set)) else [value]
    normalized = []
    for path in paths:
        if path is None:
            continue
        path_str = str(path).strip()
        if path_str:
            normalized.append(path_str)
    return tuple(dict.fromkeys(normalized))


def _normalize_split_name(value: Any) -> str:
    split_name = str(value or "train").strip().lower()
    return split_name if split_name in {"train", "validation", "test"} else "train"


def _normalize_split_ratios(value: Any) -> dict[str, float]:
    if isinstance(value, Mapping):
        raw = {
            "train": float(value.get("train", 0.7) or 0.0),
            "validation": float(value.get("validation", 0.15) or 0.0),
            "test": float(value.get("test", 0.15) or 0.0),
        }
    else:
        ratios = list(value) if isinstance(value, (list, tuple)) else []
        raw = {
            "train": float(ratios[0]) if len(ratios) > 0 else 0.7,
            "validation": float(ratios[1]) if len(ratios) > 1 else 0.15,
            "test": float(ratios[2]) if len(ratios) > 2 else 0.15,
        }
    total = sum(max(0.0, ratio) for ratio in raw.values())
    if total <= 0.0:
        return {"train": 0.7, "validation": 0.15, "test": 0.15}
    return {
        split_name: max(0.0, ratio) / total
        for split_name, ratio in raw.items()
    }


def _partition_weather_files(
    weather_files: tuple[str, ...],
    split_ratios: Mapping[str, float],
    split_seed: int,
) -> dict[str, tuple[str, ...]]:
    ordered_files = list(weather_files)
    if not ordered_files:
        return {"train": (), "validation": (), "test": ()}
    random.Random(int(split_seed)).shuffle(ordered_files)
    split_names = ("train", "validation", "test")
    raw_counts = [len(ordered_files) * float(split_ratios.get(split_name, 0.0)) for split_name in split_names]
    counts = [int(math.floor(raw_count)) for raw_count in raw_counts]
    remainder = len(ordered_files) - sum(counts)
    ranked_indices = sorted(
        range(len(split_names)),
        key=lambda idx: (raw_counts[idx] - counts[idx], -idx),
        reverse=True,
    )
    for idx in ranked_indices[:remainder]:
        counts[idx] += 1
    partitions: dict[str, tuple[str, ...]] = {}
    cursor = 0
    for split_name, count in zip(split_names, counts):
        partitions[split_name] = tuple(ordered_files[cursor: cursor + count])
        cursor += count
    return partitions


def _merge_weather_file_partitions(
    weather_files: tuple[str, ...],
    explicit_split_files: Mapping[str, tuple[str, ...]],
    split_ratios: Mapping[str, float],
    split_seed: int,
) -> dict[str, tuple[str, ...]]:
    merged = {
        "train_weather_files": tuple(explicit_split_files.get("train_weather_files", ()) or ()),
        "validation_weather_files": tuple(explicit_split_files.get("validation_weather_files", ()) or ()),
        "test_weather_files": tuple(explicit_split_files.get("test_weather_files", ()) or ()),
    }
    assigned_files = {
        path
        for files in merged.values()
        for path in files
    }
    remaining_weather_files = tuple(
        path for path in weather_files
        if path not in assigned_files
    )
    missing_split_names = [
        split_name
        for split_name in ("train", "validation", "test")
        if not merged[f"{split_name}_weather_files"]
    ]
    if not missing_split_names or not remaining_weather_files:
        return merged
    missing_split_ratios = {
        split_name: max(0.0, float(split_ratios.get(split_name, 0.0)))
        for split_name in missing_split_names
    }
    ratio_total = sum(missing_split_ratios.values())
    if ratio_total <= 0.0:
        normalized_missing_ratios = {
            split_name: 1.0 / len(missing_split_names)
            for split_name in missing_split_names
        }
    else:
        normalized_missing_ratios = {
            split_name: ratio / ratio_total
            for split_name, ratio in missing_split_ratios.items()
        }
    partitioned_files = _partition_weather_files(
        weather_files=remaining_weather_files,
        split_ratios={
            "train": normalized_missing_ratios.get("train", 0.0),
            "validation": normalized_missing_ratios.get("validation", 0.0),
            "test": normalized_missing_ratios.get("test", 0.0),
        },
        split_seed=split_seed,
    )
    for split_name in missing_split_names:
        merged[f"{split_name}_weather_files"] = partitioned_files[split_name]
    return merged


def _select_weather_files(protocol_config: Mapping[str, Any]) -> tuple[str, ...]:
    selected = protocol_config.get("selected_weather_files")
    if selected is not None:
        return tuple(selected)
    split_name = protocol_config["split_name"]
    selected = protocol_config.get(f"{split_name}_weather_files", ())
    return tuple(selected)


def _validate_protocol_weather_files(protocol_config: Mapping[str, Any]) -> None:
    split_to_files = {
        "train": tuple(protocol_config.get("train_weather_files", ()) or ()),
        "validation": tuple(protocol_config.get("validation_weather_files", ()) or ()),
        "test": tuple(protocol_config.get("test_weather_files", ()) or ()),
    }
    split_names = ("train", "validation", "test")
    for left_idx, left_name in enumerate(split_names):
        left_files = set(split_to_files[left_name])
        for right_name in split_names[left_idx + 1:]:
            overlap = sorted(left_files.intersection(split_to_files[right_name]))
            if overlap:
                raise ValueError(
                    f"weather split '{left_name}' 与 '{right_name}' 存在重叠文件: {overlap}"
                )
    selected_files = tuple(protocol_config.get("selected_weather_files", ()) or ())
    total_protocol_files = sum(len(files) for files in split_to_files.values())
    if total_protocol_files > 0 and len(selected_files) == 0:
        raise ValueError(
            f"split '{protocol_config.get('split_name', 'train')}' 未分配到任何天气文件，无法形成可复现实验协议"
        )


def resolve_phase1_task_config(config: Any | None = None, **overrides: Any) -> dict[str, Any]:
    observation = _read_value(config, "observation", {})
    action = _read_value(config, "action", {})
    reward = _read_value(config, "reward", {})
    economics = _read_value(config, "economics", {})
    protocol = _read_value(config, "protocol", {})
    protocol_split_name = _normalize_split_name(
        overrides.get(
            "split_name",
            _read_value(protocol, "split_name", "train"),
        )
    )
    protocol_weather_files = _normalize_paths(
        overrides.get(
            "weather_files",
            _read_value(protocol, "weather_files", ()),
        )
    )
    protocol_split_seed = int(
        overrides.get(
            "split_seed",
            _read_value(protocol, "split_seed", 42),
        )
    )
    protocol_split_ratios = _normalize_split_ratios(
        overrides.get(
            "split_ratios",
            _read_value(protocol, "split_ratios", (0.7, 0.15, 0.15)),
        )
    )
    explicit_split_files = {
        "train_weather_files": _normalize_paths(
            overrides.get(
                "train_weather_files",
                _read_value(protocol, "train_weather_files", ()),
            )
        ),
        "validation_weather_files": _normalize_paths(
            overrides.get(
                "validation_weather_files",
                _read_value(protocol, "validation_weather_files", ()),
            )
        ),
        "test_weather_files": _normalize_paths(
            overrides.get(
                "test_weather_files",
                _read_value(protocol, "test_weather_files", ()),
            )
        ),
    }
    if protocol_weather_files:
        explicit_split_files = _merge_weather_file_partitions(
            weather_files=protocol_weather_files,
            explicit_split_files=explicit_split_files,
            split_ratios=protocol_split_ratios,
            split_seed=protocol_split_seed,
        )
    resolved = {
        "crop_name": overrides.get("crop_name", _read_value(config, "crop_name", "wheat")),
        "mode": overrides.get("mode", _read_value(config, "mode", "all")),
        "seed": overrides.get("seed", _read_value(config, "seed", 42)),
        "observation": {
            "include_forecast": bool(
                overrides.get(
                    "include_forecast",
                    _read_value(observation, "include_forecast", False),
                )
            ),
            "forecast_horizons": _normalize_horizons(
                overrides.get(
                    "forecast_horizons",
                    _read_value(observation, "forecast_horizons", (3, 7)),
                )
            ),
        },
        "action": {
            "nitrogen_levels": tuple(
                float(x)
                for x in overrides.get(
                    "nitrogen_levels",
                    _read_value(action, "nitrogen_levels", (0.0, 30.0, 60.0, 90.0)),
                )
            ),
            "irrigation_levels": tuple(
                float(x)
                for x in overrides.get(
                    "irrigation_levels",
                    _read_value(action, "irrigation_levels", (0.0, 10.0, 20.0, 30.0)),
                )
            ),
            "decision_interval_days": int(
                overrides.get(
                    "decision_interval_days",
                    _read_value(action, "decision_interval_days", 7),
                )
            ),
            "fertilizer_budget": float(
                overrides.get(
                    "fertilizer_budget",
                    _read_value(action, "fertilizer_budget", 180.0),
                )
            ),
            "irrigation_budget": float(
                overrides.get(
                    "irrigation_budget",
                    _read_value(action, "irrigation_budget", 240.0),
                )
            ),
            "fertilizer_stage_cutoff": float(
                overrides.get(
                    "fertilizer_stage_cutoff",
                    _read_value(action, "fertilizer_stage_cutoff", 6.0),
                )
            ),
        },
        "reward": {
            "grain_delta_weight": float(
                overrides.get(
                    "grain_delta_weight",
                    _read_value(reward, "grain_delta_weight", 0.001),
                )
            ),
            "terminal_yield_weight": float(
                overrides.get(
                    "terminal_yield_weight",
                    _read_value(reward, "terminal_yield_weight", 0.05),
                )
            ),
            "fertilizer_cost_weight": float(
                overrides.get(
                    "fertilizer_cost_weight",
                    _read_value(reward, "fertilizer_cost_weight", 0.02),
                )
            ),
            "irrigation_cost_weight": float(
                overrides.get(
                    "irrigation_cost_weight",
                    _read_value(reward, "irrigation_cost_weight", 0.01),
                )
            ),
            "leaching_cost_weight": float(
                overrides.get(
                    "leaching_cost_weight",
                    _read_value(reward, "leaching_cost_weight", 0.1),
                )
            ),
            "denitrification_cost_weight": float(
                overrides.get(
                    "denitrification_cost_weight",
                    _read_value(reward, "denitrification_cost_weight", 0.1),
                )
            ),
        },
        "economics": {
            "grain_price": float(
                overrides.get(
                    "grain_price",
                    _read_value(economics, "grain_price", 0.25),
                )
            ),
            "fertilizer_unit_cost": float(
                overrides.get(
                    "fertilizer_unit_cost",
                    _read_value(economics, "fertilizer_unit_cost", 1.2),
                )
            ),
            "irrigation_unit_cost": float(
                overrides.get(
                    "irrigation_unit_cost",
                    _read_value(economics, "irrigation_unit_cost", 0.05),
                )
            ),
        },
        "protocol": {
            "split_name": protocol_split_name,
            "weather_files": protocol_weather_files,
            "split_ratios": (
                protocol_split_ratios["train"],
                protocol_split_ratios["validation"],
                protocol_split_ratios["test"],
            ),
            "split_seed": protocol_split_seed,
            "train_weather_files": explicit_split_files["train_weather_files"],
            "validation_weather_files": explicit_split_files["validation_weather_files"],
            "test_weather_files": explicit_split_files["test_weather_files"],
        },
    }
    resolved["protocol"]["selected_weather_files"] = _select_weather_files(resolved["protocol"])
    resolved["protocol"]["weather_file_counts"] = {
        "train": len(resolved["protocol"]["train_weather_files"]),
        "validation": len(resolved["protocol"]["validation_weather_files"]),
        "test": len(resolved["protocol"]["test_weather_files"]),
    }
    resolved["protocol"]["selected_weather_count"] = len(resolved["protocol"]["selected_weather_files"])
    _validate_protocol_weather_files(resolved["protocol"])
    return resolved


def build_phase1_task_env(config: Any | None = None, **overrides: Any) -> TaskEnv:
    resolved = resolve_phase1_task_config(config, **overrides)
    selected_weather_files = _select_weather_files(resolved["protocol"])
    base_env_kwargs = {
        "auxiliary_file_paths": list(selected_weather_files),
        "split_name": resolved["protocol"]["split_name"],
    }
    if "run_dssat_location" in overrides and overrides["run_dssat_location"] is not None:
        base_env_kwargs["run_dssat_location"] = overrides["run_dssat_location"]
    base_env = DssatBaseEnv(
        crop_name=resolved["crop_name"],
        mode=resolved["mode"],
        seed=resolved["seed"],
        env_kwargs=base_env_kwargs,
    )
    observation_schema = AgronomicObservationSchema(
        crop_name=resolved["crop_name"],
        include_forecast=resolved["observation"]["include_forecast"],
        forecast_horizons=resolved["observation"]["forecast_horizons"],
    )
    action_schema = WeeklyDiscreteActionSchema(
        nitrogen_levels=resolved["action"]["nitrogen_levels"],
        irrigation_levels=resolved["action"]["irrigation_levels"],
        decision_interval_days=resolved["action"]["decision_interval_days"],
        fertilizer_budget=resolved["action"]["fertilizer_budget"],
        irrigation_budget=resolved["action"]["irrigation_budget"],
        fertilizer_stage_cutoff=resolved["action"]["fertilizer_stage_cutoff"],
    )
    reward_schema = YieldCostRewardSchema(
        grain_delta_weight=resolved["reward"]["grain_delta_weight"],
        terminal_yield_weight=resolved["reward"]["terminal_yield_weight"],
        fertilizer_cost_weight=resolved["reward"]["fertilizer_cost_weight"],
        irrigation_cost_weight=resolved["reward"]["irrigation_cost_weight"],
        leaching_cost_weight=resolved["reward"]["leaching_cost_weight"],
        denitrification_cost_weight=resolved["reward"]["denitrification_cost_weight"],
    )
    reporter = EpisodeReporter(
        grain_price=resolved["economics"]["grain_price"],
        fertilizer_unit_cost=resolved["economics"]["fertilizer_unit_cost"],
        irrigation_unit_cost=resolved["economics"]["irrigation_unit_cost"],
    )
    forecast_provider = (
        WeatherWindowForecastProvider(horizons=resolved["observation"]["forecast_horizons"])
        if resolved["observation"]["include_forecast"]
        else NullForecastProvider()
    )
    return TaskEnv(
        base_env=base_env,
        observation_schema=observation_schema,
        action_schema=action_schema,
        reward_schema=reward_schema,
        episode_reporter=reporter,
        forecast_provider=forecast_provider,
    )
