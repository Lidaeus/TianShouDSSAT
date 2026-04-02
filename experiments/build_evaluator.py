from __future__ import annotations

import csv
import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .build_env import build_phase1_task_env


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(numeric_value):
        return 0.0
    return numeric_value


def _ordered_seed_items(
    results_by_seed: Mapping[int | None, "EvaluationResult"],
) -> list[tuple[int | None, "EvaluationResult"]]:
    return sorted(
        results_by_seed.items(),
        key=lambda item: (-1 if item[0] is None else 0, -1 if item[0] is None else int(item[0])),
    )


def _normalize_seed_sequence(
    seeds: list[int | None] | tuple[int | None, ...] | None,
) -> list[int | None]:
    if not seeds:
        return [None]
    normalized: list[int | None] = []
    seen: set[int | None] = set()
    for seed in seeds:
        normalized_seed = None if seed is None else int(seed)
        if normalized_seed in seen:
            continue
        seen.add(normalized_seed)
        normalized.append(normalized_seed)
    return normalized or [None]


def _aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, float | int]:
    if not reports:
        return {
            "episodes": 0,
            "weather_ids": [],
            "unique_weather_ids": [],
            "unique_weather_count": 0,
            "mean_task_return": 0.0,
            "mean_native_return": 0.0,
            "mean_total_fertilizer": 0.0,
            "mean_total_irrigation": 0.0,
            "mean_final_grain_weight": 0.0,
            "mean_final_biomass": 0.0,
            "mean_final_leaching": 0.0,
            "mean_final_denitrification": 0.0,
            "mean_nitrogen_use_efficiency": 0.0,
            "mean_water_use_efficiency": 0.0,
            "mean_gross_revenue": 0.0,
            "mean_fertilizer_input_cost": 0.0,
            "mean_irrigation_input_cost": 0.0,
            "mean_total_input_cost": 0.0,
            "mean_profit": 0.0,
        }
    weather_ids = [
        str(weather_id)
        for weather_id in (report.get("weather_id") for report in reports)
        if weather_id not in (None, "")
    ]
    return {
        "episodes": len(reports),
        "weather_ids": weather_ids,
        "unique_weather_ids": sorted(set(weather_ids)),
        "unique_weather_count": len(set(weather_ids)),
        "mean_task_return": float(np.mean([_safe_float(report.get("task_return")) for report in reports])),
        "mean_native_return": float(np.mean([_safe_float(report.get("native_return")) for report in reports])),
        "mean_total_fertilizer": float(np.mean([_safe_float(report.get("total_fertilizer")) for report in reports])),
        "mean_total_irrigation": float(np.mean([_safe_float(report.get("total_irrigation")) for report in reports])),
        "mean_final_grain_weight": float(np.mean([_safe_float(report.get("final_grain_weight")) for report in reports])),
        "mean_final_biomass": float(np.mean([_safe_float(report.get("final_biomass")) for report in reports])),
        "mean_final_leaching": float(np.mean([_safe_float(report.get("final_leaching")) for report in reports])),
        "mean_final_denitrification": float(np.mean([_safe_float(report.get("final_denitrification")) for report in reports])),
        "mean_nitrogen_use_efficiency": float(np.mean([_safe_float(report.get("nitrogen_use_efficiency")) for report in reports])),
        "mean_water_use_efficiency": float(np.mean([_safe_float(report.get("water_use_efficiency")) for report in reports])),
        "mean_gross_revenue": float(np.mean([_safe_float(report.get("gross_revenue")) for report in reports])),
        "mean_fertilizer_input_cost": float(np.mean([_safe_float(report.get("fertilizer_input_cost")) for report in reports])),
        "mean_irrigation_input_cost": float(np.mean([_safe_float(report.get("irrigation_input_cost")) for report in reports])),
        "mean_total_input_cost": float(np.mean([_safe_float(report.get("total_input_cost")) for report in reports])),
        "mean_profit": float(np.mean([_safe_float(report.get("profit")) for report in reports])),
    }


def _summarize_seed_aggregates(results_by_seed: Mapping[int | None, "EvaluationResult"]) -> dict[str, Any]:
    ordered_seed_items = _ordered_seed_items(results_by_seed)
    metric_names = sorted(
        {
            metric_name
            for _, result in ordered_seed_items
            for metric_name, metric_value in result.aggregate.items()
            if isinstance(metric_value, (int, float))
        }
    )
    metrics: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        values = np.asarray(
            [
                _safe_float(result.aggregate.get(metric_name))
                for _, result in ordered_seed_items
            ],
            dtype=np.float64,
        )
        stderr = float(np.std(values) / np.sqrt(len(values))) if len(values) > 0 else 0.0
        metrics[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "stderr": stderr,
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return {
        "seed_count": len(ordered_seed_items),
        "seeds": [seed for seed, _ in ordered_seed_items],
        "metrics": metrics,
    }


def _aggregate_metric_mean(aggregate: Mapping[str, Any], metric_name: str) -> float:
    direct_value = aggregate.get(metric_name)
    if isinstance(direct_value, (int, float)):
        return _safe_float(direct_value)
    metrics = aggregate.get("metrics", {})
    if isinstance(metrics, Mapping):
        metric_summary = metrics.get(metric_name, {})
        if isinstance(metric_summary, Mapping):
            metric_mean = metric_summary.get("mean")
            if isinstance(metric_mean, (int, float)):
                return _safe_float(metric_mean)
    return 0.0


def _normalized_multiseed_aggregate(result: "MultiSeedEvaluationResult") -> dict[str, Any]:
    summary = _summarize_seed_aggregates(result.by_seed)
    if not result.aggregate:
        return summary
    normalized = dict(summary)
    normalized.update(result.aggregate)
    summary_metrics = dict(summary.get("metrics", {}))
    aggregate_metrics = result.aggregate.get("metrics", {})
    if isinstance(aggregate_metrics, Mapping):
        for metric_name, metric_summary in aggregate_metrics.items():
            combined_metric_summary = dict(summary_metrics.get(metric_name, {}))
            if isinstance(metric_summary, Mapping):
                combined_metric_summary.update(metric_summary)
            summary_metrics[metric_name] = combined_metric_summary
    normalized["metrics"] = summary_metrics
    return normalized


def _build_leaderboard_entry(actor_name: str, aggregate: Mapping[str, Any]) -> dict[str, float | int | str]:
    return {
        "actor": actor_name,
        "episodes": int(_aggregate_metric_mean(aggregate, "episodes")),
        "mean_task_return": _aggregate_metric_mean(aggregate, "mean_task_return"),
        "mean_final_grain_weight": _aggregate_metric_mean(aggregate, "mean_final_grain_weight"),
        "mean_total_fertilizer": _aggregate_metric_mean(aggregate, "mean_total_fertilizer"),
        "mean_total_irrigation": _aggregate_metric_mean(aggregate, "mean_total_irrigation"),
        "mean_nitrogen_use_efficiency": _aggregate_metric_mean(aggregate, "mean_nitrogen_use_efficiency"),
        "mean_water_use_efficiency": _aggregate_metric_mean(aggregate, "mean_water_use_efficiency"),
        "mean_profit": _aggregate_metric_mean(aggregate, "mean_profit"),
    }


def build_actor_leaderboard(
    results: Mapping[str, "EvaluationResult" | "MultiSeedEvaluationResult"],
) -> list[dict[str, float | int | str]]:
    leaderboard = [
        _build_leaderboard_entry(
            actor_name=actor_name,
            aggregate=_result_aggregate_for_leaderboard(result),
        )
        for actor_name, result in results.items()
    ]
    return sorted(
        leaderboard,
        key=lambda row: (
            -float(row["mean_final_grain_weight"]),
            -float(row["mean_task_return"]),
            str(row["actor"]),
        ),
    )


def _serialize_seed_label(seed: int | None) -> str:
    return "default" if seed is None else str(int(seed))


def _build_actor_summary_rows(
    results: Mapping[str, "EvaluationResult" | "MultiSeedEvaluationResult"],
) -> list[dict[str, float | int | str | None]]:
    leaderboard = build_actor_leaderboard(results)
    rows: list[dict[str, float | int | str | None]] = []
    for rank, leaderboard_entry in enumerate(leaderboard, start=1):
        actor_name = str(leaderboard_entry["actor"])
        result = results[actor_name]
        aggregate = _result_aggregate_for_leaderboard(result)
        rows.append(
            {
                "rank": rank,
                "actor": actor_name,
                "seed_count": int(aggregate.get("seed_count", 1)),
                "episodes": int(_aggregate_metric_mean(aggregate, "episodes")),
                "unique_weather_count": int(_aggregate_metric_mean(aggregate, "unique_weather_count")),
                "mean_task_return": _aggregate_metric_mean(aggregate, "mean_task_return"),
                "mean_profit": _aggregate_metric_mean(aggregate, "mean_profit"),
                "mean_final_grain_weight": _aggregate_metric_mean(aggregate, "mean_final_grain_weight"),
                "mean_total_fertilizer": _aggregate_metric_mean(aggregate, "mean_total_fertilizer"),
                "mean_total_irrigation": _aggregate_metric_mean(aggregate, "mean_total_irrigation"),
                "mean_nitrogen_use_efficiency": _aggregate_metric_mean(aggregate, "mean_nitrogen_use_efficiency"),
                "mean_water_use_efficiency": _aggregate_metric_mean(aggregate, "mean_water_use_efficiency"),
            }
        )
    return rows


def _build_seed_summary_row(actor_name: str, seed: int | None, aggregate: Mapping[str, Any]) -> dict[str, float | int | str | None]:
    return {
        "actor": actor_name,
        "seed": seed,
        "seed_label": _serialize_seed_label(seed),
        "episodes": int(_aggregate_metric_mean(aggregate, "episodes")),
        "unique_weather_count": int(_aggregate_metric_mean(aggregate, "unique_weather_count")),
        "mean_task_return": _aggregate_metric_mean(aggregate, "mean_task_return"),
        "mean_profit": _aggregate_metric_mean(aggregate, "mean_profit"),
        "mean_final_grain_weight": _aggregate_metric_mean(aggregate, "mean_final_grain_weight"),
        "mean_total_fertilizer": _aggregate_metric_mean(aggregate, "mean_total_fertilizer"),
        "mean_total_irrigation": _aggregate_metric_mean(aggregate, "mean_total_irrigation"),
        "mean_nitrogen_use_efficiency": _aggregate_metric_mean(aggregate, "mean_nitrogen_use_efficiency"),
        "mean_water_use_efficiency": _aggregate_metric_mean(aggregate, "mean_water_use_efficiency"),
    }


def _build_episode_row(
    actor_name: str,
    report: Mapping[str, Any],
    episode_index: int,
    seed: int | None = None,
) -> dict[str, float | int | str | None]:
    return {
        "actor": actor_name,
        "seed": seed,
        "seed_label": _serialize_seed_label(seed),
        "episode_index": int(episode_index),
        "crop_name": report.get("crop_name"),
        "split_name": report.get("split_name"),
        "weather_id": report.get("weather_id"),
        "episode_steps": int(_safe_float(report.get("episode_steps"))),
        "task_return": _safe_float(report.get("task_return")),
        "native_return": _safe_float(report.get("native_return")),
        "total_fertilizer": _safe_float(report.get("total_fertilizer")),
        "total_irrigation": _safe_float(report.get("total_irrigation")),
        "final_grain_weight": _safe_float(report.get("final_grain_weight")),
        "final_biomass": _safe_float(report.get("final_biomass")),
        "final_leaching": _safe_float(report.get("final_leaching")),
        "final_denitrification": _safe_float(report.get("final_denitrification")),
        "nitrogen_use_efficiency": _safe_float(report.get("nitrogen_use_efficiency")),
        "water_use_efficiency": _safe_float(report.get("water_use_efficiency")),
        "gross_revenue": _safe_float(report.get("gross_revenue")),
        "fertilizer_input_cost": _safe_float(report.get("fertilizer_input_cost")),
        "irrigation_input_cost": _safe_float(report.get("irrigation_input_cost")),
        "total_input_cost": _safe_float(report.get("total_input_cost")),
        "profit": _safe_float(report.get("profit")),
    }


def build_evaluation_tables(
    results: Mapping[str, "EvaluationResult" | "MultiSeedEvaluationResult"],
) -> dict[str, list[dict[str, float | int | str | None]]]:
    actor_summary_rows = _build_actor_summary_rows(results)
    seed_summary_rows: list[dict[str, float | int | str | None]] = []
    episode_rows: list[dict[str, float | int | str | None]] = []
    for actor_name, result in results.items():
        if isinstance(result, MultiSeedEvaluationResult):
            for seed, seed_result in _ordered_seed_items(result.by_seed):
                seed_summary_rows.append(
                    _build_seed_summary_row(
                        actor_name=actor_name,
                        seed=seed,
                        aggregate=seed_result.aggregate,
                    )
                )
                for episode_index, report in enumerate(seed_result.reports, start=1):
                    episode_rows.append(
                        _build_episode_row(
                            actor_name=actor_name,
                            report=report,
                            episode_index=episode_index,
                            seed=seed,
                        )
                    )
        else:
            seed_summary_rows.append(
                _build_seed_summary_row(
                    actor_name=actor_name,
                    seed=None,
                    aggregate=result.aggregate,
                )
            )
            for episode_index, report in enumerate(result.reports, start=1):
                episode_rows.append(
                    _build_episode_row(
                        actor_name=actor_name,
                        report=report,
                        episode_index=episode_index,
                    )
                )
    tables = {
        "actor_summary": actor_summary_rows,
        "seed_summary": seed_summary_rows,
        "episode_reports": episode_rows,
    }
    return tables


def _normalize_tabular_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False)


def _collect_table_fieldnames(rows: list[Mapping[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    return fieldnames


def _write_tabular_artifact_rows(rows: list[Mapping[str, Any]], json_path: Path, csv_path: Path) -> dict[str, Any]:
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    fieldnames = _collect_table_fieldnames(rows)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for row in rows:
                writer.writerow({field: _normalize_tabular_value(row.get(field)) for field in fieldnames})
    return {
        "json": json_path.name,
        "csv": csv_path.name,
        "row_count": len(rows),
        "columns": fieldnames,
    }


def _metric_from_leaderboard_row(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, (int, float)):
        return _safe_float(value)
    return 0.0


def _artifact_run_record(payload: Mapping[str, Any], artifact_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    config = payload.get("config", {})
    protocol = config.get("protocol", {}) if isinstance(config, Mapping) else {}
    leaderboard = payload.get("leaderboard", [])
    top_actor = None
    top_actor_metrics: Mapping[str, Any] = {}
    if isinstance(leaderboard, list) and leaderboard:
        first_row = leaderboard[0]
        if isinstance(first_row, Mapping):
            top_actor = first_row.get("actor")
            top_actor_metrics = first_row
    return {
        "run_name": artifact_dir.name,
        "artifact_dir": str(artifact_dir),
        "created_at": manifest.get("created_at"),
        "algorithm": config.get("algorithm_name", "unknown") if isinstance(config, Mapping) else "unknown",
        "benchmark_preset": config.get("benchmark_preset") if isinstance(config, Mapping) else None,
        "crop": config.get("crop") if isinstance(config, Mapping) else None,
        "seed": config.get("seed") if isinstance(config, Mapping) else None,
        "split_name": protocol.get("split_name") if isinstance(protocol, Mapping) else None,
        "evaluation_split_name": protocol.get("evaluation_split_name") if isinstance(protocol, Mapping) else None,
        "evaluation_base_seeds": config.get("evaluation_base_seeds") if isinstance(config, Mapping) else None,
        "baseline_actors": config.get("baseline_actors") if isinstance(config, Mapping) else None,
        "top_actor": top_actor,
        "top_mean_task_return": _metric_from_leaderboard_row(top_actor_metrics, "mean_task_return"),
        "top_mean_profit": _metric_from_leaderboard_row(top_actor_metrics, "mean_profit"),
        "top_mean_final_grain_weight": _metric_from_leaderboard_row(top_actor_metrics, "mean_final_grain_weight"),
        "top_seed_count": int(_metric_from_leaderboard_row(top_actor_metrics, "seed_count")),
        "top_episodes": int(_metric_from_leaderboard_row(top_actor_metrics, "episodes")),
        "manifest_path": str(artifact_dir / "benchmark_artifacts_manifest.json"),
    }


def _write_cross_run_summary(index_payload: Mapping[str, Any], index_path: Path) -> dict[str, Any]:
    runs = index_payload.get("runs", [])
    summary_rows = [dict(run) for run in runs if isinstance(run, Mapping)]
    summary_json_path = index_path.parent / "benchmark_run_summary.json"
    summary_csv_path = index_path.parent / "benchmark_run_summary.csv"
    return _write_tabular_artifact_rows(summary_rows, json_path=summary_json_path, csv_path=summary_csv_path)


def _write_benchmark_runs_index(payload: Mapping[str, Any], artifact_dir: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    index_path = artifact_dir.parent / "benchmark_runs_index.json"
    index_payload: dict[str, Any] = {"runs": []}
    if index_path.exists():
        try:
            with index_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, Mapping) and isinstance(loaded.get("runs"), list):
                index_payload = {"runs": list(loaded.get("runs", []))}
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            index_payload = {"runs": []}
    run_record = _artifact_run_record(payload, artifact_dir=artifact_dir, manifest=manifest)
    existing_runs = [
        entry
        for entry in index_payload["runs"]
        if not (isinstance(entry, Mapping) and entry.get("artifact_dir") == run_record["artifact_dir"])
    ]
    existing_runs.append(run_record)
    existing_runs.sort(key=lambda entry: str(entry.get("created_at", "")))
    index_payload = {"runs": existing_runs}
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(index_payload, handle, ensure_ascii=False, indent=2)
    summary_artifacts = _write_cross_run_summary(index_payload, index_path=index_path)
    return {
        "index_path": str(index_path),
        "summary_json_path": str(index_path.parent / summary_artifacts["json"]),
        "summary_csv_path": str(index_path.parent / summary_artifacts["csv"]),
    }


def write_benchmark_artifacts(payload: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_dir / "evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    table_artifacts: dict[str, dict[str, Any]] = {}
    tables = payload.get("tables", {})
    if isinstance(tables, Mapping):
        for table_name, table_rows in tables.items():
            rows = list(table_rows) if isinstance(table_rows, list) else []
            json_path = artifact_dir / f"{table_name}.json"
            csv_path = artifact_dir / f"{table_name}.csv"
            table_artifacts[str(table_name)] = _write_tabular_artifact_rows(rows, json_path=json_path, csv_path=csv_path)

    config = payload.get("config", {})
    protocol = config.get("protocol", {}) if isinstance(config, Mapping) else {}
    manifest = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run_name": artifact_dir.name,
        "artifact_dir": str(artifact_dir),
        "algorithm_name": config.get("algorithm_name", "unknown") if isinstance(config, Mapping) else "unknown",
        "benchmark_preset": config.get("benchmark_preset") if isinstance(config, Mapping) else None,
        "crop": config.get("crop") if isinstance(config, Mapping) else None,
        "seed": config.get("seed") if isinstance(config, Mapping) else None,
        "split_name": protocol.get("split_name") if isinstance(protocol, Mapping) else None,
        "evaluation_split_name": protocol.get("evaluation_split_name") if isinstance(protocol, Mapping) else None,
        "evaluation_base_seeds": config.get("evaluation_base_seeds") if isinstance(config, Mapping) else None,
        "baseline_actors": config.get("baseline_actors") if isinstance(config, Mapping) else None,
        "summary_json": summary_path.name,
        "tables": table_artifacts,
    }
    manifest_path = artifact_dir / "benchmark_artifacts_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    batch_outputs = _write_benchmark_runs_index(payload, artifact_dir=artifact_dir, manifest=manifest)
    return {
        "summary_path": str(summary_path),
        "manifest_path": str(manifest_path),
        "batch_index_path": batch_outputs["index_path"],
        "cross_run_summary_json_path": batch_outputs["summary_json_path"],
        "cross_run_summary_csv_path": batch_outputs["summary_csv_path"],
        "manifest": manifest,
    }


def serialize_evaluation_suite(results: Mapping[str, "EvaluationResult"]) -> dict[str, Any]:
    return {
        "actors": {
            actor_name: {
                "aggregate": result.aggregate,
                "reports": result.reports,
            }
            for actor_name, result in results.items()
        },
        "leaderboard": build_actor_leaderboard(results),
        "tables": build_evaluation_tables(results),
    }


def serialize_multiseed_evaluation_suite(
    results: Mapping[str, "MultiSeedEvaluationResult"],
) -> dict[str, Any]:
    return {
        "actors": {
            actor_name: {
                "aggregate": _normalized_multiseed_aggregate(result),
                "by_seed": {
                    "default" if seed is None else str(seed): {
                        "aggregate": seed_result.aggregate,
                        "reports": seed_result.reports,
                    }
                    for seed, seed_result in _ordered_seed_items(result.by_seed)
                },
            }
            for actor_name, result in results.items()
        },
        "leaderboard": build_actor_leaderboard(results),
        "tables": build_evaluation_tables(results),
    }


def _nearest_level(levels: list[float], target: float) -> float:
    if not levels:
        return 0.0
    return min(levels, key=lambda level: (abs(level - target), level))


def _encode_discrete_grid_action(action_schema_info: Mapping[str, Any], nitrogen: float, irrigation: float) -> int:
    nitrogen_levels = [float(level) for level in action_schema_info.get("nitrogen_levels", (0.0,))]
    irrigation_levels = [float(level) for level in action_schema_info.get("irrigation_levels", (0.0,))]
    selected_nitrogen = _nearest_level(nitrogen_levels, float(nitrogen))
    selected_irrigation = _nearest_level(irrigation_levels, float(irrigation))
    n_idx = nitrogen_levels.index(selected_nitrogen)
    i_idx = irrigation_levels.index(selected_irrigation)
    return n_idx * len(irrigation_levels) + i_idx


def _feature_value(observation: Any, info: dict[str, Any], feature_name: str, scale: float = 1.0) -> float:
    features = info.get("observation_features", [])
    if not isinstance(features, list) or feature_name not in features:
        return 0.0
    feature_index = features.index(feature_name)
    if not isinstance(observation, np.ndarray) or feature_index >= len(observation):
        return 0.0
    return _safe_float(observation[feature_index]) * scale


def _normalize_reward_value(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, np.ndarray):
        return float(np.sum(value))
    if isinstance(value, (list, tuple)):
        return float(sum(_safe_float(item) for item in value))
    return _safe_float(value)


def _sanitize_action(action: Any, info: Mapping[str, Any], action_space: Any) -> Any:
    if not hasattr(action_space, "n"):
        return action
    action_count = int(getattr(action_space, "n", 0) or 0)
    if action_count <= 0:
        return action
    try:
        action_index = int(action)
    except (TypeError, ValueError):
        action_index = 0
    if action_index < 0 or action_index >= action_count:
        action_index = 0
    action_mask = info.get("action_mask")
    if action_mask is None:
        return action_index
    mask = np.asarray(action_mask).reshape(-1)
    if mask.size < action_count:
        return action_index
    valid_indices = np.flatnonzero(mask > 0)
    if valid_indices.size == 0:
        return 0
    if mask[action_index] > 0:
        return action_index
    return int(valid_indices[0])


def _fallback_report_from_info(info: Mapping[str, Any]) -> dict[str, Any]:
    management_state = info.get("management_state")
    total_fertilizer = _safe_float(
        getattr(management_state, "cumulative_fertilizer", None)
        if management_state is not None else info.get("total_fertilizer")
    )
    if total_fertilizer == 0.0:
        total_fertilizer = _safe_float((info.get("applied_action") or {}).get("anfer"))
    total_irrigation = _safe_float(
        getattr(management_state, "cumulative_irrigation", None)
        if management_state is not None else info.get("total_irrigation")
    )
    if total_irrigation == 0.0:
        total_irrigation = _safe_float((info.get("applied_action") or {}).get("amir"))
    episode_steps = int(_safe_float(
        getattr(management_state, "calendar_step", None)
        if management_state is not None else info.get("episode_steps", 0)
    ))
    if episode_steps <= 0:
        episode_steps = 1
    return {
        "crop_name": str(info.get("crop_name", "unknown")),
        "split_name": str(info.get("split_name", "unknown")),
        "weather_id": info.get("weather_id"),
        "episode_steps": episode_steps,
        "task_return": _safe_float(info.get("task_reward")),
        "native_return": _normalize_reward_value(info.get("native_reward")),
        "total_fertilizer": total_fertilizer,
        "total_irrigation": total_irrigation,
        "final_grain_weight": _safe_float(info.get("grnwt")),
        "final_biomass": _safe_float(info.get("topwt")),
        "final_leaching": _safe_float(info.get("tleachd")),
        "final_denitrification": _safe_float(info.get("tnoxd")),
        "nitrogen_use_efficiency": _safe_float(info.get("grnwt")) / total_fertilizer if total_fertilizer > 0 else 0.0,
        "water_use_efficiency": _safe_float(info.get("grnwt")) / total_irrigation if total_irrigation > 0 else 0.0,
        "gross_revenue": _safe_float(info.get("gross_revenue")),
        "fertilizer_input_cost": _safe_float(info.get("fertilizer_input_cost")),
        "irrigation_input_cost": _safe_float(info.get("irrigation_input_cost")),
        "total_input_cost": _safe_float(info.get("total_input_cost")),
        "profit": _safe_float(info.get("profit")),
    }


class ZeroActionActor:
    def compute_action(self, observation: Any, info: dict[str, Any], action_space: Any) -> int:
        return 0


class ReactiveHeuristicActor:
    def __init__(
        self,
        irrigation_trigger: float = 0.55,
        root_zone_trigger: float = 0.45,
        nitrogen_stress_trigger: float = 0.35,
        nitrogen_stage_cutoff: float | None = None,
        nitrogen_dose: float = 30.0,
        irrigation_dose: float = 20.0,
    ) -> None:
        self.irrigation_trigger = float(irrigation_trigger)
        self.root_zone_trigger = float(root_zone_trigger)
        self.nitrogen_stress_trigger = float(nitrogen_stress_trigger)
        self.nitrogen_stage_cutoff = None if nitrogen_stage_cutoff is None else float(nitrogen_stage_cutoff)
        self.nitrogen_dose = float(nitrogen_dose)
        self.irrigation_dose = float(irrigation_dose)

    def compute_action(self, observation: Any, info: dict[str, Any], action_space: Any) -> int:
        action_schema_info = info.get("action_schema", {})
        if action_schema_info.get("type") != "weekly_discrete":
            return 0
        management_state = info.get("management_state")
        decision_interval = int(action_schema_info.get("decision_interval_days", 7))
        if management_state is not None and int(getattr(management_state, "calendar_step", 0)) > 0:
            if int(getattr(management_state, "calendar_step", 0)) % max(1, decision_interval) != 0:
                return 0
        nitrogen = 0.0
        irrigation = 0.0
        current_stage = _safe_float(info.get("istage"))
        if current_stage == 0.0:
            current_stage = _feature_value(observation, info, "istage_norm", scale=8.0)
        effective_cutoff = self.nitrogen_stage_cutoff
        if effective_cutoff is None:
            effective_cutoff = _safe_float(action_schema_info.get("fertilizer_stage_cutoff", 6.0))
        n_stress = _safe_float(info.get("nstres"))
        if n_stress == 0.0:
            n_stress = _feature_value(observation, info, "nstres_norm")
        swfac = _safe_float(info.get("swfac"))
        if swfac == 0.0:
            swfac = _feature_value(observation, info, "swfac_norm")
        root_zone_sw_ratio = _safe_float(info.get("root_zone_sw_ratio"))
        if root_zone_sw_ratio == 0.0:
            root_zone_sw_ratio = _feature_value(observation, info, "root_zone_sw_ratio")
        if management_state is None or float(getattr(management_state, "remaining_fertilizer", 0.0)) > 0:
            if current_stage < effective_cutoff and n_stress >= self.nitrogen_stress_trigger:
                nitrogen = self.nitrogen_dose
        if management_state is None or float(getattr(management_state, "remaining_irrigation", 0.0)) > 0:
            if swfac <= self.irrigation_trigger or root_zone_sw_ratio <= self.root_zone_trigger:
                irrigation = self.irrigation_dose
        return _encode_discrete_grid_action(action_schema_info, nitrogen=nitrogen, irrigation=irrigation)


class RuleBasedActor:
    def __init__(
        self,
        irrigation_trigger: float = 0.5,
        root_zone_trigger: float = 0.35,
        nitrogen_stress_trigger: float = 0.45,
        min_crop_stage: float = 1.0,
        nitrogen_stage_cutoff: float | None = None,
        min_fertilizer_gap_days: float = 10.0,
        min_irrigation_gap_days: float = 7.0,
        nitrogen_dose: float = 30.0,
        irrigation_dose: float = 20.0,
    ) -> None:
        self.irrigation_trigger = float(irrigation_trigger)
        self.root_zone_trigger = float(root_zone_trigger)
        self.nitrogen_stress_trigger = float(nitrogen_stress_trigger)
        self.min_crop_stage = float(min_crop_stage)
        self.nitrogen_stage_cutoff = None if nitrogen_stage_cutoff is None else float(nitrogen_stage_cutoff)
        self.min_fertilizer_gap_days = float(min_fertilizer_gap_days)
        self.min_irrigation_gap_days = float(min_irrigation_gap_days)
        self.nitrogen_dose = float(nitrogen_dose)
        self.irrigation_dose = float(irrigation_dose)

    def compute_action(self, observation: Any, info: dict[str, Any], action_space: Any) -> int:
        action_schema_info = info.get("action_schema", {})
        if action_schema_info.get("type") != "weekly_discrete":
            return 0
        management_state = info.get("management_state")
        decision_interval = int(action_schema_info.get("decision_interval_days", 7))
        calendar_step = int(getattr(management_state, "calendar_step", 0)) if management_state is not None else 0
        if calendar_step > 0 and calendar_step % max(1, decision_interval) != 0:
            return 0
        current_stage = _safe_float(info.get("istage"))
        if current_stage == 0.0:
            current_stage = _feature_value(observation, info, "istage_norm", scale=8.0)
        if current_stage < self.min_crop_stage:
            return 0
        n_stress = _safe_float(info.get("nstres"))
        if n_stress == 0.0:
            n_stress = _feature_value(observation, info, "nstres_norm")
        swfac = _safe_float(info.get("swfac"))
        if swfac == 0.0:
            swfac = _feature_value(observation, info, "swfac_norm")
        root_zone_sw_ratio = _safe_float(info.get("root_zone_sw_ratio"))
        if root_zone_sw_ratio == 0.0:
            root_zone_sw_ratio = _feature_value(observation, info, "root_zone_sw_ratio")
        remaining_fertilizer = float(getattr(management_state, "remaining_fertilizer", 0.0)) if management_state is not None else float("inf")
        remaining_irrigation = float(getattr(management_state, "remaining_irrigation", 0.0)) if management_state is not None else float("inf")
        days_since_fertilizer = float(getattr(management_state, "days_since_fertilizer", self.min_fertilizer_gap_days)) if management_state is not None else self.min_fertilizer_gap_days
        days_since_irrigation = float(getattr(management_state, "days_since_irrigation", self.min_irrigation_gap_days)) if management_state is not None else self.min_irrigation_gap_days
        effective_cutoff = self.nitrogen_stage_cutoff
        if effective_cutoff is None:
            effective_cutoff = _safe_float(action_schema_info.get("fertilizer_stage_cutoff", 6.0))
        nitrogen = 0.0
        irrigation = 0.0
        if remaining_fertilizer > 0.0 and days_since_fertilizer >= self.min_fertilizer_gap_days:
            if current_stage < effective_cutoff and n_stress >= self.nitrogen_stress_trigger:
                nitrogen = self.nitrogen_dose
        if remaining_irrigation > 0.0 and days_since_irrigation >= self.min_irrigation_gap_days:
            if swfac <= self.irrigation_trigger or root_zone_sw_ratio <= self.root_zone_trigger:
                irrigation = self.irrigation_dose
        return _encode_discrete_grid_action(action_schema_info, nitrogen=nitrogen, irrigation=irrigation)


class FixedCalendarActor:
    def __init__(
        self,
        fertilizer_days: tuple[int, ...] = (0, 14, 28),
        irrigation_days: tuple[int, ...] = (0, 7, 14, 21, 28, 35, 42, 49),
        fertilizer_dose: float = 30.0,
        irrigation_dose: float = 10.0,
    ) -> None:
        self.fertilizer_days = tuple(sorted({int(day) for day in fertilizer_days if int(day) >= 0}))
        self.irrigation_days = tuple(sorted({int(day) for day in irrigation_days if int(day) >= 0}))
        self.fertilizer_dose = float(fertilizer_dose)
        self.irrigation_dose = float(irrigation_dose)

    def compute_action(self, observation: Any, info: dict[str, Any], action_space: Any) -> int:
        action_schema_info = info.get("action_schema", {})
        if action_schema_info.get("type") != "weekly_discrete":
            return 0
        management_state = info.get("management_state")
        calendar_step = int(getattr(management_state, "calendar_step", 0)) if management_state is not None else 0
        nitrogen = self.fertilizer_dose if calendar_step in self.fertilizer_days else 0.0
        irrigation = self.irrigation_dose if calendar_step in self.irrigation_days else 0.0
        return _encode_discrete_grid_action(action_schema_info, nitrogen=nitrogen, irrigation=irrigation)


def build_baseline_actor(name: str, **kwargs: Any) -> Any:
    normalized_name = name.strip().lower()
    if normalized_name in {"zero", "no_input", "null"}:
        return ZeroActionActor()
    if normalized_name in {"reactive", "heuristic", "threshold"}:
        return ReactiveHeuristicActor(**kwargs)
    if normalized_name in {"rule", "rule_based", "rule-based", "stable_rule", "stable_reactive"}:
        return RuleBasedActor(**kwargs)
    if normalized_name in {"calendar", "fixed_calendar", "fixed-calendar"}:
        return FixedCalendarActor(**kwargs)
    raise ValueError(f"不支持的 baseline actor: {name}")


def _resolve_action(
    actor: Callable[..., Any] | Any | None,
    observation: Any,
    info: dict[str, Any],
    action_space: Any,
) -> Any:
    if actor is None:
        return _sanitize_action(action_space.sample(), info, action_space)
    if hasattr(actor, "compute_action"):
        try:
            action = actor.compute_action(observation=observation, info=info, action_space=action_space)
        except TypeError:
            action = actor.compute_action(obs=observation, info=info)
        return _sanitize_action(action, info, action_space)
    if callable(actor):
        return _sanitize_action(actor(observation, info, action_space), info, action_space)
    raise TypeError("actor 必须为 None、可调用对象，或实现 compute_action 方法")


@dataclass
class EvaluationResult:
    reports: list[dict[str, Any]]
    aggregate: dict[str, float | int]


@dataclass
class MultiSeedEvaluationResult:
    by_seed: dict[int | None, EvaluationResult]
    aggregate: dict[str, Any]


def _result_aggregate_for_leaderboard(
    result: "EvaluationResult" | "MultiSeedEvaluationResult",
) -> Mapping[str, Any]:
    if isinstance(result, MultiSeedEvaluationResult):
        return _normalized_multiseed_aggregate(result)
    return result.aggregate


class EpisodeEvaluator:
    def __init__(self, env_factory: Callable[[], Any], evaluation_seed_stride: int = 1000) -> None:
        self.env_factory = env_factory
        self.evaluation_seed_stride = int(evaluation_seed_stride)

    def evaluate(
        self,
        actor: Callable[..., Any] | Any | None = None,
        num_episodes: int = 3,
        seed: int | None = None,
    ) -> EvaluationResult:
        reports: list[dict[str, Any]] = []
        for episode_idx in range(int(num_episodes)):
            env = self.env_factory()
            try:
                episode_seed = None
                if seed is not None:
                    episode_seed = int(seed) + episode_idx * self.evaluation_seed_stride
                observation, info = env.reset(seed=episode_seed)
                terminated = False
                truncated = False
                while not (terminated or truncated):
                    action = _resolve_action(actor, observation, info, env.action_space)
                    observation, _, terminated, truncated, info = env.step(action)
                report = dict(info.get("episode_report", {}))
                if not report:
                    report = dict(info.get("episode_summary", {}))
                if not report:
                    report = _fallback_report_from_info(info)
                reports.append(report)
            finally:
                env.close()
        return EvaluationResult(reports=reports, aggregate=_aggregate_reports(reports))

    def evaluate_actor_suite(
        self,
        actors: Mapping[str, Callable[..., Any] | Any | None],
        num_episodes: int = 3,
        seed: int | None = None,
    ) -> dict[str, EvaluationResult]:
        results: dict[str, EvaluationResult] = {}
        for actor_name, actor in actors.items():
            results[actor_name] = self.evaluate(actor=actor, num_episodes=num_episodes, seed=seed)
        return results

    def evaluate_over_seeds(
        self,
        actor: Callable[..., Any] | Any | None = None,
        seeds: list[int | None] | tuple[int | None, ...] = (None,),
        num_episodes: int = 3,
    ) -> MultiSeedEvaluationResult:
        normalized_seeds = _normalize_seed_sequence(seeds)
        results_by_seed: dict[int | None, EvaluationResult] = {}
        for seed in normalized_seeds:
            results_by_seed[seed] = self.evaluate(actor=actor, num_episodes=num_episodes, seed=seed)
        return MultiSeedEvaluationResult(
            by_seed=results_by_seed,
            aggregate=_summarize_seed_aggregates(results_by_seed),
        )

    def evaluate_actor_suite_over_seeds(
        self,
        actors: Mapping[str, Callable[..., Any] | Any | None],
        seeds: list[int | None] | tuple[int | None, ...] = (None,),
        num_episodes: int = 3,
    ) -> dict[str, MultiSeedEvaluationResult]:
        results: dict[str, MultiSeedEvaluationResult] = {}
        for actor_name, actor in actors.items():
            results[actor_name] = self.evaluate_over_seeds(
                actor=actor,
                seeds=seeds,
                num_episodes=num_episodes,
            )
        return results


def build_phase1_evaluator(config: Any | None = None, **overrides: Any) -> EpisodeEvaluator:
    evaluation_seed_stride = int(overrides.get("evaluation_seed_stride", getattr(config, "evaluation_seed_stride", 1000)))
    task_config = getattr(config, "task", config)
    return EpisodeEvaluator(
        env_factory=lambda: build_phase1_task_env(task_config, **overrides),
        evaluation_seed_stride=evaluation_seed_stride,
    )
