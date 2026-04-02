import os
import sys
import argparse
import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OffPolicyTrainer, OffPolicyTrainerParams
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter

from experiments.build_env import build_phase1_task_env
from experiments.build_evaluator import (
    build_baseline_actor,
    build_phase1_evaluator,
    serialize_evaluation_suite,
    serialize_multiseed_evaluation_suite,
    write_benchmark_artifacts,
)
from experiments.experiment_config import build_phase1_benchmark_config
from experiments.build_policy import build_rainbow_components

PHASE1_SINGLE_CROP_BENCHMARK = "phase1_single_crop"


def get_args():
    parser = argparse.ArgumentParser(description="在 DSSAT 环境下训练 Rainbow DQN")
    parser.add_argument("--crop", type=str, default="maize", choices=["maize", "tomato", "wheat"], help="训练作物类型")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="批次大小")
    parser.add_argument("--hidden-size", type=int, default=128, help="网络隐藏层大小")
    parser.add_argument("--log-dir", type=str, default="logs", help="日志保存目录")
    parser.add_argument("--train-env-num", type=int, default=1, help="训练环境数量")
    parser.add_argument("--test-env-num", type=int, default=1, help="测试环境数量")
    parser.add_argument("--include-forecast", action="store_true", help="是否启用未来天气预报摘要 observation")
    parser.add_argument("--forecast-horizons", type=int, nargs="+", default=[3, 7], help="天气预报摘要窗口，例如 3 7")
    parser.add_argument("--split-name", type=str, default="train", help="实验协议 split 名称: train / validation / test")
    parser.add_argument("--evaluation-split-name", type=str, default=None, help="训练后统一评估使用的 split，默认与训练 split 相同")
    parser.add_argument("--weather-files", type=str, nargs="*", default=None, help="统一天气文件列表，按 split_ratios + split_seed 自动切分")
    parser.add_argument("--train-weather-files", type=str, nargs="*", default=None, help="显式指定训练 split 的天气文件")
    parser.add_argument("--validation-weather-files", type=str, nargs="*", default=None, help="显式指定验证 split 的天气文件")
    parser.add_argument("--test-weather-files", type=str, nargs="*", default=None, help="显式指定测试 split 的天气文件")
    parser.add_argument("--weather-split-ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15], help="天气文件切分比例，顺序为 train validation test")
    parser.add_argument("--split-seed", type=int, default=42, help="天气文件切分随机种子")
    parser.add_argument("--run-dssat-location", type=str, default=None, help="可选 DSSAT 可执行文件路径，未提供时自动解析")
    parser.add_argument("--replay-buffer-size", type=int, default=10000, help="经验回放容量")
    parser.add_argument("--epoch-num-steps", type=int, default=500, help="每个 epoch 的环境步数")
    parser.add_argument("--collection-step-num-env-steps", type=int, default=10, help="每次采样的环境步数")
    parser.add_argument("--test-step-num-episodes", type=int, default=1, help="训练期间每轮测试 episode 数量")
    parser.add_argument("--eval-episodes", type=int, default=2, help="训练后统一评估的 episode 数量")
    parser.add_argument("--evaluation-seed-stride", type=int, default=1000, help="统一评估时每个 episode 的 seed 步长")
    parser.add_argument("--grain-price", type=float, default=0.25, help="经济 KPI 使用的产量单价")
    parser.add_argument("--fertilizer-unit-cost", type=float, default=1.2, help="经济 KPI 使用的施肥单位成本")
    parser.add_argument("--irrigation-unit-cost", type=float, default=0.05, help="经济 KPI 使用的灌溉单位成本")
    parser.add_argument(
        "--baseline-actors",
        type=str,
        nargs="*",
        default=["zero", "calendar", "rule_based"],
        help="训练后一起评估的 baseline actor 名称",
    )
    parser.add_argument(
        "--benchmark-preset",
        type=str,
        default="none",
        choices=["none", PHASE1_SINGLE_CROP_BENCHMARK],
        help="应用推荐的单作物 benchmark 协议默认值",
    )
    parser.add_argument(
        "--evaluation-base-seeds",
        type=int,
        nargs="*",
        default=None,
        help="训练后统一评估使用的基础 seed 列表；提供多个值时自动输出多 seed 汇总",
    )
    return parser.parse_args()


def _normalize_forecast_horizons(horizons):
    normalized = sorted({int(horizon) for horizon in (horizons or []) if int(horizon) > 0})
    return normalized or [3, 7]


def _normalize_actor_names(actor_names):
    normalized = []
    for actor_name in actor_names or []:
        name = str(actor_name).strip()
        if name and name not in normalized:
            normalized.append(name)
    return normalized


def _normalize_optional_split_name(split_name):
    if split_name is None:
        return None
    normalized = str(split_name).strip().lower()
    if not normalized:
        return None
    return normalized if normalized in {"train", "validation", "test"} else "train"


def _apply_benchmark_preset(args):
    preset_name = str(getattr(args, "benchmark_preset", "none") or "none").strip().lower()
    args.benchmark_preset = preset_name if preset_name in {"none", PHASE1_SINGLE_CROP_BENCHMARK} else "none"
    if args.benchmark_preset != PHASE1_SINGLE_CROP_BENCHMARK:
        return args
    preset = build_phase1_benchmark_config(
        crop_name=getattr(args, "crop", "wheat"),
        seed=getattr(args, "seed", 42),
        split_name=getattr(args, "split_name", "train"),
        evaluation_split_name=getattr(args, "evaluation_split_name", None) or "test",
        weather_files=getattr(args, "weather_files", None),
        train_weather_files=getattr(args, "train_weather_files", None),
        validation_weather_files=getattr(args, "validation_weather_files", None),
        test_weather_files=getattr(args, "test_weather_files", None),
    )
    if not bool(getattr(args, "include_forecast", False)):
        args.include_forecast = preset.task.observation.include_forecast
    if not getattr(args, "evaluation_split_name", None):
        args.evaluation_split_name = preset.evaluation_split_name
    if getattr(args, "evaluation_base_seeds", None) is None:
        args.evaluation_base_seeds = list(preset.evaluation_base_seeds)
    default_baselines = ["zero", "calendar", "rule_based"]
    if list(getattr(args, "baseline_actors", []) or []) == default_baselines:
        args.baseline_actors = list(preset.baseline_actors)
    if int(getattr(args, "eval_episodes", 2)) == 2:
        args.eval_episodes = preset.evaluation_episodes
    if str(getattr(args, "log_dir", "logs")).strip() == "logs":
        args.log_dir = preset.log_dir
    return args


def _normalize_cli_args(args):
    args = _apply_benchmark_preset(args)
    args.forecast_horizons = _normalize_forecast_horizons(getattr(args, "forecast_horizons", None))
    args.baseline_actors = _normalize_actor_names(getattr(args, "baseline_actors", None))
    args.split_name = _normalize_optional_split_name(getattr(args, "split_name", "train")) or "train"
    args.evaluation_split_name = _normalize_optional_split_name(getattr(args, "evaluation_split_name", None))
    numeric_constraints = {
        "train_env_num": 1,
        "test_env_num": 1,
        "epochs": 1,
        "batch_size": 1,
        "hidden_size": 1,
        "replay_buffer_size": 1,
        "epoch_num_steps": 1,
        "collection_step_num_env_steps": 1,
        "test_step_num_episodes": 1,
        "eval_episodes": 1,
        "evaluation_seed_stride": 1,
    }
    for field_name, minimum in numeric_constraints.items():
        field_value = int(getattr(args, field_name))
        if field_value < minimum:
            raise ValueError(f"{field_name} 必须大于等于 {minimum}")
        setattr(args, field_name, field_value)
    weather_split_ratios = [float(value) for value in getattr(args, "weather_split_ratios", [])]
    if len(weather_split_ratios) != 3:
        raise ValueError("weather_split_ratios 必须提供 train/validation/test 三个值")
    if any(value < 0.0 for value in weather_split_ratios):
        raise ValueError("weather_split_ratios 不能包含负数")
    if sum(weather_split_ratios) <= 0.0:
        raise ValueError("weather_split_ratios 总和必须大于 0")
    args.weather_split_ratios = weather_split_ratios
    if getattr(args, "evaluation_base_seeds", None) is not None:
        args.evaluation_base_seeds = [int(seed) for seed in args.evaluation_base_seeds]
    return args


def _task_env_overrides(args, seed, split_name=None):
    overrides = {
        "crop_name": args.crop,
        "seed": seed,
        "include_forecast": args.include_forecast,
        "forecast_horizons": tuple(args.forecast_horizons),
        "split_name": split_name or args.split_name,
        "split_ratios": tuple(args.weather_split_ratios),
        "split_seed": args.split_seed,
        "grain_price": args.grain_price,
        "fertilizer_unit_cost": args.fertilizer_unit_cost,
        "irrigation_unit_cost": args.irrigation_unit_cost,
    }
    optional_sequence_fields = (
        "weather_files",
        "train_weather_files",
        "validation_weather_files",
        "test_weather_files",
    )
    for field_name in optional_sequence_fields:
        field_value = getattr(args, field_name, None)
        if field_value:
            overrides[field_name] = tuple(field_value)
    run_dssat_location = getattr(args, "run_dssat_location", None)
    if run_dssat_location:
        overrides["run_dssat_location"] = run_dssat_location
    return overrides


def make_env_factory(crop_name, seed, include_forecast, forecast_horizons, env_overrides=None):
    def _make():
        base_overrides = dict(env_overrides or {})
        base_overrides.update({
            "crop_name": crop_name,
            "seed": seed,
            "include_forecast": include_forecast,
            "forecast_horizons": tuple(forecast_horizons),
        })
        return build_phase1_task_env(**base_overrides)

    return _make


def make_vector_env_factories(crop_name, base_seed, env_num, include_forecast, forecast_horizons, env_overrides=None):
    factories = []
    for idx in range(env_num):
        env_seed = None if base_seed is None else base_seed + idx
        factories.append(
            make_env_factory(
                crop_name=crop_name,
                seed=env_seed,
                include_forecast=include_forecast,
                forecast_horizons=forecast_horizons,
                env_overrides=env_overrides,
            )
        )
    return factories


def evaluate_actor_suite(policy, args, log_path):
    args = _normalize_cli_args(args)
    evaluation_split_name = args.evaluation_split_name or args.split_name
    protocol_overrides = _task_env_overrides(args, args.seed, split_name=evaluation_split_name)
    evaluator = build_phase1_evaluator(
        evaluation_seed_stride=args.evaluation_seed_stride,
        **protocol_overrides,
    )
    actor_suite = {"rainbow": policy}
    for actor_name in args.baseline_actors:
        actor_suite[actor_name] = build_baseline_actor(actor_name)
    policy.eval()
    evaluation_base_seeds = args.evaluation_base_seeds
    if evaluation_base_seeds:
        normalized_base_seeds = list(dict.fromkeys(int(seed) for seed in evaluation_base_seeds))
        results = evaluator.evaluate_actor_suite_over_seeds(
            actors=actor_suite,
            num_episodes=args.eval_episodes,
            seeds=normalized_base_seeds,
        )
        payload = serialize_multiseed_evaluation_suite(results)
    else:
        results = evaluator.evaluate_actor_suite(
            actors=actor_suite,
            num_episodes=args.eval_episodes,
            seed=None if args.seed is None else args.seed + 5000,
        )
        payload = serialize_evaluation_suite(results)
    payload["config"] = {
        "algorithm_name": "rainbow",
        "run_name": os.path.basename(log_path),
        "crop": args.crop,
        "seed": args.seed,
        "benchmark_preset": args.benchmark_preset,
        "include_forecast": bool(args.include_forecast),
        "forecast_horizons": list(args.forecast_horizons),
        "protocol": {
            "split_name": args.split_name,
            "evaluation_split_name": evaluation_split_name,
            "weather_files": None if args.weather_files is None else list(args.weather_files),
            "train_weather_files": None if args.train_weather_files is None else list(args.train_weather_files),
            "validation_weather_files": None if args.validation_weather_files is None else list(args.validation_weather_files),
            "test_weather_files": None if args.test_weather_files is None else list(args.test_weather_files),
            "weather_split_ratios": list(args.weather_split_ratios),
            "split_seed": int(args.split_seed),
        },
        "economics": {
            "grain_price": float(args.grain_price),
            "fertilizer_unit_cost": float(args.fertilizer_unit_cost),
            "irrigation_unit_cost": float(args.irrigation_unit_cost),
        },
        "eval_episodes": int(args.eval_episodes),
        "baseline_actors": list(args.baseline_actors),
        "evaluation_seed_stride": int(args.evaluation_seed_stride),
        "evaluation_base_seeds": None if args.evaluation_base_seeds is None else list(args.evaluation_base_seeds),
    }
    artifact_outputs = write_benchmark_artifacts(payload, log_path)
    summary_path = artifact_outputs["summary_path"]
    print("\n统一评估汇总:")
    print(f"评估 split: {evaluation_split_name}")
    if evaluation_base_seeds:
        for actor_name, result in results.items():
            metrics = result.aggregate.get("metrics", {})
            print(
                f"- {actor_name}: "
                f"yield={metrics.get('mean_final_grain_weight', {}).get('mean', 0.0):.2f}±{metrics.get('mean_final_grain_weight', {}).get('std', 0.0):.2f}, "
                f"task_return={metrics.get('mean_task_return', {}).get('mean', 0.0):.4f}±{metrics.get('mean_task_return', {}).get('std', 0.0):.4f}, "
                f"profit={metrics.get('mean_profit', {}).get('mean', 0.0):.2f}±{metrics.get('mean_profit', {}).get('std', 0.0):.2f}, "
                f"N={metrics.get('mean_total_fertilizer', {}).get('mean', 0.0):.2f}, "
                f"I={metrics.get('mean_total_irrigation', {}).get('mean', 0.0):.2f}, "
                f"NUE={metrics.get('mean_nitrogen_use_efficiency', {}).get('mean', 0.0):.2f}, "
                f"WUE={metrics.get('mean_water_use_efficiency', {}).get('mean', 0.0):.2f}"
            )
    else:
        for actor_name, result in results.items():
            aggregate = result.aggregate
            print(
                f"- {actor_name}: "
                f"yield={aggregate.get('mean_final_grain_weight', 0.0):.2f}, "
                f"task_return={aggregate.get('mean_task_return', 0.0):.4f}, "
                f"profit={aggregate.get('mean_profit', 0.0):.2f}, "
                f"N={aggregate.get('mean_total_fertilizer', 0.0):.2f}, "
                f"I={aggregate.get('mean_total_irrigation', 0.0):.2f}, "
                f"NUE={aggregate.get('mean_nitrogen_use_efficiency', 0.0):.2f}, "
                f"WUE={aggregate.get('mean_water_use_efficiency', 0.0):.2f}"
            )
    print(f"评估结果已保存: {summary_path}")
    print(f"benchmark 产物清单已保存: {artifact_outputs['manifest_path']}")
    print(f"benchmark 批次索引已保存: {artifact_outputs['batch_index_path']}")
    print(f"benchmark 跨 run 汇总已保存: {artifact_outputs['cross_run_summary_csv_path']}")
    return results


def _safe_close(resource):
    if resource is None:
        return
    close = getattr(resource, "close", None)
    if callable(close):
        close()


def train(args):
    args = _normalize_cli_args(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"rainbow_{args.crop}_{timestamp}"
    log_path = os.path.join(str(PROJECT_ROOT), args.log_dir, log_name)
    os.makedirs(log_path, exist_ok=True)
    
    print(f"--- 启动训练: {args.crop.upper()} (设备: {device}) ---")
    print(f"日志路径: {log_path}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_envs = None
    test_envs = None
    example_env = None
    writer = None
    try:
        env_overrides = _task_env_overrides(args, args.seed)
        train_envs = DummyVectorEnv(
            make_vector_env_factories(
                crop_name=args.crop,
                base_seed=args.seed,
                env_num=args.train_env_num,
                include_forecast=args.include_forecast,
                forecast_horizons=args.forecast_horizons,
                env_overrides=env_overrides,
            )
        )
        test_envs = DummyVectorEnv(
            make_vector_env_factories(
                crop_name=args.crop,
                base_seed=None if args.seed is None else args.seed + 1000,
                env_num=args.test_env_num,
                include_forecast=args.include_forecast,
                forecast_horizons=args.forecast_horizons,
                env_overrides=env_overrides,
            )
        )

        example_env = make_env_factory(
            crop_name=args.crop,
            seed=args.seed,
            include_forecast=args.include_forecast,
            forecast_horizons=args.forecast_horizons,
            env_overrides=env_overrides,
        )()
        state_shape = example_env.observation_space.shape
        action_shape = example_env.action_space.n
        print(f"观察空间维度: {state_shape}, 动作空间: Discrete({action_shape})")
        example_obs, example_info = example_env.reset(seed=args.seed)
        print(f"首个观测维度: {example_obs.shape[0]}")
        if "weather_forecast" in example_info:
            print(f"天气预报窗口: {tuple(args.forecast_horizons)}")

        algo, policy = build_rainbow_components(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_size=args.hidden_size,
            device=device,
        )

        train_collector = Collector(policy, train_envs, VectorReplayBuffer(args.replay_buffer_size, len(train_envs)))
        test_collector = Collector(policy, test_envs)

        writer = SummaryWriter(log_path)
        logger = TensorboardLogger(writer)

        params = OffPolicyTrainerParams(
            max_epochs=args.epochs,
            epoch_num_steps=args.epoch_num_steps,
            training_collector=train_collector,
            test_collector=test_collector,
            test_step_num_episodes=args.test_step_num_episodes,
            batch_size=args.batch_size,
            collection_step_num_env_steps=args.collection_step_num_env_steps,
            logger=logger,
            save_best_fn=lambda policy: torch.save(policy.state_dict(), os.path.join(log_path, "policy_best.pth"))
        )

        trainer = OffPolicyTrainer(algorithm=algo, params=params)
        result = trainer.run()

        print(f"\n训练结束: {result}")
        torch.save(policy.state_dict(), os.path.join(log_path, "policy_final.pth"))
        evaluate_actor_suite(policy=policy, args=args, log_path=log_path)
        return {
            "trainer_result": result,
            "log_path": log_path,
        }
    finally:
        _safe_close(writer)
        _safe_close(example_env)
        _safe_close(train_envs)
        _safe_close(test_envs)

if __name__ == "__main__":
    args = get_args()
    train(args)
