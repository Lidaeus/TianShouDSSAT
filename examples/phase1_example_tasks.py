import argparse
import json
import os
import sys
from types import SimpleNamespace

PROJECT_ROOT = "/home/lidaeus/TianShouDSSAT"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.build_env import build_phase1_task_env
from experiments.build_evaluator import build_baseline_actor, build_phase1_evaluator
from experiments.build_policy import build_rainbow_components
from train_dssat import train as run_training_entry


def _make_output_dir(output_dir: str) -> str:
    resolved = os.path.abspath(output_dir)
    os.makedirs(resolved, exist_ok=True)
    return resolved


def run_env_showcase(output_dir: str, crop_name: str, steps: int) -> dict:
    env = build_phase1_task_env(
        crop_name=crop_name,
        include_forecast=True,
        forecast_horizons=(3, 7),
    )
    try:
        observation, info = env.reset(seed=42)
        reset_info = dict(info)
        transitions = []
        for step_idx in range(int(steps)):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            transitions.append(
                {
                    "step": step_idx + 1,
                    "action": int(action),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "applied_action": dict(info.get("applied_action", {})),
                }
            )
            if terminated or truncated:
                break
        result = {
            "task": "env_showcase",
            "crop_name": crop_name,
            "observation_dim": int(observation.shape[0]),
            "forecast_keys": sorted((reset_info.get("weather_forecast") or {}).keys()),
            "forecast_summary": dict(reset_info.get("weather_forecast", {})),
            "feature_count": len(reset_info.get("observation_features", [])),
            "action_schema": dict(reset_info.get("action_schema", {})),
            "transitions": transitions,
            "episode_report": dict(info.get("episode_report", {})),
        }
    finally:
        env.close()
    output_path = os.path.join(output_dir, "env_showcase.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return result


def run_evaluator_showcase(output_dir: str, crop_name: str, episodes: int) -> dict:
    probe_env = build_phase1_task_env(
        crop_name=crop_name,
        include_forecast=True,
        forecast_horizons=(3, 7),
    )
    try:
        algorithm, rainbow_policy = build_rainbow_components(
            state_shape=probe_env.observation_space.shape,
            action_shape=probe_env.action_space.n,
            hidden_size=64,
            device="cpu",
        )
        del algorithm
    finally:
        probe_env.close()
    evaluator = build_phase1_evaluator(
        crop_name=crop_name,
        include_forecast=True,
        forecast_horizons=(3, 7),
        evaluation_seed_stride=50,
    )
    actor_suite = {
        "random": None,
        "zero": build_baseline_actor("zero"),
        "reactive": build_baseline_actor("reactive"),
        "rainbow_untrained": rainbow_policy,
    }
    results = evaluator.evaluate_actor_suite(
        actors=actor_suite,
        num_episodes=int(episodes),
        seed=123,
    )
    serialized = {
        actor_name: {
            "aggregate": result.aggregate,
            "reports": result.reports,
        }
        for actor_name, result in results.items()
    }
    output_path = os.path.join(output_dir, "evaluator_showcase.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(serialized, handle, ensure_ascii=False, indent=2)
    return serialized


def run_training_showcase(output_dir: str, crop_name: str) -> dict:
    training_log_dir = os.path.relpath(os.path.join(output_dir, "training_logs"), PROJECT_ROOT)
    result = run_training_entry(
        SimpleNamespace(
            crop=crop_name,
            seed=7,
            epochs=1,
            batch_size=8,
            hidden_size=32,
            log_dir=training_log_dir,
            train_env_num=1,
            test_env_num=1,
            include_forecast=False,
            forecast_horizons=[3, 7],
            replay_buffer_size=128,
            epoch_num_steps=10,
            collection_step_num_env_steps=5,
            test_step_num_episodes=1,
            eval_episodes=1,
            evaluation_seed_stride=100,
            baseline_actors=["zero", "reactive"],
        )
    )
    payload = {
        "task": "training_showcase",
        "crop_name": crop_name,
        "log_path": result["log_path"],
        "trainer_result": str(result["trainer_result"]),
    }
    output_path = os.path.join(output_dir, "training_showcase.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 Phase1 重构后的示例任务集合")
    parser.add_argument("--crop", type=str, default="wheat", choices=["wheat", "maize", "tomato"])
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "logs", "example_tasks"))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["env", "evaluate", "train"],
        choices=["env", "evaluate", "train"],
    )
    args = parser.parse_args()

    output_dir = _make_output_dir(args.output_dir)
    results: dict[str, dict] = {}
    if "env" in args.tasks:
        results["env"] = run_env_showcase(output_dir=output_dir, crop_name=args.crop, steps=args.steps)
    if "evaluate" in args.tasks:
        results["evaluate"] = run_evaluator_showcase(output_dir=output_dir, crop_name=args.crop, episodes=args.episodes)
    if "train" in args.tasks:
        results["train"] = run_training_showcase(output_dir=output_dir, crop_name=args.crop)

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"output_dir": output_dir, "tasks": list(results.keys())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
