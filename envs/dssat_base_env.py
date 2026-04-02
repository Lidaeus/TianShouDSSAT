import importlib
import os
import sys
from typing import Any

import gymnasium as gym
import numpy as np

from .resource_resolver import resolve_crop_resources


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GYM_DSSAT_PDI_PATH = os.path.join(PROJECT_ROOT, "lib", "gym_dssat_pdi_official", "gym-dssat-pdi")


def _load_dssat_pdi_class():
    try:
        module = importlib.import_module("gym_dssat_pdi.envs.dssat_pdi")
        return module.DssatPdi
    except ModuleNotFoundError as exc:
        if os.path.isdir(GYM_DSSAT_PDI_PATH) and GYM_DSSAT_PDI_PATH not in sys.path:
            sys.path.insert(0, GYM_DSSAT_PDI_PATH)
        try:
            module = importlib.import_module("gym_dssat_pdi.envs.dssat_pdi")
            return module.DssatPdi
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "未能导入 gym_dssat_pdi.envs.dssat_pdi，请检查 vendored gym_dssat_pdi 依赖是否完整。"
            ) from exc


try:
    DssatPdi = _load_dssat_pdi_class()
except ModuleNotFoundError:
    DssatPdi = None


def _to_scalar_reward(reward: Any) -> float:
    if isinstance(reward, np.ndarray):
        return float(np.sum(reward))
    if isinstance(reward, (list, tuple)):
        return float(np.sum(reward))
    if reward is None:
        return 0.0
    return float(reward)


def _normalize_auxiliary_file_inputs(auxiliary_file_paths: Any) -> list[str]:
    if auxiliary_file_paths is None:
        return []
    if isinstance(auxiliary_file_paths, (str, os.PathLike)):
        candidates = [auxiliary_file_paths]
    else:
        try:
            candidates = list(auxiliary_file_paths)
        except TypeError as exc:
            raise TypeError("auxiliary_file_paths 必须为路径序列或单个路径字符串") from exc
    normalized: list[str] = []
    for raw_path in candidates:
        path = str(raw_path).strip()
        if not path:
            raise ValueError("auxiliary_file_paths 中包含空路径")
        normalized.append(path)
    return normalized


class DssatBaseEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        crop_name: str,
        mode: str = "all",
        seed: int | None = None,
        env_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.crop_name = crop_name
        self.mode = mode
        self.seed_value = seed
        env_kwargs = dict(env_kwargs or {})
        self.split_name = str(env_kwargs.pop("split_name", "train"))
        self.env_kwargs = env_kwargs
        extra_auxiliary_files = _normalize_auxiliary_file_inputs(self.env_kwargs.get("auxiliary_file_paths"))
        self.resources = resolve_crop_resources(
            crop_name=crop_name,
            run_dssat_location=self.env_kwargs.get("run_dssat_location"),
            extra_auxiliary_files=extra_auxiliary_files,
        )
        if not self.resources.auxiliary_file_paths:
            raise ValueError("DssatBaseEnv 未解析到任何 auxiliary_file_paths")
        final_env_kwargs: dict[str, Any] = {
            "run_dssat_location": self.resources.run_dssat_location,
            "cultivar": crop_name,
            "mode": mode,
            "seed": seed,
            "fileX_template_path": self.resources.filex_template_path,
            "pdi_template_path": self.resources.pdi_template_path,
            "auxiliary_file_paths": self.resources.auxiliary_file_paths,
        }
        final_env_kwargs.update(self.resources.extra_env_kwargs)
        final_env_kwargs.update(self.env_kwargs)
        dssat_pdi_class = DssatPdi or _load_dssat_pdi_class()
        self.env = dssat_pdi_class(**final_env_kwargs)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self._episode_steps = 0
        self._native_return = 0.0
        self._cumulative_actions = {"anfer": 0.0, "amir": 0.0}
        self._last_observation: dict[str, Any] | None = None
        self._last_context: dict[str, Any] = {}

    def _build_summary(self, observation: dict[str, Any] | None) -> dict[str, Any]:
        observation = observation or {}
        return {
            "crop_name": self.crop_name,
            "split_name": self.split_name,
            "episode_steps": self._episode_steps,
            "native_return": self._native_return,
            "total_anfer": self._cumulative_actions["anfer"],
            "total_amir": self._cumulative_actions["amir"],
            "final_grnwt": float(observation.get("grnwt", 0.0) or 0.0),
            "final_topwt": float(observation.get("topwt", 0.0) or 0.0),
            "final_tleachd": float(observation.get("tleachd", 0.0) or 0.0),
            "final_tnoxd": float(observation.get("tnoxd", 0.0) or 0.0),
            "final_trnu": float(observation.get("trnu", 0.0) or 0.0),
        }

    def _weather_file_paths(self) -> list[str]:
        return [path for path in self.resources.auxiliary_file_paths if path.lower().endswith(".wth")]

    def _weather_id(self) -> str | None:
        weather_paths = self._weather_file_paths()
        if not weather_paths:
            return None
        return os.path.splitext(os.path.basename(weather_paths[0]))[0]

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        info = dict(info or {})
        self._episode_steps = 0
        self._native_return = 0.0
        self._cumulative_actions = {"anfer": 0.0, "amir": 0.0}
        self._last_observation = observation
        self._last_context = dict(info.get("context", {}) or {})
        info["crop_name"] = self.crop_name
        info["split_name"] = self.split_name
        info["native_reward"] = 0.0
        info["weather_file_paths"] = self._weather_file_paths()
        info["primary_weather_path"] = info["weather_file_paths"][0] if info["weather_file_paths"] else None
        info["weather_id"] = self._weather_id()
        info["episode_summary"] = self._build_summary(observation)
        return observation, info

    def step(self, action):
        observation, native_reward, terminated, truncated, info = self.env.step(action)
        info = dict(info or {})
        scalar_reward = _to_scalar_reward(native_reward)
        self._episode_steps += 1
        self._native_return += scalar_reward
        self._cumulative_actions["anfer"] += float(action.get("anfer", 0.0) or 0.0)
        self._cumulative_actions["amir"] += float(action.get("amir", 0.0) or 0.0)
        self._last_observation = observation
        self._last_context = dict(info.get("context", {}) or {})
        info["crop_name"] = self.crop_name
        info["split_name"] = self.split_name
        info["native_reward"] = scalar_reward
        info["weather_file_paths"] = self._weather_file_paths()
        info["primary_weather_path"] = info["weather_file_paths"][0] if info["weather_file_paths"] else None
        info["weather_id"] = self._weather_id()
        info["applied_action"] = dict(action)
        info["episode_summary"] = self._build_summary(observation)
        return observation, scalar_reward, terminated, truncated, info

    def close(self):
        return self.env.close()
