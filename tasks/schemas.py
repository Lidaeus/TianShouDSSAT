from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from envs.crop_registry import get_crop_config

from .forecast import BaseForecastProvider, NullForecastProvider


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (list, tuple, np.ndarray)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip(value: float, low: float, high: float) -> float:
    return float(min(high, max(low, value)))


def _weighted_root_zone_ratio(
    sw: list[float] | np.ndarray | None,
    ll: list[float] | np.ndarray | None,
    dul: list[float] | np.ndarray | None,
    dlayr: list[float] | np.ndarray | None,
    root_depth: float,
) -> float:
    if sw is None or ll is None or dul is None or dlayr is None:
        return 0.0
    sw_arr = np.asarray(sw, dtype=np.float32)
    ll_arr = np.asarray(ll, dtype=np.float32)
    dul_arr = np.asarray(dul, dtype=np.float32)
    dlayr_arr = np.asarray(dlayr, dtype=np.float32)
    if not (len(sw_arr) == len(ll_arr) == len(dul_arr) == len(dlayr_arr)):
        return 0.0
    if len(sw_arr) == 0 or root_depth <= 0:
        return 0.0
    remaining_depth = float(root_depth)
    numer = 0.0
    denom = 0.0
    for sw_i, ll_i, dul_i, depth_i in zip(sw_arr, ll_arr, dul_arr, dlayr_arr):
        if remaining_depth <= 0:
            break
        used_depth = min(float(depth_i), remaining_depth)
        remaining_depth -= used_depth
        capacity = max(1e-6, float(dul_i) - float(ll_i))
        ratio = _clip((float(sw_i) - float(ll_i)) / capacity, 0.0, 1.5)
        numer += ratio * used_depth
        denom += used_depth
    if denom <= 0:
        return 0.0
    return _clip(numer / denom, 0.0, 1.5)


def _profile_ratio(
    sw: list[float] | np.ndarray | None,
    ll: list[float] | np.ndarray | None,
    dul: list[float] | np.ndarray | None,
) -> float:
    if sw is None or ll is None or dul is None:
        return 0.0
    sw_arr = np.asarray(sw, dtype=np.float32)
    ll_arr = np.asarray(ll, dtype=np.float32)
    dul_arr = np.asarray(dul, dtype=np.float32)
    if not (len(sw_arr) == len(ll_arr) == len(dul_arr)) or len(sw_arr) == 0:
        return 0.0
    capacity = np.maximum(dul_arr - ll_arr, 1e-6)
    ratio = np.clip((sw_arr - ll_arr) / capacity, 0.0, 1.5)
    return float(np.mean(ratio))


def _normalize_reward_value(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, np.ndarray):
        return float(np.sum(value))
    if isinstance(value, (list, tuple)):
        return float(sum(_safe_float(item) for item in value))
    return _safe_float(value)


@dataclass
class ManagementState:
    calendar_step: int = 0
    cumulative_fertilizer: float = 0.0
    cumulative_irrigation: float = 0.0
    days_since_fertilizer: float = 0.0
    days_since_irrigation: float = 0.0
    fertilizer_budget: float = 180.0
    irrigation_budget: float = 240.0

    @property
    def remaining_fertilizer(self) -> float:
        return max(0.0, self.fertilizer_budget - self.cumulative_fertilizer)

    @property
    def remaining_irrigation(self) -> float:
        return max(0.0, self.irrigation_budget - self.cumulative_irrigation)


class BaseObservationSchema(ABC):
    @property
    @abstractmethod
    def observation_space(self) -> gym.Space:
        raise NotImplementedError

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def encode(
        self,
        raw_state: dict[str, Any],
        context: dict[str, Any],
        management_state: ManagementState,
        metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        raise NotImplementedError


class AgronomicObservationSchema(BaseObservationSchema):
    def __init__(
        self,
        crop_name: str,
        include_forecast: bool = False,
        forecast_horizons: tuple[int, ...] = (3, 7),
    ) -> None:
        self.crop_name = crop_name
        self.include_forecast = include_forecast
        self.forecast_horizons = tuple(sorted(set(int(h) for h in forecast_horizons if int(h) > 0)))
        self.crop_config = get_crop_config(crop_name)
        self._feature_names = [
            "istage_norm",
            "vstage_norm",
            "canopy_norm",
            "grnwt_norm",
            "topwt_norm",
            "rtdep_norm",
            "swfac_norm",
            "nstres_norm",
            "root_zone_sw_ratio",
            "profile_sw_ratio",
            "wtdep_norm",
            "trnu_norm",
            "tleachd_norm",
            "tnoxd_norm",
            "cumulative_fertilizer_norm",
            "cumulative_irrigation_norm",
            "rain_norm",
            "tmin_norm",
            "tmax_norm",
            "srad_norm",
            "days_since_fertilizer_norm",
            "days_since_irrigation_norm",
            "remaining_fertilizer_ratio",
            "remaining_irrigation_ratio",
        ]
        if self.include_forecast:
            for horizon in self.forecast_horizons:
                self._feature_names.extend([
                    f"forecast_rain_{horizon}d_norm",
                    f"forecast_tmean_{horizon}d_norm",
                    f"forecast_srad_{horizon}d_norm",
                ])
        self._observation_space = gym.spaces.Box(
            low=-2.0,
            high=2.0,
            shape=(len(self._feature_names),),
            dtype=np.float32,
        )

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def feature_groups(self) -> dict[str, tuple[str, ...]]:
        groups: dict[str, tuple[str, ...]] = {
            "crop_state": (
                "istage_norm",
                "vstage_norm",
                "canopy_norm",
                "grnwt_norm",
                "topwt_norm",
                "rtdep_norm",
                "swfac_norm",
                "nstres_norm",
            ),
            "soil_water": (
                "root_zone_sw_ratio",
                "profile_sw_ratio",
                "wtdep_norm",
            ),
            "nitrogen_flux": (
                "trnu_norm",
                "tleachd_norm",
                "tnoxd_norm",
            ),
            "management_memory": (
                "cumulative_fertilizer_norm",
                "cumulative_irrigation_norm",
                "days_since_fertilizer_norm",
                "days_since_irrigation_norm",
                "remaining_fertilizer_ratio",
                "remaining_irrigation_ratio",
            ),
            "current_weather": (
                "rain_norm",
                "tmin_norm",
                "tmax_norm",
                "srad_norm",
            ),
        }
        if self.include_forecast:
            groups["forecast_weather"] = tuple(
                feature_name
                for feature_name in self._feature_names
                if feature_name.startswith("forecast_")
            )
        return groups

    def describe(self) -> dict[str, Any]:
        return {
            "type": "agronomic",
            "crop_name": self.crop_name,
            "include_forecast": self.include_forecast,
            "forecast_horizons": self.forecast_horizons,
            "feature_names": self.feature_names,
            "feature_groups": self.feature_groups(),
        }

    def encode(
        self,
        raw_state: dict[str, Any],
        context: dict[str, Any],
        management_state: ManagementState,
        metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        metadata = metadata or {}
        canopy_value = raw_state.get(self.crop_config.canopy_variable, raw_state.get("xlai", raw_state.get("lai", 0.0)))
        root_depth = _safe_float(raw_state.get("rtdep"))
        sw = raw_state.get("sw", context.get("sw"))
        ll = raw_state.get("ll", context.get("ll"))
        dul = raw_state.get("dul", context.get("dul"))
        dlayr = raw_state.get("dlayr", context.get("dlayr"))
        forecast = metadata.get("weather_forecast", {}) or {}
        features = [
            _clip(_safe_float(raw_state.get("istage")) / 8.0, 0.0, 1.5),
            _clip(_safe_float(raw_state.get("vstage")) / 30.0, 0.0, 1.5),
            _clip(_safe_float(canopy_value) / 10.0, 0.0, 1.5),
            _clip(_safe_float(raw_state.get("grnwt")) / 30000.0, 0.0, 1.5),
            _clip(_safe_float(raw_state.get("topwt")) / 50000.0, 0.0, 1.5),
            _clip(root_depth / 300.0, 0.0, 1.5),
            _clip(_safe_float(raw_state.get("swfac")), 0.0, 1.5),
            _clip(_safe_float(raw_state.get("nstres")), 0.0, 1.5),
            _weighted_root_zone_ratio(sw=sw, ll=ll, dul=dul, dlayr=dlayr, root_depth=root_depth),
            _profile_ratio(sw=sw, ll=ll, dul=dul),
            _clip(_safe_float(raw_state.get("wtdep")) / 1000.0, 0.0, 1.5),
            _clip(_safe_float(raw_state.get("trnu")) / 10.0, 0.0, 1.5),
            _clip(_safe_float(raw_state.get("tleachd")) / 100.0, 0.0, 1.5),
            _clip(_safe_float(raw_state.get("tnoxd")) / 10.0, 0.0, 1.5),
            _clip(management_state.cumulative_fertilizer / max(1.0, management_state.fertilizer_budget), 0.0, 1.5),
            _clip(management_state.cumulative_irrigation / max(1.0, management_state.irrigation_budget), 0.0, 1.5),
            _clip(_safe_float(raw_state.get("rain")) / 100.0, 0.0, 1.5),
            _clip(_safe_float(raw_state.get("tmin")) / 50.0, -1.5, 1.5),
            _clip(_safe_float(raw_state.get("tmax")) / 50.0, -1.5, 1.5),
            _clip(_safe_float(raw_state.get("srad")) / 40.0, 0.0, 1.5),
            _clip(management_state.days_since_fertilizer / 60.0, 0.0, 1.5),
            _clip(management_state.days_since_irrigation / 60.0, 0.0, 1.5),
            _clip(management_state.remaining_fertilizer / max(1.0, management_state.fertilizer_budget), 0.0, 1.5),
            _clip(management_state.remaining_irrigation / max(1.0, management_state.irrigation_budget), 0.0, 1.5),
        ]
        if self.include_forecast:
            for horizon in self.forecast_horizons:
                features.extend([
                    _clip(_safe_float(forecast.get(f"rain_{horizon}d")) / max(50.0, 30.0 * horizon), 0.0, 1.5),
                    _clip(_safe_float(forecast.get(f"tmean_{horizon}d")) / 50.0, -1.5, 1.5),
                    _clip(_safe_float(forecast.get(f"srad_{horizon}d")) / 40.0, 0.0, 1.5),
                ])
        return np.asarray(features, dtype=np.float32)


class BaseActionSchema(ABC):
    @property
    @abstractmethod
    def action_space(self) -> gym.Space:
        raise NotImplementedError

    @abstractmethod
    def initial_management_state(self) -> ManagementState:
        raise NotImplementedError

    @abstractmethod
    def decode(self, action: Any, raw_state: dict[str, Any], management_state: ManagementState) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def update_management(self, management_state: ManagementState, applied_action: dict[str, float]) -> ManagementState:
        raise NotImplementedError


class WeeklyDiscreteActionSchema(BaseActionSchema):
    def __init__(
        self,
        nitrogen_levels: tuple[float, ...] = (0.0, 30.0, 60.0, 90.0),
        irrigation_levels: tuple[float, ...] = (0.0, 10.0, 20.0, 30.0),
        decision_interval_days: int = 7,
        fertilizer_budget: float = 180.0,
        irrigation_budget: float = 240.0,
        fertilizer_stage_cutoff: float = 6.0,
    ) -> None:
        self.nitrogen_levels = tuple(float(x) for x in nitrogen_levels)
        self.irrigation_levels = tuple(float(x) for x in irrigation_levels)
        self.decision_interval_days = int(decision_interval_days)
        self.fertilizer_budget = float(fertilizer_budget)
        self.irrigation_budget = float(irrigation_budget)
        self.fertilizer_stage_cutoff = float(fertilizer_stage_cutoff)
        self._action_space = gym.spaces.Discrete(len(self.nitrogen_levels) * len(self.irrigation_levels))

    @property
    def action_space(self) -> gym.Space:
        return self._action_space

    def valid_action_mask(self, raw_state: dict[str, Any], management_state: ManagementState) -> np.ndarray:
        mask = np.zeros(self._action_space.n, dtype=np.int8)
        if management_state.calendar_step > 0 and management_state.calendar_step % self.decision_interval_days != 0:
            mask[0] = 1
            return mask
        istage = _safe_float(raw_state.get("istage"))
        max_fertilizer = 0.0 if istage >= self.fertilizer_stage_cutoff else management_state.remaining_fertilizer
        max_irrigation = management_state.remaining_irrigation
        for action_idx in range(self._action_space.n):
            n_idx = action_idx // len(self.irrigation_levels)
            i_idx = action_idx % len(self.irrigation_levels)
            nitrogen = self.nitrogen_levels[n_idx]
            irrigation = self.irrigation_levels[i_idx]
            if nitrogen <= max_fertilizer and irrigation <= max_irrigation:
                mask[action_idx] = 1
        if not np.any(mask):
            mask[0] = 1
        return mask

    def describe(self) -> dict[str, Any]:
        return {
            "type": "weekly_discrete",
            "nitrogen_levels": self.nitrogen_levels,
            "irrigation_levels": self.irrigation_levels,
            "decision_interval_days": self.decision_interval_days,
            "fertilizer_budget": self.fertilizer_budget,
            "irrigation_budget": self.irrigation_budget,
            "fertilizer_stage_cutoff": self.fertilizer_stage_cutoff,
            "supports_action_mask": True,
        }

    def initial_management_state(self) -> ManagementState:
        return ManagementState(
            calendar_step=0,
            cumulative_fertilizer=0.0,
            cumulative_irrigation=0.0,
            days_since_fertilizer=0.0,
            days_since_irrigation=0.0,
            fertilizer_budget=self.fertilizer_budget,
            irrigation_budget=self.irrigation_budget,
        )

    def decode(self, action: Any, raw_state: dict[str, Any], management_state: ManagementState) -> dict[str, float]:
        action_idx = int(action)
        if action_idx < 0 or action_idx >= self._action_space.n:
            raise ValueError(f"动作索引越界: {action_idx}")
        if management_state.calendar_step > 0 and management_state.calendar_step % self.decision_interval_days != 0:
            return {"anfer": 0.0, "amir": 0.0}
        n_idx = action_idx // len(self.irrigation_levels)
        i_idx = action_idx % len(self.irrigation_levels)
        nitrogen = self.nitrogen_levels[n_idx]
        irrigation = self.irrigation_levels[i_idx]
        istage = _safe_float(raw_state.get("istage"))
        if istage >= self.fertilizer_stage_cutoff:
            nitrogen = 0.0
        nitrogen = min(nitrogen, management_state.remaining_fertilizer)
        irrigation = min(irrigation, management_state.remaining_irrigation)
        return {"anfer": float(nitrogen), "amir": float(irrigation)}

    def update_management(self, management_state: ManagementState, applied_action: dict[str, float]) -> ManagementState:
        anfer = float(applied_action.get("anfer", 0.0) or 0.0)
        amir = float(applied_action.get("amir", 0.0) or 0.0)
        return ManagementState(
            calendar_step=management_state.calendar_step + 1,
            cumulative_fertilizer=management_state.cumulative_fertilizer + anfer,
            cumulative_irrigation=management_state.cumulative_irrigation + amir,
            days_since_fertilizer=0.0 if anfer > 0 else management_state.days_since_fertilizer + 1.0,
            days_since_irrigation=0.0 if amir > 0 else management_state.days_since_irrigation + 1.0,
            fertilizer_budget=management_state.fertilizer_budget,
            irrigation_budget=management_state.irrigation_budget,
        )


class BaseRewardSchema(ABC):
    @abstractmethod
    def compute(
        self,
        previous_state: dict[str, Any] | None,
        next_state: dict[str, Any],
        management_state: ManagementState,
        applied_action: dict[str, float],
        terminated: bool,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[float, dict[str, float]]:
        raise NotImplementedError


class YieldCostRewardSchema(BaseRewardSchema):
    def __init__(
        self,
        grain_delta_weight: float = 0.001,
        terminal_yield_weight: float = 0.05,
        fertilizer_cost_weight: float = 0.02,
        irrigation_cost_weight: float = 0.01,
        leaching_cost_weight: float = 0.1,
        denitrification_cost_weight: float = 0.1,
    ) -> None:
        self.grain_delta_weight = float(grain_delta_weight)
        self.terminal_yield_weight = float(terminal_yield_weight)
        self.fertilizer_cost_weight = float(fertilizer_cost_weight)
        self.irrigation_cost_weight = float(irrigation_cost_weight)
        self.leaching_cost_weight = float(leaching_cost_weight)
        self.denitrification_cost_weight = float(denitrification_cost_weight)

    def compute(
        self,
        previous_state: dict[str, Any] | None,
        next_state: dict[str, Any],
        management_state: ManagementState,
        applied_action: dict[str, float],
        terminated: bool,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[float, dict[str, float]]:
        previous_grnwt = _safe_float((previous_state or {}).get("grnwt"))
        next_grnwt = _safe_float(next_state.get("grnwt"))
        grain_delta = max(0.0, next_grnwt - previous_grnwt)
        fertilizer_cost = self.fertilizer_cost_weight * float(applied_action.get("anfer", 0.0) or 0.0)
        irrigation_cost = self.irrigation_cost_weight * float(applied_action.get("amir", 0.0) or 0.0)
        leaching_cost = self.leaching_cost_weight * _safe_float(next_state.get("tleachd"))
        denitrification_cost = self.denitrification_cost_weight * _safe_float(next_state.get("tnoxd"))
        terminal_yield_bonus = self.terminal_yield_weight * (next_grnwt / 1000.0) if terminated else 0.0
        grain_progress = self.grain_delta_weight * (grain_delta / 1000.0)
        reward = grain_progress + terminal_yield_bonus - fertilizer_cost - irrigation_cost - leaching_cost - denitrification_cost
        components = {
            "grain_progress": grain_progress,
            "terminal_yield_bonus": terminal_yield_bonus,
            "fertilizer_cost": fertilizer_cost,
            "irrigation_cost": irrigation_cost,
            "leaching_cost": leaching_cost,
            "denitrification_cost": denitrification_cost,
            "remaining_fertilizer": management_state.remaining_fertilizer,
            "remaining_irrigation": management_state.remaining_irrigation,
        }
        return float(reward), components

    def describe(self) -> dict[str, Any]:
        return {
            "type": "yield_cost",
            "grain_delta_weight": self.grain_delta_weight,
            "terminal_yield_weight": self.terminal_yield_weight,
            "fertilizer_cost_weight": self.fertilizer_cost_weight,
            "irrigation_cost_weight": self.irrigation_cost_weight,
            "leaching_cost_weight": self.leaching_cost_weight,
            "denitrification_cost_weight": self.denitrification_cost_weight,
        }


class TaskEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        base_env: gym.Env,
        observation_schema: BaseObservationSchema,
        action_schema: BaseActionSchema,
        reward_schema: BaseRewardSchema,
        episode_reporter: Any | None = None,
        forecast_provider: BaseForecastProvider | None = None,
    ) -> None:
        self.base_env = base_env
        self.observation_schema = observation_schema
        self.action_schema = action_schema
        self.reward_schema = reward_schema
        self.episode_reporter = episode_reporter
        self.forecast_provider = forecast_provider or NullForecastProvider()
        self.action_space = action_schema.action_space
        self.observation_space = observation_schema.observation_space
        self.management_state = action_schema.initial_management_state()
        self._last_raw_state: dict[str, Any] | None = None
        self._last_info: dict[str, Any] = {}

    def _enrich_info(self, raw_state: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        enriched_info = dict(info or {})
        enriched_info.update(self.forecast_provider.enrich_metadata(raw_state, enriched_info))
        return enriched_info

    def _standardize_info(self, raw_state: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        standardized = dict(info or {})
        context = dict(standardized.get("context", {}) or {})
        sw = raw_state.get("sw", context.get("sw"))
        ll = raw_state.get("ll", context.get("ll"))
        dul = raw_state.get("dul", context.get("dul"))
        dlayr = raw_state.get("dlayr", context.get("dlayr"))
        rtdep = _safe_float(raw_state.get("rtdep"))
        standardized.update({
            "istage": _safe_float(raw_state.get("istage")),
            "vstage": _safe_float(raw_state.get("vstage")),
            "swfac": _safe_float(raw_state.get("swfac")),
            "nstres": _safe_float(raw_state.get("nstres")),
            "grnwt": _safe_float(raw_state.get("grnwt")),
            "topwt": _safe_float(raw_state.get("topwt")),
            "rtdep": rtdep,
            "wtdep": _safe_float(raw_state.get("wtdep")),
            "trnu": _safe_float(raw_state.get("trnu")),
            "tleachd": _safe_float(raw_state.get("tleachd")),
            "tnoxd": _safe_float(raw_state.get("tnoxd")),
            "rain": _safe_float(raw_state.get("rain")),
            "tmin": _safe_float(raw_state.get("tmin")),
            "tmax": _safe_float(raw_state.get("tmax")),
            "srad": _safe_float(raw_state.get("srad")),
            "yrdoy": _safe_float(raw_state.get("yrdoy")),
            "root_zone_sw_ratio": _weighted_root_zone_ratio(sw=sw, ll=ll, dul=dul, dlayr=dlayr, root_depth=rtdep),
            "profile_sw_ratio": _profile_ratio(sw=sw, ll=ll, dul=dul),
        })
        return standardized

    def reset(self, **kwargs):
        raw_state, info = self.base_env.reset(**kwargs)
        self.management_state = self.action_schema.initial_management_state()
        self._last_raw_state = raw_state
        self._last_info = self._standardize_info(raw_state, self._enrich_info(raw_state, dict(info or {})))
        if self.episode_reporter is not None:
            self.episode_reporter.reset(
                crop_name=self._last_info.get("crop_name", "unknown"),
                metadata=self._last_info,
            )
        observation = self.observation_schema.encode(
            raw_state=raw_state,
            context=dict(self._last_info.get("context", {}) or {}),
            management_state=self.management_state,
            metadata=self._last_info,
        )
        info = dict(self._last_info)
        info["observation_features"] = self.observation_schema.feature_names
        if hasattr(self.observation_schema, "describe"):
            info["observation_schema"] = self.observation_schema.describe()
        if hasattr(self.action_schema, "describe"):
            info["action_schema"] = self.action_schema.describe()
        if hasattr(self.action_schema, "valid_action_mask"):
            info["action_mask"] = self.action_schema.valid_action_mask(raw_state, self.management_state)
        if hasattr(self.reward_schema, "describe"):
            info["reward_schema"] = self.reward_schema.describe()
        return observation, info

    def step(self, action):
        action_dict = self.action_schema.decode(
            action=action,
            raw_state=self._last_raw_state or {},
            management_state=self.management_state,
        )
        raw_state, native_reward, terminated, truncated, info = self.base_env.step(action_dict)
        info = self._standardize_info(raw_state, self._enrich_info(raw_state, dict(info or {})))
        self.management_state = self.action_schema.update_management(self.management_state, action_dict)
        native_reward = _normalize_reward_value(native_reward)
        task_reward, reward_components = self.reward_schema.compute(
            previous_state=self._last_raw_state,
            next_state=raw_state,
            management_state=self.management_state,
            applied_action=action_dict,
            terminated=bool(terminated or truncated),
            metadata=info,
        )
        observation = self.observation_schema.encode(
            raw_state=raw_state,
            context=dict(info.get("context", {}) or {}),
            management_state=self.management_state,
            metadata=info,
        )
        self._last_raw_state = raw_state
        self._last_info = info
        if self.episode_reporter is not None:
            self.episode_reporter.record_transition(
                raw_state=raw_state,
                applied_action=action_dict,
                task_reward=task_reward,
                native_reward=native_reward,
                terminated=bool(terminated or truncated),
            )
            if terminated or truncated:
                report = self.episode_reporter.finalize()
                summary = dict(info.get("episode_summary", {}))
                summary.update(report)
                info["episode_report"] = report
                info["episode_summary"] = summary
        info["task_reward"] = task_reward
        info["native_reward"] = native_reward
        info["reward_components"] = reward_components
        info["applied_action"] = action_dict
        info["management_state"] = self.management_state
        info["observation_features"] = self.observation_schema.feature_names
        if hasattr(self.observation_schema, "describe"):
            info["observation_schema"] = self.observation_schema.describe()
        if hasattr(self.action_schema, "describe"):
            info["action_schema"] = self.action_schema.describe()
        if hasattr(self.action_schema, "valid_action_mask"):
            info["action_mask"] = self.action_schema.valid_action_mask(raw_state, self.management_state)
        if hasattr(self.reward_schema, "describe"):
            info["reward_schema"] = self.reward_schema.describe()
        return observation, task_reward, terminated, truncated, info

    def close(self):
        return self.base_env.close()
