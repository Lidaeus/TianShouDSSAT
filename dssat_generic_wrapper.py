import os
import sys
import glob
import tempfile
from datetime import date, timedelta
import gymnasium as gym
import numpy as np
from typing import Dict, List, Any, Optional

try:
    from tasks.forecast import WeatherWindowForecastProvider
except ModuleNotFoundError:
    WeatherWindowForecastProvider = None

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GYM_DSSAT_PDI_PATH = os.path.join(PROJECT_ROOT, "lib", "gym_dssat_pdi_official", "gym-dssat-pdi")
if GYM_DSSAT_PDI_PATH not in sys.path:
    sys.path.insert(0, GYM_DSSAT_PDI_PATH)

from gym_dssat_pdi.envs.dssat_pdi import DssatPdi

# 作物特定配置矩阵：避免过度抽象，明确定义差异
CROP_SPECIFIC_CONFIGS = {
    "maize": {
        "cultivar_code": "UFGA8201",
        "obs_keys": [
            "swfac", "vstage", "wtdep", "grnwt", 
            "topwt", "lai", "pcntn", "stresn"
        ],
        "default_template": "templates/maize/UFGA8201.jinja2",
        "is_transplanted": False
    },
    "tomato": {
        "cultivar_code": "UFGA0602",
        "obs_keys": [
            "swfac", "istage", "rtdep", "grnwt", 
            "topwt", "xlai", "nstres"
        ],
        "default_template": "templates/tomato/UFGA0602.jinja2",
        "is_transplanted": True
    },
    "wheat": {
        "cultivar_code": "KSAS8101",
        "obs_keys": [
            "swfac", "istage", "vstage", "grnwt",
            "topwt", "xlai", "nstres", "rtdep"
        ],
        "default_template": "templates/wheat/KSAS8101.jinja2",
        "is_transplanted": False
    }
}


def _find_existing_path(candidates: List[str]) -> str:
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"未找到可用资源文件: {candidates}")


def _find_venv_wheat_config(filename: str) -> str:
    matches = sorted(glob.glob(os.path.join(PROJECT_ROOT, "venv", "lib", "python*", "site-packages",
                                            "gym_dssat_pdi", "envs", "configs", "wheat", filename)))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"未找到 Wheat 配置文件: {filename}")


def _prepare_wheat_pdi_template(template_path: str) -> str:
    with open(template_path, "r", encoding="utf-8") as f:
        content = f.read()

    content = content.replace("pdi_early_stopping.itemset(is_early_stopping)", "pdi_early_stopping[...] = is_early_stopping")
    content = content.replace("anfer.itemset(action['anfer'])", "anfer[...] = action['anfer']")
    content = content.replace("amir.itemset(action['amir'])", "amir[...] = action['amir']")
    content = content.replace(
        "          is_early_stopping = False\n          is_last_send = False\n          print('Client started')",
        "          is_early_stopping = False\n"
        "          is_last_send = False\n"
        "          yrdoy = 0\n"
        "          nstres = 0.0\n"
        "          istage = 0\n"
        "          vstage = 0.0\n"
        "          grnwt = 0.0\n"
        "          topwt = 0.0\n"
        "          pltpop = 0.0\n"
        "          swfac = 0.0\n"
        "          pcngrn = 0.0\n"
        "          tleachd = 0.0\n"
        "          tnoxd = 0.0\n"
        "          wtnup = 0.0\n"
        "          cleach = 0.0\n"
        "          cnox = 0.0\n"
        "          cumsumfert = 0.0\n"
        "          xlai = 0.0\n"
        "          trnu = 0.0\n"
        "          rain = 0.0\n"
        "          tmin = 0.0\n"
        "          tmax = 0.0\n"
        "          srad = 0.0\n"
        "          dtt = 0.0\n"
        "          dap = 0\n"
        "          es = 0.0\n"
        "          eo = 0.0\n"
        "          eop = 0.0\n"
        "          eos = 0.0\n"
        "          ep = 0.0\n"
        "          runoff = 0.0\n"
        "          wtdep = 0.0\n"
        "          rtdep = 0.0\n"
        "          totaml = 0.0\n"
        "          totir = 0.0\n"
        "          ll = []\n"
        "          dul = []\n"
        "          sw = []\n"
        "          ds = []\n"
        "          sat = []\n"
        "          dlayr = []\n"
        "          print('Client started')"
    )

    tmp = tempfile.NamedTemporaryFile("w", suffix="_wheat_pdi.jinja2", delete=False, encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return tmp.name


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _expand_year(year: int) -> int:
    if year >= 100:
        return year
    return 1900 + year if year >= 50 else 2000 + year


def _ordinal_from_yrdoy(value: Any) -> Optional[int]:
    numeric = _safe_int(value)
    if numeric is None:
        return None
    digits = str(abs(numeric))
    if len(digits) >= 7:
        year = int(digits[:-3])
        doy = int(digits[-3:])
    elif len(digits) == 5:
        year = _expand_year(int(digits[:2]))
        doy = int(digits[2:])
    else:
        return None
    if doy <= 0:
        return None
    try:
        return (date(year, 1, 1) + timedelta(days=doy - 1)).toordinal()
    except ValueError:
        return None


def _normalize_daily_weather(day: Dict[str, Any]) -> Dict[str, float]:
    tmin = _safe_float(day.get("tmin", day.get("TMIN")))
    tmax = _safe_float(day.get("tmax", day.get("TMAX")))
    tmean = _safe_float(day.get("tmean", day.get("TMEAN")))
    if tmean == 0.0 and (tmin != 0.0 or tmax != 0.0):
        tmean = (tmin + tmax) / 2.0
    return {
        "rain": _safe_float(day.get("rain", day.get("RAIN"))),
        "tmin": tmin,
        "tmax": tmax,
        "tmean": tmean,
        "srad": _safe_float(day.get("srad", day.get("SRAD"))),
    }


def _summarize_daily_weather(daily_weather: List[Dict[str, Any]], horizons: List[int]) -> Dict[str, float]:
    normalized = [_normalize_daily_weather(day) for day in daily_weather]
    summary: Dict[str, float] = {}
    for horizon in horizons:
        window = normalized[:horizon]
        if not window:
            summary[f"rain_{horizon}d"] = 0.0
            summary[f"tmean_{horizon}d"] = 0.0
            summary[f"srad_{horizon}d"] = 0.0
            continue
        summary[f"rain_{horizon}d"] = float(sum(day["rain"] for day in window))
        summary[f"tmean_{horizon}d"] = float(sum(day["tmean"] for day in window) / len(window))
        summary[f"srad_{horizon}d"] = float(sum(day["srad"] for day in window) / len(window))
    return summary


def _dedupe_paths(paths: List[str]) -> List[str]:
    deduped: List[str] = []
    seen: set[str] = set()
    for path in paths:
        normalized = str(path).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


class _FallbackWeatherWindowForecastProvider:
    def __init__(self, horizons: tuple[int, ...]):
        self.horizons = tuple(sorted({int(h) for h in horizons if int(h) > 0}))

    def enrich_metadata(self, raw_obs: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        daily_weather = info.get("weather_forecast_daily")
        if not isinstance(daily_weather, list):
            daily_weather = []
        forecast_summary = _summarize_daily_weather(
            [day for day in daily_weather if isinstance(day, dict)],
            list(self.horizons),
        )
        return {"weather_forecast": forecast_summary}

class DssatGenericWrapper(gym.Wrapper):
    """
    Tianshou 2.0 兼容的通用 DSSAT 包装器。
    支持多作物切换，通过配置驱动解决作物间的生理差异。
    """
    def __init__(
        self, 
        crop_name: str,
        env_kwargs: Optional[Dict[str, Any]] = None,
        history_len: int = 5,
        seed: Optional[int] = None,
        include_forecast: bool = False,
        forecast_horizons: tuple[int, ...] = (3, 7),
    ):
        if crop_name not in CROP_SPECIFIC_CONFIGS:
            raise ValueError(f"不支持的作物: {crop_name}. 可选: {list(CROP_SPECIFIC_CONFIGS.keys())}")
        
        self.crop_name = crop_name
        self.config = CROP_SPECIFIC_CONFIGS[crop_name]
        self.history_len = history_len
        self.include_forecast = include_forecast
        self.forecast_horizons = sorted({int(h) for h in forecast_horizons if int(h) > 0})
        forecast_provider_cls = WeatherWindowForecastProvider or _FallbackWeatherWindowForecastProvider
        self._forecast_provider = forecast_provider_cls(tuple(self.forecast_horizons)) if include_forecast else None
        self._generated_temp_paths = []
        self._weather_records_cache: Dict[str, List[Dict[str, float]]] = {}
        env_kwargs = dict(env_kwargs or {})
        
        # 1. 准备初始化参数
        final_env_kwargs = {
            "run_dssat_location": "/opt/dssat_env/inst/run_dssat",
            "cultivar": self.crop_name,
            "fileX_template_path": self.config["default_template"],
            "pdi_template_path": f"templates/{self.crop_name}/dssat_pdi.jinja2",
            "seed": seed
        }
        
        # 自动补齐资源文件路径
        aux_files = list(env_kwargs.get("auxiliary_file_paths", []))
        if self.crop_name == "tomato":
            aux_files.extend(["/opt/dssat_env/data/UFGA.CLI", "/opt/dssat_env/data/SOIL.SOL", "/opt/dssat_env/data/UFGA0601.WTH"])
        elif self.crop_name == "maize":
            aux_files.extend(["/opt/dssat_env/data/UFGA.CLI", "/opt/dssat_env/data/SOIL.SOL", "/opt/dssat_env/data/UFGA8201.WTH"])
        elif self.crop_name == "wheat":
            wheat_pdi_template = _prepare_wheat_pdi_template(_find_venv_wheat_config("dssat_pdi.jinja2"))
            self._generated_temp_paths.append(wheat_pdi_template)
            final_env_kwargs.update({
                "fileX_template_path": _find_existing_path([
                    os.path.join(PROJECT_ROOT, "lib", "gym_dssat_pdi_official", "dssat-csm-data", "Wheat", "KSAS8101.WHX"),
                ]),
                "pdi_template_path": wheat_pdi_template,
                "random_weather": False,
            })
            aux_files.extend(["/opt/dssat_env/data/SOIL.SOL", "/opt/dssat_env/data/KSAS8101.WTH"])
        
        final_env_kwargs.update(env_kwargs)
        final_env_kwargs["auxiliary_file_paths"] = _dedupe_paths(aux_files)
        self.weather_file_paths = [
            path for path in final_env_kwargs["auxiliary_file_paths"]
            if isinstance(path, str) and path.lower().endswith(".wth")
        ]
        
        self.obs_keys = self.config["obs_keys"]
        self._history_buffer = []
        
        env = DssatPdi(**final_env_kwargs)
        super().__init__(env)
        
        num_features = len(self.obs_keys) * self.history_len
        if self.include_forecast:
            num_features += len(self.forecast_horizons) * 3
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32
        )

        # 5. 定义离散动作空间 (动作离散化)
        # 氮肥 (kg/ha): 0, 40, 80, 120, 160, 200
        self.anfer_buckets = [0, 40, 80, 120, 160, 200]
        # 灌溉 (mm): 0, 10, 20, 30, 40, 50
        self.amir_buckets = [0, 10, 20, 30, 40, 50]
        
        self.num_anfer = len(self.anfer_buckets)
        self.num_amir = len(self.amir_buckets)
        self.action_space = gym.spaces.Discrete(self.num_anfer * self.num_amir)

    @staticmethod
    def summarize_forecast(daily_weather: List[Dict[str, Any]], horizons: tuple[int, ...] = (3, 7)) -> Dict[str, float]:
        normalized_horizons = sorted({int(h) for h in horizons if int(h) > 0})
        return _summarize_daily_weather(daily_weather, normalized_horizons)

    def _load_weather_records(self, weather_file_path: str) -> List[Dict[str, float]]:
        cached_records = self._weather_records_cache.get(weather_file_path)
        if cached_records is not None:
            return cached_records
        header_tokens: List[str] = []
        rows: List[Dict[str, float]] = []
        with open(weather_file_path, "r", encoding="utf-8") as weather_file:
            for raw_line in weather_file:
                line = raw_line.strip()
                if not line or line.startswith("*") or line.startswith("!"):
                    continue
                if line.startswith("@"):
                    header_tokens = line[1:].split()
                    continue
                if not header_tokens or "DATE" not in header_tokens:
                    continue
                values = line.split()
                if len(values) < len(header_tokens):
                    continue
                row = dict(zip(header_tokens, values))
                ordinal = _ordinal_from_yrdoy(row.get("DATE"))
                if ordinal is None:
                    continue
                normalized = _normalize_daily_weather(row)
                normalized["ordinal"] = float(ordinal)
                rows.append(normalized)
        rows.sort(key=lambda item: item["ordinal"])
        self._weather_records_cache[weather_file_path] = rows
        return rows

    def _extract_daily_forecast(self, info: Dict[str, Any]) -> List[Dict[str, Any]]:
        for key in ("weather_forecast_daily", "daily_forecast", "forecast_daily", "weather_forecast_raw"):
            value = info.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        weather_forecast = info.get("weather_forecast")
        if isinstance(weather_forecast, dict):
            daily = weather_forecast.get("daily")
            if isinstance(daily, list):
                return [item for item in daily if isinstance(item, dict)]
        return []

    def _build_forecast_payload(self, raw_obs: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        if self._forecast_provider is None:
            return {"weather_forecast": {}}
        enriched_info = dict(info)
        enriched_info["weather_file_paths"] = list(self.weather_file_paths)
        enriched_info["primary_weather_path"] = self.weather_file_paths[0] if self.weather_file_paths else None
        return self._forecast_provider.enrich_metadata(raw_obs, enriched_info)

    def _map_action(self, action_idx: int) -> Dict[str, float]:
        """将离散动作索引映射为物理值字典"""
        action_idx = int(action_idx) % self.action_space.n
        
        # 解码索引: action_idx = anfer_idx * num_amir + amir_idx
        anfer_idx = action_idx // self.num_amir
        amir_idx = action_idx % self.num_amir
        
        return {
            "anfer": float(self.anfer_buckets[anfer_idx]),
            "amir": float(self.amir_buckets[amir_idx])
        }

    def _extract_features(self, raw_obs: Dict[str, Any]) -> np.ndarray:
        vals = []
        for key in self.obs_keys:
            val = raw_obs.get(key, 0.0)
            if val is None: val = 0.0
            vals.append(float(val))
        return np.array(vals, dtype=np.float32)

    def _update_history(self, feat: np.ndarray) -> np.ndarray:
        if len(self._history_buffer) == 0:
            self._history_buffer = [feat] * self.history_len
        else:
            self._history_buffer.append(feat)
            self._history_buffer.pop(0)
        return np.concatenate(self._history_buffer)

    def _extract_forecast_features(self, forecast: Dict[str, Any]) -> np.ndarray:
        if not self.include_forecast:
            return np.array([], dtype=np.float32)
        features: List[float] = []
        for horizon in self.forecast_horizons:
            features.extend([
                _safe_float(forecast.get(f"rain_{horizon}d")),
                _safe_float(forecast.get(f"tmean_{horizon}d")),
                _safe_float(forecast.get(f"srad_{horizon}d")),
            ])
        return np.array(features, dtype=np.float32)

    def _build_observation(self, raw_obs: Dict[str, Any], info: Dict[str, Any]) -> np.ndarray:
        feat = self._extract_features(raw_obs)
        stacked_obs = self._update_history(feat)
        forecast_payload = self._build_forecast_payload(raw_obs, info) if self.include_forecast else {"weather_forecast": {}}
        info.update(forecast_payload)
        forecast_features = self._extract_forecast_features(info.get("weather_forecast", {}))
        if forecast_features.size == 0:
            return stacked_obs
        return np.concatenate([stacked_obs, forecast_features]).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._history_buffer = [] 
        if info is None: info = {}
        info["weather_file_paths"] = list(self.weather_file_paths)
        info["primary_weather_path"] = self.weather_file_paths[0] if self.weather_file_paths else None
        return self._build_observation(obs, info), info

    def step(self, action):
        action_dict = self._map_action(action)
        
        obs, reward, terminated, truncated, info = self.env.step(action_dict)
        
        if info is None: info = {}
        info["weather_file_paths"] = list(self.weather_file_paths)
        info["primary_weather_path"] = self.weather_file_paths[0] if self.weather_file_paths else None
        if isinstance(reward, list):
            final_reward = float(np.sum(reward))
        elif reward is not None:
            final_reward = float(reward)
        else:
            final_reward = 0.0
        info["applied_action"] = dict(action_dict)
        info["action_index"] = int(action) % self.action_space.n
        info["native_reward"] = final_reward
        stacked_obs = self._build_observation(obs, info)
            
        return stacked_obs, final_reward, terminated, truncated, info
    def close(self):
        try:
            return self.env.close()
        finally:
            for path in self._generated_temp_paths:
                if os.path.exists(path):
                    os.remove(path)
