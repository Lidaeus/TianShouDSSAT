from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field


def _normalize_positive_int_tuple(values, default: tuple[int, ...]) -> tuple[int, ...]:
    if values is None:
        return default
    candidates = values if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) else [values]
    normalized = sorted({int(value) for value in candidates if int(value) > 0})
    return tuple(normalized) or default


def _normalize_float_tuple(values, default: tuple[float, ...]) -> tuple[float, ...]:
    if values is None:
        return default
    candidates = values if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) else [values]
    normalized = [float(value) for value in candidates]
    return tuple(normalized) or default


def _normalize_string_tuple(values) -> tuple[str, ...]:
    if values is None:
        return ()
    candidates = values if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) else [values]
    normalized = []
    for value in candidates:
        item = str(value).strip()
        if item and item not in normalized:
            normalized.append(item)
    return tuple(normalized)


def _normalize_split_name(value) -> str:
    split_name = str(value or "train").strip().lower()
    return split_name if split_name in {"train", "validation", "test"} else "train"


def _normalize_optional_split_name(value) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    return _normalize_split_name(normalized)


def _normalize_split_ratios(values) -> tuple[float, float, float]:
    default = (0.7, 0.15, 0.15)
    if isinstance(values, Mapping):
        raw = (
            float(values.get("train", default[0]) or 0.0),
            float(values.get("validation", default[1]) or 0.0),
            float(values.get("test", default[2]) or 0.0),
        )
    else:
        normalized = _normalize_float_tuple(values, default)
        raw = (
            normalized[0] if len(normalized) > 0 else default[0],
            normalized[1] if len(normalized) > 1 else default[1],
            normalized[2] if len(normalized) > 2 else default[2],
        )
    clipped = tuple(max(0.0, float(value)) for value in raw)
    total = sum(clipped)
    if total <= 0.0:
        return default
    return tuple(value / total for value in clipped)


def _coerce_dataclass_config(value, config_cls):
    if isinstance(value, config_cls):
        return value
    if isinstance(value, Mapping):
        return config_cls(**value)
    return config_cls() if value is None else config_cls(value)


def _normalize_actor_names(values) -> tuple[str, ...]:
    default = ("zero", "calendar", "rule_based")
    normalized = _normalize_string_tuple(values)
    return normalized or default


def _normalize_seed_tuple(values) -> tuple[int, ...]:
    if values is None:
        return ()
    candidates = values if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) else [values]
    normalized = []
    for value in candidates:
        seed = int(value)
        if seed not in normalized:
            normalized.append(seed)
    return tuple(normalized)


@dataclass
class ObservationConfig:
    include_forecast: bool = False
    forecast_horizons: tuple[int, ...] = (3, 7)

    def __post_init__(self) -> None:
        self.include_forecast = bool(self.include_forecast)
        self.forecast_horizons = _normalize_positive_int_tuple(self.forecast_horizons, (3, 7))


@dataclass
class ActionConfig:
    nitrogen_levels: tuple[float, ...] = (0.0, 30.0, 60.0, 90.0)
    irrigation_levels: tuple[float, ...] = (0.0, 10.0, 20.0, 30.0)
    decision_interval_days: int = 7
    fertilizer_budget: float = 180.0
    irrigation_budget: float = 240.0
    fertilizer_stage_cutoff: float = 6.0

    def __post_init__(self) -> None:
        self.nitrogen_levels = _normalize_float_tuple(self.nitrogen_levels, (0.0, 30.0, 60.0, 90.0))
        self.irrigation_levels = _normalize_float_tuple(self.irrigation_levels, (0.0, 10.0, 20.0, 30.0))
        self.decision_interval_days = max(1, int(self.decision_interval_days))
        self.fertilizer_budget = float(self.fertilizer_budget)
        self.irrigation_budget = float(self.irrigation_budget)
        self.fertilizer_stage_cutoff = float(self.fertilizer_stage_cutoff)


@dataclass
class RewardConfig:
    grain_delta_weight: float = 0.001
    terminal_yield_weight: float = 0.05
    fertilizer_cost_weight: float = 0.02
    irrigation_cost_weight: float = 0.01
    leaching_cost_weight: float = 0.1
    denitrification_cost_weight: float = 0.1

    def __post_init__(self) -> None:
        self.grain_delta_weight = float(self.grain_delta_weight)
        self.terminal_yield_weight = float(self.terminal_yield_weight)
        self.fertilizer_cost_weight = float(self.fertilizer_cost_weight)
        self.irrigation_cost_weight = float(self.irrigation_cost_weight)
        self.leaching_cost_weight = float(self.leaching_cost_weight)
        self.denitrification_cost_weight = float(self.denitrification_cost_weight)


@dataclass
class EconomicsConfig:
    grain_price: float = 0.25
    fertilizer_unit_cost: float = 1.2
    irrigation_unit_cost: float = 0.05

    def __post_init__(self) -> None:
        self.grain_price = float(self.grain_price)
        self.fertilizer_unit_cost = float(self.fertilizer_unit_cost)
        self.irrigation_unit_cost = float(self.irrigation_unit_cost)


@dataclass
class ProtocolConfig:
    split_name: str = "train"
    weather_files: tuple[str, ...] = ()
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
    split_seed: int = 42
    train_weather_files: tuple[str, ...] = ()
    validation_weather_files: tuple[str, ...] = ()
    test_weather_files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.split_name = _normalize_split_name(self.split_name)
        self.weather_files = _normalize_string_tuple(self.weather_files)
        self.split_ratios = _normalize_split_ratios(self.split_ratios)
        self.split_seed = int(self.split_seed)
        self.train_weather_files = _normalize_string_tuple(self.train_weather_files)
        self.validation_weather_files = _normalize_string_tuple(self.validation_weather_files)
        self.test_weather_files = _normalize_string_tuple(self.test_weather_files)


@dataclass
class Phase1TaskConfig:
    crop_name: str = "wheat"
    mode: str = "all"
    seed: int | None = 42
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    economics: EconomicsConfig = field(default_factory=EconomicsConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)

    def __post_init__(self) -> None:
        self.crop_name = str(self.crop_name).strip() or "wheat"
        self.mode = str(self.mode).strip() or "all"
        self.seed = None if self.seed is None else int(self.seed)
        self.observation = _coerce_dataclass_config(self.observation, ObservationConfig)
        self.action = _coerce_dataclass_config(self.action, ActionConfig)
        self.reward = _coerce_dataclass_config(self.reward, RewardConfig)
        self.economics = _coerce_dataclass_config(self.economics, EconomicsConfig)
        self.protocol = _coerce_dataclass_config(self.protocol, ProtocolConfig)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentConfig:
    task: Phase1TaskConfig = field(default_factory=Phase1TaskConfig)
    algorithm_name: str = "rainbow"
    train_env_num: int = 1
    test_env_num: int = 1
    log_dir: str = "logs"
    evaluation_episodes: int = 3
    evaluation_seed_stride: int = 1000
    evaluation_split_name: str | None = None
    evaluation_base_seeds: tuple[int, ...] = ()
    baseline_actors: tuple[str, ...] = ("zero", "calendar", "rule_based")

    def __post_init__(self) -> None:
        self.task = _coerce_dataclass_config(self.task, Phase1TaskConfig)
        self.algorithm_name = str(self.algorithm_name).strip() or "rainbow"
        self.train_env_num = max(1, int(self.train_env_num))
        self.test_env_num = max(1, int(self.test_env_num))
        self.log_dir = str(self.log_dir).strip() or "logs"
        self.evaluation_episodes = max(1, int(self.evaluation_episodes))
        self.evaluation_seed_stride = max(1, int(self.evaluation_seed_stride))
        self.evaluation_split_name = _normalize_optional_split_name(self.evaluation_split_name)
        self.evaluation_base_seeds = _normalize_seed_tuple(self.evaluation_base_seeds)
        self.baseline_actors = _normalize_actor_names(self.baseline_actors)

    def to_dict(self) -> dict:
        return asdict(self)


def build_phase1_benchmark_config(
    crop_name: str = "wheat",
    seed: int | None = 42,
    split_name: str = "train",
    evaluation_split_name: str | None = "test",
    weather_files=None,
    train_weather_files=None,
    validation_weather_files=None,
    test_weather_files=None,
) -> ExperimentConfig:
    return ExperimentConfig(
        task=Phase1TaskConfig(
            crop_name=crop_name,
            seed=seed,
            observation=ObservationConfig(
                include_forecast=True,
                forecast_horizons=(3, 7),
            ),
            protocol=ProtocolConfig(
                split_name=split_name,
                weather_files=weather_files,
                train_weather_files=train_weather_files,
                validation_weather_files=validation_weather_files,
                test_weather_files=test_weather_files,
            ),
        ),
        log_dir="logs/benchmarks/phase1_single_crop",
        evaluation_episodes=3,
        evaluation_seed_stride=1000,
        evaluation_split_name=evaluation_split_name,
        evaluation_base_seeds=(11, 23, 37),
        baseline_actors=("zero", "calendar", "rule_based"),
    )
