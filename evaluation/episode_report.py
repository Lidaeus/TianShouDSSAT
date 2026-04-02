from dataclasses import asdict, dataclass
from typing import Any


EPISODE_REPORT_VERSION = "phase1.v1"


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


@dataclass
class EpisodeReport:
    crop_name: str
    split_name: str
    weather_id: str | None
    episode_steps: int
    task_return: float
    native_return: float
    total_fertilizer: float
    total_irrigation: float
    final_grain_weight: float
    final_biomass: float
    final_leaching: float
    final_denitrification: float
    nitrogen_use_efficiency: float
    water_use_efficiency: float
    gross_revenue: float
    fertilizer_input_cost: float
    irrigation_input_cost: float
    total_input_cost: float
    profit: float

    def to_dict(self) -> dict[str, float | str | int]:
        return asdict(self)

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(cls.__dataclass_fields__.keys())

    @classmethod
    def metric_groups(cls) -> dict[str, tuple[str, ...]]:
        return {
            "identity": ("crop_name", "split_name", "weather_id"),
            "episode": ("episode_steps", "task_return", "native_return"),
            "inputs": ("total_fertilizer", "total_irrigation"),
            "outputs": ("final_grain_weight", "final_biomass"),
            "environmental_costs": ("final_leaching", "final_denitrification"),
            "efficiency": ("nitrogen_use_efficiency", "water_use_efficiency"),
            "economics": (
                "gross_revenue",
                "fertilizer_input_cost",
                "irrigation_input_cost",
                "total_input_cost",
                "profit",
            ),
        }

    @classmethod
    def describe(cls) -> dict[str, Any]:
        return {
            "version": EPISODE_REPORT_VERSION,
            "fields": cls.field_names(),
            "metric_groups": cls.metric_groups(),
        }


class EpisodeReporter:
    def __init__(
        self,
        grain_price: float = 0.25,
        fertilizer_unit_cost: float = 1.2,
        irrigation_unit_cost: float = 0.05,
    ) -> None:
        self.grain_price = float(grain_price)
        self.fertilizer_unit_cost = float(fertilizer_unit_cost)
        self.irrigation_unit_cost = float(irrigation_unit_cost)
        self.reset(crop_name="unknown")

    def reset(self, crop_name: str, metadata: dict[str, Any] | None = None) -> None:
        metadata = dict(metadata or {})
        self.crop_name = crop_name
        self.split_name = str(metadata.get("split_name", "unknown"))
        self.weather_id = metadata.get("weather_id")
        self.episode_steps = 0
        self.task_return = 0.0
        self.native_return = 0.0
        self.total_fertilizer = 0.0
        self.total_irrigation = 0.0
        self.final_state: dict[str, Any] = {}

    def record_transition(
        self,
        raw_state: dict[str, Any],
        applied_action: dict[str, float],
        task_reward: float,
        native_reward: float,
        terminated: bool,
    ) -> None:
        self.episode_steps += 1
        self.task_return += float(task_reward)
        self.native_return += float(native_reward)
        self.total_fertilizer += float(applied_action.get("anfer", 0.0) or 0.0)
        self.total_irrigation += float(applied_action.get("amir", 0.0) or 0.0)
        if terminated:
            self.final_state = dict(raw_state or {})

    def finalize_report(self) -> EpisodeReport:
        final_grain_weight = _safe_float(self.final_state.get("grnwt"))
        total_fertilizer = self.total_fertilizer
        total_irrigation = self.total_irrigation
        gross_revenue = final_grain_weight * self.grain_price
        fertilizer_input_cost = total_fertilizer * self.fertilizer_unit_cost
        irrigation_input_cost = total_irrigation * self.irrigation_unit_cost
        total_input_cost = fertilizer_input_cost + irrigation_input_cost
        return EpisodeReport(
            crop_name=self.crop_name,
            split_name=self.split_name,
            weather_id=self.weather_id,
            episode_steps=self.episode_steps,
            task_return=self.task_return,
            native_return=self.native_return,
            total_fertilizer=total_fertilizer,
            total_irrigation=total_irrigation,
            final_grain_weight=final_grain_weight,
            final_biomass=_safe_float(self.final_state.get("topwt")),
            final_leaching=_safe_float(self.final_state.get("tleachd")),
            final_denitrification=_safe_float(self.final_state.get("tnoxd")),
            nitrogen_use_efficiency=final_grain_weight / total_fertilizer if total_fertilizer > 0 else 0.0,
            water_use_efficiency=final_grain_weight / total_irrigation if total_irrigation > 0 else 0.0,
            gross_revenue=gross_revenue,
            fertilizer_input_cost=fertilizer_input_cost,
            irrigation_input_cost=irrigation_input_cost,
            total_input_cost=total_input_cost,
            profit=gross_revenue - total_input_cost,
        )

    def finalize(self) -> dict[str, float | str | int]:
        report = self.finalize_report().to_dict()
        report["report_version"] = EPISODE_REPORT_VERSION
        report["report_schema"] = EpisodeReport.describe()
        return report
