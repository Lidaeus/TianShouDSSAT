from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int | None:
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


def _ordinal_from_yrdoy(value: Any) -> int | None:
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


def _ordinal_from_date_like(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().toordinal()
    if isinstance(value, date):
        return value.toordinal()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = int(value)
        if numeric > 1_000_000:
            try:
                return datetime.strptime(str(numeric), "%Y%m%d").date().toordinal()
            except ValueError:
                return _ordinal_from_yrdoy(numeric)
        yrdoy_ordinal = _ordinal_from_yrdoy(numeric)
        if yrdoy_ordinal is not None:
            return yrdoy_ordinal
        if numeric > 500_000:
            return numeric
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        for pattern in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y-%j", "%Y/%j"):
            try:
                return datetime.strptime(stripped, pattern).date().toordinal()
            except ValueError:
                continue
        if stripped.isdigit():
            return _ordinal_from_date_like(int(stripped))
        return None
    return None


def _extract_entry_ordinal(day: dict[str, Any]) -> int | None:
    for key in ("ordinal", "date", "DATE", "wdate", "WDATE", "yrdoy", "YRDOY"):
        ordinal = _ordinal_from_date_like(day.get(key))
        if ordinal is not None:
            return ordinal
    year = _safe_int(day.get("year", day.get("YEAR")))
    doy = _safe_int(day.get("doy", day.get("DOY")))
    if year is not None and doy is not None:
        return _ordinal_from_yrdoy(int(f"{year}{doy:03d}"))
    return None


def _normalize_daily_forecast(raw_daily_forecast: list[dict[str, Any]]) -> list[dict[str, float]]:
    normalized: list[dict[str, float]] = []
    for day in raw_daily_forecast:
        rain = _safe_float(day.get("rain", day.get("RAIN")))
        tmin = _safe_float(day.get("tmin", day.get("TMIN")))
        tmax = _safe_float(day.get("tmax", day.get("TMAX")))
        srad = _safe_float(day.get("srad", day.get("SRAD")))
        tmean = _safe_float(day.get("tmean", day.get("TMEAN")))
        if tmean == 0.0 and (tmin != 0.0 or tmax != 0.0):
            tmean = (tmin + tmax) / 2.0
        normalized.append({
            "rain": rain,
            "tmin": tmin,
            "tmax": tmax,
            "tmean": tmean,
            "srad": srad,
        })
    return normalized


def _prepare_daily_forecast(raw_daily_forecast: list[dict[str, Any]], current_ordinal: int | None = None) -> list[dict[str, float]]:
    dated_entries: list[tuple[int, dict[str, float]]] = []
    undated_entries: list[dict[str, float]] = []
    for day in raw_daily_forecast:
        normalized_day = _normalize_daily_forecast([day])[0]
        ordinal = _extract_entry_ordinal(day)
        if ordinal is None:
            undated_entries.append(normalized_day)
        else:
            dated_entries.append((ordinal, normalized_day))
    if current_ordinal is None or not dated_entries:
        return [entry for _, entry in sorted(dated_entries, key=lambda item: item[0])] + undated_entries if dated_entries else undated_entries
    future_entries = [entry for ordinal, entry in sorted(dated_entries, key=lambda item: item[0]) if ordinal > current_ordinal]
    if future_entries:
        return future_entries + undated_entries
    return undated_entries


def _summarize_daily_forecast(raw_daily_forecast: list[dict[str, Any]], horizons: tuple[int, ...]) -> dict[str, float]:
    daily_forecast = _prepare_daily_forecast(raw_daily_forecast)
    summary: dict[str, float] = {}
    for horizon in horizons:
        window = daily_forecast[:horizon]
        if not window:
            summary[f"rain_{horizon}d"] = 0.0
            summary[f"tmean_{horizon}d"] = 0.0
            summary[f"srad_{horizon}d"] = 0.0
            continue
        summary[f"rain_{horizon}d"] = sum(day["rain"] for day in window)
        summary[f"tmean_{horizon}d"] = sum(day["tmean"] for day in window) / len(window)
        summary[f"srad_{horizon}d"] = sum(day["srad"] for day in window) / len(window)
    return summary


class BaseForecastProvider(ABC):
    @abstractmethod
    def enrich_metadata(self, raw_state: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class NullForecastProvider(BaseForecastProvider):
    def enrich_metadata(self, raw_state: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        return {}


class WeatherWindowForecastProvider(BaseForecastProvider):
    def __init__(self, horizons: tuple[int, ...] = (3, 7)) -> None:
        self.horizons = tuple(sorted(set(int(h) for h in horizons if int(h) > 0)))
        self._weather_cache: dict[str, list[dict[str, float | int]]] = {}

    def _extract_daily_forecast(self, info: dict[str, Any]) -> list[dict[str, Any]]:
        for key in (
            "weather_forecast_daily",
            "daily_forecast",
            "forecast_daily",
            "weather_forecast_raw",
        ):
            value = info.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        weather_forecast = info.get("weather_forecast")
        if isinstance(weather_forecast, dict) and "daily" in weather_forecast and isinstance(weather_forecast["daily"], list):
            return [item for item in weather_forecast["daily"] if isinstance(item, dict)]
        return []

    def _extract_weather_file_paths(self, info: dict[str, Any]) -> list[str]:
        candidates: list[str] = []
        for key in ("weather_file_path", "primary_weather_path"):
            value = info.get(key)
            if isinstance(value, str) and value:
                candidates.append(value)
        for key in ("weather_file_paths", "auxiliary_file_paths"):
            value = info.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item:
                        candidates.append(item)
        unique_candidates: list[str] = []
        for candidate in candidates:
            if candidate.lower().endswith(".wth") and candidate not in unique_candidates:
                unique_candidates.append(candidate)
        return unique_candidates

    def _load_weather_file(self, weather_file_path: str) -> list[dict[str, float | int]]:
        resolved_path = str(Path(weather_file_path).resolve())
        cached = self._weather_cache.get(resolved_path)
        if cached is not None:
            return cached
        rows: list[dict[str, float | int]] = []
        header_tokens: list[str] = []
        with open(resolved_path, "r", encoding="utf-8") as file:
            for raw_line in file:
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
                rows.append({
                    "ordinal": ordinal,
                    "rain": _safe_float(row.get("RAIN")),
                    "tmin": _safe_float(row.get("TMIN")),
                    "tmax": _safe_float(row.get("TMAX")),
                    "tmean": _safe_float(row.get("TMEAN")),
                    "srad": _safe_float(row.get("SRAD")),
                })
        rows.sort(key=lambda item: int(item["ordinal"]))
        self._weather_cache[resolved_path] = rows
        return rows

    def _build_from_weather_file(self, raw_state: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        current_ordinal = _ordinal_from_yrdoy(raw_state.get("yrdoy", info.get("yrdoy")))
        if current_ordinal is None:
            return {}
        for weather_file_path in self._extract_weather_file_paths(info):
            try:
                daily_records = self._load_weather_file(weather_file_path)
            except OSError:
                continue
            future_records = [record for record in daily_records if int(record["ordinal"]) > current_ordinal]
            if not future_records:
                continue
            normalized_future = _normalize_daily_forecast([
                {
                    "rain": record["rain"],
                    "tmin": record["tmin"],
                    "tmax": record["tmax"],
                    "tmean": record["tmean"],
                    "srad": record["srad"],
                }
                for record in future_records
            ])
            return {
                "weather_forecast": _summarize_daily_forecast(normalized_future, self.horizons),
                "weather_forecast_raw": normalized_future[:max(self.horizons, default=0)],
                "weather_forecast_available_days": len(normalized_future),
                "weather_forecast_source": weather_file_path,
            }
        return {}

    def enrich_metadata(self, raw_state: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        existing_forecast = info.get("weather_forecast")
        if isinstance(existing_forecast, dict):
            required_keys = {
                f"{metric}_{horizon}d"
                for horizon in self.horizons
                for metric in ("rain", "tmean", "srad")
            }
            if required_keys.issubset(existing_forecast.keys()):
                return {"weather_forecast": existing_forecast}
        daily_forecast = self._extract_daily_forecast(info)
        if not daily_forecast:
            return self._build_from_weather_file(raw_state, info) or {"weather_forecast": {}}
        current_ordinal = _ordinal_from_yrdoy(raw_state.get("yrdoy", info.get("yrdoy")))
        normalized_daily_forecast = _prepare_daily_forecast(daily_forecast, current_ordinal=current_ordinal)
        return {
            "weather_forecast": _summarize_daily_forecast(normalized_daily_forecast, self.horizons),
            "weather_forecast_raw": normalized_daily_forecast,
            "weather_forecast_available_days": len(normalized_daily_forecast),
            "weather_forecast_source": "daily_forecast",
        }
