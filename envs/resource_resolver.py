import glob
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .crop_registry import PROJECT_ROOT, CropConfig, get_crop_config


@dataclass
class ResolvedCropResources:
    crop_name: str
    run_dssat_location: str
    filex_template_path: str
    pdi_template_path: str
    auxiliary_file_paths: list[str]
    extra_env_kwargs: dict[str, object] = field(default_factory=dict)


def _find_existing_path(candidates: list[str]) -> str:
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"未找到可用资源文件: {candidates}")


def _candidate_data_directories() -> list[str]:
    dssat_home = os.environ.get("DSSAT_HOME")
    candidates = [
        os.environ.get("DSSAT_DATA_DIR"),
        None if dssat_home is None else str(Path(dssat_home) / "data"),
        dssat_home,
        str(PROJECT_ROOT / "lib" / "gym_dssat_pdi_official" / "flatten_data"),
        str(PROJECT_ROOT / "lib" / "gym_dssat_pdi_official" / "dssat-csm-data"),
        "/opt/dssat_env/data",
    ]
    resolved: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = str(Path(candidate).expanduser())
        if os.path.isdir(candidate_path) and candidate_path not in resolved:
            resolved.append(candidate_path)
    return resolved


def _resolve_auxiliary_path(path_or_name: str) -> str:
    candidate = str(path_or_name).strip()
    if not candidate:
        raise FileNotFoundError("辅助文件路径不能为空")
    expanded_candidate = str(Path(candidate).expanduser())
    if os.path.isabs(expanded_candidate) or os.path.exists(expanded_candidate):
        if os.path.exists(expanded_candidate):
            return expanded_candidate
        raise FileNotFoundError(f"未找到辅助文件: {candidate}")
    basename = os.path.basename(expanded_candidate)
    search_roots = _candidate_data_directories()
    for root in search_roots:
        direct_candidates = [
            os.path.join(root, expanded_candidate),
            os.path.join(root, basename),
        ]
        for direct_candidate in direct_candidates:
            if os.path.exists(direct_candidate):
                return direct_candidate
        recursive_matches = sorted(glob.glob(os.path.join(root, "**", basename), recursive=True))
        if recursive_matches:
            return recursive_matches[0]
    raise FileNotFoundError(f"未找到辅助文件 {candidate}; 已搜索目录: {search_roots}")


def _resolve_run_dssat_location(run_dssat_location: str | None) -> str:
    explicit = None if run_dssat_location is None else str(run_dssat_location).strip()
    if explicit:
        expanded = str(Path(explicit).expanduser())
        if os.path.exists(expanded):
            return expanded
        resolved = shutil.which(explicit)
        if resolved is not None:
            return resolved
    environment_candidates = [
        os.environ.get("DSSAT_RUN_PATH"),
        str(PROJECT_ROOT / "lib" / "gym_dssat_pdi_official" / "install_dir" / "run_dssat"),
        "/opt/dssat_env/inst/run_dssat",
    ]
    for candidate in environment_candidates:
        if not candidate:
            continue
        expanded = str(Path(candidate).expanduser())
        if os.path.exists(expanded):
            return expanded
    system_candidate = shutil.which("run_dssat")
    if system_candidate is not None:
        return system_candidate
    return explicit or "run_dssat"


def _find_venv_wheat_config(filename: str) -> str:
    pattern = str(PROJECT_ROOT / "venv" / "lib" / "python*" / "site-packages" / "gym_dssat_pdi" / "envs" / "configs" / "wheat" / filename)
    matches = sorted(glob.glob(pattern))
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


def _merge_auxiliary_files(default_files: list[str], extra_files: list[str] | None) -> list[str]:
    merged = list(default_files)
    resolved_paths: list[str] = []
    seen: set[str] = set()
    if not extra_files:
        for path in merged:
            resolved = _resolve_auxiliary_path(path)
            if resolved not in seen:
                seen.add(resolved)
                resolved_paths.append(resolved)
        return resolved_paths
    extra_weather_files = [path for path in extra_files if str(path).lower().endswith(".wth")]
    if extra_weather_files:
        merged = [path for path in merged if not str(path).lower().endswith(".wth")]
    merged.extend(str(path) for path in extra_files if str(path).strip())
    for path in merged:
        resolved = _resolve_auxiliary_path(path)
        if resolved not in seen:
            seen.add(resolved)
            resolved_paths.append(resolved)
    return resolved_paths


def resolve_crop_resources(
    crop_name: str,
    run_dssat_location: str | None = None,
    extra_auxiliary_files: list[str] | None = None,
) -> ResolvedCropResources:
    crop_config: CropConfig = get_crop_config(crop_name)
    auxiliary_file_paths = list(crop_config.auxiliary_files)
    extra_env_kwargs: dict[str, object] = {}
    filex_template_path = crop_config.filex_template
    pdi_template_path = crop_config.pdi_template

    if crop_name == "wheat":
        pdi_template_path = _prepare_wheat_pdi_template(_find_venv_wheat_config("dssat_pdi.jinja2"))
        filex_template_path = _find_existing_path([
            str(PROJECT_ROOT / "templates" / "wheat" / "KSAS8101.jinja2"),
            str(PROJECT_ROOT / "lib" / "gym_dssat_pdi_official" / "dssat-csm-data" / "Wheat" / "KSAS8101.WHX"),
        ])
        extra_env_kwargs["random_weather"] = False

    auxiliary_file_paths = _merge_auxiliary_files(auxiliary_file_paths, extra_auxiliary_files)

    return ResolvedCropResources(
        crop_name=crop_name,
        run_dssat_location=_resolve_run_dssat_location(run_dssat_location),
        filex_template_path=filex_template_path,
        pdi_template_path=pdi_template_path,
        auxiliary_file_paths=auxiliary_file_paths,
        extra_env_kwargs=extra_env_kwargs,
    )
