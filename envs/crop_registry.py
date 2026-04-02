from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class CropConfig:
    name: str
    cultivar_code: str
    filex_template: str
    pdi_template: str
    canopy_variable: str
    is_transplanted: bool
    auxiliary_files: tuple[str, ...]


_CROP_REGISTRY: dict[str, CropConfig] = {
    "maize": CropConfig(
        name="maize",
        cultivar_code="UFGA8201",
        filex_template=str(PROJECT_ROOT / "templates" / "maize" / "UFGA8201.jinja2"),
        pdi_template=str(PROJECT_ROOT / "templates" / "maize" / "dssat_pdi.jinja2"),
        canopy_variable="lai",
        is_transplanted=False,
        auxiliary_files=(
            "UFGA.CLI",
            "SOIL.SOL",
            "UFGA8201.WTH",
        ),
    ),
    "tomato": CropConfig(
        name="tomato",
        cultivar_code="UFGA0602",
        filex_template=str(PROJECT_ROOT / "templates" / "tomato" / "UFGA0602.jinja2"),
        pdi_template=str(PROJECT_ROOT / "templates" / "tomato" / "dssat_pdi.jinja2"),
        canopy_variable="xlai",
        is_transplanted=True,
        auxiliary_files=(
            "UFGA.CLI",
            "SOIL.SOL",
            "UFGA0601.WTH",
        ),
    ),
    "wheat": CropConfig(
        name="wheat",
        cultivar_code="KSAS8101",
        filex_template=str(PROJECT_ROOT / "templates" / "wheat" / "KSAS8101.jinja2"),
        pdi_template=str(PROJECT_ROOT / "templates" / "wheat" / "dssat_pdi.jinja2"),
        canopy_variable="xlai",
        is_transplanted=False,
        auxiliary_files=(
            "SOIL.SOL",
            "KSAS8101.WTH",
        ),
    ),
}


def get_crop_config(crop_name: str) -> CropConfig:
    if crop_name not in _CROP_REGISTRY:
        raise ValueError(f"不支持的作物: {crop_name}. 可选: {list(_CROP_REGISTRY.keys())}")
    return _CROP_REGISTRY[crop_name]


def list_supported_crops() -> list[str]:
    return list(_CROP_REGISTRY.keys())
