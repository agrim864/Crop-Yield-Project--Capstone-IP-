from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import logging
import yaml


@dataclass
class Paths:
    data: Path = Path("compiled data FINAL (3).xlsx")
    outputs_dir: Path = Path("outputs")


@dataclass
class CVConfig:
    n_splits: int = 4
    min_samples_cv: int = 8


@dataclass
class ScenarioConfig:
    years_ahead: int = 10
    clean_scale: float = 0.8
    polluted_scale: float = 1.2
    aod_cols: tuple[str, ...] = ("AOD", "Aod")
    pm_cols: tuple[str, ...] = ("PM2.5 (μgm-3)", "pm2.5", "pm2.5 μgm-3")


@dataclass
class ExperimentConfig:
    paths: Paths = field(default_factory=Paths)
    cv: CVConfig = field(default_factory=CVConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    random_state: int = 42


CONFIG = ExperimentConfig()


def load_yaml_config(yaml_path: Path = Path("config.yaml")) -> None:
    if not yaml_path.exists():
        return
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # paths
    if "data_path" in cfg:
        CONFIG.paths.data = Path(cfg["data_path"])
    if "outputs_dir" in cfg:
        CONFIG.paths.outputs_dir = Path(cfg["outputs_dir"])

    # random state
    if "random_state" in cfg:
        CONFIG.random_state = int(cfg["random_state"])

    # cv
    cv = cfg.get("cv", {})
    if "n_splits" in cv:
        CONFIG.cv.n_splits = int(cv["n_splits"])
    if "min_samples_cv" in cv:
        CONFIG.cv.min_samples_cv = int(cv["min_samples_cv"])

    # forecast/scenario
    forecast = cfg.get("forecast", {})
    if "years_ahead" in forecast:
        CONFIG.scenario.years_ahead = int(forecast["years_ahead"])

    scen = cfg.get("scenario", {})
    if "clean_scale" in scen:
        CONFIG.scenario.clean_scale = float(scen["clean_scale"])
    if "polluted_scale" in scen:
        CONFIG.scenario.polluted_scale = float(scen["polluted_scale"])
    if "aod_cols" in scen:
        CONFIG.scenario.aod_cols = tuple(scen["aod_cols"])
    if "pm_cols" in scen:
        CONFIG.scenario.pm_cols = tuple(scen["pm_cols"])


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    logging.getLogger(__name__).info("Logging configured.")


def ensure_output_dirs() -> None:
    CONFIG.paths.outputs_dir.mkdir(parents=True, exist_ok=True)
    (CONFIG.paths.outputs_dir / "models").mkdir(parents=True, exist_ok=True)
    (CONFIG.paths.outputs_dir / "reports").mkdir(parents=True, exist_ok=True)
