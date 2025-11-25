from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging


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
    paths: Paths = Paths()
    cv: CVConfig = CVConfig()
    scenario: ScenarioConfig = ScenarioConfig()
    random_state: int = 42


CONFIG = ExperimentConfig()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging configured.")


def ensure_output_dirs() -> None:
    CONFIG.paths.outputs_dir.mkdir(parents=True, exist_ok=True)
