import argparse
import subprocess
import sys
import time
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)

PIPELINE_MODULES = [
    "src.macro.data_extraction",
    "src.macro.data_processing",
    "src.macro.generate_drought_labels",
    "src.macro.feature_extraction",
    "src.macro.model_training",
    "src.macro.inference",
]

MICRO_MODULES = [
    "src.micro.stand_drought_analysis",
    "src.micro.multiscale_coupling",
]


def run_step(module_name: str) -> None:
    rel_path = Path(*module_name.split(".")).with_suffix(".py")
    if not rel_path.exists():
        logger.warning(f"Skipping {module_name}: file {rel_path} not found.")
        return

    logger.info(f"Running step: {module_name}")
    start = time.time()
    result = subprocess.run([sys.executable, "-m", module_name])

    if result.returncode != 0:
        logger.error(f"Error in {module_name} (exit code {result.returncode}). Aborting.")
        sys.exit(result.returncode)

    logger.info(f"Finished {module_name} in {time.time() - start:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Scale GeoAI Forest Vulnerability Pipeline")
    parser.add_argument(
        "--include-micro",
        action="store_true",
        help="Also run stand-level Sentinel-2 analysis and macro-micro multiscale coupling.",
    )
    args = parser.parse_args()

    start_total = time.time()
    modules = list(PIPELINE_MODULES) + (MICRO_MODULES if args.include_micro else [])
    for module in modules:
        run_step(module)

    elapsed_min = (time.time() - start_total) / 60
    logger.info(f"Pipeline completed in {elapsed_min:.1f} min.")


if __name__ == "__main__":
    main()
