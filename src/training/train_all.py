"""
Master training orchestrator.
Runs all training scripts in order.

Usage (from project root):
    python src/training/train_all.py

    # Train only tabular models:
    python src/training/train_all.py --tabular

    # Train only image models:
    python src/training/train_all.py --image
"""

import subprocess
import logging
import sys
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# All scripts relative to project root
TABULAR_SCRIPTS = [
    "src/training/train_diabetes.py",
    "src/training/train_heart.py",
    "src/training/train_breast_cancer.py",
    "src/training/train_kidney.py",
    "src/training/train_liver.py",
]

IMAGE_SCRIPTS = [
    "src/training/train_malaria.py",
    "src/training/train_pneumonia.py",
]


def run_script(script: str) -> bool:
    logging.info(f"━━━ Starting: {script} ━━━")
    result = subprocess.run(
        [sys.executable, script],
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
    )
    if result.returncode == 0:
        logging.info(f"✓ Completed: {script}")
        return True
    else:
        logging.error(f"✗ Failed:    {script}  (exit code {result.returncode})")
        return False


def main():
    args = sys.argv[1:]
    if "--tabular" in args:
        scripts = TABULAR_SCRIPTS
    elif "--image" in args:
        scripts = IMAGE_SCRIPTS
    else:
        scripts = TABULAR_SCRIPTS + IMAGE_SCRIPTS

    failed = []
    for script in scripts:
        if not run_script(script):
            failed.append(script)

    print("\n" + "=" * 60)
    if failed:
        logging.error(f"Training finished with {len(failed)} failure(s):")
        for s in failed:
            logging.error(f"  ✗ {s}")
        sys.exit(1)
    else:
        logging.info(f"All {len(scripts)} training script(s) completed successfully! ✓")


if __name__ == "__main__":
    main()
