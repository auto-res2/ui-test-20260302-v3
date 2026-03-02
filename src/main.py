"""
Main orchestrator for Chain-of-Thought prompt tuning experiments.
Uses Hydra for configuration management.
"""

import subprocess
import sys
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for experiment execution.
    Orchestrates inference runs based on configuration.

    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print(f"Starting experiment: {cfg.run.run_id}")
    print(f"Mode: {cfg.mode}")
    print("=" * 80)

    # Print configuration
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    # Apply mode-specific overrides
    cfg = apply_mode_overrides(cfg)

    # Create results directory
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Determine task type and run appropriate script
    # For this experiment, we're doing inference-only
    print("\nTask type: Inference-only (prompt tuning)")

    # Run inference as a subprocess
    run_inference_subprocess(cfg)

    print("\n" + "=" * 80)
    print(f"Experiment completed: {cfg.run.run_id}")
    print("=" * 80)


def apply_mode_overrides(cfg: DictConfig) -> DictConfig:
    """
    Apply mode-specific configuration overrides.

    Args:
        cfg: Original configuration

    Returns:
        Modified configuration
    """
    if cfg.mode == "sanity_check":
        print("\nApplying sanity_check mode overrides:")

        # Override WandB project for sanity runs
        if not OmegaConf.is_readonly(cfg):
            original_project = cfg.wandb.project
            cfg.wandb.project = f"{original_project}-sanity"
            print(f"  - WandB project: {original_project} -> {cfg.wandb.project}")

        # Dataset limits are already small for sanity_check,
        # but we ensure they're applied in inference.py
        print(f"  - Sample limit: 10 (applied in inference.py)")
        print(f"  - WandB mode: {cfg.wandb.mode}")

    return cfg


def run_inference_subprocess(cfg: DictConfig) -> None:
    """
    Run inference script as a subprocess.

    Args:
        cfg: Configuration
    """
    print("\n" + "-" * 80)
    print("Running inference script...")
    print("-" * 80)

    # Prepare command
    cmd = [
        sys.executable,
        "-u",  # Unbuffered output
        "-c",
        """
import sys
sys.path.insert(0, '.')

from omegaconf import OmegaConf
import yaml

# Load config from stdin
cfg_dict = yaml.safe_load(sys.stdin.read())
cfg = OmegaConf.create(cfg_dict)

# Run inference
from src.inference import run_inference
run_inference(cfg)
""",
    ]

    # Convert config to YAML for passing via stdin
    cfg_yaml = OmegaConf.to_yaml(cfg)

    # Run subprocess
    try:
        result = subprocess.run(
            cmd,
            input=cfg_yaml,
            text=True,
            check=True,
            capture_output=False,  # Stream output to console
        )

        print("\n" + "-" * 80)
        print("Inference completed successfully")
        print("-" * 80)

    except subprocess.CalledProcessError as e:
        print(f"\nError running inference: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
