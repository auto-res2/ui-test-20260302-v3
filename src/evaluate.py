"""
Evaluation script for comparing multiple runs.
Fetches results from WandB and generates comparison plots.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import matplotlib
import wandb

# Use non-interactive backend for PDF generation
matplotlib.use("Agg")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate and compare experiment runs")
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Results directory"
    )
    parser.add_argument(
        "--run_ids", type=str, required=True, help="JSON list of run IDs to compare"
    )
    parser.add_argument(
        "--wandb_entity", type=str, help="WandB entity (optional, can use env var)"
    )
    parser.add_argument(
        "--wandb_project", type=str, help="WandB project (optional, can use env var)"
    )

    args = parser.parse_args()

    # Parse run_ids
    run_ids = json.loads(args.run_ids)
    print(f"Evaluating runs: {run_ids}")

    # Get WandB config
    entity = args.wandb_entity or os.environ.get("WANDB_ENTITY", "airas")
    project = args.wandb_project or os.environ.get(
        "WANDB_PROJECT", "ui-test-20260302-v3"
    )

    print(f"WandB: {entity}/{project}")

    # Initialize WandB API
    api = wandb.Api()

    # Fetch runs
    runs_data = {}
    for run_id in run_ids:
        print(f"\nFetching data for run: {run_id}")
        run_data = fetch_run_data(api, entity, project, run_id)
        runs_data[run_id] = run_data

        # Export per-run metrics
        export_run_metrics(args.results_dir, run_id, run_data)

        # Generate per-run figures
        generate_run_figures(args.results_dir, run_id, run_data)

    # Generate comparison metrics and figures
    generate_comparison(args.results_dir, runs_data)

    print("\n" + "=" * 80)
    print("Evaluation completed successfully")
    print("=" * 80)


def fetch_run_data(
    api: wandb.Api, entity: str, project: str, run_id: str
) -> Dict[str, Any]:
    """
    Fetch run data from WandB API.

    Args:
        api: WandB API instance
        entity: WandB entity
        project: WandB project
        run_id: Run display name

    Returns:
        Dictionary with run data
    """
    # Find run by display name
    runs = api.runs(
        f"{entity}/{project}", filters={"display_name": run_id}, order="-created_at"
    )

    if not runs:
        print(f"Warning: No run found with name '{run_id}'")
        return {"run_id": run_id, "summary": {}, "config": {}, "history": []}

    run = runs[0]  # Most recent run with this name
    print(f"  Found run: {run.name} (ID: {run.id})")

    # Get summary metrics
    summary = dict(run.summary)

    # Get config
    config = dict(run.config)

    # Get history (if any logged metrics over time)
    history = run.history()
    history_list = history.to_dict("records") if not history.empty else []

    return {
        "run_id": run_id,
        "wandb_id": run.id,
        "summary": summary,
        "config": config,
        "history": history_list,
    }


def export_run_metrics(results_dir: str, run_id: str, run_data: Dict) -> None:
    """
    Export metrics for a single run.

    Args:
        results_dir: Results directory
        run_id: Run ID
        run_data: Run data from WandB
    """
    output_dir = Path(results_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = output_dir / "metrics.json"

    # Extract key metrics from summary
    metrics = {"run_id": run_id, **run_data["summary"]}

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Exported metrics to: {metrics_file}")


def generate_run_figures(results_dir: str, run_id: str, run_data: Dict) -> None:
    """
    Generate figures for a single run.

    Args:
        results_dir: Results directory
        run_id: Run ID
        run_data: Run data from WandB
    """
    output_dir = Path(results_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = run_data["summary"]

    # Generate metrics bar chart
    if summary:
        fig, ax = plt.subplots(figsize=(8, 6))

        # Filter numeric metrics
        numeric_metrics = {
            k: v
            for k, v in summary.items()
            if isinstance(v, (int, float)) and k != "total_samples"
        }

        if numeric_metrics:
            metrics_names = list(numeric_metrics.keys())
            metrics_values = list(numeric_metrics.values())

            ax.barh(metrics_names, metrics_values)
            ax.set_xlabel("Value")
            ax.set_title(f"Metrics for {run_id}")
            ax.grid(axis="x", alpha=0.3)

            plt.tight_layout()

            output_file = output_dir / f"{run_id}_metrics.pdf"
            plt.savefig(output_file)
            plt.close()

            print(f"  Generated figure: {output_file}")


def generate_comparison(results_dir: str, runs_data: Dict[str, Dict]) -> None:
    """
    Generate comparison metrics and figures.

    Args:
        results_dir: Results directory
        runs_data: Dictionary mapping run_id to run data
    """
    comparison_dir = Path(results_dir) / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Collect metrics across all runs
    all_metrics = {}
    for run_id, run_data in runs_data.items():
        all_metrics[run_id] = run_data["summary"]

    # Determine primary metric (accuracy)
    primary_metric = "accuracy"

    # Calculate aggregated metrics
    aggregated = {"primary_metric": primary_metric, "metrics_by_run": {}}

    for run_id, metrics in all_metrics.items():
        aggregated["metrics_by_run"][run_id] = metrics

    # Identify best proposed and baseline
    proposed_runs = {k: v for k, v in all_metrics.items() if "proposed" in k.lower()}
    baseline_runs = {
        k: v
        for k, v in all_metrics.items()
        if "comparative" in k.lower() or "baseline" in k.lower()
    }

    if proposed_runs and primary_metric in list(proposed_runs.values())[0]:
        best_proposed_id = max(
            proposed_runs.keys(), key=lambda k: proposed_runs[k].get(primary_metric, 0)
        )
        aggregated["best_proposed"] = {
            "run_id": best_proposed_id,
            primary_metric: proposed_runs[best_proposed_id].get(primary_metric, 0),
        }

    if baseline_runs and primary_metric in list(baseline_runs.values())[0]:
        best_baseline_id = max(
            baseline_runs.keys(), key=lambda k: baseline_runs[k].get(primary_metric, 0)
        )
        aggregated["best_baseline"] = {
            "run_id": best_baseline_id,
            primary_metric: baseline_runs[best_baseline_id].get(primary_metric, 0),
        }

    # Calculate gap
    if "best_proposed" in aggregated and "best_baseline" in aggregated:
        gap = (
            aggregated["best_proposed"][primary_metric]
            - aggregated["best_baseline"][primary_metric]
        )
        aggregated["gap"] = gap

    # Export aggregated metrics
    aggregated_file = comparison_dir / "aggregated_metrics.json"
    with open(aggregated_file, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nExported aggregated metrics to: {aggregated_file}")

    # Generate comparison figures
    generate_comparison_figures(comparison_dir, runs_data, primary_metric)


def generate_comparison_figures(
    comparison_dir: Path, runs_data: Dict[str, Dict], primary_metric: str
) -> None:
    """
    Generate comparison figures across all runs.

    Args:
        comparison_dir: Directory for comparison outputs
        runs_data: Dictionary mapping run_id to run data
        primary_metric: Primary metric to highlight
    """
    # Extract all common metrics
    all_summaries = {run_id: data["summary"] for run_id, data in runs_data.items()}

    # Find common numeric metrics
    all_metric_names = set()
    for summary in all_summaries.values():
        all_metric_names.update(
            k for k, v in summary.items() if isinstance(v, (int, float))
        )

    # Generate bar chart for each metric
    for metric_name in sorted(all_metric_names):
        # Skip total_samples
        if metric_name == "total_samples":
            continue

        # Collect values for this metric
        metric_values = {}
        for run_id, summary in all_summaries.items():
            if metric_name in summary:
                metric_values[run_id] = summary[metric_name]

        if len(metric_values) < 2:
            continue  # Skip if not enough data

        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        run_ids = list(metric_values.keys())
        values = list(metric_values.values())

        # Color proposed runs differently
        colors = [
            "#1f77b4" if "proposed" in rid.lower() else "#ff7f0e" for rid in run_ids
        ]

        bars = ax.bar(range(len(run_ids)), values, color=colors)
        ax.set_xticks(range(len(run_ids)))
        ax.set_xticklabels(run_ids, rotation=45, ha="right")
        ax.set_ylabel(metric_name.replace("_", " ").title())
        ax.set_title(f"Comparison: {metric_name.replace('_', ' ').title()}")
        ax.grid(axis="y", alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#1f77b4", label="Proposed"),
            Patch(facecolor="#ff7f0e", label="Baseline"),
        ]
        ax.legend(handles=legend_elements, loc="best")

        plt.tight_layout()

        output_file = comparison_dir / f"comparison_{metric_name}.pdf"
        plt.savefig(output_file)
        plt.close()

        print(f"Generated comparison figure: {output_file}")


if __name__ == "__main__":
    main()
