"""
Inference script for Chain-of-Thought prompt tuning experiments.
Supports both C2D-CoT (proposed) and baseline CoT methods.
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from omegaconf import DictConfig, OmegaConf
import wandb
from openai import OpenAI

from src.preprocess import load_gsm8k, check_correctness, normalize_prediction


def run_inference(cfg: DictConfig) -> None:
    """
    Main inference function for both C2D-CoT and baseline CoT.

    Args:
        cfg: Hydra configuration
    """
    # Initialize WandB
    wandb_enabled = cfg.wandb.mode == "online"
    if wandb_enabled:
        # For sanity check mode, use separate project namespace
        project = cfg.wandb.project
        if cfg.mode == "sanity_check":
            project = f"{cfg.wandb.project}-sanity"

        wandb.init(
            entity=cfg.wandb.entity,
            project=project,
            name=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print(f"WandB run URL: {wandb.run.get_url()}")

    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Determine max samples based on mode
    if cfg.mode == "sanity_check":
        max_samples = 10  # Process 10 samples for sanity check
        print(f"Running in sanity_check mode: processing {max_samples} samples")
    else:
        max_samples = cfg.run.dataset.max_samples
        print(f"Running in main mode: processing {max_samples} samples")

    # Load dataset
    print(
        f"Loading GSM8K dataset (split={cfg.run.dataset.split}, max_samples={max_samples})"
    )
    examples = load_gsm8k(
        split=cfg.run.dataset.split,
        max_samples=max_samples,
        cache_dir=cfg.run.inference.cache_dir,
    )
    print(f"Loaded {len(examples)} examples")

    # Run inference based on method type
    method_type = cfg.run.method.type
    print(f"Running inference with method: {method_type}")

    if method_type == "c2d_cot":
        results = run_c2d_cot(client, cfg, examples)
    elif method_type == "baseline_cot":
        results = run_baseline_cot(client, cfg, examples)
    else:
        raise ValueError(f"Unknown method type: {method_type}")

    # Calculate metrics
    metrics = calculate_metrics(results)
    print("\nFinal Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Log to WandB
    if wandb_enabled:
        wandb.summary.update(metrics)
        # Log per-example results table
        table = wandb.Table(
            columns=["id", "question", "prediction", "gold_answer", "correct"],
            data=[
                [
                    r["id"],
                    r["question"][:100],
                    r["prediction"][:100],
                    r["gold_answer"],
                    r["correct"],
                ]
                for r in results[:100]
            ],
        )
        wandb.log({"predictions": table})

    # Save results
    save_results(cfg, results, metrics)

    # Sanity validation
    if cfg.mode == "sanity_check":
        run_sanity_validation(results, metrics)

    if wandb_enabled:
        wandb.finish()


def run_baseline_cot(
    client: OpenAI, cfg: DictConfig, examples: List[Dict]
) -> List[Dict]:
    """
    Run baseline single-pass CoT inference.

    Args:
        client: OpenAI client
        cfg: Configuration
        examples: List of examples to process

    Returns:
        List of results with predictions
    """
    results = []
    prompt_template = cfg.run.method.prompt_template

    for i, example in enumerate(examples):
        if (i + 1) % 10 == 0:
            print(f"Processing example {i + 1}/{len(examples)}")

        # Format prompt
        prompt = prompt_template.format(question=example["question"])

        # Call API
        try:
            response = client.chat.completions.create(
                model=cfg.run.model.name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=cfg.run.method.max_tokens,
                temperature=cfg.openai.temperature,
            )

            prediction_text = response.choices[0].message.content

            # Check correctness
            correct = check_correctness(prediction_text, example["answer"])

            results.append(
                {
                    "id": example["id"],
                    "question": example["question"],
                    "gold_answer": example["answer"],
                    "prediction": prediction_text,
                    "correct": correct,
                    "method": "baseline_cot",
                }
            )

        except Exception as e:
            print(f"Error processing example {i}: {e}")
            results.append(
                {
                    "id": example["id"],
                    "question": example["question"],
                    "gold_answer": example["answer"],
                    "prediction": "",
                    "correct": False,
                    "method": "baseline_cot",
                    "error": str(e),
                }
            )

        # Rate limiting
        time.sleep(0.1)

    return results


def run_c2d_cot(client: OpenAI, cfg: DictConfig, examples: List[Dict]) -> List[Dict]:
    """
    Run C2D-CoT inference with calibrated gating.

    Args:
        client: OpenAI client
        cfg: Configuration
        examples: List of examples to process

    Returns:
        List of results with predictions
    """
    # First, calibrate threshold on a subset if in main mode
    if cfg.mode == "main" and hasattr(cfg.run.dataset, "calibration_samples"):
        calibration_size = cfg.run.dataset.calibration_samples
        print(f"Calibrating threshold on first {calibration_size} examples")

        calibration_examples = examples[:calibration_size]
        test_examples = examples[calibration_size:]

        threshold = calibrate_threshold(client, cfg, calibration_examples)
        print(f"Calibrated threshold: {threshold:.4f}")
    else:
        # Use default threshold for sanity check
        threshold = cfg.run.method.confidence.threshold
        test_examples = examples

    # Run inference on test examples
    results = []
    num_verified = 0

    for i, example in enumerate(test_examples):
        if (i + 1) % 10 == 0:
            print(f"Processing example {i + 1}/{len(test_examples)}")

        try:
            # Draft pass
            draft_prompt = cfg.run.method.draft.prompt_template.format(
                question=example["question"]
            )

            draft_response = client.chat.completions.create(
                model=cfg.run.model.name,
                messages=[{"role": "user", "content": draft_prompt}],
                max_tokens=cfg.run.method.draft.max_tokens,
                temperature=cfg.openai.temperature,
                logprobs=True,
                top_logprobs=2,
            )

            draft_text = draft_response.choices[0].message.content

            # Calculate confidence score from logprobs
            confidence = calculate_confidence(draft_response)

            # Decide whether to verify
            use_verification = confidence < threshold

            if use_verification:
                num_verified += 1

                # Verify pass
                verify_prompt = cfg.run.method.verify.prompt_template.format(
                    question=example["question"], draft_answer=draft_text
                )

                verify_response = client.chat.completions.create(
                    model=cfg.run.model.name,
                    messages=[{"role": "user", "content": verify_prompt}],
                    max_tokens=cfg.run.method.verify.max_tokens,
                    temperature=cfg.openai.temperature,
                    logprobs=True,
                )

                verify_text = verify_response.choices[0].message.content

                # Score both answers using standardized prompt
                draft_score = score_answer(client, cfg, example["question"], draft_text)
                verify_score = score_answer(
                    client, cfg, example["question"], verify_text
                )

                # Select answer with higher score
                if verify_score > draft_score:
                    final_prediction = verify_text
                    used_verification = True
                else:
                    final_prediction = draft_text
                    used_verification = False
            else:
                final_prediction = draft_text
                used_verification = False

            # Check correctness
            correct = check_correctness(final_prediction, example["answer"])

            results.append(
                {
                    "id": example["id"],
                    "question": example["question"],
                    "gold_answer": example["answer"],
                    "prediction": final_prediction,
                    "draft": draft_text,
                    "confidence": confidence,
                    "used_verification": used_verification,
                    "correct": correct,
                    "method": "c2d_cot",
                }
            )

        except Exception as e:
            print(f"Error processing example {i}: {e}")
            results.append(
                {
                    "id": example["id"],
                    "question": example["question"],
                    "gold_answer": example["answer"],
                    "prediction": "",
                    "correct": False,
                    "method": "c2d_cot",
                    "error": str(e),
                }
            )

        # Rate limiting
        time.sleep(0.1)

    verification_rate = num_verified / len(test_examples) if test_examples else 0
    print(
        f"Verification rate: {verification_rate:.2%} ({num_verified}/{len(test_examples)})"
    )

    return results


def calculate_confidence(response) -> float:
    """
    Calculate confidence score from API response logprobs.
    Uses margin between top-1 and top-2 logprobs.

    Args:
        response: OpenAI API response with logprobs

    Returns:
        Confidence score (higher = more confident)
    """
    try:
        # Get logprobs from the last token (answer token)
        logprobs = response.choices[0].logprobs
        if logprobs and logprobs.content:
            # Average margin across all tokens
            margins = []
            for token_logprob in logprobs.content:
                if token_logprob.top_logprobs and len(token_logprob.top_logprobs) >= 2:
                    top1 = token_logprob.top_logprobs[0].logprob
                    top2 = token_logprob.top_logprobs[1].logprob
                    margin = top1 - top2
                    margins.append(margin)

            if margins:
                # Return average margin (higher = more confident)
                return float(np.mean(margins))
    except Exception as e:
        print(f"Error calculating confidence: {e}")

    return 0.0


def score_answer(client: OpenAI, cfg: DictConfig, question: str, answer: str) -> float:
    """
    Score an answer using standardized answer-only prompt.

    Args:
        client: OpenAI client
        cfg: Configuration
        question: Question text
        answer: Answer text to score

    Returns:
        Log probability score
    """
    try:
        # Use answer scoring template
        prompt = cfg.run.method.answer_scoring_template.format(question=question)

        # Get logprob for the answer
        response = client.chat.completions.create(
            model=cfg.run.model.name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.0,
            logprobs=True,
        )

        # Sum logprobs as score
        logprobs = response.choices[0].logprobs
        if logprobs and logprobs.content:
            total_logprob = sum(t.logprob for t in logprobs.content)
            return float(total_logprob)
    except Exception as e:
        print(f"Error scoring answer: {e}")

    return -1000.0  # Low score on error


def calibrate_threshold(
    client: OpenAI, cfg: DictConfig, calibration_examples: List[Dict]
) -> float:
    """
    Calibrate confidence threshold to achieve target verification rate.

    Args:
        client: OpenAI client
        cfg: Configuration
        calibration_examples: Examples for calibration

    Returns:
        Calibrated threshold
    """
    # Run draft pass on calibration examples
    confidences = []

    for example in calibration_examples:
        try:
            draft_prompt = cfg.run.method.draft.prompt_template.format(
                question=example["question"]
            )

            response = client.chat.completions.create(
                model=cfg.run.model.name,
                messages=[{"role": "user", "content": draft_prompt}],
                max_tokens=cfg.run.method.draft.max_tokens,
                temperature=cfg.openai.temperature,
                logprobs=True,
                top_logprobs=2,
            )

            confidence = calculate_confidence(response)
            confidences.append(confidence)

            time.sleep(0.1)
        except Exception as e:
            print(f"Error in calibration: {e}")
            continue

    if not confidences:
        print("Warning: No confidences collected, using default threshold")
        return cfg.run.method.confidence.threshold

    # Find threshold that achieves target verification rate
    target_rate = cfg.run.method.confidence.calibration_target_rate
    sorted_conf = sorted(confidences)

    # Items below threshold will be verified
    threshold_idx = int(len(sorted_conf) * target_rate)
    threshold = (
        sorted_conf[threshold_idx]
        if threshold_idx < len(sorted_conf)
        else sorted_conf[-1]
    )

    return threshold


def calculate_metrics(results: List[Dict]) -> Dict[str, float]:
    """
    Calculate evaluation metrics from results.

    Args:
        results: List of result dictionaries

    Returns:
        Dictionary of metrics
    """
    total = len(results)
    correct = sum(1 for r in results if r.get("correct", False))
    accuracy = correct / total if total > 0 else 0.0

    metrics = {
        "accuracy": accuracy,
        "total_samples": float(total),
        "correct_samples": float(correct),
    }

    # Add C2D-specific metrics if available
    verified = [r for r in results if r.get("used_verification", False)]
    if any("used_verification" in r for r in results):
        verification_rate = len(verified) / total if total > 0 else 0.0
        metrics["verification_rate"] = verification_rate

        # Accuracy on verified vs non-verified
        if verified:
            verified_acc = sum(1 for r in verified if r["correct"]) / len(verified)
            metrics["verified_accuracy"] = verified_acc

        non_verified = [r for r in results if not r.get("used_verification", False)]
        if non_verified:
            non_verified_acc = sum(1 for r in non_verified if r["correct"]) / len(
                non_verified
            )
            metrics["non_verified_accuracy"] = non_verified_acc

    return metrics


def save_results(
    cfg: DictConfig, results: List[Dict], metrics: Dict[str, float]
) -> None:
    """
    Save results to disk.

    Args:
        cfg: Configuration
        results: List of results
        metrics: Calculated metrics
    """
    output_dir = Path(cfg.results_dir) / cfg.run.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_file}")

    # Save metrics
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_file}")


def run_sanity_validation(results: List[Dict], metrics: Dict[str, float]) -> None:
    """
    Run sanity validation checks and print verdict.

    Args:
        results: List of results
        metrics: Calculated metrics
    """
    # Validation criteria for inference tasks
    samples_processed = len(results)
    outputs_valid = sum(1 for r in results if r.get("prediction", "").strip())

    # Check for unique outputs
    predictions = [r.get("prediction", "")[:50] for r in results if r.get("prediction")]
    outputs_unique = len(set(predictions)) > 1 if predictions else False

    # All metrics should be finite
    all_finite = all(np.isfinite(v) for v in metrics.values())

    # Print summary
    summary = {
        "samples": samples_processed,
        "outputs_valid": outputs_valid,
        "outputs_unique": outputs_unique,
        "accuracy": metrics.get("accuracy", 0.0),
    }
    print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(summary)}")

    # Determine pass/fail
    if samples_processed < 5:
        print("SANITY_VALIDATION: FAIL reason=insufficient_samples")
    elif outputs_valid < samples_processed:
        print("SANITY_VALIDATION: FAIL reason=invalid_outputs")
    elif not outputs_unique:
        print("SANITY_VALIDATION: FAIL reason=identical_outputs")
    elif not all_finite:
        print("SANITY_VALIDATION: FAIL reason=non_finite_metrics")
    else:
        print("SANITY_VALIDATION: PASS")


if __name__ == "__main__":
    print("Error: This script should be called from main.py via subprocess")
    sys.exit(1)
