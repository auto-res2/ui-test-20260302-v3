"""
Data preprocessing for GSM8K dataset.
"""

import re
from typing import List, Dict, Any
from datasets import load_dataset


def load_gsm8k(
    split: str = "test", max_samples: int = None, cache_dir: str = ".cache"
) -> List[Dict[str, Any]]:
    """
    Load GSM8K dataset.

    Args:
        split: Dataset split (train or test)
        max_samples: Maximum number of samples to load
        cache_dir: Directory for caching dataset

    Returns:
        List of examples with 'question' and 'answer' fields
    """
    dataset = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)

    examples = []
    for i, item in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break

        # Extract numeric answer from the answer string
        answer_text = item["answer"]
        numeric_answer = extract_numeric_answer(answer_text)

        examples.append(
            {
                "id": i,
                "question": item["question"],
                "answer_text": answer_text,
                "answer": numeric_answer,
            }
        )

    return examples


def extract_numeric_answer(text: str) -> float:
    """
    Extract numeric answer from GSM8K answer text.
    The answer is typically after "####" in the format.

    Args:
        text: Answer text from GSM8K

    Returns:
        Numeric answer as float
    """
    # GSM8K answers are in format: "reasoning #### answer"
    if "####" in text:
        answer_part = text.split("####")[-1].strip()
    else:
        answer_part = text.strip()

    # Remove commas and extract number
    answer_part = answer_part.replace(",", "")

    # Extract the numeric value (handles negative numbers and decimals)
    matches = re.findall(r"-?\d+\.?\d*", answer_part)
    if matches:
        return float(matches[-1])

    raise ValueError(f"Could not extract numeric answer from: {text}")


def normalize_prediction(prediction: str) -> float:
    """
    Normalize a prediction string to extract numeric answer.

    Args:
        prediction: Raw prediction text

    Returns:
        Numeric answer as float
    """
    # Look for "Final: <number>" pattern
    if "Final:" in prediction or "final:" in prediction.lower():
        # Extract text after "Final:"
        parts = re.split(r"[Ff]inal\s*:", prediction)
        if len(parts) > 1:
            answer_part = parts[-1].strip()
        else:
            answer_part = prediction
    else:
        answer_part = prediction

    # Remove commas
    answer_part = answer_part.replace(",", "")

    # Extract numeric value
    matches = re.findall(r"-?\d+\.?\d*", answer_part)
    if matches:
        return float(matches[-1])

    raise ValueError(f"Could not extract numeric answer from prediction: {prediction}")


def check_correctness(
    prediction: str, gold_answer: float, tolerance: float = 1e-3
) -> bool:
    """
    Check if prediction matches gold answer.

    Args:
        prediction: Raw prediction text
        gold_answer: Gold numeric answer
        tolerance: Numerical tolerance for comparison

    Returns:
        True if correct, False otherwise
    """
    try:
        pred_value = normalize_prediction(prediction)
        return abs(pred_value - gold_answer) < tolerance
    except (ValueError, TypeError):
        return False
