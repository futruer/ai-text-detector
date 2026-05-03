"""
Evaluation metrics for AI text detection.
Token-level F1 and Boundary F1@K.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score


def compute_token_level_f1(
    y_true: List[List[int]],
    y_pred: List[List[int]],
    mask: List[List[int]]
) -> Dict[str, float]:
    """Compute token-level metrics for class 1 (AI-generated).

    Ignores -100 labels and padding positions (mask=0).
    """
    all_true_tokens = []
    all_pred_tokens = []

    for true_seq, pred_seq, mask_seq in zip(y_true, y_pred, mask):
        for t, p, m in zip(true_seq, pred_seq, mask_seq):
            if m == 1 and t != -100:  # Valid token, not ignored
                all_true_tokens.append(t)
                all_pred_tokens.append(p)

    all_true_tokens = np.array(all_true_tokens)
    all_pred_tokens = np.array(all_pred_tokens)

    # Calculate metrics for class 1 (AI)
    precision = precision_score(all_true_tokens, all_pred_tokens, pos_label=1, zero_division=0)
    recall = recall_score(all_true_tokens, all_pred_tokens, pos_label=1, zero_division=0)
    f1 = f1_score(all_true_tokens, all_pred_tokens, pos_label=1, zero_division=0)

    return {
        'token_f1': f1,
        'token_precision': precision,
        'token_recall': recall
    }


def extract_boundaries(labels: List[int]) -> List[int]:
    """Extract boundary positions from label sequence.

    Boundary = position where label changes (0->1 or 1->0).
    Example: [0, 0, 1, 1, 0] -> boundaries at positions [2, 4]
    """
    boundaries = []
    prev_label = None
    for i, label in enumerate(labels):
        if label == -100:
            continue
        if prev_label is not None and label != prev_label:
            boundaries.append(i)
        prev_label = label
    return boundaries


def compute_boundary_f1_at_k(
    y_true_boundaries: List[List[int]],
    y_pred_boundaries: List[List[int]],
    k: int = 5
) -> float:
    """Compute Boundary F1@K.

    Compares predicted boundaries against true boundaries.
    For each sample, takes top-K predicted boundaries.
    """
    total_precision = 0.0
    total_recall = 0.0

    for true_bounds, pred_bounds in zip(y_true_boundaries, y_pred_boundaries):
        if len(pred_bounds) > k:
            pred_bounds = pred_bounds[:k]

        true_set = set(true_bounds)
        pred_set = set(pred_bounds)

        if len(pred_set) > 0:
            precision = len(true_set & pred_set) / len(pred_set)
        else:
            precision = 0.0

        if len(true_set) > 0:
            recall = len(true_set & pred_set) / len(true_set)
        else:
            recall = 1.0 if len(pred_set) == 0 else 0.0

        total_precision += precision
        total_recall += recall

    n = len(y_true_boundaries)
    avg_precision = total_precision / n if n > 0 else 0.0
    avg_recall = total_recall / n if n > 0 else 0.0

    if avg_precision + avg_recall > 0:
        f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
    else:
        f1 = 0.0

    return f1


def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    k_boundary: int = 5
) -> Dict[str, float]:
    """Full evaluation of model on test set.

    Returns:
        Dictionary with 'token_f1', 'token_precision', 'token_recall', 'boundary_f1_at_k'
    """
    model.eval()

    all_true_labels = []
    all_pred_labels = []
    all_masks = []

    if len(test_loader) == 0:
        return {
            'token_f1': 0.0,
            'token_precision': 0.0,
            'token_recall': 0.0,
            'boundary_f1_at_k': 0.0
        }

    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            mask = batch['mask'].to(device)

            # Decode (inference mode - tags=None)
            decoded = model(features, tags=None, mask=mask)

            true_labels = batch['labels'].cpu().numpy().tolist()
            pred_labels = decoded  # Already a list of lists from CRF decode
            masks = mask.cpu().numpy().tolist()

            all_true_labels.extend(true_labels)
            all_pred_labels.extend(pred_labels)
            all_masks.extend(masks)

    # Compute token-level metrics
    token_metrics = compute_token_level_f1(all_true_labels, all_pred_labels, all_masks)

    # Extract boundaries for boundary F1@K
    true_boundaries = [extract_boundaries(labels) for labels in all_true_labels]
    pred_boundaries = [extract_boundaries(labels) for labels in all_pred_labels]

    boundary_f1 = compute_boundary_f1_at_k(true_boundaries, pred_boundaries, k=k_boundary)

    return {
        **token_metrics,
        'boundary_f1_at_k': boundary_f1
    }


if __name__ == '__main__':
    # Simple test
    y_true = [[0, 0, 1, 1, 0, -100, -100]]
    y_pred = [[0, 0, 1, 0, 0, 0, 0]]
    mask = [[1, 1, 1, 1, 1, 0, 0]]

    metrics = compute_token_level_f1(y_true, y_pred, mask)
    print(f"Token metrics: {metrics}")

    true_bounds = extract_boundaries(y_true[0])
    pred_bounds = extract_boundaries(y_pred[0])
    print(f"True boundaries: {true_bounds}")
    print(f"Pred boundaries: {pred_bounds}")

    boundary_f1 = compute_boundary_f1_at_k([true_bounds], [pred_bounds], k=5)
    print(f"Boundary F1@5: {boundary_f1}")