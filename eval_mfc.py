#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FactNet-Bench: MFC Evaluation Script

Evaluates Multilingual Fact Checking predictions.
Metrics: Accuracy, Macro F1, R@5 (Evidence Recall), Span F1

Usage:
    python eval_mfc.py \
        --predictions predictions.jsonl \
        --gold mfc_bench/en/test.jsonl \
        --output results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import Counter, defaultdict


# -------------------- Label Metrics --------------------

LABELS = ['SUPPORTED', 'REFUTED', 'NEI']


def compute_accuracy(predictions: List[str], gold: List[str]) -> float:
    """Compute accuracy."""
    if not predictions or not gold:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, gold) if p == g)
    return correct / len(gold)


def compute_macro_f1(predictions: List[str], gold: List[str]) -> Tuple[float, Dict[str, float]]:
    """
    Compute macro F1 across labels.
    
    Returns (macro_f1, per_label_f1).
    """
    if not predictions or not gold:
        return 0.0, {}
    
    per_label_f1 = {}
    
    for label in LABELS:
        tp = sum(1 for p, g in zip(predictions, gold) if p == label and g == label)
        fp = sum(1 for p, g in zip(predictions, gold) if p == label and g != label)
        fn = sum(1 for p, g in zip(predictions, gold) if p != label and g == label)
        
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        per_label_f1[label] = round(f1, 4)
    
    macro_f1 = sum(per_label_f1.values()) / len(LABELS)
    return round(macro_f1, 4), per_label_f1


# -------------------- Evidence Metrics --------------------

def compute_evidence_recall_at_k(
    predictions: List[Dict],
    gold: List[Dict],
    k: int = 5
) -> float:
    """
    Compute evidence unit Recall@K.
    
    For verifiable instances (SUPPORTED/REFUTED), check if any of top-K
    predicted evidence units matches a gold evidence unit.
    """
    hits = 0
    total = 0
    
    for pred, g in zip(predictions, gold):
        label = g.get('label', '')
        if label not in ['SUPPORTED', 'REFUTED']:
            continue
        
        gold_evidence = g.get('evidence', [])
        if not gold_evidence:
            continue
        
        total += 1
        
        # Get gold evidence pointers
        gold_pointers = {e.get('evidence_pointer', '') for e in gold_evidence}
        gold_pointers.discard('')
        
        if not gold_pointers:
            continue
        
        # Get predicted evidence pointers (top-K)
        pred_evidence = pred.get('predicted_evidence', [])[:k]
        pred_pointers = {e.get('evidence_pointer', '') if isinstance(e, dict) else str(e)
                        for e in pred_evidence}
        
        # Check for any match
        if gold_pointers & pred_pointers:
            hits += 1
    
    return round(hits / max(1, total), 4)


def compute_span_f1(
    predictions: List[Dict],
    gold: List[Dict]
) -> float:
    """
    Compute span-level Evidence F1.
    
    For matched evidence units, compute token-level F1 on spans.
    """
    f1_scores = []
    
    for pred, g in zip(predictions, gold):
        label = g.get('label', '')
        if label not in ['SUPPORTED', 'REFUTED']:
            continue
        
        gold_evidence = g.get('evidence', [])
        if not gold_evidence:
            continue
        
        pred_evidence = pred.get('predicted_evidence', [])
        
        # Index gold evidence by pointer
        gold_by_pointer = {}
        for e in gold_evidence:
            ptr = e.get('evidence_pointer', '')
            if ptr:
                gold_by_pointer[ptr] = e
        
        # Find best matching evidence
        best_f1 = 0.0
        
        for pe in pred_evidence:
            if isinstance(pe, str):
                continue
            
            ptr = pe.get('evidence_pointer', '')
            if ptr not in gold_by_pointer:
                continue
            
            ge = gold_by_pointer[ptr]
            
            # Get spans
            gold_spans = ge.get('spans', [])
            pred_spans = pe.get('spans', [])
            
            if not gold_spans:
                # No gold spans, just pointer match is success
                best_f1 = max(best_f1, 1.0)
                continue
            
            if not pred_spans:
                continue
            
            # Convert spans to character sets
            gold_chars = set()
            for span in gold_spans:
                if isinstance(span, list) and len(span) == 2:
                    gold_chars.update(range(span[0], span[1]))
            
            pred_chars = set()
            for span in pred_spans:
                if isinstance(span, list) and len(span) == 2:
                    pred_chars.update(range(span[0], span[1]))
            
            if not gold_chars or not pred_chars:
                continue
            
            # Compute F1
            intersection = len(gold_chars & pred_chars)
            precision = intersection / len(pred_chars)
            recall = intersection / len(gold_chars)
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                best_f1 = max(best_f1, f1)
        
        f1_scores.append(best_f1)
    
    if not f1_scores:
        return 0.0
    
    return round(sum(f1_scores) / len(f1_scores), 4)


# -------------------- Main Evaluation --------------------

def evaluate_mfc(
    predictions: List[Dict],
    gold: List[Dict]
) -> Dict[str, Any]:
    """
    Evaluate MFC predictions.
    
    Each item should have:
    - id: instance identifier
    - predicted_label: predicted label (SUPPORTED/REFUTED/NEI)
    - label: gold label
    - predicted_evidence: list of evidence dicts with evidence_pointer and optional spans
    - evidence: gold evidence list
    """
    # Index gold by ID
    gold_by_id = {g['id']: g for g in gold}
    
    # Align predictions with gold
    aligned_preds = []
    aligned_gold = []
    
    for pred in predictions:
        pred_id = pred.get('id')
        if pred_id in gold_by_id:
            aligned_preds.append(pred)
            aligned_gold.append(gold_by_id[pred_id])
    
    if not aligned_preds:
        return {
            'accuracy': 0.0,
            'macro_f1': 0.0,
            'total_instances': len(gold),
            'evaluated_instances': 0
        }
    
    # Extract labels
    pred_labels = [p.get('predicted_label', 'NEI') for p in aligned_preds]
    gold_labels = [g.get('label', 'NEI') for g in aligned_gold]
    
    # Compute label metrics
    accuracy = compute_accuracy(pred_labels, gold_labels)
    macro_f1, per_label_f1 = compute_macro_f1(pred_labels, gold_labels)
    
    # Compute evidence metrics
    recall_at_5 = compute_evidence_recall_at_k(aligned_preds, aligned_gold, k=5)
    span_f1 = compute_span_f1(aligned_preds, aligned_gold)
    
    # Label distribution
    pred_dist = Counter(pred_labels)
    gold_dist = Counter(gold_labels)
    
    results = {
        'accuracy': round(accuracy, 4),
        'macro_f1': macro_f1,
        'recall_at_5': recall_at_5,
        'span_f1': span_f1,
        'total_instances': len(gold),
        'evaluated_instances': len(aligned_preds),
        'per_label_f1': per_label_f1,
        'predicted_distribution': dict(pred_dist),
        'gold_distribution': dict(gold_dist)
    }
    
    return results


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate MFC (Multilingual Fact Checking) predictions'
    )
    parser.add_argument(
        '--predictions',
        type=str,
        required=True,
        help='Path to predictions JSONL file'
    )
    parser.add_argument(
        '--gold',
        type=str,
        required=True,
        help='Path to gold test JSONL file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for results JSON (optional)'
    )
    parser.add_argument(
        '--hypothesis-only',
        action='store_true',
        help='Evaluate hypothesis-only baseline (no evidence)'
    )
    args = parser.parse_args()
    
    print(f"Loading predictions from {args.predictions}...")
    predictions = load_jsonl(args.predictions)
    print(f"  Loaded {len(predictions)} predictions")
    
    print(f"Loading gold data from {args.gold}...")
    gold = load_jsonl(args.gold)
    print(f"  Loaded {len(gold)} gold instances")
    
    print("\nEvaluating...")
    results = evaluate_mfc(predictions, gold)
    
    print("\n" + "=" * 50)
    print("MFC Evaluation Results")
    print("=" * 50)
    print(f"  Accuracy:   {results['accuracy']:.4f}")
    print(f"  Macro F1:   {results['macro_f1']:.4f}")
    print(f"  R@5:        {results['recall_at_5']:.4f}")
    print(f"  Span F1:    {results['span_f1']:.4f}")
    print(f"\n  Per-label F1:")
    for label, f1 in results.get('per_label_f1', {}).items():
        print(f"    {label}: {f1:.4f}")
    print(f"\n  Gold distribution: {results.get('gold_distribution', {})}")
    print(f"  Pred distribution: {results.get('predicted_distribution', {})}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    return results


if __name__ == "__main__":
    main()
