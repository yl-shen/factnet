#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FactNet-Bench: MKQA Evaluation Script

Evaluates Multilingual KG Question Answering predictions.
Metrics: Macro F1, Valid%

Usage:
    python eval_mkqa.py \
        --predictions predictions.jsonl \
        --gold mkqa_bench/en/test.jsonl \
        --output results.json
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional, Any


# -------------------- Logical Form Parsing --------------------

def parse_logical_form(lf_str: str) -> Optional[Dict]:
    """
    Parse a logical form string into structured form.
    
    Returns None if invalid.
    """
    lf_str = lf_str.strip()
    if not lf_str.startswith('(') or not lf_str.endswith(')'):
        return None
    
    # Handle nested constraint
    inner = lf_str[1:-1].strip()
    
    if '(' in inner:
        main_part, constraint_part = inner.split('(', 1)
        main_tokens = main_part.strip().split()
        constraint_str = '(' + constraint_part.rstrip(')')
    else:
        main_tokens = inner.split()
        constraint_str = None
    
    if not main_tokens:
        return None
    
    form_type = main_tokens[0]
    
    if form_type == "hop1" and len(main_tokens) >= 3:
        return {
            'type': 'hop1',
            'subject': main_tokens[1],
            'predicates': [main_tokens[2]]
        }
    elif form_type == "hop2" and len(main_tokens) >= 4:
        return {
            'type': 'hop2',
            'subject': main_tokens[1],
            'predicates': [main_tokens[2], main_tokens[3]]
        }
    elif form_type == "hop2c" and len(main_tokens) >= 4:
        return {
            'type': 'hop2c',
            'subject': main_tokens[1],
            'predicates': [main_tokens[2], main_tokens[3]],
            'constraint': constraint_str
        }
    
    return None


def is_valid_lf(lf_str: str) -> bool:
    """Check if a logical form is syntactically valid."""
    return parse_logical_form(lf_str) is not None


def validate_qid(qid: str) -> bool:
    """Check if string is a valid QID format."""
    return bool(re.match(r'^Q\d+$', qid))


def validate_pid(pid: str) -> bool:
    """Check if string is a valid PID format."""
    return bool(re.match(r'^P\d+$', pid))


# -------------------- Answer Normalization --------------------

def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison."""
    if not answer:
        return ""
    answer = str(answer).strip().lower()
    # Remove common prefixes
    answer = re.sub(r'^(the|a|an)\s+', '', answer)
    return answer


def normalize_answer_set(answers: List[str]) -> Set[str]:
    """Normalize a set of answers."""
    return {normalize_answer(a) for a in answers if a}


# -------------------- Metrics --------------------

def compute_set_f1(predicted: Set[str], gold: Set[str]) -> float:
    """Compute F1 between two sets."""
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    
    intersection = len(predicted & gold)
    precision = intersection / len(predicted)
    recall = intersection / len(gold)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def evaluate_mkqa(
    predictions: List[Dict],
    gold: List[Dict]
) -> Dict[str, Any]:
    """
    Evaluate MKQA predictions.
    
    Each item should have:
    - id: instance identifier
    - predicted_lf: predicted logical form string (for predictions)
    - logical_form: gold logical form (for gold)
    - predicted_answers: predicted answer set (for predictions)
    - answers: gold answer set (for gold)
    """
    # Index gold by ID
    gold_by_id = {g['id']: g for g in gold}
    
    results = {
        'total_instances': len(gold),
        'valid_predictions': 0,
        'invalid_predictions': 0,
        'f1_scores': [],
        'by_type': defaultdict(lambda: {'count': 0, 'f1_sum': 0, 'valid': 0})
    }
    
    for pred in predictions:
        pred_id = pred.get('id')
        if pred_id not in gold_by_id:
            continue
        
        gold_item = gold_by_id[pred_id]
        
        # Check validity
        predicted_lf = pred.get('predicted_lf', '')
        is_valid = is_valid_lf(predicted_lf)
        
        if is_valid:
            results['valid_predictions'] += 1
        else:
            results['invalid_predictions'] += 1
            # Invalid predictions get F1 = 0
            results['f1_scores'].append(0.0)
            continue
        
        # Compare answers
        pred_answers = normalize_answer_set(pred.get('predicted_answers', []))
        gold_answers = normalize_answer_set(gold_item.get('answers', []))
        
        f1 = compute_set_f1(pred_answers, gold_answers)
        results['f1_scores'].append(f1)
        
        # Track by query type
        parsed = parse_logical_form(gold_item.get('logical_form', ''))
        if parsed:
            qtype = parsed['type']
            results['by_type'][qtype]['count'] += 1
            results['by_type'][qtype]['f1_sum'] += f1
            results['by_type'][qtype]['valid'] += 1
    
    # Compute final metrics
    n = len(results['f1_scores'])
    if n == 0:
        macro_f1 = 0.0
    else:
        macro_f1 = sum(results['f1_scores']) / n
    
    total_preds = results['valid_predictions'] + results['invalid_predictions']
    valid_pct = 100 * results['valid_predictions'] / max(1, total_preds)
    
    final_results = {
        'macro_f1': round(macro_f1, 4),
        'valid_percent': round(valid_pct, 2),
        'total_instances': len(gold),
        'evaluated_instances': n,
        'valid_predictions': results['valid_predictions'],
        'invalid_predictions': results['invalid_predictions']
    }
    
    # By type breakdown
    for qtype, data in results['by_type'].items():
        if data['count'] > 0:
            final_results[f'{qtype}_f1'] = round(data['f1_sum'] / data['count'], 4)
            final_results[f'{qtype}_count'] = data['count']
    
    return final_results


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
        description='Evaluate MKQA (Multilingual KG QA) predictions'
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
    args = parser.parse_args()
    
    print(f"Loading predictions from {args.predictions}...")
    predictions = load_jsonl(args.predictions)
    print(f"  Loaded {len(predictions)} predictions")
    
    print(f"Loading gold data from {args.gold}...")
    gold = load_jsonl(args.gold)
    print(f"  Loaded {len(gold)} gold instances")
    
    print("\nEvaluating...")
    results = evaluate_mkqa(predictions, gold)
    
    print("\n" + "=" * 50)
    print("MKQA Evaluation Results")
    print("=" * 50)
    print(f"  Macro F1:   {results['macro_f1']:.4f}")
    print(f"  Valid%:     {results['valid_percent']:.2f}%")
    print(f"  Total:      {results['total_instances']}")
    print(f"  Evaluated:  {results['evaluated_instances']}")
    
    if 'hop1_f1' in results:
        print(f"\n  Hop1 F1:    {results['hop1_f1']:.4f} (n={results.get('hop1_count', 0)})")
    if 'hop2_f1' in results:
        print(f"  Hop2 F1:    {results['hop2_f1']:.4f} (n={results.get('hop2_count', 0)})")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    return results


if __name__ == "__main__":
    main()
