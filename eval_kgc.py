#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FactNet-Bench: KGC Evaluation Script

Evaluates Knowledge Graph Completion (link prediction) models.
Metrics: MRR, Hits@1, Hits@3, Hits@10

Usage:
    python eval_kgc.py \
        --predictions predictions.tsv \
        --gold kgc_bench/test.tsv \
        --all-true kgc_bench/all_true.tsv \
        --output results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict


def load_triples(path: str) -> List[Tuple[str, str, str]]:
    """Load triples from TSV file."""
    triples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                triples.append((parts[0], parts[1], parts[2]))
    return triples


def load_all_true_set(path: str) -> Set[Tuple[str, str, str]]:
    """Load all true triples for filtering."""
    return set(load_triples(path))


def load_predictions(path: str) -> Dict[str, List[str]]:
    """
    Load predictions from file.
    
    Expected format (one line per test query):
    query_type\tsubject\tpredicate\tobject\tranked_candidates
    
    where query_type is 'head' or 'tail'
    and ranked_candidates is comma-separated list of candidate entities
    """
    predictions = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            query_type = parts[0]
            subject = parts[1]
            predicate = parts[2]
            obj = parts[3]
            candidates = parts[4].split(',') if parts[4] else []
            
            # Key: (query_type, s, p, o)
            key = f"{query_type}_{subject}_{predicate}_{obj}"
            predictions[key] = candidates
    
    return predictions


def compute_filtered_rank(
    gold: str,
    candidates: List[str],
    all_true: Set[Tuple[str, str, str]],
    query_type: str,
    subject: str,
    predicate: str,
    obj: str
) -> int:
    """
    Compute filtered rank of gold entity.
    
    Filtering removes all other true entities from ranking.
    """
    if not candidates:
        return len(candidates) + 1  # Gold not found
    
    rank = 1
    for i, cand in enumerate(candidates):
        if cand == gold:
            return rank
        
        # Check if candidate forms a true triple (should be filtered out)
        if query_type == 'tail':
            test_triple = (subject, predicate, cand)
        else:  # head
            test_triple = (cand, predicate, obj)
        
        if test_triple not in all_true:
            rank += 1
    
    # Gold not in candidates
    return len(candidates) + 1


def evaluate_kgc(
    gold_triples: List[Tuple[str, str, str]],
    predictions: Dict[str, List[str]],
    all_true: Set[Tuple[str, str, str]]
) -> Dict[str, float]:
    """
    Evaluate KGC predictions.
    
    For each test triple, we evaluate both head and tail prediction.
    """
    ranks = []
    missing_predictions = 0
    
    for subj, pred, obj in gold_triples:
        # Tail prediction: (s, p, ?) -> o
        tail_key = f"tail_{subj}_{pred}_{obj}"
        if tail_key in predictions:
            tail_rank = compute_filtered_rank(
                obj, predictions[tail_key], all_true,
                'tail', subj, pred, obj
            )
            ranks.append(tail_rank)
        else:
            missing_predictions += 1
            ranks.append(len(all_true))  # Worst case
        
        # Head prediction: (?, p, o) -> s
        head_key = f"head_{subj}_{pred}_{obj}"
        if head_key in predictions:
            head_rank = compute_filtered_rank(
                subj, predictions[head_key], all_true,
                'head', subj, pred, obj
            )
            ranks.append(head_rank)
        else:
            missing_predictions += 1
            ranks.append(len(all_true))
    
    # Compute metrics
    n = len(ranks)
    if n == 0:
        return {
            'mrr': 0.0,
            'hits@1': 0.0,
            'hits@3': 0.0,
            'hits@10': 0.0,
            'num_queries': 0,
            'missing_predictions': missing_predictions
        }
    
    mrr = sum(1.0 / r for r in ranks) / n
    hits1 = sum(1 for r in ranks if r <= 1) / n
    hits3 = sum(1 for r in ranks if r <= 3) / n
    hits10 = sum(1 for r in ranks if r <= 10) / n
    
    return {
        'mrr': round(mrr, 4),
        'hits@1': round(hits1, 4),
        'hits@3': round(hits3, 4),
        'hits@10': round(hits10, 4),
        'num_queries': n,
        'num_triples': len(gold_triples),
        'missing_predictions': missing_predictions,
        'mean_rank': round(sum(ranks) / n, 2)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate KGC (Knowledge Graph Completion) predictions'
    )
    parser.add_argument(
        '--predictions',
        type=str,
        required=True,
        help='Path to predictions file'
    )
    parser.add_argument(
        '--gold',
        type=str,
        required=True,
        help='Path to gold test triples (TSV)'
    )
    parser.add_argument(
        '--all-true',
        type=str,
        required=True,
        help='Path to all true triples for filtering (TSV)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for results JSON (optional)'
    )
    args = parser.parse_args()
    
    print(f"Loading gold triples from {args.gold}...")
    gold_triples = load_triples(args.gold)
    print(f"  Loaded {len(gold_triples)} test triples")
    
    print(f"Loading all true triples from {args.all_true}...")
    all_true = load_all_true_set(args.all_true)
    print(f"  Loaded {len(all_true)} true triples")
    
    print(f"Loading predictions from {args.predictions}...")
    predictions = load_predictions(args.predictions)
    print(f"  Loaded {len(predictions)} predictions")
    
    print("\nEvaluating...")
    results = evaluate_kgc(gold_triples, predictions, all_true)
    
    print("\n" + "=" * 50)
    print("KGC Evaluation Results")
    print("=" * 50)
    print(f"  MRR:      {results['mrr']:.4f}")
    print(f"  Hits@1:   {results['hits@1']:.4f}")
    print(f"  Hits@3:   {results['hits@3']:.4f}")
    print(f"  Hits@10:  {results['hits@10']:.4f}")
    print(f"  Mean Rank: {results['mean_rank']:.2f}")
    print(f"  Queries:  {results['num_queries']}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    return results


if __name__ == "__main__":
    main()
