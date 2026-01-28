#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FactNet-Bench: Knowledge Graph Completion (KGC) Benchmark Construction

This script constructs the KGC benchmark from FactNet FactSynsets:
1. Extract entity-valued synsets
2. Project to (S, P, O) triples
3. Apply synset-level split assignment
4. Handle cross-split collisions
5. Filter to top K properties

Usage:
    python 09_build_kgc_bench.py \
        --outdir ./kgc_bench \
        --top-k-relations 320 \
        --log-dir ./logs
"""

import argparse
import json
import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass

from bench_utils import (
    get_es_client, 
    scroll_all_synsets,
    get_synset_split,
    extract_qid_from_value,
    sha256_short,
    write_tsv,
    DEFAULT_BUILD_ID,
    KGCTriple
)

from es_config import ES_FACTSYNSET_INDEX

# -------------------- Logging --------------------

def setup_logging(log_dir: Path, name: str = "kgc_bench_builder") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        for h in logger.handlers[:]:
            logger.removeHandler(h)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(ch)
    
    log_file = log_dir / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.handlers.RotatingFileHandler(
        str(log_file),
        maxBytes=50*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    
    return logger


# -------------------- Triple Extraction --------------------

def extract_triple_from_synset(synset: Dict) -> Optional[Tuple[str, str, str, str]]:
    """
    Extract (subject, predicate, object, synset_id) from a synset if entity-valued.
    
    Args:
        synset: FactSynset document
        
    Returns:
        Tuple of (subject_qid, property_pid, object_qid, synset_id) or None
    """
    synset_id = synset.get('synset_id')
    subject_qid = synset.get('subject_qid')
    property_pid = synset.get('property_pid')
    normalized_value = synset.get('normalized_value', '')
    
    if not (synset_id and subject_qid and property_pid):
        return None
    
    # Extract object QID from normalized_value
    object_qid = extract_qid_from_value(normalized_value)
    
    if not object_qid:
        return None
    
    return (subject_qid, property_pid, object_qid, synset_id)


# -------------------- Main Pipeline --------------------

def build_kgc_benchmark(
    outdir: Path,
    build_id: str,
    top_k_relations: int,
    batch_size: int,
    max_synsets: Optional[int],
    logger: logging.Logger
):
    """
    Build the KGC benchmark.
    
    Steps:
    1. Stream all synsets and extract entity-valued triples
    2. Assign splits based on synset_id
    3. Group by triple key (S, P, O) and handle cross-split collisions
    4. Filter to top-K relations by train frequency
    5. Output train/dev/test TSV files
    """
    outdir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting KGC benchmark construction")
    logger.info(f"  Output directory: {outdir}")
    logger.info(f"  Build ID: {build_id}")
    logger.info(f"  Top-K relations: {top_k_relations}")
    
    es = get_es_client()
    
    # -------------------- Phase 1: Extract all entity-valued triples --------------------
    logger.info("Phase 1: Extracting entity-valued triples from synsets...")
    
    # triple_key -> list of (synset_id, split)
    triple_to_synsets: Dict[Tuple[str, str, str], List[Tuple[str, str]]] = defaultdict(list)
    
    # Count property frequencies (in train split)
    property_train_freq = Counter()
    
    total_synsets = 0
    entity_valued_synsets = 0
    
    source_fields = ['synset_id', 'subject_qid', 'property_pid', 'normalized_value']
    
    for batch in scroll_all_synsets(
        es, 
        index_name=ES_FACTSYNSET_INDEX,
        batch_size=batch_size,
        max_synsets=max_synsets,
        source_fields=source_fields
    ):
        for synset in batch:
            total_synsets += 1
            
            result = extract_triple_from_synset(synset)
            if result is None:
                continue
            
            subject, predicate, obj, synset_id = result
            entity_valued_synsets += 1
            
            # Compute split
            split = get_synset_split(synset_id, build_id)
            
            # Store triple -> synset mapping
            triple_key = (subject, predicate, obj)
            triple_to_synsets[triple_key].append((synset_id, split))
            
            # Count property frequency in train
            if split == "train":
                property_train_freq[predicate] += 1
        
        if total_synsets % 100000 == 0:
            logger.info(f"  Processed {total_synsets} synsets, {entity_valued_synsets} entity-valued")
    
    logger.info(f"Phase 1 complete: {total_synsets} total synsets, {entity_valued_synsets} entity-valued")
    logger.info(f"  Unique triples (before filtering): {len(triple_to_synsets)}")
    logger.info(f"  Unique properties: {len(property_train_freq)}")
    
    # -------------------- Phase 2: Select top-K properties --------------------
    logger.info(f"Phase 2: Selecting top {top_k_relations} properties by train frequency...")
    
    top_properties = set(p for p, _ in property_train_freq.most_common(top_k_relations))
    logger.info(f"  Selected {len(top_properties)} properties")
    
    # -------------------- Phase 3: Handle cross-split collisions --------------------
    logger.info("Phase 3: Handling cross-split collisions and de-duplication...")
    
    # Separate triples by split, handling collisions
    train_triples = []
    dev_triples = []
    test_triples = []
    
    cross_split_removed = 0
    property_filtered = 0
    
    for triple_key, synset_list in triple_to_synsets.items():
        subject, predicate, obj = triple_key
        
        # Filter by property
        if predicate not in top_properties:
            property_filtered += 1
            continue
        
        # Get unique splits for this triple
        splits = set(s for _, s in synset_list)
        
        if len(splits) > 1:
            # Cross-split collision
            # Keep in train if any synset is train, remove from dev/test
            if "train" in splits:
                # Pick one synset_id (deterministic: first alphabetically)
                train_synsets = [(sid, sp) for sid, sp in synset_list if sp == "train"]
                train_synsets.sort(key=lambda x: x[0])
                synset_id = train_synsets[0][0]
                train_triples.append(KGCTriple(subject, predicate, obj, synset_id, "train"))
            cross_split_removed += 1
        else:
            # Single split
            split = list(splits)[0]
            synset_id = sorted(synset_list, key=lambda x: x[0])[0][0]  # Deterministic
            
            triple = KGCTriple(subject, predicate, obj, synset_id, split)
            
            if split == "train":
                train_triples.append(triple)
            elif split == "dev":
                dev_triples.append(triple)
            else:
                test_triples.append(triple)
    
    logger.info(f"Phase 3 complete:")
    logger.info(f"  Property filtered: {property_filtered}")
    logger.info(f"  Cross-split collisions removed from dev/test: {cross_split_removed}")
    logger.info(f"  Train triples: {len(train_triples)}")
    logger.info(f"  Dev triples: {len(dev_triples)}")
    logger.info(f"  Test triples: {len(test_triples)}")
    
    # -------------------- Phase 4: Collect entities and relations --------------------
    logger.info("Phase 4: Collecting entity and relation vocabularies...")
    
    all_entities: Set[str] = set()
    all_relations: Set[str] = set()
    
    for triple in train_triples + dev_triples + test_triples:
        all_entities.add(triple.subject)
        all_entities.add(triple.object)
        all_relations.add(triple.predicate)
    
    entities_list = sorted(all_entities)
    relations_list = sorted(all_relations)
    
    logger.info(f"  Unique entities: {len(entities_list)}")
    logger.info(f"  Unique relations: {len(relations_list)}")
    
    # -------------------- Phase 5: Write output files --------------------
    logger.info("Phase 5: Writing output files...")
    
    # Entity and relation files
    write_tsv(entities_list, outdir / "entities.txt")
    write_tsv(relations_list, outdir / "relations.txt")
    logger.info(f"  Written entities.txt and relations.txt")
    
    # Train/Dev/Test TSV files
    train_lines = [t.to_tsv_line() for t in train_triples]
    dev_lines = [t.to_tsv_line() for t in dev_triples]
    test_lines = [t.to_tsv_line() for t in test_triples]
    
    write_tsv(train_lines, outdir / "train.tsv")
    write_tsv(dev_lines, outdir / "dev.tsv")
    write_tsv(test_lines, outdir / "test.tsv")
    logger.info(f"  Written train.tsv, dev.tsv, test.tsv")
    
    # All true triples (for filtered evaluation)
    all_lines = train_lines + dev_lines + test_lines
    write_tsv(all_lines, outdir / "all_true.tsv")
    logger.info(f"  Written all_true.tsv")
    
    # Statistics
    stats = {
        "build_id": build_id,
        "total_synsets_processed": total_synsets,
        "entity_valued_synsets": entity_valued_synsets,
        "top_k_relations": top_k_relations,
        "cross_split_removed": cross_split_removed,
        "train_triples": len(train_triples),
        "dev_triples": len(dev_triples),
        "test_triples": len(test_triples),
        "total_triples": len(train_triples) + len(dev_triples) + len(test_triples),
        "unique_entities": len(entities_list),
        "unique_relations": len(relations_list),
        "avg_degree": round(2 * len(all_lines) / max(1, len(entities_list)), 2),
        "property_frequencies": {p: c for p, c in property_train_freq.most_common(100)}
    }
    
    with open(outdir / "stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"  Written stats.json")
    
    # -------------------- Summary --------------------
    logger.info("=" * 60)
    logger.info("KGC Benchmark Construction Complete!")
    logger.info("=" * 60)
    logger.info(f"  Train triples: {len(train_triples):,}")
    logger.info(f"  Dev triples: {len(dev_triples):,}")
    logger.info(f"  Test triples: {len(test_triples):,}")
    logger.info(f"  Total triples: {len(all_lines):,}")
    logger.info(f"  Entities: {len(entities_list):,}")
    logger.info(f"  Relations: {len(relations_list)}")
    logger.info(f"  Average degree: {stats['avg_degree']}")
    logger.info(f"  Output directory: {outdir}")
    
    return stats


# -------------------- CLI --------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='FactNet-Bench: Build KGC (Knowledge Graph Completion) Benchmark'
    )
    parser.add_argument(
        '--outdir', 
        type=str, 
        required=True,
        help='Output directory for benchmark files'
    )
    parser.add_argument(
        '--build-id',
        type=str,
        default=DEFAULT_BUILD_ID,
        help=f'Build ID for deterministic split assignment (default: {DEFAULT_BUILD_ID})'
    )
    parser.add_argument(
        '--top-k-relations',
        type=int,
        default=320,
        help='Number of top relations to keep by train frequency (default: 320)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=2000,
        help='Batch size for ES queries (default: 2000)'
    )
    parser.add_argument(
        '--max-synsets',
        type=int,
        default=None,
        help='Max synsets to process (for testing, default: all)'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='./logs',
        help='Directory for log files (default: ./logs)'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate existing benchmark files'
    )
    return parser.parse_args()


def validate_benchmark(outdir: Path, logger: logging.Logger) -> bool:
    """Validate existing benchmark files."""
    required_files = ['train.tsv', 'dev.tsv', 'test.tsv', 'all_true.tsv', 
                      'entities.txt', 'relations.txt', 'stats.json']
    
    all_exist = True
    for f in required_files:
        path = outdir / f
        if path.exists():
            logger.info(f"  ✓ {f} exists")
        else:
            logger.error(f"  ✗ {f} missing")
            all_exist = False
    
    if not all_exist:
        return False
    
    # Load stats
    with open(outdir / "stats.json") as f:
        stats = json.load(f)
    
    logger.info(f"\nStatistics:")
    logger.info(f"  Train triples: {stats['train_triples']:,}")
    logger.info(f"  Dev triples: {stats['dev_triples']:,}")
    logger.info(f"  Test triples: {stats['test_triples']:,}")
    logger.info(f"  Entities: {stats['unique_entities']:,}")
    logger.info(f"  Relations: {stats['unique_relations']}")
    
    # Check for overlap between splits
    logger.info("\nChecking split isolation...")
    
    train_triples = set()
    with open(outdir / "train.tsv") as f:
        for line in f:
            train_triples.add(line.strip())
    
    dev_triples = set()
    with open(outdir / "dev.tsv") as f:
        for line in f:
            dev_triples.add(line.strip())
    
    test_triples = set()
    with open(outdir / "test.tsv") as f:
        for line in f:
            test_triples.add(line.strip())
    
    train_dev_overlap = train_triples & dev_triples
    train_test_overlap = train_triples & test_triples
    dev_test_overlap = dev_triples & test_triples
    
    if train_dev_overlap:
        logger.error(f"  ✗ Train-Dev overlap: {len(train_dev_overlap)} triples")
        return False
    else:
        logger.info("  ✓ No Train-Dev overlap")
    
    if train_test_overlap:
        logger.error(f"  ✗ Train-Test overlap: {len(train_test_overlap)} triples")
        return False
    else:
        logger.info("  ✓ No Train-Test overlap")
    
    if dev_test_overlap:
        logger.error(f"  ✗ Dev-Test overlap: {len(dev_test_overlap)} triples")
        return False
    else:
        logger.info("  ✓ No Dev-Test overlap")
    
    logger.info("\n✓ Validation passed!")
    return True


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    log_dir = Path(args.log_dir)
    
    logger = setup_logging(log_dir)
    logger.info(f"Args: {args}")
    
    if args.validate_only:
        logger.info(f"Validating benchmark at {outdir}...")
        validate_benchmark(outdir, logger)
        return
    
    try:
        build_kgc_benchmark(
            outdir=outdir,
            build_id=args.build_id,
            top_k_relations=args.top_k_relations,
            batch_size=args.batch_size,
            max_synsets=args.max_synsets,
            logger=logger
        )
    except Exception as e:
        logger.error(f"Build failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
