#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FactNet-Bench: Multilingual Fact Checking (MFC) Benchmark Construction

This script constructs the MFC benchmark from FactNet:
1. Generate SUPPORTED claims from synsets with FactSenses
2. Generate REFUTED claims by value replacement
3. Generate NEI claims with no matching synsets
4. Associate gold evidence units with character spans

Usage:
    python 11_build_mfc_bench.py \
        --outdir ./mfc_bench \
        --log-dir ./logs
"""

import argparse
import json
import logging
import logging.handlers
import os
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict

from bench_utils import (
    get_es_client,
    scroll_all_synsets,
    scroll_all_factsenses,
    query_labels_for_qids,
    get_synset_split,
    extract_qid_from_value,
    normalize_value,
    sha256_short,
    write_jsonl,
    DEFAULT_BUILD_ID,
    TARGET_LANGUAGES,
    MFCInstance
)

from es_config import (
    ES_FACTSYNSET_INDEX, 
    ES_FACTSENSE_INDEX, 
    ES_LABELS_INDEX
)

# -------------------- Claim Templates --------------------

# Templates for generating SUPPORTED/REFUTED claims
CLAIM_TEMPLATES = {
    'en': [
        "The {property} of {subject} is {value}.",
        "{subject}'s {property} is {value}.",
        "{subject} has {value} as {property}."
    ],
    'zh': [
        "{subject}的{property}是{value}。",
        "{subject}的{property}为{value}。"
    ],
    'es': [
        "El/La {property} de {subject} es {value}.",
        "{subject} tiene {value} como {property}."
    ],
    'fr': [
        "Le/La {property} de {subject} est {value}.",
        "{subject} a {value} comme {property}."
    ],
    'de': [
        "Der/Die/Das {property} von {subject} ist {value}.",
        "{subject} hat {value} als {property}."
    ],
    'ru': [
        "{property} {subject} — {value}.",
        "У {subject} {property} — {value}."
    ],
    'ar': [
        "{property} {subject} هو/هي {value}.",
    ],
    'hi': [
        "{subject} का/की {property} {value} है।",
    ],
    'id': [
        "{property} dari {subject} adalah {value}.",
        "{subject} memiliki {value} sebagai {property}."
    ],
    'it': [
        "Il/La {property} di {subject} è {value}.",
        "{subject} ha {value} come {property}."
    ],
    'ja': [
        "{subject}の{property}は{value}です。",
    ],
    'ko': [
        "{subject}의 {property}은/는 {value}입니다.",
    ],
    'nl': [
        "De/Het {property} van {subject} is {value}.",
        "{subject} heeft {value} als {property}."
    ],
    'pl': [
        "{property} {subject} to {value}.",
    ],
    'pt': [
        "O/A {property} de {subject} é {value}.",
        "{subject} tem {value} como {property}."
    ],
    'th': [
        "{property}ของ{subject}คือ{value}",
    ],
    'tr': [
        "{subject}'in {property}'i {value}'dir.",
    ],
    'vi': [
        "{property} của {subject} là {value}.",
    ]
}

# Common property labels cache
PROPERTY_LABELS: Dict[str, Dict[str, str]] = {}

# -------------------- Logging --------------------

def setup_logging(log_dir: Path, name: str = "mfc_bench_builder") -> logging.Logger:
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


# -------------------- Claim Generation --------------------

def generate_claim(
    subject_label: str,
    property_label: str,
    value_label: str,
    language: str,
    seed: int
) -> str:
    """Generate a claim from labels using a template."""
    templates = CLAIM_TEMPLATES.get(language, CLAIM_TEMPLATES['en'])
    
    random.seed(seed)
    template = random.choice(templates)
    
    return template.format(
        subject=subject_label,
        property=property_label,
        value=value_label
    )


def find_span_in_text(text: str, target: str) -> Optional[Tuple[int, int]]:
    """
    Find character span of target in text (case-insensitive).
    
    Returns (start, end) as half-open interval, or None if not found.
    """
    if not text or not target:
        return None
    
    text_lower = text.lower()
    target_lower = target.lower()
    
    idx = text_lower.find(target_lower)
    if idx >= 0:
        return (idx, idx + len(target))
    return None


# -------------------- Evidence Collection --------------------

def query_factsenses_for_synset(
    es,
    synset_id: str,
    statement_ids: List[str],
    language: str,
    index_name: str = "factnet_factsense_v1"
) -> List[Dict]:
    """
    Query FactSenses for a synset in a specific language.
    
    Returns list of evidence dicts with {evidence_pointer, text, spans}.
    """
    if not statement_ids:
        return []
    
    query = {
        "query": {
            "bool": {
                "must": [
                    {"terms": {"belongs_to_statement_id": statement_ids}},
                    {"term": {"language": language}}
                ]
            }
        },
        "_source": ["factsense_id", "sentence", "page_id", "page_title", 
                    "match_type", "confidence", "value_label"],
        "size": 50
    }
    
    try:
        resp = es.search(index=index_name, body=query, request_timeout=30)
        evidences = []
        
        for hit in resp.get('hits', {}).get('hits', []):
            src = hit['_source']
            sentence = src.get('sentence', '')
            if not sentence or len(sentence) < 20:
                continue
            
            evidence_pointer = f"{language}_{src.get('page_id', 0)}_sent_{src.get('factsense_id', '')[:8]}"
            
            # Try to find value span in sentence
            value_label = src.get('value_label', '')
            spans = []
            if value_label:
                span = find_span_in_text(sentence, value_label)
                if span:
                    spans.append(list(span))
            
            evidences.append({
                'evidence_pointer': evidence_pointer,
                'text': sentence[:500],  # Limit length
                'spans': spans,
                'confidence': src.get('confidence', 0.5),
                'match_type': src.get('match_type', '')
            })
        
        # Sort by confidence
        evidences.sort(key=lambda x: x['confidence'], reverse=True)
        return evidences[:5]  # Top 5 evidence units
        
    except Exception as e:
        return []


# -------------------- NEI Claim Generation --------------------

def generate_nei_claim(
    subject_label: str,
    property_label: str,
    fake_value_label: str,
    language: str,
    seed: int
) -> str:
    """Generate an NEI claim with a fake value."""
    return generate_claim(subject_label, property_label, fake_value_label, language, seed)


def get_fake_value_for_property(
    es,
    property_pid: str,
    original_value: str,
    all_values: Dict[str, List[str]],  # property -> [values]
    seed: int
) -> Optional[str]:
    """
    Get a fake value for a property that's different from original.
    Uses values seen in other synsets with the same property.
    """
    if property_pid not in all_values or len(all_values[property_pid]) < 2:
        return None
    
    candidates = [v for v in all_values[property_pid] if v != original_value]
    if not candidates:
        return None
    
    random.seed(seed)
    return random.choice(candidates)


# -------------------- Main Pipeline --------------------

def build_mfc_benchmark(
    outdir: Path,
    build_id: str,
    languages: List[str],
    max_instances_per_lang: int,
    label_ratios: Dict[str, float],  # {"SUPPORTED": 0.34, "REFUTED": 0.33, "NEI": 0.33}
    batch_size: int,
    max_synsets: Optional[int],
    logger: logging.Logger
):
    """
    Build the MFC benchmark.
    
    Steps:
    1. Stream synsets with FactSenses for SUPPORTED claims
    2. Generate REFUTED claims by value replacement
    3. Generate NEI claims with fake values
    4. Associate gold evidence
    5. Output per-language JSONL files
    """
    outdir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting MFC benchmark construction")
    logger.info(f"  Output directory: {outdir}")
    logger.info(f"  Build ID: {build_id}")
    logger.info(f"  Languages: {languages}")
    logger.info(f"  Max instances per language: {max_instances_per_lang}")
    logger.info(f"  Label ratios: {label_ratios}")
    
    es = get_es_client()
    
    # -------------------- Phase 1: Collect synsets and values --------------------
    logger.info("Phase 1: Collecting synsets with entity values...")
    
    # synset_id -> synset data
    synset_data = {}
    # property -> [value_qids]
    property_values: Dict[str, List[str]] = defaultdict(list)
    
    source_fields = ['synset_id', 'subject_qid', 'property_pid', 'normalized_value',
                     'member_statement_ids', 'aggregate_confidence']
    
    total_synsets = 0
    for batch in scroll_all_synsets(
        es,
        index_name=ES_FACTSYNSET_INDEX,
        batch_size=batch_size,
        max_synsets=max_synsets,
        source_fields=source_fields
    ):
        for synset in batch:
            total_synsets += 1
            
            synset_id = synset.get('synset_id')
            subject_qid = synset.get('subject_qid')
            property_pid = synset.get('property_pid')
            value = synset.get('normalized_value', '')
            
            if not (synset_id and subject_qid and property_pid):
                continue
            
            value_qid = extract_qid_from_value(value)
            if not value_qid:
                continue
            
            # Parse member_statement_ids
            member_ids = synset.get('member_statement_ids', [])
            if isinstance(member_ids, str):
                try:
                    member_ids = json.loads(member_ids)
                except:
                    member_ids = []
            
            synset_data[synset_id] = {
                'synset_id': synset_id,
                'subject_qid': subject_qid,
                'property_pid': property_pid,
                'value_qid': value_qid,
                'member_statement_ids': member_ids,
                'confidence': synset.get('aggregate_confidence', 0.5),
                'split': get_synset_split(synset_id, build_id)
            }
            
            property_values[property_pid].append(value_qid)
        
        if total_synsets % 100000 == 0:
            logger.info(f"  Processed {total_synsets} synsets, {len(synset_data)} candidates")
    
    logger.info(f"Phase 1 complete: {len(synset_data)} candidate synsets")
    
    # -------------------- Phase 2: Load labels --------------------
    logger.info("Phase 2: Loading entity and property labels...")
    
    all_qids = set()
    all_pids = set()
    for sid, data in synset_data.items():
        all_qids.add(data['subject_qid'])
        all_qids.add(data['value_qid'])
        all_pids.add(data['property_pid'])
    
    logger.info(f"  Loading labels for {len(all_qids)} entities and {len(all_pids)} properties...")
    
    entity_labels = query_labels_for_qids(es, list(all_qids), languages, ES_LABELS_INDEX)
    logger.info(f"  Loaded labels for {len(entity_labels)} entities")
    
    # Load property labels (stored as entities in same index or separate)
    property_labels = query_labels_for_qids(es, list(all_pids), languages, ES_LABELS_INDEX)
    logger.info(f"  Loaded labels for {len(property_labels)} properties")
    
    # -------------------- Phase 3: Generate claims --------------------
    logger.info("Phase 3: Generating claims...")
    
    instances_by_split = {
        'train': defaultdict(list),
        'dev': defaultdict(list),
        'test': defaultdict(list)
    }
    
    stats = {
        'total_synsets': len(synset_data),
        'generated': {'SUPPORTED': 0, 'REFUTED': 0, 'NEI': 0},
        'by_language': defaultdict(lambda: {'SUPPORTED': 0, 'REFUTED': 0, 'NEI': 0}),
        'by_split': Counter(),
        'evidence_found': 0,
        'no_evidence': 0
    }
    
    synset_list = list(synset_data.items())
    random.seed(42)
    random.shuffle(synset_list)
    
    for idx, (synset_id, data) in enumerate(synset_list):
        if idx % 10000 == 0:
            logger.info(f"  Processing synset {idx}/{len(synset_list)}")
        
        subject_qid = data['subject_qid']
        property_pid = data['property_pid']
        value_qid = data['value_qid']
        split = data['split']
        member_ids = data['member_statement_ids']
        
        subject_labels = entity_labels.get(subject_qid, {})
        value_labels = entity_labels.get(value_qid, {})
        prop_labels = property_labels.get(property_pid, {})
        
        for lang in languages:
            # Check quotas
            current_counts = {
                'SUPPORTED': len([i for i in instances_by_split[split][lang] if i.label == 'SUPPORTED']),
                'REFUTED': len([i for i in instances_by_split[split][lang] if i.label == 'REFUTED']),
                'NEI': len([i for i in instances_by_split[split][lang] if i.label == 'NEI'])
            }
            
            target_per_split = {
                'train': int(max_instances_per_lang * 0.8),
                'dev': int(max_instances_per_lang * 0.1),
                'test': int(max_instances_per_lang * 0.1)
            }
            target_total = target_per_split[split]
            
            # Get labels
            subj_data = subject_labels.get(lang, {})
            subj_label = subj_data.get('label') if isinstance(subj_data, dict) else None
            
            val_data = value_labels.get(lang, {})
            val_label = val_data.get('label') if isinstance(val_data, dict) else None
            
            prop_data = prop_labels.get(lang, {})
            prop_label = prop_data.get('label') if isinstance(prop_data, dict) else None
            
            if not (subj_label and val_label and prop_label):
                continue
            
            seed = hash(f"{synset_id}_{lang}")
            
            # -------------------- SUPPORTED claim --------------------
            supported_target = int(target_total * label_ratios.get('SUPPORTED', 0.34))
            if current_counts['SUPPORTED'] < supported_target:
                # Query evidence
                evidences = query_factsenses_for_synset(
                    es, synset_id, member_ids, lang, ES_FACTSENSE_INDEX
                )
                
                if evidences:
                    stats['evidence_found'] += 1
                    
                    claim = generate_claim(subj_label, prop_label, val_label, lang, seed)
                    instance_id = f"mfc_{sha256_short(f'{synset_id}_{lang}_sup', 12)}_{lang}"
                    
                    instance = MFCInstance(
                        id=instance_id,
                        claim=claim,
                        label='SUPPORTED',
                        language=lang,
                        evidence=evidences,
                        source_synset_id=synset_id,
                        split=split
                    )
                    
                    instances_by_split[split][lang].append(instance)
                    stats['generated']['SUPPORTED'] += 1
                    stats['by_language'][lang]['SUPPORTED'] += 1
                    stats['by_split'][split] += 1
                else:
                    stats['no_evidence'] += 1
            
            # -------------------- REFUTED claim --------------------
            refuted_target = int(target_total * label_ratios.get('REFUTED', 0.33))
            if current_counts['REFUTED'] < refuted_target:
                # Get a different value for the same property
                fake_value_qid = get_fake_value_for_property(
                    es, property_pid, value_qid, property_values, seed + 1
                )
                
                if fake_value_qid:
                    fake_val_data = entity_labels.get(fake_value_qid, {}).get(lang, {})
                    fake_val_label = fake_val_data.get('label') if isinstance(fake_val_data, dict) else None
                    
                    if fake_val_label:
                        # Evidence is the evidence for the TRUE synset (which refutes the claim)
                        evidences = query_factsenses_for_synset(
                            es, synset_id, member_ids, lang, ES_FACTSENSE_INDEX
                        )
                        
                        if evidences:
                            claim = generate_claim(subj_label, prop_label, fake_val_label, lang, seed + 1)
                            instance_id = f"mfc_{sha256_short(f'{synset_id}_{lang}_ref', 12)}_{lang}"
                            
                            instance = MFCInstance(
                                id=instance_id,
                                claim=claim,
                                label='REFUTED',
                                language=lang,
                                evidence=evidences,
                                source_synset_id=synset_id,
                                split=split
                            )
                            
                            instances_by_split[split][lang].append(instance)
                            stats['generated']['REFUTED'] += 1
                            stats['by_language'][lang]['REFUTED'] += 1
                            stats['by_split'][split] += 1
            
            # -------------------- NEI claim --------------------
            nei_target = int(target_total * label_ratios.get('NEI', 0.33))
            if current_counts['NEI'] < nei_target:
                # Generate with a completely fake value (not in property_values)
                fake_nei_label = f"Unknown_{sha256_short(str(seed + 2), 6)}"
                
                claim = generate_nei_claim(subj_label, prop_label, fake_nei_label, lang, seed + 2)
                instance_id = f"mfc_{sha256_short(f'{synset_id}_{lang}_nei', 12)}_{lang}"
                
                instance = MFCInstance(
                    id=instance_id,
                    claim=claim,
                    label='NEI',
                    language=lang,
                    evidence=[],  # No gold evidence for NEI
                    source_synset_id=synset_id,
                    split=split
                )
                
                instances_by_split[split][lang].append(instance)
                stats['generated']['NEI'] += 1
                stats['by_language'][lang]['NEI'] += 1
                stats['by_split'][split] += 1
    
    # -------------------- Phase 4: Write output files --------------------
    logger.info("Phase 4: Writing output files...")
    
    total_written = 0
    
    for lang in languages:
        lang_dir = outdir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        
        for split in ['train', 'dev', 'test']:
            instances = instances_by_split[split][lang]
            if instances:
                output_path = lang_dir / f"{split}.jsonl"
                write_jsonl([i.to_dict() for i in instances], str(output_path))
                total_written += len(instances)
                logger.info(f"  Written {len(instances)} instances to {output_path}")
    
    # Write retrieval index metadata
    index_meta = {
        'train_only_index': 'evidence_index_train',
        'full_index': 'evidence_index_full',
        'description': 'Train-only index excludes Dev/Test synset evidence for training retrievers'
    }
    with open(outdir / "index_metadata.json", 'w') as f:
        json.dump(index_meta, f, indent=2)
    
    # Write stats
    stats['by_language'] = {k: dict(v) for k, v in stats['by_language'].items()}
    stats['by_split'] = dict(stats['by_split'])
    stats['total_written'] = total_written
    stats['languages'] = languages
    stats['build_id'] = build_id
    stats['label_ratios'] = label_ratios
    
    with open(outdir / "stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # -------------------- Summary --------------------
    logger.info("=" * 60)
    logger.info("MFC Benchmark Construction Complete!")
    logger.info("=" * 60)
    logger.info(f"  Total instances: {total_written}")
    logger.info(f"  By label: {stats['generated']}")
    logger.info(f"  By split: {dict(stats['by_split'])}")
    logger.info(f"  Evidence found: {stats['evidence_found']}")
    logger.info(f"  No evidence: {stats['no_evidence']}")
    logger.info(f"  Languages: {len(languages)}")
    logger.info(f"  Output directory: {outdir}")
    
    return stats


# -------------------- CLI --------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='FactNet-Bench: Build MFC (Multilingual Fact Checking) Benchmark'
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
        '--languages',
        type=str,
        nargs='+',
        default=TARGET_LANGUAGES,
        help=f'Target languages (default: all 18)'
    )
    parser.add_argument(
        '--max-instances-per-lang',
        type=int,
        default=5000,
        help='Max instances per language (default: 5000)'
    )
    parser.add_argument(
        '--supported-ratio',
        type=float,
        default=0.34,
        help='Ratio of SUPPORTED claims (default: 0.34)'
    )
    parser.add_argument(
        '--refuted-ratio',
        type=float,
        default=0.33,
        help='Ratio of REFUTED claims (default: 0.33)'
    )
    parser.add_argument(
        '--nei-ratio',
        type=float,
        default=0.33,
        help='Ratio of NEI claims (default: 0.33)'
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
        help='Max synsets to process (for testing)'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='./logs',
        help='Directory for log files'
    )
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Only show statistics for existing benchmark'
    )
    return parser.parse_args()


def show_stats(outdir: Path, logger: logging.Logger):
    """Show statistics for existing benchmark."""
    stats_path = outdir / "stats.json"
    if not stats_path.exists():
        logger.error(f"Stats file not found: {stats_path}")
        return
    
    with open(stats_path) as f:
        stats = json.load(f)
    
    logger.info(f"\nBenchmark Statistics:")
    logger.info(f"  Total instances: {stats.get('total_written', 0):,}")
    logger.info(f"  By label: {stats.get('generated', {})}")
    logger.info(f"  Languages: {len(stats.get('languages', []))}")
    
    # Check label distribution
    generated = stats.get('generated', {})
    total = sum(generated.values())
    if total > 0:
        logger.info(f"\n  Label distribution:")
        for label, count in generated.items():
            pct = 100 * count / total
            logger.info(f"    {label}: {count:,} ({pct:.1f}%)")


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    log_dir = Path(args.log_dir)
    
    logger = setup_logging(log_dir)
    logger.info(f"Args: {args}")
    
    if args.stats_only:
        show_stats(outdir, logger)
        return
    
    label_ratios = {
        'SUPPORTED': args.supported_ratio,
        'REFUTED': args.refuted_ratio,
        'NEI': args.nei_ratio
    }
    
    # Normalize ratios
    total_ratio = sum(label_ratios.values())
    label_ratios = {k: v / total_ratio for k, v in label_ratios.items()}
    
    try:
        build_mfc_benchmark(
            outdir=outdir,
            build_id=args.build_id,
            languages=args.languages,
            max_instances_per_lang=args.max_instances_per_lang,
            label_ratios=label_ratios,
            batch_size=args.batch_size,
            max_synsets=args.max_synsets,
            logger=logger
        )
    except Exception as e:
        logger.error(f"Build failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
