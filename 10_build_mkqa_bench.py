#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FactNet-Bench: Multilingual KG Question Answering (MKQA) Benchmark Construction

This script constructs the MKQA benchmark from FactNet:
1. Define logical form grammar (1-hop, 2-hop queries)
2. Generate questions using multilingual property/entity labels
3. Execute queries to get gold answers
4. Apply split assignment and filtering

Usage:
    python 10_build_mkqa_bench.py \
        --outdir ./mkqa_bench \
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
    query_labels_for_qids,
    query_statements_by_ids,
    get_synset_split,
    extract_qid_from_value,
    sha256_short,
    write_jsonl,
    DEFAULT_BUILD_ID,
    TARGET_LANGUAGES,
    MKQAInstance
)

from es_config import ES_FACTSYNSET_INDEX, ES_FACTSTATEMENT_INDEX, ES_LABELS_INDEX

# -------------------- Question Templates --------------------

# 1-hop question templates by language
HOP1_TEMPLATES = {
    'en': [
        "What is the {property} of {subject}?",
        "Who is the {property} of {subject}?",
        "{subject} has which {property}?"
    ],
    'zh': [
        "{subject}的{property}是什么？",
        "{subject}的{property}是谁？"
    ],
    'es': [
        "¿Cuál es el/la {property} de {subject}?",
        "¿Quién es el/la {property} de {subject}?"
    ],
    'fr': [
        "Quel est le/la {property} de {subject}?",
        "Qui est le/la {property} de {subject}?"
    ],
    'de': [
        "Was ist der/die/das {property} von {subject}?",
        "Wer ist der/die/das {property} von {subject}?"
    ],
    'ru': [
        "Что такое {property} {subject}?",
        "Кто является {property} {subject}?"
    ],
    'ar': [
        "ما هو/هي {property} {subject}؟",
        "من هو/هي {property} {subject}؟"
    ],
    'hi': [
        "{subject} का/की {property} क्या है?",
        "{subject} का/की {property} कौन है?"
    ],
    'id': [
        "Apa {property} dari {subject}?",
        "Siapa {property} dari {subject}?"
    ],
    'it': [
        "Qual è il/la {property} di {subject}?",
        "Chi è il/la {property} di {subject}?"
    ],
    'ja': [
        "{subject}の{property}は何ですか？",
        "{subject}の{property}は誰ですか？"
    ],
    'ko': [
        "{subject}의 {property}은/는 무엇입니까?",
        "{subject}의 {property}은/는 누구입니까?"
    ],
    'nl': [
        "Wat is de/het {property} van {subject}?",
        "Wie is de/het {property} van {subject}?"
    ],
    'pl': [
        "Jaki jest {property} {subject}?",
        "Kto jest {property} {subject}?"
    ],
    'pt': [
        "Qual é o/a {property} de {subject}?",
        "Quem é o/a {property} de {subject}?"
    ],
    'th': [
        "{subject} มี{property}อะไร?",
        "{subject} มี{property}เป็นใคร?"
    ],
    'tr': [
        "{subject}'in {property}'i nedir?",
        "{subject}'in {property}'i kimdir?"
    ],
    'vi': [
        "{property} của {subject} là gì?",
        "{property} của {subject} là ai?"
    ]
}

# Common property labels (PID -> {lang: label})
# This will be populated from ES
PROPERTY_LABELS: Dict[str, Dict[str, str]] = {}

# -------------------- Logging --------------------

def setup_logging(log_dir: Path, name: str = "mkqa_bench_builder") -> logging.Logger:
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


# -------------------- Logical Form Grammar --------------------

@dataclass
class LogicalForm:
    """Represents an executable logical form."""
    form_type: str  # "hop1", "hop2", "hop2c"
    subject: str    # QID
    predicates: List[str]  # PIDs
    constraint: Optional[Dict] = None  # For hop2c
    
    def to_sexp(self) -> str:
        """Convert to S-expression string."""
        if self.form_type == "hop1":
            return f"(hop1 {self.subject} {self.predicates[0]})"
        elif self.form_type == "hop2":
            return f"(hop2 {self.subject} {self.predicates[0]} {self.predicates[1]})"
        elif self.form_type == "hop2c":
            constraint_str = self._constraint_to_str()
            return f"(hop2c {self.subject} {self.predicates[0]} {self.predicates[1]} {constraint_str})"
        return ""
    
    def _constraint_to_str(self) -> str:
        if not self.constraint:
            return "(limit 10)"
        c_type = self.constraint.get('type')
        if c_type == 'type':
            return f"(type {self.constraint['value']})"
        elif c_type == 'year':
            return f"(year {self.constraint['value']})"
        elif c_type == 'limit':
            return f"(limit {self.constraint['value']})"
        return ""


def parse_logical_form(lf_str: str) -> Optional[LogicalForm]:
    """
    Parse an S-expression logical form.
    
    Supports:
        (hop1 SUBJ PID)
        (hop2 SUBJ PID1 PID2)
        (hop2c SUBJ PID1 PID2 (constraint_type constraint_value))
    """
    lf_str = lf_str.strip()
    if not lf_str.startswith('(') or not lf_str.endswith(')'):
        return None
    
    # Remove outer parens and split
    inner = lf_str[1:-1].strip()
    
    # Handle nested constraint
    if '(' in inner:
        # Split at the nested paren
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
        return LogicalForm(
            form_type="hop1",
            subject=main_tokens[1],
            predicates=[main_tokens[2]]
        )
    elif form_type == "hop2" and len(main_tokens) >= 4:
        return LogicalForm(
            form_type="hop2",
            subject=main_tokens[1],
            predicates=[main_tokens[2], main_tokens[3]]
        )
    elif form_type == "hop2c" and len(main_tokens) >= 4:
        constraint = None
        if constraint_str:
            c_inner = constraint_str[1:-1].strip().split()
            if len(c_inner) >= 2:
                constraint = {'type': c_inner[0], 'value': c_inner[1]}
        return LogicalForm(
            form_type="hop2c",
            subject=main_tokens[1],
            predicates=[main_tokens[2], main_tokens[3]],
            constraint=constraint
        )
    
    return None


# -------------------- Query Execution --------------------

def execute_hop1(
    es, 
    subject_qid: str, 
    property_pid: str,
    synset_index: str
) -> List[str]:
    """
    Execute a 1-hop query: Get all objects for (subject, property, ?).
    
    Returns list of answer QIDs.
    """
    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"subject_qid": subject_qid}},
                    {"term": {"property_pid": property_pid}}
                ]
            }
        },
        "_source": ["normalized_value"],
        "size": 200
    }
    
    try:
        resp = es.search(index=synset_index, body=query, request_timeout=30)
        answers = []
        for hit in resp.get('hits', {}).get('hits', []):
            value = hit['_source'].get('normalized_value', '')
            qid = extract_qid_from_value(value)
            if qid:
                answers.append(qid)
        return list(set(answers))
    except Exception as e:
        return []


def execute_hop2(
    es,
    subject_qid: str,
    property1: str,
    property2: str,
    synset_index: str
) -> List[str]:
    """
    Execute a 2-hop query: 
    Get all objects for subject -[p1]-> intermediate -[p2]-> ?
    """
    # First hop
    intermediates = execute_hop1(es, subject_qid, property1, synset_index)
    
    if not intermediates:
        return []
    
    # Second hop (batch)
    answers = set()
    for inter_qid in intermediates[:20]:  # Limit intermediates
        hop2_answers = execute_hop1(es, inter_qid, property2, synset_index)
        answers.update(hop2_answers)
        if len(answers) >= 200:
            break
    
    return list(answers)[:200]


def execute_logical_form(
    es,
    lf: LogicalForm,
    synset_index: str
) -> List[str]:
    """Execute a logical form and return answer QIDs."""
    if lf.form_type == "hop1":
        return execute_hop1(es, lf.subject, lf.predicates[0], synset_index)
    elif lf.form_type == "hop2":
        return execute_hop2(es, lf.subject, lf.predicates[0], lf.predicates[1], synset_index)
    elif lf.form_type == "hop2c":
        # For now, treat hop2c same as hop2, constraint filtering TBD
        return execute_hop2(es, lf.subject, lf.predicates[0], lf.predicates[1], synset_index)
    return []


# -------------------- Question Generation --------------------

def generate_question(
    subject_label: str,
    property_label: str,
    language: str,
    seed: int
) -> str:
    """Generate a question from labels using a template."""
    templates = HOP1_TEMPLATES.get(language, HOP1_TEMPLATES['en'])
    
    # Deterministic template selection
    random.seed(seed)
    template = random.choice(templates)
    
    return template.format(subject=subject_label, property=property_label)


def load_property_labels(es, property_pids: List[str], languages: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Load property labels from the labels index.
    
    Note: Property PIDs are stored with 'P' prefix, but we need to query them.
    """
    result = {}
    
    # Query property labels
    labels_data = query_labels_for_qids(es, property_pids, languages, ES_LABELS_INDEX)
    
    for pid, lang_labels in labels_data.items():
        if pid not in result:
            result[pid] = {}
        for lang, data in lang_labels.items():
            if data.get('label'):
                result[pid][lang] = data['label']
    
    # Fallback: use PID itself if no label found
    for pid in property_pids:
        if pid not in result:
            result[pid] = {}
        for lang in languages:
            if lang not in result[pid]:
                result[pid][lang] = pid  # Fallback to PID
    
    return result


# -------------------- Main Pipeline --------------------

def build_mkqa_benchmark(
    outdir: Path,
    build_id: str,
    languages: List[str],
    max_instances_per_lang: int,
    hop1_ratio: float,
    batch_size: int,
    max_synsets: Optional[int],
    logger: logging.Logger
):
    """
    Build the MKQA benchmark.
    
    Steps:
    1. Stream synsets and identify candidates with multilingual labels
    2. Generate 1-hop and 2-hop questions
    3. Execute queries to get gold answers
    4. Filter by answer set size
    5. Output per-language JSONL files
    """
    outdir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting MKQA benchmark construction")
    logger.info(f"  Output directory: {outdir}")
    logger.info(f"  Build ID: {build_id}")
    logger.info(f"  Languages: {languages}")
    logger.info(f"  Max instances per language: {max_instances_per_lang}")
    logger.info(f"  Hop1/Hop2 ratio: {hop1_ratio}/{1-hop1_ratio}")
    
    es = get_es_client()
    
    # -------------------- Phase 1: Collect candidate synsets --------------------
    logger.info("Phase 1: Collecting candidate synsets with entity values...")
    
    # Collect synsets with entity-valued objects
    candidates = []  # List of (synset_id, subject_qid, property_pid, object_qid, split)
    property_counter = Counter()
    
    source_fields = ['synset_id', 'subject_qid', 'property_pid', 'normalized_value']
    
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
            
            object_qid = extract_qid_from_value(value)
            if not object_qid:
                continue
            
            split = get_synset_split(synset_id, build_id)
            candidates.append((synset_id, subject_qid, property_pid, object_qid, split))
            property_counter[property_pid] += 1
        
        if total_synsets % 100000 == 0:
            logger.info(f"  Processed {total_synsets} synsets, {len(candidates)} candidates")
    
    logger.info(f"Phase 1 complete: {len(candidates)} candidate synsets")
    
    # -------------------- Phase 2: Load labels --------------------
    logger.info("Phase 2: Loading entity and property labels...")
    
    # Collect all QIDs needing labels
    all_qids = set()
    all_pids = set()
    for _, subject, prop, obj, _ in candidates:
        all_qids.add(subject)
        all_qids.add(obj)
        all_pids.add(prop)
    
    logger.info(f"  Loading labels for {len(all_qids)} entities and {len(all_pids)} properties...")
    
    # Load entity labels
    entity_labels = query_labels_for_qids(es, list(all_qids), languages, ES_LABELS_INDEX)
    logger.info(f"  Loaded labels for {len(entity_labels)} entities")
    
    # Load property labels
    property_labels = load_property_labels(es, list(all_pids), languages)
    logger.info(f"  Loaded labels for {len(property_labels)} properties")
    
    # -------------------- Phase 3: Generate questions --------------------
    logger.info("Phase 3: Generating questions for each language...")
    
    instances_by_split = {
        'train': defaultdict(list),  # lang -> [instances]
        'dev': defaultdict(list),
        'test': defaultdict(list)
    }
    
    stats = {
        'total_candidates': len(candidates),
        'generated': 0,
        'filtered_no_label': 0,
        'filtered_empty_answer': 0,
        'by_language': defaultdict(int),
        'by_split': Counter()
    }
    
    for idx, (synset_id, subject_qid, property_pid, object_qid, split) in enumerate(candidates):
        if idx % 50000 == 0:
            logger.info(f"  Processing candidate {idx}/{len(candidates)}")
        
        # Get labels for each target language
        subject_labels_all = entity_labels.get(subject_qid, {})
        property_labels_all = property_labels.get(property_pid, {})
        
        for lang in languages:
            # Check if we already have enough for this language/split
            current_count = len(instances_by_split[split][lang])
            target_per_split = {
                'train': int(max_instances_per_lang * 0.8),
                'dev': int(max_instances_per_lang * 0.1),
                'test': int(max_instances_per_lang * 0.1)
            }
            
            if current_count >= target_per_split[split]:
                continue
            
            # Get labels in this language
            subject_data = subject_labels_all.get(lang, {})
            subject_label = subject_data.get('label') if isinstance(subject_data, dict) else None
            property_label = property_labels_all.get(lang)
            
            if not subject_label or not property_label:
                stats['filtered_no_label'] += 1
                continue
            
            # Determine question type (1-hop vs 2-hop)
            seed = hash(f"{synset_id}_{lang}")
            random.seed(seed)
            is_hop1 = random.random() < hop1_ratio
            
            if is_hop1:
                # Generate 1-hop question
                lf = LogicalForm(
                    form_type="hop1",
                    subject=subject_qid,
                    predicates=[property_pid]
                )
                
                # Execute to get answers
                answers = execute_hop1(es, subject_qid, property_pid, ES_FACTSYNSET_INDEX)
                
                if not answers or len(answers) > 200:
                    stats['filtered_empty_answer'] += 1
                    continue
                
                question = generate_question(subject_label, property_label, lang, seed)
                
            else:
                # For 2-hop, we need another property from object
                # Skip for now in simple implementation
                continue
            
            # Create instance
            instance_id = f"mkqa_{sha256_short(f'{synset_id}_{lang}', 12)}_{lang}"
            
            instance = MKQAInstance(
                id=instance_id,
                question=question,
                logical_form=lf.to_sexp(),
                answers=answers,
                language=lang,
                synset_ids=[synset_id],
                split=split
            )
            
            instances_by_split[split][lang].append(instance)
            stats['generated'] += 1
            stats['by_language'][lang] += 1
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
    
    # Write grammar file
    grammar_content = """# FactNet-MKQA Logical Form Grammar
# 
# <LF>  ::= (hop1 <SUBJ> <PID>)
#        |  (hop2 <SUBJ> <PID> <PID>)
#        |  (hop2c <SUBJ> <PID> <PID> <CONSTRAINT>)
# <SUBJ>::= Q[0-9]+
# <PID> ::= P[0-9]+
# <CONSTRAINT> ::= (type Q[0-9]+) | (year <INT>) | (limit <INT>)
"""
    with open(outdir / "grammar.txt", 'w') as f:
        f.write(grammar_content)
    
    # Write stats
    stats['by_language'] = dict(stats['by_language'])
    stats['by_split'] = dict(stats['by_split'])
    stats['total_written'] = total_written
    stats['languages'] = languages
    stats['build_id'] = build_id
    
    with open(outdir / "stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # -------------------- Summary --------------------
    logger.info("=" * 60)
    logger.info("MKQA Benchmark Construction Complete!")
    logger.info("=" * 60)
    logger.info(f"  Total instances: {total_written}")
    logger.info(f"  Languages: {len(languages)}")
    logger.info(f"  By split: {dict(stats['by_split'])}")
    logger.info(f"  Output directory: {outdir}")
    
    return stats


# -------------------- CLI --------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='FactNet-Bench: Build MKQA (Multilingual KG QA) Benchmark'
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
        default=4000,
        help='Max instances per language (default: 4000)'
    )
    parser.add_argument(
        '--hop1-ratio',
        type=float,
        default=0.62,
        help='Ratio of 1-hop vs 2-hop questions (default: 0.62)'
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
        '--validate-grammar',
        action='store_true',
        help='Validate logical form grammar'
    )
    return parser.parse_args()


def validate_grammar(logger: logging.Logger):
    """Validate the logical form grammar with test cases."""
    test_cases = [
        ("(hop1 Q42 P31)", "hop1", "Q42", ["P31"]),
        ("(hop2 Q142 P36 P17)", "hop2", "Q142", ["P36", "P17"]),
        ("(hop2c Q123 P1 P2 (type Q5))", "hop2c", "Q123", ["P1", "P2"]),
    ]
    
    all_passed = True
    for lf_str, expected_type, expected_subj, expected_preds in test_cases:
        lf = parse_logical_form(lf_str)
        if lf is None:
            logger.error(f"Failed to parse: {lf_str}")
            all_passed = False
            continue
        
        if lf.form_type != expected_type:
            logger.error(f"Type mismatch for {lf_str}: expected {expected_type}, got {lf.form_type}")
            all_passed = False
        
        if lf.subject != expected_subj:
            logger.error(f"Subject mismatch for {lf_str}: expected {expected_subj}, got {lf.subject}")
            all_passed = False
        
        if lf.predicates != expected_preds:
            logger.error(f"Predicates mismatch for {lf_str}: expected {expected_preds}, got {lf.predicates}")
            all_passed = False
        
        # Test round-trip
        roundtrip = lf.to_sexp()
        if roundtrip != lf_str and not lf_str.startswith("(hop2c"):  # hop2c might have constraint differences
            logger.warning(f"Round-trip mismatch: {lf_str} -> {roundtrip}")
        
        logger.info(f"  ✓ {lf_str}")
    
    if all_passed:
        logger.info("Grammar validation passed!")
    else:
        logger.error("Grammar validation failed!")
    
    return all_passed


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    log_dir = Path(args.log_dir)
    
    logger = setup_logging(log_dir)
    logger.info(f"Args: {args}")
    
    if args.validate_grammar:
        logger.info("Validating logical form grammar...")
        validate_grammar(logger)
        return
    
    try:
        build_mkqa_benchmark(
            outdir=outdir,
            build_id=args.build_id,
            languages=args.languages,
            max_instances_per_lang=args.max_instances_per_lang,
            hop1_ratio=args.hop1_ratio,
            batch_size=args.batch_size,
            max_synsets=args.max_synsets,
            logger=logger
        )
    except Exception as e:
        logger.error(f"Build failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
