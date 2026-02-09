# FactNet: A Billion-Scale Knowledge Graph for Multilingual Factual Grounding

<div align="center">

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)

**A three-layer multilingual fact knowledge graph built from Wikidata and Wikipedia**

</div>

## 🎯 Overview

FactNet is an open multilingual fact knowledge graph that transforms structured Wikidata statements and multilingual Wikipedia pages into a unified, hierarchical fact representation system. It consists of three core layers:

| Layer | Description |
|-------|-------------|
| **FactStatement** | Language-neutral, atomic fact units directly mapped from Wikidata statements |
| **FactSense** | Multilingual, natural language expressions of facts extracted from Wikipedia pages |
| **FactSynset** | Semantic equivalence classes aggregating similar FactStatements with normalized values |

The FactSynset layer supports rich inter-synset relations (hypernym, causal, temporal, geographic, etc.) enabling advanced reasoning and cross-lingual fact retrieval.

## 🏗️ Architecture

```
Wikidata JSON Dump          Wikipedia XML Dumps
        │                          │
        ▼                          ▼
┌─────────────────┐     ┌─────────────────────┐
│  FactStatement  │     │  Wikipedia Pages    │
│  Extraction     │     │  + PageLinks        │
│  (01_parse)     │     │  + Redirects        │
└────────┬────────┘     └──────────┬──────────┘
         │                         │
         │      ┌──────────────────┘
         │      │
         ▼      ▼
    ┌──────────────────┐
    │  Elasticsearch   │  ← Indexing for fast retrieval
    │  (factnet_*)     │
    └────────┬─────────┘
             │
    ┌────────┴────────────────────────────┐
    │                                      │
    ▼                                      ▼
┌──────────────┐                  ┌───────────────────┐
│  FactSense   │                  │   FactSynset      │
│  Generation  │                  │   Aggregation     │
│  (06_build)  │                  │   (07_build)      │
└──────────────┘                  └─────────┬─────────┘
                                            │
                                            ▼
                                  ┌───────────────────┐
                                  │  Synset Relations │
                                  │  (08_build)       │
                                  └───────────────────┘
```

## 📁 Project Structure

```
factnet_prj/
├── 01_parse_wikidata_factstatement.py  # Extract FactStatements from Wikidata
├── 02_wiki_extract.py                   # Extract Wikipedia pages using WikiExtractor
├── 03_sql_to_parquet.py                 # Parse pagelinks/redirects SQL dumps
├── 04_wiki_es_store.py                  # Index Wikipedia pages to Elasticsearch
├── 05_factstatement_to_es.py            # Index FactStatements to Elasticsearch
├── 05_label_to_es.py                    # Index entity labels to Elasticsearch
├── 06_build_factsense.py                # Generate FactSense instances
├── 07_build_factsynset.py               # Aggregate FactStatements into FactSynsets
├── 08_build_synset_relations.py         # Build inter-synset relation edges
├── 09_build_kgc_bench.py                # Build KG Completion benchmark
├── 10_build_mkqa_bench.py               # Build Multilingual KG QA benchmark
├── 11_build_mfc_bench.py                # Build Multilingual Fact Checking benchmark
├── bench_utils.py                        # Shared utilities for benchmark construction
├── eval_kgc.py                           # KGC evaluation script
├── eval_mkqa.py                          # MKQA evaluation script
├── eval_mfc.py                           # MFC evaluation script
├── es_config.py                          # Elasticsearch configuration
└── proposal.md                           # Research proposal document
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Elasticsearch cluster (with sufficient storage)
- Required Python packages:

```bash
pip install pandas pyarrow elasticsearch nltk tqdm psutil
```

### Configuration

1. Configure Elasticsearch connection in `es_config.py`:

```python
ES_IP_LIST = ["your_es_host_1", "your_es_host_2"]
ES_PORT = "9200"
ES_USER = "your_username"
ES_PASSWARD = "your_password"
```

### Data Preparation

Download the required data dumps:
- Wikidata JSON dump: `latest-all.json.bz2`
- Wikipedia XML dumps: `{lang}wiki-*-pages-articles.xml.bz2`
- Wikipedia SQL dumps: `*-pagelinks.sql.gz`, `*-redirect.sql.gz`

## 📋 Pipeline Execution

### Phase 1: Data Extraction

**Step 1: Parse Wikidata FactStatements**
```bash
python 01_parse_wikidata_factstatement.py \
  --input /path/to/latest-all.json.bz2 \
  --outdir /path/to/01_factstatement \
  --format parquet \
  --batch-size 50000 \
  --workers 30
```

**Step 2: Extract Wikipedia Pages**
```bash
python 02_wiki_extract.py \
  --orig /path/to/wikipedia_dumps \
  --out /path/to/wikiextractor_out \
  --processes 16
```

**Step 3: Parse PageLinks and Redirects**
```bash
python 03_sql_to_parquet.py \
  --pagelinks-dir /path/to/wiki_pagelinks \
  --redirects-dir /path/to/wiki_redirect \
  --out-root /path/to/sql_to_parquet \
  --workers 16
```

### Phase 2: Elasticsearch Indexing

**Step 4: Index Wikipedia Pages**
```bash
python 04_wiki_es_store.py \
  --resume \
  --langs en zh de fr \
  --wikiextractor-out /path/to/wikiextractor_out \
  --parquet-out /path/to/sql_to_parquet \
  --index factnet_pages_v1 \
  --workers 32
```

**Step 5: Index FactStatements and Labels**
```bash
# Index FactStatements
python 05_factstatement_to_es.py \
  --parquet-dir /path/to/01_factstatement \
  --index factnet_factstatements_v1

# Index Labels
python 05_label_to_es.py \
  --parquet-dir /path/to/01_factstatement \
  --es-index-prefix factnet
```

### Phase 3: FactNet Construction

**Step 6: Build FactSense**
```bash
python 06_build_factsense.py \
  --file-list /path/to/factstatements_part_*.parquet \
  --outdir /path/to/factsense_out \
  --workers 32 \
  --batch-size 500
```

**Step 7: Build FactSynset**
```bash
python 07_build_factsynset.py \
  --factstatements-dir /path/to/01_factstatement \
  --outdir /path/to/factsynset_out \
  --workers 32
```

**Step 8: Build Synset Relations**
```bash
python 08_build_synset_relations.py \
  --outdir /path/to/synset_relations_out \
  --workers 16
```

## 📊 Benchmarks

FactNet provides three evaluation benchmarks:

### Knowledge Graph Completion (KGC)

Evaluates link prediction capabilities. Metrics: MRR, Hits@1/3/10.

```bash
# Build benchmark
python 09_build_kgc_bench.py \
  --outdir /path/to/kgc_bench \
  --top-k-relations 500

# Evaluate
python eval_kgc.py \
  --predictions predictions.tsv \
  --gold kgc_bench/test.tsv \
  --all-true kgc_bench/all_true.tsv
```

### Multilingual KG QA (MKQA)

Evaluates multilingual question answering. Metrics: Macro F1, Valid%.

Supported languages: en, zh, de, es, fr, it, ja, ko, nl, pl, pt, th, tr, vi

```bash
# Build benchmark
python 10_build_mkqa_bench.py \
  --outdir /path/to/mkqa_bench \
  --languages en zh de fr \
  --max-instances-per-lang 5000

# Evaluate
python eval_mkqa.py \
  --predictions predictions.jsonl \
  --gold mkqa_bench/en/test.jsonl
```

### Multilingual Fact Checking (MFC)

Evaluates fact verification. Metrics: Accuracy, Macro F1, Evidence R@5, Span F1.

Labels: SUPPORTED, REFUTED, NEI (Not Enough Information)

```bash
# Build benchmark
python 11_build_mfc_bench.py \
  --outdir /path/to/mfc_bench \
  --languages en zh de \
  --max-instances-per-lang 10000

# Evaluate
python eval_mfc.py \
  --predictions predictions.jsonl \
  --gold mfc_bench/en/test.jsonl
```

## 📈 Elasticsearch Indices

| Index Name | Description |
|------------|-------------|
| `factnet_pages_v1` | Wikipedia pages with sentences, pagelinks, redirects |
| `factnet_labels_v1` | Entity labels and aliases per language |
| `factnet_factstatements_v1` | Wikidata statements with metadata |
| `factnet_factsense_v1` | FactSense instances linking statements to text |
| `factnet_factsynset_v1` | Aggregated FactSynsets with canonical mentions |
| `factnet_synset_relations_v1` | Inter-synset relation edges |

## 🔗 Synset Relation Types

FactNet supports diverse relation types between synsets:

| Category | Relations |
|----------|-----------|
| **Hierarchy** | `hypernym` (subclass_of, instance_of), `part_of`, `has_part`, `member_of` |
| **Temporal/Spatial** | `geographic_location`, `geographic_contains`, `adjacent`, `follows`, `followed_by`, `temporal_before` |
| **Causal** | `causal` (has_cause, has_effect), `influenced_by`, `influences` |
| **Attribution** | `created_by` (author, director, composer), `employed_by`, `affiliated_with` |
| **Semantic** | `equivalent`, `contradiction`, `opposite_of`, `similar_to`, `support`, `refute` |

## 🛠️ Key Features

- **Checkpoint & Resume**: All pipeline scripts support checkpointing for fault tolerance
- **Parallel Processing**: Multi-worker processing with configurable batch sizes
- **Memory Management**: Adaptive batch sizing based on file size and memory pressure
- **Flexible Formats**: Support for both Parquet and CSV output formats
- **Graceful Shutdown**: Signal handling for clean interruption (SIGINT/SIGTERM)

## 📄 License

This project is licensed under the MIT License.

## 📚 Citation

If you use FactNet in your research, please cite:

```bibtex
@article{shen2026factnet,
  title={FactNet: A Billion-Scale Knowledge Graph for Multilingual Factual Grounding},
  author={Shen, Yingli and Lai, Wen and Zhou, Jie and Zhang, Xueren and Wang, Yudong and Luo, Kangyang and Wang, Shuo and Gao, Ge and Fraser, Alexander and Sun, Maosong},
  journal={arXiv preprint arXiv:2602.03417},
  year={2026}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
