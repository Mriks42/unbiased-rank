# unbiased-rank

Learning-to-rank from position-biased click feedback: measuring what naive training loses, and what propensity correction actually recovers.

## The question

Production ranking systems are trained on click logs. Clicks are biased by position — a document shown at rank 1 is examined far more often than the same document at rank 10, regardless of relevance. A ranker trained naively on those clicks partly learns *where things were shown* rather than *what was relevant*.

This project measures that gap, and measures how much Inverse Propensity Scoring recovers — including the conditions under which it stops helping.

## Honest scope

Read this before interpreting any number this repo produces.

- **Clicks are simulated, not observed.** Human relevance grades from the ESCI dataset are treated as ground truth; a position-bias model generates synthetic click logs from them. This is standard methodology in the unbiased-LTR literature, but it is a simulation, and every conclusion is conditional on the click model being a reasonable description of reality.
- **No production traffic.** Any latency or throughput figure reported here is measured on a stated, modest configuration and is never extrapolated.
- **The grade-to-relevance mapping is an assumption**, not a measurement. A robustness variant checks that conclusions are not an artifact of the chosen values.

## Status

| Milestone | State |
|---|---|
| M1 — Data foundation (ingest, validation, quarantine, query-level split) | **Complete** |
| M2 — Retrieval baseline (BM25 + dense + RRF) | **Complete** — see [EVALUATION.md](EVALUATION.md) |
| M3 — Click simulator | Not started |
| M4 — The experiment (four arms, sweeps, statistics) | Not started |
| M5 — Cross-encoder and distillation | Not started |
| M6 — Serving | Not started |
| M7 — Deploy and CI | Not started |

## Results so far

Retrieval baseline on the test split (19,339 queries, 100 candidates per query,
95% bootstrap CIs). Full methodology and caveats in [EVALUATION.md](EVALUATION.md).

| Arm | NDCG@10 | 95% CI |
|---|---|---|
| random | 0.1543 | [0.1524, 0.1562] |
| bm25 | 0.8895 | [0.8872, 0.8917] |
| dense | 0.8937 | [0.8915, 0.8959] |
| **rrf** | **0.9056** | [0.9036, 0.9076] |

The random arm is not decoration. It exposed that ESCI's judged-only candidate
sets are ~89% relevant, so random ranking scored 0.83 and left almost no
headroom for any effect to show up in. Candidate sets are now padded to 100 with
sampled negatives, which is also what makes position bias meaningful to simulate
in M3.

## Getting the data

ESCI is not vendored in this repository. Download it from
[amazon-science/esci-data](https://github.com/amazon-science/esci-data) and place these files in `data/raw/`:

```
shopping_queries_dataset_examples.parquet
shopping_queries_dataset_products.parquet
```

**Review the dataset license before deploying this project publicly.** Research-release terms can restrict redistribution and commercial use.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Running

```powershell
# Full test suite — works without the ESCI download (synthetic fixtures)
.\.venv\Scripts\python.exe -m pytest

# Ingestion (requires the ESCI files above)
.\.venv\Scripts\python.exe -c "import logging; logging.basicConfig(level=logging.INFO); from unbiased_rank.data.ingest import ingest; print(ingest())"
```

## Design notes

### Splits are query-level, not row-level

ESCI holds many judged products per query. A row-level split would put some judgments for a query in train and others in test, letting a ranker memorise that query's relevant products and be rewarded for it at evaluation. `find_leaked_queries` asserts this cannot happen, and a property test injects a deliberate leak to confirm the detector actually fires.

### Splits are hashed, not shuffled

Assignment is a pure function of `(seed, query_id)` via `hashlib.blake2b`. This makes it reproducible without persisting a split file, independent of row order, and stable under dataset growth — adding queries never reassigns existing ones. The builtin `hash()` is deliberately avoided: it is randomised per process for strings and would produce a different split on every run.

### Row-level failures are quarantined; structural failures raise

A bad value diverts one row to `data/quarantine/` with a counted reason. A missing or mistyped column raises instead — that means the input is not the dataset we think it is, and quarantining every row would report a 100% quarantine rate while hiding the real cause.

### The test-set floor is enforced

`min_test_queries` defaults to 5,000, derived from the power analysis for a 0.005 NDCG@10 minimum detectable effect. Ingestion fails loudly rather than silently producing an underpowered comparison.

## Layout

```
src/unbiased_rank/
  config.py            # typed, YAML-backed configuration
  data/
    ingest.py          # load -> validate -> quarantine -> split -> persist
    schemas.py         # Pandera schemas, quarantine partitioning
    splits.py          # query-level hashed splits, leak detection
tests/
  unit/                # schema and split behaviour
  property/            # Hypothesis invariants (leakage, determinism, stability)
  integration/         # full pipeline against real parquet on disk
```
