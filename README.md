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
| M3 — Click simulator | **Complete** — harness gate passes |
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

### Simulator validation (M3)

With position bias switched off, clicks are an unbiased sample of relevance, so
a ranker trained on them should match one trained on true grades. It does:

| Trained on | NDCG@10 | 95% CI |
|---|---|---|
| true grades | 0.9079 | [0.9033, 0.9124] |
| clicks at η=0 | 0.9016 | [0.8968, 0.9063] |
| **difference** | **−0.0063** | [−0.0080, −0.0046] |

Tolerance ±0.02 — **PASS**. The criterion is deliberately on magnitude rather
than statistical significance: at 4,000 queries even a trivial gap is
detectable, and the question is whether the harness is broken, not whether two
arms differ at all.

Worth noting for M4: the trained ceiling (0.9079) sits only just above the RRF
baseline (0.9056), with BM25 alone at 0.8895. The usable band between "no
learning" and "perfect labels" is therefore about 0.02 wide. If naive training
on biased clicks does not fall well below 0.8895, the effect will be small
relative to that band — which would itself be a result worth reporting rather
than tuning around.

## Getting the data

ESCI is not vendored in this repository. Download it from
[amazon-science/esci-data](https://github.com/amazon-science/esci-data) and place these files in `data/raw/`:

```
shopping_queries_dataset_examples.parquet
shopping_queries_dataset_products.parquet
```

**License:** ESCI is distributed under **Apache-2.0** (verified against the
repository's `LICENSE` file, not just its README). Commercial use and
redistribution are permitted with attribution. Cite Reddy et al. 2022 — the
citation is in this repo's [LICENSE](LICENSE).

**Download gotcha:** the `raw.githubusercontent.com` URLs return 133-byte
git-lfs *pointer* files, not data. The real files come from
`media.githubusercontent.com`:

```powershell
$base = "https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset"
curl.exe -L -o "data\raw\shopping_queries_dataset_examples.parquet" "$base/shopping_queries_dataset_examples.parquet"
curl.exe -L -o "data\raw\shopping_queries_dataset_products.parquet" "$base/shopping_queries_dataset_products.parquet"
```

Roughly 1.1 GB total (examples 49 MB, products 1.06 GB).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

No GPU is required for M1–M4; everything runs CPU-only. M5 (cross-encoder
fine-tuning) will need one.

## Running

```powershell
# Test suite — works without the ESCI download (synthetic fixtures throughout)
.\.venv\Scripts\python.exe -m pytest -m "not requires_model"

# 1. Ingest: validate, quarantine, assign leak-free query-level splits
.\.venv\Scripts\python.exe -m unbiased_rank.data.ingest

# 2. Encode. Scoped to the splits you need -- the test split touches 313k of
#    1.2M products (~24 min on 6 CPU cores); "all" takes closer to two hours.
.\.venv\Scripts\python.exe -m unbiased_rank.indexing.catalog --splits test

# 3. Retrieval baseline: BM25 vs dense vs RRF vs a random floor
.\.venv\Scripts\python.exe -m unbiased_rank.experiments.baseline --split test

# 4. Simulator calibration: CTR-by-rank across the eta sweep
.\.venv\Scripts\python.exe -m unbiased_rank.experiments.calibrate_simulator

# 5. Harness validation gate -- run this before trusting any M4 result
.\.venv\Scripts\python.exe -m unbiased_rank.experiments.harness_gate
```

Committed results live in [`outputs/`](outputs/) so the numbers are readable
without re-running the encode.

## Design notes

### Splits are query-level, not row-level

ESCI holds many judged products per query. A row-level split would put some judgments for a query in train and others in test, letting a ranker memorise that query's relevant products and be rewarded for it at evaluation. `find_leaked_queries` asserts this cannot happen, and a property test injects a deliberate leak to confirm the detector actually fires.

### Splits are hashed, not shuffled

Assignment is a pure function of `(seed, query_id)` via `hashlib.blake2b`. This makes it reproducible without persisting a split file, independent of row order, and stable under dataset growth — adding queries never reassigns existing ones. The builtin `hash()` is deliberately avoided: it is randomised per process for strings and would produce a different split on every run.

### Row-level failures are quarantined; structural failures raise

A bad value diverts one row to `data/quarantine/` with a counted reason. A missing or mistyped column raises instead — that means the input is not the dataset we think it is, and quarantining every row would report a 100% quarantine rate while hiding the real cause.

### The test-set floor is enforced

`min_test_queries` defaults to 5,000, derived from the power analysis for a 0.005 NDCG@10 minimum detectable effect. Ingestion fails loudly rather than silently producing an underpowered comparison. The measured `σ_d` turned out to be 0.076, so 1,833 queries suffice — the test split has 19,339.

### Candidate sets are padded with sampled negatives

ESCI's judged sets are ~89% relevant, so ordering them is nearly impossible to get wrong. Padding to 100 candidates restores the metric's dynamic range and is also what makes position bias meaningful to simulate — examination probability only matters when many candidates compete for few visible slots. Details and the assumption this rests on are in [EVALUATION.md](EVALUATION.md).

### Features are computed by one shared module

`ranking/features.py` is used by both training and serving. Train/serve skew — features computed slightly differently in the two paths — is the classic silent production failure in ranking, and sharing the code makes it impossible by construction rather than by discipline.

### BM25 scores candidates, not the corpus

Scoring slices candidates first, then the query's terms. An earlier version densified one full corpus column per query term: correct, but allocating ~10 MB per lookup at 1.2M products, which is the difference between a seconds-long evaluation and an hours-long one. A regression test asserts the cost no longer scales with corpus size — the correctness tests passed both before and after, so they would not have caught it.

### Embedding caches carry a fingerprint

Model name, corpus size, `max_seq_length` and a sampled text digest. A stale cache reused after the corpus or model changed would corrupt every downstream number while the run looked perfectly healthy.

## Layout

```
src/unbiased_rank/
  config.py                    # typed, YAML-backed configuration
  data/
    ingest.py                  # load -> validate -> quarantine -> split -> persist
    schemas.py                 # Pandera schemas, quarantine partitioning
    splits.py                  # query-level hashed splits, leak detection
  indexing/
    text.py                    # tokenisation
    lexical.py                 # BM25 over a sparse term-document matrix
    dense.py                   # bi-encoder, fingerprinted cache, FAISS index
    fusion.py                  # reciprocal rank fusion
    catalog.py                 # scoped catalogue/query encoding
  ranking/
    candidates.py              # per-query candidate sets, sampled negatives
    features.py                # shared train/serve feature extraction
    lambdamart.py              # LightGBM lambdarank wrapper
    labels.py                  # per-arm labels: grades, clicks, IPS weights
  simulation/
    position_bias.py           # PBM propensity curve, IPS weights
    click_model.py             # grade -> relevance -> click
    logger.py                  # impression and click-log generation
  evaluation/
    metrics.py                 # NDCG, MRR, Recall (per query, not averaged)
    statistics.py              # paired bootstrap, power analysis, BH-FDR
  experiments/
    baseline.py                # M2 retrieval baseline
    calibrate_simulator.py     # M3 CTR-vs-propensity calibration
    harness_gate.py            # M3 validation gate
tests/
  unit/                        # module behaviour and calibration
  property/                    # Hypothesis invariants (leakage, determinism)
  integration/                 # full pipeline against real parquet on disk
outputs/                       # committed result JSON
```
