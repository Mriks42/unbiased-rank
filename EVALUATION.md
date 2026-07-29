# Evaluation

Results and methodology for the retrieval baseline (M2). All numbers come from
`outputs/baseline_test_neg100.json`, reproducible with:

```powershell
python -m unbiased_rank.experiments.baseline --split test --candidates-per-query 100
```

## Setting

**Re-ranking, not full-corpus retrieval.** ESCI judges ~19 products per query
out of a 1.2M catalogue. Retrieving from the whole corpus would return mostly
*unjudged* documents, and scoring those as irrelevant penalises any system that
surfaces good-but-unjudged products — a bias unrelated to what this project
measures. Ranking a known-relevance candidate set is how LETOR, MSLR and
Istella are arranged.

**Candidate sets are padded to 100 with sampled negatives.** See below for why.

| | |
|---|---|
| Split | test (query-level, leak-free) |
| Queries | 19,339 |
| Candidates per query | 100 (median 16 judged + sampled negatives) |
| Metric | NDCG@10, exponential gain `2^g − 1`, grades E=3 S=2 C=1 I=0 |
| Uncertainty | Percentile bootstrap, 10,000 resamples, paired across arms |

## Results

| Arm | NDCG@10 | 95% CI | MRR |
|---|---|---|---|
| random | 0.1543 | [0.1524, 0.1562] | 0.3584 |
| bm25 | 0.8895 | [0.8872, 0.8917] | 0.9725 |
| dense | 0.8937 | [0.8915, 0.8959] | 0.9748 |
| **rrf** | **0.9056** | [0.9036, 0.9076] | 0.9787 |

Paired differences against BM25:

| Comparison | Δ NDCG@10 | 95% CI | Significant |
|---|---|---|---|
| dense − bm25 | +0.0042 | [+0.0024, +0.0060] | yes |
| rrf − bm25 | +0.0161 | [+0.0150, +0.0172] | yes |
| random − bm25 | −0.7352 | [−0.7379, −0.7323] | yes |

**Latency:** BM25 1.02 ms/query over the 1.2M-document index; dense scoring
39 µs/query (embeddings precomputed).

**Measured `σ_d` (rrf vs bm25): 0.0764**, implying 1,833 queries for a 0.005
minimum detectable effect at α=0.05, power=0.80. The Stage 3.1 protocol assumed
0.12; the measured value is lower, so M4 has roughly 10× headroom on the test
split.

## Why candidate sets are padded with negatives

The judged-only configuration is **not a usable evaluation setting**, and the
random-floor arm is what revealed it.

| | judged-only (median 16) | padded (100) |
|---|---|---|
| random NDCG@10 | 0.8322 | 0.1543 |
| headroom to perfect | 0.20 | 0.85 |

ESCI's test split is 68.3% Exact, 20.5% Substitute, 2.3% Complement, 9.0%
Irrelevant. Per query, a mean of 89% of judged candidates are relevant and 64%
are Exact. Ordering 16 products of which 14 are relevant is nearly impossible to
get wrong, so random ranking scored 0.80 and every effect had to fit in the
remaining 0.20.

It also made position bias degenerate: examination probability only matters when
many candidates compete for few visible slots. With 16 candidates and a 20-slot
window, everything is examined and there is no bias to simulate — which would
have quietly hollowed out M3 and M4.

Padding to 100 matches how production rankers actually operate (order 100–1000
retrieved candidates, mostly irrelevant).

## What changed when the setting was corrected

| Comparison | judged-only | padded | Change |
|---|---|---|---|
| dense − bm25 | +0.0065 | +0.0042 | **−35%** |
| rrf − bm25 | +0.0083 | +0.0161 | **+94%** |

Two things worth stating plainly:

1. **Dense-alone's advantage over BM25 shrinks by about a third** once the task
   is realistically hard. The judged-only setting overstates it.
2. **Hybrid fusion's advantage nearly doubles.** Lexical and semantic matching
   fail on *different* candidates, so combining them pays off most when there
   are many hard negatives to reject — exactly the regime the easy setting
   hides.

## A process error, recorded

An intermediate run on a 3,000-query sample gave dense − bm25 = +0.0021, CI
[−0.0024, +0.0066], and was initially read as "the dense advantage disappears
under realistic candidates." That was wrong. The full split shows the effect is
real, merely smaller than the easy setting suggested; the 3,000-query sample was
underpowered to detect it, and the project's own power analysis said as much
(≈2,570 queries needed for an effect of that size).

It is kept here because it is the exact failure mode this project studies:
treating a non-significant result from an underpowered sample as evidence of no
effect.

## Threats to validity

- **Sampled negatives are *assumed* irrelevant.** With ~19 judged products out
  of 1.2M, a randomly drawn product is almost certainly irrelevant, and this is
  standard practice in the learning-to-rank literature. It is still an
  assumption, not a measurement.
- **Negatives are drawn uniformly** from products judged for *other* queries.
  BM25-retrieved hard negatives would be more confusable and are a documented
  extension, not implemented.
- **The grade mapping (E=3, S=2, C=1, I=0) is a modelling choice**, not
  measured. Conclusions about *ordering* between arms are robust to it;
  absolute NDCG values are not.
- **Single dataset, single locale** (ESCI US). No claim is made about transfer
  to other domains or languages.
- **Full-corpus Recall@100 is not reported.** In the re-ranking setting it is
  trivially 1.0 and carries no information; a genuine corpus-wide retrieval
  check remains an open item.

## Reproducing

```powershell
# Ingest (needs the ESCI parquet files in data/raw/)
python -m unbiased_rank.data.ingest

# Encode the products the test split references (~24 min on 6 CPU cores)
python -m unbiased_rank.indexing.catalog --splits test

# Baseline
python -m unbiased_rank.experiments.baseline --split test --candidates-per-query 100

# The judged-only configuration, for comparison
python -m unbiased_rank.experiments.baseline --split test --candidates-per-query 0
```
