# LTHCM improvement plan

Reviewed 29 September 2026. This is a code and artifact review, not a fresh run of the full real-data benchmark fleet. The published numbers below come from `BENCHMARK_REAL_RESULTS.md` and the local `runs/*_summary.json` files; they should be treated as reported results until a clean, independently repeatable run is archived.

## Executive assessment

The project has substantial mathematical prototypes, a useful set of real-data comparisons, and unusually candid negative results. Its best current evidence is **retrieval within supplied multi-hop paragraph pools**, especially MuSiQue: mined graph Joint@2 is 0.408 versus 0.274 for flat two-step on 500 questions. The case for a general-purpose, high-capacity LLM memory system is much weaker. The public engine is not the system being scored by most real benchmarks, important arms use hand-set components, and the reported artifacts are insufficient to replay individual decisions.

The fastest route to a step change is to narrow the product claim to **evidence retrieval with learned graph expansion**, make one production path evaluable, and spend effort on edge quality, candidate recall, and cost-aware routing. More mathematical modules will not address the measured bottlenecks.

## What the existing numbers actually say

| Evaluation | Strongest relevant comparison in the report | Reading |
| --- | --- | --- |
| MuSiQue, 500 local question pools | Mined graph Joint@2 0.408; flat two-step 0.274; random graph 0.188 | A promising local-pool retrieval gain. Still 0.716 with oracle edges, so edge/seed recall is the main headroom. |
| HotpotQA distractor, 500 stratified questions | Mined plus rerank 0.546; flat plus rerank 0.532; routed plus rerank 0.544 | Small pooled gain; the bridge slice is 0.352 versus 0.328. Comparison questions gain little from graph work. Needs a larger held-out run and end-to-end cost measurement. |
| 2WikiMultiHopQA, 100 questions | Mined Joint@2 0.420; flat two-step 0.370 | Directional, but the reported comparison has p=0.1588. |
| SciFact, 300 queries | Mined plus rerank nDCG@10 0.6910; flat plus rerank 0.6936 | Graph expansion does not add value after reranking on this dataset. |
| VitaminC, 1,000 pairs | NLI-grounded sheaf accuracy 0.492; raw NLI 0.517 | Sheaf underperforms its own semantic model. Pure untrained sheaf is 0.400, below 0.495 majority accuracy. |
| WordNet, 250 nodes | Poincare MAP 0.1393; Euclidean 0.1418 | No hierarchy gain shown. |

The report also identifies RULER as saturated and the 32-frontier/case-study material as mechanism demonstrations. Keep those separate from competitive benchmark claims.

## Priority 0: make the evidence trustworthy and the product real

### 1. Benchmark the same path users call

**Gap.** `LTHCMEngine.retrieve_and_verify` in `lthcm/engine/pipeline.py` has its own candidate generation, filtering, and Fisher-Rao final sort. The real benchmark files implement separate ranking formulas and graph bonuses; the real suites do not call this method. In the engine, the optional Wasserstein similarity and several intermediate graph scores are returned or logged but do not enter the final sort. The default Titans update changes state during reads. As a result, benchmark improvements cannot be assumed to carry over to the advertised engine.

**Change.** Extract one deterministic `Retriever` interface with `index(corpus)`, `search(query, k, config)`, ranked document IDs, scores, and stage traces. Make the real runners call it. Put graph mining, reranking, temporal filtering, contradiction flags, and online updates behind explicit configuration. Default evaluation to read-only inference; evaluate updates as a separate continual-memory experiment. Add an integration test that scores the public API and benchmark adapter identically on the same fixed corpus.

**Done when:** the same config and corpus produce identical ranked IDs/scores through the library API and benchmark CLI, and each scored module is visible in an ablation.

### 2. Publish replayable run artifacts

**Gap.** `runs/` is ignored by Git. The checked-in report lists hashes of local summary JSONs, but the summaries omit ranked per-query/per-document outputs. `validate_artifacts()` checks nonempty files, a shared run ID, and `is_synthetic`, not dataset revision, query IDs, model revision, actual commit, or the reported values. The Markdown renderer contains fixed case-study numbers, hardware text, sample sizes, conclusions, and even some fixed router counts. Its saved-run mode stamps the current time and estimates total runtime from per-suite values plus a constant. A regenerated report can therefore look like a new verified run without a new evaluation.

**Change.** Save a run bundle for each experiment: config, Git SHA and dirty flag, dataset name/split/revision/checksum, selected query IDs, model IDs and revisions, seed, device, dependency lock, corpus and edge hashes, TREC run files or equivalent ranked JSONL, per-query metrics, stage timings, peak RAM/VRAM, and aggregate metrics. Put a small immutable manifest plus scripts under version control or a permanent artifact release. Render every numeric claim and environment field from the bundle; fail generation on missing fields. Keep qualitative examples in a separately labeled static section.

**Done when:** a clean checkout can regenerate the report byte-for-byte from a versioned bundle, and a reviewer can recompute every aggregate and paired comparison from per-query records.

### 3. Remove ambiguity around the “learned graph”

**Gap.** `build_edges(method="learned")` creates `LearnedEdgeScorer()` when no scorer is passed. Its default coefficients are hard-coded in `lthcm/graph/learned_edge_scorer.py`; the HotpotQA benchmark calls the method without loading a fitted checkpoint. The file calls these defaults pre-calibrated on HotpotQA train, but no training provenance or calibration artifact is attached to this arm. The scorer can fit on training pairs, yet that is not the path used by the published evaluation.

**Change.** Rename the current arm `fixed_weight_edges`. Train a scorer on training questions only, freeze a versioned checkpoint, and require that checkpoint for a `learned_edges` arm. Record the train IDs, feature schema, negative sampling, dev tuning, calibration curve/Brier score, and edge recall/precision. Never insert gold edges into the candidate proposal set when measuring candidate-generator recall; report proposal recall and reranker quality separately.

**Done when:** a fresh process loads the same checkpoint, reproduces edge scores, and the trained arm improves over fixed weights and random edges on untouched validation questions.

### 4. Correct the public claim hierarchy

**Gap.** The README opens with 100% synthetic mechanism results and calls several hand-built examples “real-world benchmarks” or “SOTA,” while the competitive real-data table contains neutral and negative outcomes. Its test badge/count and directory description also appear older than the current tree. This makes the strongest claim harder to find and the system easier to overinterpret.

**Change.** Lead with a compact real-data scorecard and explicit task definitions: paragraph retrieval, not answer accuracy; supplied 10-paragraph HotpotQA pools, not open-corpus search; oracle as upper bound. Move synthetic mechanism tables below real results. Tie the badge to CI. Separate implemented, integrated, and empirically validated modules.

**Done when:** a reader can identify the strongest baseline, sample/split, metric, and limitation for each headline result without reading the paper.

## Priority 1: change the benchmark so gains transfer

### 5. Evaluate larger and harder retrieval settings

**Gap.** HotpotQA uses 500 stratified questions although the validation split is larger. MuSiQue and 2Wiki score each question's provided paragraph pool. These are useful controlled tasks but do not establish an external long-term memory or open-corpus advantage. The smaller slices also magnify selection and multiple-comparison risk.

**Change.** Freeze development choices now, then run the full HotpotQA distractor validation split, all eligible MuSiQue/2Wiki validation examples, and at least two additional BEIR datasets with different domains. Add an open-corpus retrieval task with stable passage IDs and explicit corpus construction. Report results by bridge/comparison, hop count, answer location, corpus size, and graph-edge coverage. Compare against BM25, exact dense, hybrid BM25+dense, flat two-step, and the same reranker with the same candidate budget. Use a separate development split for weights and routing thresholds.

**Done when:** the gain survives the full held-out split against the strongest equal-budget baseline and its confidence interval excludes zero on a preregistered primary metric.

### 6. Make edge and seed failure measurable

**Gap.** MuSiQue mined Joint@2 of 0.408 is far from its oracle 0.716. In the current multi-hop runners, expansion begins from a single top seed or a small set. The graph cannot recover a gold second hop if the first hop is missed or the required edge is absent.

**Change.** Record per-question first-hop recall@k, candidate proposal recall, gold-edge coverage, reachable-gold rate, second-hop rank, and failure category. Try top-k seeds with a bounded beam, learned query-conditioned edge scores, and a no-graph fallback. Tune k/beam/bonus on dev, then lock. Compare a simple title-link graph, current heuristics, trained edges, and random degree-matched edges with identical query and rerank budgets.

**Done when:** improvement is attributable to higher reachable-gold and joint retrieval, not only a changed score scale; no material regression on comparison questions.

### 7. Measure the trade-off users would pay

**Gap.** The report has limited routing timing, but no consistent end-to-end cost table across all arms. `SuperpositionMemoryStore.add_document` appends to a dense tensor, and `WormholeGraphIndex.query` computes direct scores in a Python loop over every node. That is linear-scan behavior before graph traversal, so “high-capacity” is unproven at scale.

**Change.** Measure indexing time, graph-build time, p50/p95 query latency, queries/sec, resident RAM, peak VRAM, serialized index size, and incremental-update cost at 10k, 100k, and 1m passages. Include embedding and reranker time, cache cold/warm status, and hardware. Replace repeated tensor concatenation with batched ingestion and a persistent vector index; keep graph adjacency sparse. Plot effectiveness versus latency/cost and choose a default on the Pareto frontier.

**Done when:** the recommended configuration meets explicit latency/memory targets and beats a strong hybrid-plus-reranker baseline at the same budget.

## Priority 2: simplify and validate the research stack

### 8. Retire or isolate modules that have no measured contribution

The engine instantiates superposition, gauge, spectral graph refinement, Lorentzian filtering, sheaf, Fisher-Rao, stochastic resonance, hyperbolic storage, Titans, and more. A mathematical invariant test establishes implementation properties, not retrieval utility. Use a module-by-module ablation on the public path, with the same candidate set, seed, and budget. Keep a module in the default path only if it improves a relevant held-out metric or supplies a documented product guarantee. Put exploratory modules behind an `experimental` namespace/config.

The temporal gate needs a specific time-stamped corpus and anachronism metric. The sheaf needs multi-document consistency tests with calibrated abstention, NLI-only and NLI-plus-rule controls. The hyperbolic store needs a larger hierarchy task. The online memories need ordered stream tests for retention, interference, and privacy-safe deletion. Without such tests, present them as prototypes.

### 9. Fix benchmark statistics and protocol drift

The paired bootstrap helper reports a two-sided p-value from the fraction of ordinary bootstrap means crossing zero. Use a paired randomization/permutation test for the null p-value and a paired bootstrap for the confidence interval, or document a validated bootstrap test. Avoid displaying `p=0.0000`; report a bound set by the number of resamples. Choose primary comparisons and metrics in advance; account for multiple suites, slices, and exploratory arms. Persist per-query scores so uncertainty and error analyses can be audited.

### 10. Make setup and CI reproducible

`pyproject.toml` uses lower bounds and `requirements.txt` omits benchmark packages; the CUDA lock is hardware-specific. Add a tested CPU lock or constraints file, a documented optional GPU profile, dataset/model download instructions with revisions, and a single smoke command that never silently reports synthetic fixtures as real. Split fast deterministic unit tests from external-data benchmark jobs. Add CI gates for the public retrieval API, graph leak checks, report rendering from a tiny fixture, and package installation from a clean environment.

## Proposed benchmark contract

1. **Question and data:** one row per dataset/split with corpus size, query count, task, qrels or supporting-fact definition, and exact selected IDs. State whether the corpus is question-local or open-corpus.
2. **Shared resources:** all arms use the same text, encoder, initial candidate budget, reranker model, and final `k`; only the named component changes. Count graph build and query generation in cost.
3. **No gold leakage:** mine edges from allowed corpus text or a versioned training split. Freeze graph/model before test queries. Keep oracle results in an upper-bound appendix.
4. **Ablations:** flat, hybrid lexical+dense, two-step, rerank-only, graph-only, graph-plus-rerank, random graph, and oracle. Test router against always-graph and always-flat.
5. **Primary metrics:** Joint@2 for local multi-hop pools; nDCG@10 and Recall@100 for open-corpus IR; macro-F1 and risk/coverage for verification. Include p95 latency and peak memory alongside quality.
6. **Analysis:** paired per-query deltas, 95% intervals, a valid null test, failure buckets, and untouched full-split confirmation. Keep demos separate.

## First execution sequence

| Step | Deliverable | Gate to move on |
| --- | --- | --- |
| 1 | One benchmark adapter for the public retriever, plus parity test | Published arms correspond to callable product behavior. |
| 2 | Immutable run bundle and data-driven report renderer | All tables regenerate from ranked outputs and provenance. |
| 3 | Full-split baseline rerun with preregistered primary comparisons | Current gains and losses replicate. |
| 4 | Trained edge scorer and top-k seed/beam ablation | Gains exceed fixed heuristic and strong two-step/rerank baselines. |
| 5 | 10k/100k/1m scaling and quality-versus-cost report | A default configuration is defensible for real workloads. |

If steps 3–5 do not show a robust graph gain, position the repository as an experimental retrieval research toolkit and retain only the modules with demonstrated utility in the default engine.

## Review pointers

- `lthcm/engine/pipeline.py`: main retrieval path, final Fisher-Rao sort, optional scores, online update.
- `lthcm/benchmarks/real/{multihop,ir,contradiction,hierarchy}`: independent evaluation implementations.
- `lthcm/graph/{build_edges,learned_edge_scorer}.py`: graph proposals and fixed “learned” weights.
- `lthcm/benchmarks/run_real_benchmarks_to_md.py`: fixed prose/values and artifact validation.
- `lthcm/core/{superposition,wormhole}.py`: current ingestion and query scaling behavior.
- `BENCHMARK_REAL_RESULTS.md`, `runs/*_summary.json`, `README.md`, `pyproject.toml`, `.gitignore`: claims, evidence, packaging, persistence.
