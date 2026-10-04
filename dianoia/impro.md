# Dianoia improvement and benchmark plan

Reviewed: 2026-09-29. Scope: the Dianoia application at the repository root and its pinned LTHCM submodule (`8a71605379ee32059e714feb3648cf74f4b42dc6`). This is a code and documentation review, not a new model evaluation.

## Executive assessment

Dianoia has a coherent prototype loop and useful parser, logging, and mocked controller tests. Its main missing proof is that **LLM guided walking improves answers over a simpler retrieval and synthesis pipeline at the same evidence and compute budget**. The published benchmark report in `LTHCM/BENCHMARK_REAL_RESULTS.md` measures LTHCM retrieval and verification variants, not Dianoia's complete walk, answer, rewalk, cost, or reliability. Treat those results as motivation and baselines, not as Dianoia results.

The largest improvement opportunity is to make evidence selection robust before adding more model complexity: passage level indexing, all seed access, bounded neighbor selection, citations, and explicit abstention. Then benchmark against strong flat retrieval plus reranking and the same answer model.

## Observed gaps, in priority order

| Priority | Gap and evidence | Impact | Concrete change |
| --- | --- | --- | --- |
| P0 | `walk()` retrieves `top_k` seeds but starts at `seeded_node_ids[0]` and has no action to switch roots (`dianoia/walk.py:533-559`, `:633-660`). | Relevant evidence in seeds 2 through k is inaccessible unless connected by a traversable edge. `top_k` currently overstates exploration breadth. | Add a seed frontier and `SWITCH_SEED`/frontier action, or let the controller choose among seeds initially. Test disconnected gold evidence. |
| P0 | The CLI makes one node per entire `.txt`/`.md` file (`dianoia/cli.py:33-62`, `:93-103`), then inserts full node content into prompts (`dianoia/walk.py:573-580`, `:450-463`). | Long files dilute embeddings, exceed context limits, hide passage provenance, and inflate latency. | Chunk by tokens/sections with overlap; retain file path, heading, offsets, stable chunk ID, and parent document ID. Limit prompt tokens and select bounded evidence. |
| P0 | After a conflict, `run()` returns the rewalk answer even if re-verification still finds conflicts (`dianoia/walk.py:754-770`, `:807-817`). On rewalk failure it returns the original conflicted answer (`:771-781`). | A known unsupported or contradicted answer can be displayed as the final answer. | Define `verified`, `conflicted`, `insufficient_evidence`, and `verification_unavailable` outcomes. Return an explicit abstention or clearly marked draft on unresolved conflict; never label it verified. |
| P0 | NLI model loading catches every exception and silently falls back to geometric scores (`LTHCM/lthcm/core/sheaf.py:153-163`). Dianoia reports whether NLI is active (`dianoia/verify.py:96-103`) but still treats `conflicts_found=False` as success. | A missing model or download failure changes verification semantics without an operational failure. The LTHCM report itself shows its NLI sheaf arm losing to raw NLI on VitaminC (49.2% vs 51.7%, `LTHCM/BENCHMARK_REAL_RESULTS.md:21`, `:180-191`). | Make verifier availability a first class state; fail closed or explicitly disable verification. Compare raw NLI and sheaf as separate arms, calibrate thresholds on dev data, and measure false negatives. |
| P0 | The index is loaded with `pickle.load()` (`dianoia/cli.py:156-176`). | Loading an index from another person can execute arbitrary code. | Replace pickle with a versioned data-only format and explicit validation. Until then, document that index files must be trusted and reject untrusted sources. |
| P1 | The graph is mandatory even for a single document or a disconnected corpus (`dianoia/cli.py:119-146`; `dianoia/walk.py:301-315`). | The quick-start `--docs` path fails for small or weakly linked collections, although ordinary retrieval could answer them. | Add a flat retrieval fallback and mark which route was used. Measure graph coverage and isolated-node rate. |
| P1 | Neighbor display shows every outgoing neighbor (`dianoia/walk.py:328-356`, `:481-490`); no question relevance ranking or hard token cap is applied. The controller can revisit old nodes (`:637-659`). | Hub nodes enlarge prompts and loops consume steps without adding evidence. | Rank a bounded frontier by question similarity, edge quality, novelty, and diversity; track visited edges; stop on low expected gain. Log skipped candidates for analysis. |
| P1 | No source citations or claim to passage mapping in the final answer (`dianoia/prompts.py:63-85`, `dianoia/walk.py:450-475`). | Users cannot inspect support, and answer level NLI over whole passages may miss unsupported individual claims. | Require cited chunk IDs per factual claim; verify claim by claim against cited spans; score citation precision and recall. |
| P1 | Retrieved document text is interpolated directly into controller and synthesis prompts (`dianoia/prompts.py:46-60`, `:74-84`). | A document can instruct the LLM to change actions or ignore the question. | Delimit retrieved material as untrusted data, use structured tool output or constrained actions where supported, and add malicious document regression cases. |
| P1 | Packaging declares only `dianoia*`; the CLI manually adds the checkout's `LTHCM` directory to `sys.path` (`pyproject.toml`, `dianoia/cli.py:351-358`). | Editable checkout works differently from an installed wheel or direct `WalkController` import. | Declare and pin an installable LTHCM dependency or package both components explicitly; test installation from a clean wheel in CI. |
| P1 | `--index` loading does not check the embedding model name or dimension against the index (`dianoia/cli.py:156-216`, `:360-383`). | A saved index may be queried with an incompatible encoder or misleading embedding geometry. | Store index schema, LTHCM revision, encoder ID/revision, dimension, normalization, chunker, and graph builder settings. Validate on load. |
| P1 | A Dianoia run constructs the full LTHCM engine, which initializes many modules beyond the superposition store, graph, resonance seeder, and sheaf filter used by the walk (`LTHCM/lthcm/engine/pipeline.py:59-84`; `dianoia/walk.py:320-356`, `:833-850`). | Startup, VRAM use, and serialization can be dominated by unused components. | Profile initialization and split a lean retrieval/verification interface from optional research modules; report memory and startup deltas before changing architecture. |
| P2 | JSONL logs include raw questions, model outputs, and final answers and append without rotation (`dianoia/log.py:39-125`). | Sensitive content can persist indefinitely and logs can grow without bound. | Make logging opt-in or configurable by retention level; redact or hash content when requested, rotate files, and retain structured timings and IDs. |
| P2 | Model calls have no explicit request timeout, retry budget, per-question token cap, or latency accounting (`dianoia/walk.py:415-448`). | A walk can hang or incur unpredictable cost. | Add deadlines, bounded retries, token/step budgets, and phase-level timing. Surface parse failure and fallback rates. |
| P2 | CLI inputs such as `--max-steps` and `--top-k` are not validated as positive (`dianoia/cli.py:321-332`). | Zero/negative values can produce misleading forced-answer behavior or empty seeds. | Validate arguments and return a clear error before indexing or contacting models. |

## Benchmark status today

- Dianoia has **no end-to-end benchmark runner or saved Dianoia benchmark results** in this repository. Its 56 discovered unit tests exercise parsing, prompts, logging, CLI behavior, and mocked walk logic; they are not answer-quality measurements.
- In this review environment, `python -m unittest discover -s tests -v` ran 56 tests: 33 passed and 23 were skipped because PyTorch is absent. `pytest` is also absent. The actual walk, NLI, model server, and GPU paths were therefore not validated here.
- No CI workflow or lockfile for Dianoia is present at the repository root. Add a small unit and packaging job, then a separate, opt-in GPU integration job.
- LTHCM's report provides useful reference points: HotpotQA flat plus rerank Joint@2 0.532 versus mined plus rerank 0.546, SciFact flat plus rerank nDCG@10 0.6936 versus mined plus rerank 0.6910, and VitaminC raw NLI accuracy 51.7% versus NLI sheaf 49.2% (`LTHCM/BENCHMARK_REAL_RESULTS.md:17-23`, `:132-157`, `:175-191`, `:221-236`). These mixed outcomes argue for routing and for measuring marginal benefit rather than assuming every graph step helps.
- The report lists `runs/*.json` and checksums (`LTHCM/BENCHMARK_REAL_RESULTS.md:273-284`), but `LTHCM/runs` is absent from this checkout. Results could not be independently reproduced from included artifacts. Its Hotpot narrative also uses a 250/250 stratified slice in one section and 242/258 production routing counts in another (`:133-165`); preserve both only with explicit sampling and denominator descriptions.

## End-to-end Dianoia benchmark to build

### Questions and evaluation unit

1. **Primary:** HotpotQA distractor validation and MuSiQue validation, using each question's provided candidate paragraphs. Score answer exact match and token F1, both supporting paragraphs found, supporting fact recall, citation precision/recall, and abstention accuracy. Report bridge/comparison and hop count slices. Build graph edges only from candidate corpus text and disclosed metadata; never from supporting facts or labels.
2. **Open corpus transfer:** BEIR SciFact and at least one further BEIR collection. Score retrieval nDCG@10 and Recall@k, plus grounded answer correctness on a separately labeled answer subset. Avoid claiming answer quality from retrieval metrics alone.
3. **Verification:** VitaminC/FEVER for claim-level verifier calibration, plus a multi-source conflicting-evidence set. Report macro-F1, contradiction recall, false alarm rate, and risk-coverage curves for abstention. Use raw NLI as a required comparator.
4. **Reliability and safety:** disconnected graph, single file, long document, misleading high-degree hub, missing NLI weights, malformed controller output, OOM/fallback, prompt injection in retrieved text, and an untrusted index load attempt. These are regression cases, not competitive benchmark scores.

### Paired arms

Use the same corpus, chunks, encoder, answer model, decoding settings, evidence token budget, and question IDs across arms. Record both equal evidence-budget and equal latency-budget comparisons.

| Arm | Purpose |
| --- | --- |
| A: BM25 or flat dense top-k + identical synthesis model | Basic retrieval-augmented generation baseline. |
| B: hybrid BM25+dense + cross-encoder rerank + identical synthesis model | Strong practical baseline. |
| C: LTHCM retrieval top-k + identical synthesis model, no walk | Isolate value of the memory layer. |
| D: Dianoia walk with graph edges, no verifier | Isolate value of deliberative traversal. |
| E: Dianoia walk + raw NLI claim verifier | Test the simpler verifier. |
| F: Dianoia walk + sheaf/NLI verifier and one rewalk | Full system; quantify repair success and extra cost. |
| Controls: random degree-matched edges, shuffled edges, oracle graph (upper bound only), and no-edge fallback | Detect benefit from actual graph structure and show graph headroom. |

### Reproducible protocol

- Freeze corpus, graph, and encoder before test questions are evaluated. Tune prompts, thresholds, routing, and step limits on train/dev only; use a held-out test once. Prevent model and dataset contamination claims from exceeding what can be checked.
- Publish dataset revision, split, sample IDs, qrels, graph construction mode, model and quantization hashes, code and submodule SHAs, package lock, seed, hardware, exact commands, and complete predictions/rankings. Store a machine-readable run manifest and per-question JSONL output with separate attempt IDs.
- Track answer EM/F1, factuality by human or audited rubric on a subset, evidence coverage, citation quality, conflict detection, rewalk repair rate, abstention risk/coverage, parse failure, fallback, timeout, p50/p95 latency, generated and prompt tokens, VRAM, and index build time/size.
- Compare paired per-question deltas with bootstrap confidence intervals; show absolute scores and costs. Report all failures in the denominator. Predeclare a primary metric and smallest useful gain before tuning. Include several seeds for stochastic model decisions.
- Add a small offline smoke fixture to CI and a separate opt-in full benchmark command for local GPU runs. CI should validate schema, determinism where expected, and that every arm receives the same question set; it should never assert a research score must improve.

## Delivery sequence and acceptance gates

### Phase 1: make the prototype dependable

1. Replace whole-file nodes with stable chunks and provenance; cap prompt size. **Gate:** long-document and duplicate-filename fixtures index without truncation or ID collision, and answer citations resolve to source spans.
2. Expose all initial seeds and a bounded, ranked graph frontier. **Gate:** disconnected gold evidence in any top-k seed remains reachable; no repeated-edge loop exceeds the step budget.
3. Add explicit verification and abstention states. **Gate:** unresolved contradiction, missing NLI model, and failed rewalk never appear as verified answers.
4. Version and validate index metadata; remove unsafe loading for untrusted files. **Gate:** clean wheel install works; mismatched encoder/index is rejected with a clear error.

### Phase 2: establish trustworthy measurements

1. Implement dataset adapters, paired arms, per-question outputs, and a manifest. **Gate:** one command reruns a small fixture and produces a validated report; full runs can be audited from saved artifacts.
2. Run HotpotQA and MuSiQue first, then SciFact and verification sets. **Gate:** report quality and cost against flat plus rerank; show confidence intervals and failure rates by question type.
3. Review individual wins and losses. **Gate:** at least 50 error cases are categorized by seed miss, graph miss, controller choice, synthesis, verifier error, and missing evidence; priorities are updated from those counts.

### Phase 3: improve only where evidence supports it

- Route simple or comparison questions to the cheaper flat path; invoke walking for questions whose graph expansion has measured value.
- Train or tune frontier ranking on dev data before training a full controller. Use logs only after adding outcome labels and privacy controls; raw trajectories alone do not supply a correct policy target.
- Consider stronger graph edges and learned policies only if their gain survives the strong rerank baseline and equal-budget evaluation.

## Suggested first benchmark success criterion

Pre-register a target such as **at least +3 percentage points in supporting-paragraph Joint@2 on bridge questions over hybrid retrieval plus rerank, with a 95% paired confidence interval above zero, while keeping answer F1 no worse and p95 latency within a stated budget**. This is a proposed gate, not a measured result. If the graph cannot meet it, ship the flat route and keep graph walking as an opt-in research feature while investigating the failure categories.
