A runnable guide for two regimes: standard flat-vector benchmarks, and the specific harness changes a topological system needs, done without gold leakage.

Part 1: Standard
Part 2: Topological

Part 1: the standard protocol
Goal: numbers a reviewer can reproduce and compare against published baselines. Real data, standard metrics, no LTHCM-specific structure. LTHCM enters only as a ranker over the same embeddings ("LTHCM-flat": Fisher-Rao / POVM / stochastic resonance, zero wormhole edges).

Encoder
sentence-transformers/all-MiniLM-L6-v2, 384-d, normalized, default max_seq_length
Dense baseline
faiss.IndexFlatIP (exact), top-100
Lexical baseline
BM25 (rank_bm25 BM25Okapi, or Pyserini for the published BEIR setup)
Metrics tool
pytrec_eval, never hand-rolled metrics
Tuning
Any threshold or weight fit on train/dev only. Test is touched once.
0. Environment
python -m venv .venv && source .venv/bin/activate
# CPU box: the pinned CUDA wheels in requirements.txt are not needed
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers faiss-cpu rank_bm25 pytrec_eval-terrier datasets pandas networkx scipy
# add these to requirements.txt (sentence-transformers is missing today)
git rev-parse HEAD > runs/SHA.txt; pip freeze > runs/freeze.txt
Log the git SHA, package versions, device and seed with every run. Save every run file (query to ranked doc ids with scores), not just the averages.

1. BEIR SciFact (tested, reference numbers below)
Data
HF BeIR/scifact (corpus 5,183 docs, queries) + BeIR/scifact-qrels test.tsv (300 test queries)
Doc text
title + " " + text
Metrics
nDCG@10 (primary), Recall@10, Recall@100, MAP@100, MRR (pytrec recip_rank over the top-100 run; truncate the run to 10 for strict MRR@10)
from datasets import load_dataset
import pandas as pd, numpy as np, faiss, pytrec_eval
from sentence_transformers import SentenceTransformer

corpus  = load_dataset("BeIR/scifact", "corpus",  split="corpus")
queries = load_dataset("BeIR/scifact", "queries", split="queries")
qr = pd.read_csv("hf://datasets/BeIR/scifact-qrels/test.tsv", sep="\t", dtype=str)
qrels = {}
for q, d, s in qr.itertuples(index=False):
    qrels.setdefault(q, {})[d] = int(s)
qids  = [q["_id"] for q in queries if q["_id"] in qrels]      # 300
qtext = {q["_id"]: q["text"] for q in queries}
dids  = [d["_id"] for d in corpus]
dtext = [(d["title"] + " " + d["text"]).strip() for d in corpus]

m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
D = m.encode(dtext, batch_size=64, normalize_embeddings=True).astype("float32")
Q = m.encode([qtext[q] for q in qids], normalize_embeddings=True).astype("float32")
ix = faiss.IndexFlatIP(D.shape[1]); ix.add(D)
S, I = ix.search(Q, 100)
run = {q: {dids[j]: float(s) for s, j in zip(S[n], I[n])} for n, q in enumerate(qids)}

ev = pytrec_eval.RelevanceEvaluator(
    qrels, {"ndcg_cut.10", "recall.10", "recall.100", "map_cut.100", "recip_rank"})
per_q = ev.evaluate(run)   # keep per-query scores for significance tests
My run of the code above, 24 Sep 2026, CPU. Use as a sanity check for your setup.
Attribute	BM25 (rank_bm25, \w+ lowercase)1	Flat FAISS, MiniLM-L6-v22
nDCG@10	0.652	0.645
Recall@10	0.774	0.783
Recall@100	0.873	0.925
MAP@100	0.613	0.603
MRR (over top-100)	0.623	0.611
Compare with the synthetic set
The synthetic harness reports flat nDCG@10 0.940 and MRR 1.000. On real SciFact flat MiniLM is 0.645. The real set has far more headroom, which is where a genuine LTHCM gain would show.
Then run LTHCM-flat over the same D and Q and score it with the same evaluator. Repeat on NFCorpus, TREC-COVID and FiQA (same BeIR/* loaders) so one dataset cannot carry the result.

2. HotpotQA (multi-hop retrieval)
Primary data
HF hotpotqa/hotpot_qa, config distractor, validation split (7,405 questions)
Setting
Rank each question's own 10 paragraphs (2 gold + 8 distractors). Gold = titles in supporting_facts.
Harder setting
BeIR/hotpotqa (open corpus of Wikipedia passages, test queries) with the same pytrec_eval pipeline. Budget RAM for a flat index of that size.
Metrics
Both-gold-in-top-2 (joint / EM@2), Recall@2, Recall@5, nDCG@10, MAP
ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
for ex in ds:
    titles = ex["context"]["title"]
    paras  = [t + ". " + " ".join(s) for t, s in zip(titles, ex["context"]["sentences"])]
    gold   = set(ex["supporting_facts"]["title"])       # used ONLY for scoring
    # rank paras against ex["question"] with BM25 / flat / LTHCM-flat
    # joint@2 = 1 if both gold titles are in the top 2
Every system returns a full ranking of the pool. No system gets a special 2-slot construction.
Report the flat baselines: BM25, flat single-shot, and a simple flat 2-step baseline (retrieve top-1, append its text to the query, retrieve again). Without the 2-step baseline, any multi-hop method beats a single-shot retriever and proves little.
Report by question type (bridge vs comparison). Bridge questions are where hop 2 matters.
3. FEVER and VitaminC (claim verification)
VitaminC
HF tals/vitaminc, test split (55,197 claim/evidence pairs), labels SUPPORTS / REFUTES / NOT ENOUGH INFO
FEVER
Official FEVER 1.0 shared-task files, or HF copenlu/fever_gold_evidence (claim + gold evidence sentences; validation 15,935). The copenlu set is a community re-format, so say which one you used.
Task
Given claim + gold evidence, predict the label. Report 3-way, and 2-way (drop NEI) separately.
Metrics
Accuracy, macro-F1, per-class F1, confusion matrix
Baseline A: cosine threshold, tuned on the dev split, applied once to test.
Baseline B (required): cross-encoder/nli-deberta-v3-xsmall alone, argmax over (contradiction, entailment, neutral), mapped to REFUTES / SUPPORTS / NEI. LTHCM uses this model, so it is the real baseline.
Baseline B2: the same cross-encoder with the exact symmetric rule LTHCM uses (average P(contra) both directions, entailment gate 0.35). This isolates what the sheaf layer adds beyond the rule.
Class balance: FEVER and VitaminC are not 50/50. Report majority-class accuracy too.
4. Reporting rules for Part 1
1
Every table row names its dataset, split and query count.
2
Paired significance over queries: paired bootstrap (10,000 resamples) or permutation test on per-query scores, with 95% CIs. With n = 8 or 10 queries, as in the synthetic sets, no difference is significant.
3
Publish run files and the exact command lines.
4
Label the synthetic suites as synthetic smoke tests and keep them out of any table that names HotpotQA, BEIR or FEVER.

Part 2: where the harness must change for a topological system
Your 4.2 point is fair: a benchmark that treats documents as isolated points never lets a graph do anything. The fix is to let LTHCM build and use its graph, under one rule that keeps the result honest.

The graph may be built only from corpus text (plus any external source you disclose), before any query is seen, with no access to qrels, supporting facts, labels, case ids or hop tags. It is then frozen and shared by all queries.

Step 0: decouple unit tests from benchmark scores
1
Move the hand-written suites to lthcm/benchmarks/synthetic/ and rename them (for example demo_multihop.py, demo_scifact_style.py, demo_contradiction.py). Retitle the docstrings: "Synthetic smoke demo. Not HotpotQA / BEIR / FEVER."
2
Rewrite test_standard_community_benchmarks_suite to check that the suites run and return the right schema (keys present, metrics in [0, 1], deterministic across two runs). Drop the >= 0.95 thresholds and the "LTHCM > flat" assertions. A unit test that requires a benchmark score locks in the gold wiring.
3
Do the same for test_universal_32_frontiers_benchmark_suite (it asserts 32/32 pass): test execution, not outcome.
4
Put the real benchmarks in lthcm/benchmarks/real/ behind a pytest marker (for example @pytest.mark.benchmark, skipped unless RUN_BENCHMARKS=1). They need downloads and minutes of compute, so they should not run in the unit suite.
5
Delete the gold add_wormhole calls from the synthetic scripts too, or keep them only in a function named oracle_edges() that is reported as an oracle.
Step 1: build an edge builder (LTHCM does not have one yet)
lthcm/core has WormholeGraphIndex.add_wormhole but no entity extraction or automatic linking. Every edge today is written by hand. Add a module such as lthcm/graph/build_edges.py and treat it as a named component in the paper, since the result depends on it.

def build_edges(docs, method="title_mention+ner", max_out=5):
    """docs: list of {id, title, text}. Sees corpus text only.
    Returns [(src, dst, relation, permeability)]. No labels, no queries."""
    # 1. title-mention: A -> B if B's title (or alias) appears in A's text
    # 2. shared named entities (spaCy en_core_web_sm; scispaCy for biomedical),
    #    weighted by IDF of the entity across the corpus
    # 3. cap out-degree at max_out; permeability from a fixed rule
    #    (e.g. 0.85 * idf_norm), set on dev, never on test
    ...

edges = build_edges(corpus)                 # once, before the query loop
json.dump(edges, open("runs/edges.json", "w"))
for s, d, rel, p in edges:
    index.add_wormhole(s, d, rel, permeability=p)
Report edge stats: number of edges, mean out-degree, share of distractor docs with edges, and the share of gold pairs the builder happened to connect ("gold-pair coverage"). Coverage is a diagnostic you compute after the fact, never an input.
Node metadata must not carry labels. hotpot_eval.py currently sets cluster="case_<case_id>" on every gold node. Even if nothing reads it today, remove it.
HotpotQA: what changes
1
Replace lines 269-285 (gold hop1 <-> hop2 wiring) with build_edges() over every paragraph in the pool, distractors included. In the distractor setting the 8 distractors often mention the same entities, so they get edges too. That is the point: the graph must earn its gain against distractors that also have bridges.
2
Keep the Re-Wormholing Loop, but it is only meaningful once distractors can have edges. Change "first candidate with any outgoing edge" to "candidate whose best bridge has the highest combined score" (seed score x permeability x bridge-query similarity). With the current rule plus gold-only edges it is a label lookup.
3
Return a full ranked list and score it with the same pytrec_eval code as Part 1. If you want a 2-slot [seed, bridge] output, give the flat 2-step baseline the same 2-slot budget.
4
Run four arms on the same questions: (a) no edges, (b) random edges with the same degree distribution, (c) build_edges(), (d) gold edges labeled ORACLE. The claim lives in (c) minus (a) and (c) minus (b). (d) is an upper bound only.
What this entitles you to claim
If (c) beats flat 2-step with a significant paired test: "Automatically extracted entity bridges improve multi-hop paragraph retrieval over flat and iterative dense baselines." It does not let you claim HotpotQA QA accuracy (no answer extraction was scored), and it does not let you use (d) as a result.
Suggested language: "We evaluate on the HotpotQA distractor dev set (7,405 questions) under a modified retrieval harness in which LTHCM constructs a wormhole graph from corpus text via title-mention and named-entity linking, without access to supporting facts. The graph is frozen before querying. We report no-edge, random-edge and oracle-edge ablations."

BEIR SciFact: what changes
1
Replace lines 246-249 (four hand-picked edges between co-relevant docs) with edges mined over all 5,183 docs: shared biomedical entities (genes, drugs, proteins via scispaCy), IDF-weighted, capped out-degree. A kNN embedding graph is a weaker second option; say which one you used.
2
SciFact ships no citation graph. If you add real citations (OpenAlex or Semantic Scholar), build them for every doc in the corpus, disclose the source, and report it as "LTHCM + external citation graph".
3
Same four arms: no edges, random edges, mined edges, oracle.
4
Repeat on at least two more BEIR sets (NFCorpus, TREC-COVID). A graph tuned on one small set often does not transfer.
What this entitles you to claim
"Text-mined relational edges improve nDCG@10 on SciFact (and X, Y) over flat MiniLM-L6-v2 retrieval." Any nDCG gain should be read against the real flat baseline of 0.645, not the synthetic 0.940. It does not support "vocabulary gap bridged" in general unless it holds across datasets.
Suggested language: "Edges are derived solely from corpus text (entity co-occurrence, IDF-weighted, max out-degree 5) and never from relevance judgments."

FEVER / VitaminC: what changes
1
Keep the symmetric sheaf setup. It is a legitimate aggregation rule.
2
Name the NLI model as a component in every table: "LTHCM sheaf + nli-deberta-v3-xsmall". Report Baseline B (NLI alone) and B2 (NLI with the same symmetric rule) next to it.
3
Fit the 2.5 obstruction threshold and the 0.35 entailment gate on dev only, then freeze them.
4
Where the sheaf can add something beyond pairwise NLI is multi-passage consistency: several evidence sentences per claim, contradictions across a set. Test that directly (FEVER claims with multi-sentence evidence) instead of single pairs.
What this entitles you to claim
Only the delta over B2. On the 20 synthetic pairs that delta is zero: the cross-encoder with the same symmetric rule also gets 20/20. "Cosine similarity cannot detect contradiction" is true but is not an LTHCM result, and the cosine row should not be the headline comparison.
Suggested language: "Contradiction scores are produced by a pretrained NLI cross-encoder (nli-deberta-v3-xsmall) and aggregated through a symmetric sheaf restriction map. We report the cross-encoder alone and with the identical symmetric rule as baselines."

How to present the two regimes in the paper
Replace the current Table 1 / Table 2 framing with these.
Attribute	Regime A: standard (Part 1)1	Regime B: topological harness (Part 2)2
Data	Real test splits, full size	Same splits
Graph	None	Built from corpus text, frozen, logged
Baselines	BM25, flat FAISS	Plus flat 2-step, NLI-only, random edges
Claim	Comparable to published numbers	Gain from the graph, measured by ablation
Suggested wording for 4.1: "Regime A follows standard BEIR / HotpotQA / FEVER protocols with no structure beyond embeddings. Regime B permits LTHCM to construct a relational graph from corpus text before querying. No regime uses relevance judgments to build structure. Oracle-edge results are reported separately as an upper bound."

For the current Table 1: either drop its HotpotQA rows or relabel them "synthetic set, oracle edges, no backtracking". Describing them as "flat-vector constrained" is not accurate, because the gold edges were already present.

Run checklist
1
Step 0 done: tests pass without score thresholds; synthetic suites renamed.
2
Part 1 numbers for BM25, flat, LTHCM-flat on SciFact, HotpotQA distractor, VitaminC / FEVER. Sanity-check SciFact flat nDCG@10 ≈ 0.645.
3
build_edges() committed, edges.json saved per dataset, edge stats reported.
4
Four-arm ablation per dataset with paired bootstrap CIs.
5
grep the harness for supporting_facts, qrels, label, case_id, hop inside anything that runs before scoring. Nothing should match.