Goal: Replace the Tier 3/4 frontier items (#11 Complex-Analytic Embeddings, #13 H^3 Temporal Paradox, #14 Quantum Error Correction, #15 HoTT) with five additions that each (a) attack a gap the real benchmark fleet actually measured, (b) have a published baseline and runnable reference code, and (c) can win or lose on a real benchmark within a day of CPU/GPU time.

Why These Five (Measured Gaps, Not Hypothetical Ones)
All numbers below come from the independent second-POV run on HEAD 2c8f544. Harnesses were unmodified; the only reviewer additions were an embedding disk cache and per-query metric capture. Setup: MiniLM-L6-v2, CPU, seed 42.

Benchmark (exact run)	Flat Dense	LTHCM Mined	LTHCM Oracle	What it tells us
HotpotQA distractor dev, bridge (N=1,000 of stratified 2,000), Joint@2	26.2% (1-shot) / 33.7% (2-step)	32.9%	47.8%	Mined edges carry real signal (+8.5 over random at 24.4%), but capture only about ⅓ of the oracle headroom. Edge recall is the bottleneck.
HotpotQA distractor dev, bridge, Recall@2	58.7%	60.7%	69.5%	About 40% of gold bridge paragraphs never reach the top 2. Query-side recall is the second bottleneck.
HotpotQA distractor dev, comparison (N=1,000), Joint@2	59.1%	52.0%	81.8%	Graph bonus hurts comparison questions (−7.1). Routing by question type matters.
BEIR SciFact test (N=300, 5,183 docs), nDCG@10	0.6451	0.6413	0.6449	Statistical tie. Mined vs random: Δ = −0.0002, 95% CI [−0.0061, +0.0059]. Even oracle edges add nothing under the current top-1-seed reranker. Precision gains must come from a stronger reranking stage, not edges.
VitaminC test (first 1,000 pairs), 3-way accuracy	Raw DeBERTa NLI 51.7%	NLI-grounded sheaf (C2) 49.2%	n/a	C2 loses to raw NLI. The pure sheaf (C1) scores 39.1%, below the 49.5% majority class, with NEI F1 = 0.0. Geometry cannot substitute for claim-level semantics.
Replacement Map
Retired Frontier Item	Why It Is Retired	Replaced By
#11 Complex-Analytic Embeddings	No benchmark where only \mathbb{C}^D wins; re-derives every existing module in complex space	R5 Contrastive Claim-Pair Fine-Tuning (richer representation, trainable and measurable)
#13 H^3 Temporal Paradox Detection	Temporal paradoxes are cycle detection in a time-annotated digraph; H^1 + timestamps suffice	R4 GraphRAG Community Summaries (multi-document consistency that is actually runnable)
#14 Quantum Error Correction	Embedding corruption is classical noise; stabilizer codes answer a quantum measurement problem	R2 Late-Interaction / Cross-Encoder Rerank (precision and robustness through a proven mechanism)
#15 Homotopy Type Theory	Natural-language equivalence has no decision procedure	R1 Learned Edge Scorer (formal-ish guarantees replaced by a supervised signal you can ablate)
n/a (bonus)	n/a	R3 HyDE + Query Decomposition (fixes bridge recall directly)
Tier R - Evidence-Driven Replacements (Published Baselines, Measured Headroom)
R1. 🎯 Learned Edge Scorer for the Mined Wormhole Graph
Problem it solves: build_edges() ranks candidate edges with fixed heuristics: title mention at permeability 0.85, and a shared-entity score 0.50 + 0.10\cdot\sum \text{IDF} capped at 0.92, then top-max_out. On HotpotQA bridge questions this gives 32.9% Joint@2 against an oracle ceiling of 47.8%. The graph machinery works (oracle proves it), but the miner picks the wrong edges or misses the right ones.
Theory: Treat edge selection as supervised link prediction over pairs the miner already proposes. For a candidate pair (i, j), build a feature vector from signals already computed in build_edges.py, plus cheap geometric ones:
\phi(i,j) = \big[\,\mathbb{1}_{\text{title}_j \subset \text{text}_i},\ \textstyle\sum_{e \in E_i \cap E_j}\text{IDF}(e),\ |E_i \cap E_j|,\ \cos(\mathbf{d}_i,\mathbf{d}_j),\ \text{pos}_i(\text{title}_j),\ \deg^{\text{in}}(j)\,\big]
where \text{pos}_i(\text{title}_j) is the normalized sentence index of the first mention (the discourse-distance feature; early mentions are more often the bridge entity). Fit a calibrated scorer

p_\theta(e_{ij}=1 \mid \phi) = \sigma\!\left(\mathbf{w}^\top \phi(i,j) + b\right)
(or a gradient-boosted tree) with binary cross-entropy against gold supporting-paragraph pairs:

\mathcal{L}(\theta) = -\sum_{(i,j)} \Big[y_{ij}\log p_\theta + (1-y_{ij})\log(1-p_\theta)\Big]
Set permeability := p_\theta(e_{ij}=1) and keep top-max_out by p_\theta. An optional query-conditioned variant \phi(q,i,j) adds \cos(\mathbf{q},\mathbf{d}_j) and turns the scorer into a second-hop reranker in the style of path retrievers.

Leak-freedom rule (non-negotiable): Labels come only from the HotpotQA / 2WikiMultiHopQA train splits. The scorer is frozen before touching validation. Features stay corpus-only, preserving the existing guarantee in build_edges.py. Report scorer-selected edges alongside the unchanged random and oracle arms.
Impact on LTHCM: Adds method="learned" to lthcm/graph/build_edges.py. Permeabilities become calibrated probabilities that flow into wormhole.py (get_bridges(min_permeability=...) becomes a meaningful probability threshold) and spectral_wormhole.py. compute_edge_statistics(gold_pairs=...) already reports gold-pair coverage, so edge recall becomes a first-class tracked metric. Primary success criterion: close part of the 32.9% → 47.8% bridge gap while mined still beats random under the existing bootstrap.
Implementation complexity: 🟢 Low-Medium. Logistic regression or LightGBM over about 6 features. Training pairs come from the 90,447-question HotpotQA train split. No GPU required.
Reference: Yang et al., HotpotQA (EMNLP 2018); Ho et al., Constructing a Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps (2WikiMultiHopQA, COLING 2020); Asai et al., Learning to Retrieve Reasoning Paths over Wikipedia Graph for Question Answering (ICLR 2020); Xiong et al., Answering Complex Open-Domain Questions with Multi-Hop Dense Retrieval (ICLR 2021); Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017) for the GNN upgrade path.
R2. 🔎 Late-Interaction / Cross-Encoder Rerank Stage
Problem it solves: On SciFact every LTHCM arm ties flat dense (0.6413–0.6449 vs 0.6451 nDCG@10). The current graph reranker only adds a bonus from the top-1 seed, so even perfect oracle edges cannot move precision. Reviewers will also ask the obvious question: does the geometric stack add anything beyond a strong standard reranker? Today there is no ablation that answers it.
Theory: Insert a reranking stage between first-stage retrieval (FAISS / BM25) and the geometric stack. Two proven options:
Late interaction (ColBERT / ColBERTv2): encode query and document into token matrices E_q \in \mathbb{R}^{|q|\times d} and E_d \in \mathbb{R}^{|d|\times d}, then score by summed maximum similarity:
s(q,d) = \sum_{i=1}^{|q|} \max_{j=1}^{|d|} \; E_{q_i} \cdot E_{d_j}^{\top}
ColBERTv2 adds residual compression and denoised supervision, so token-level indexes stay tractable.

Cross-encoder (monoBERT / bge-reranker): jointly encode the pair and read out a relevance logit:
s(q,d) = \mathbf{w}^\top \,\text{BERT}\big([\text{CLS}]\, q\, [\text{SEP}]\, d\big)_{[\text{CLS}]}
Rerank the top-k (e.g. 100) FAISS candidates.

The rerank score then becomes the base score the wormhole bonus modifies, replacing raw cosine.

The ablation this enables (the real value): a 2×2 grid on SciFact and HotpotQA
no rerank	rerank
flat	measured	new
+ LTHCM (mined edges)	measured	new
If LTHCM + rerank beats rerank alone with a CI excluding zero, the geometry earns its place. If not, the paper should say so.

Impact on LTHCM: New optional stage in lthcm/engine/pipeline.py ahead of the wormhole and Fisher-Rao stages. scifact_eval.py and hotpot_real_eval.py gain flat+rerank and lthcm+rerank arms reusing paired_bootstrap_test. The fisher_rao.py blend (currently 0.65\cdot\cos + 0.35\cdot s_{FR}) should be re-tuned on top of rerank scores, on a dev split, never the test split.
Implementation complexity: 🟢 Low for a cross-encoder (sentence-transformers CrossEncoder, already a dependency via nli-deberta-v3-xsmall). 🟡 Medium for ColBERTv2 (separate index; GPU recommended for corpora beyond SciFact size).
Reference: Khattab & Zaharia, ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT (SIGIR 2020); Santhanam et al., ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction (NAACL 2022); Nogueira & Cho, Passage Re-ranking with BERT (2019); Thakur et al., BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models (NeurIPS 2021 Datasets & Benchmarks); Xiao et al., C-Pack: Packaged Resources To Advance General Chinese Embedding (2023, BGE models and rerankers); Formal et al., SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking (SIGIR 2021) as a stronger first stage.
R3. 🧩 HyDE + Query Decomposition for Multi-Hop Recall
Problem it solves: Bridge Recall@2 is 60.7% for mined LTHCM (58.7% flat). Roughly 40% of the gold bridge paragraphs never reach the top 2. The second hop is often lexically invisible from the question ("the director of the film that…"), so no edge bonus can help a paragraph that was never scored highly enough to be a seed or a bridge target. Note also that the harness's naive 2-step iterative baseline (append top-1 text to the query) collapses on comparison questions (34.0% vs 59.1% 1-shot), so query rewriting must be type-aware.
Theory:
HyDE (Hypothetical Document Embeddings): prompt a generator to write N hypothetical answer passages \hat{h}_1..\hat{h}_N for q, embed them with the same encoder f, and retrieve with the averaged vector:
\mathbf{v}_{\text{HyDE}} = \frac{1}{N+1}\Big(f(q) + \sum_{k=1}^{N} f(\hat{h}_k)\Big)
The hypothetical passage may be factually wrong. It only needs to land in the right neighbourhood of document space.

Question decomposition: split a bridge question into sub-questions q \rightarrow (q_1, q_2[\text{ans}_1]), retrieve for q_1, fill the bridge entity into q_2, retrieve again. The final score fuses both hops, e.g. reciprocal-rank fusion
\text{RRF}(d) = \sum_{h \in \{1,2\}} \frac{1}{k + \text{rank}_h(d)}, \quad k = 60
Type routing: apply decomposition only to bridge-type questions (detected by a light classifier or the generator itself). Comparison questions keep single-shot retrieval over both entities.
Impact on LTHCM: New query-side module feeding pipeline.py before stochastic_resonance.py (SR's noisy probes and HyDE's hypothetical probes are both query-ensemble methods, so evaluate them jointly, not stacked blindly). The decomposition's hop-1 top result is a better seed for wormhole.py bridges, so R1 and R3 compound. Success criterion: raise bridge Recall@2 above 60.7% and Joint@2 above the 33.7% 2-step baseline without regressing comparison below 59.1%.
Implementation complexity: 🟡 Medium. Needs a generator (a small local instruction model works offline; an API model is faster to prototype). Generation cost dominates, so cache generations per question ID for reproducibility.
Reference: Gao, Ma, Lin & Callan, Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE, ACL 2023); Min et al., Multi-hop Reading Comprehension through Question Decomposition and Rescoring (DecompRC, ACL 2019); Press et al., Measuring and Narrowing the Compositionality Gap in Language Models (Self-Ask, Findings of EMNLP 2023); Trivedi et al., Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions (IRCoT, ACL 2023); Cormack, Clarke & Büttcher, Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods (SIGIR 2009).
R4. 🏘️ GraphRAG Community Summaries vs Sheaf Energy for Multi-Evidence Consistency
Problem it solves: Lever 3 wants multi-document consistency: does a set of retrieved passages agree? The sheaf is LTHCM's answer, but on real data the NLI-grounded sheaf (C2) currently reduces to NLI contradiction probabilities (sheaf.py overwrites Dirichlet energies with 10 \times \bar{p}_{\text{contra}}), and the pure geometric sheaf is below majority class. GraphRAG is the practical, widely adopted competitor for multi-document synthesis, and any reviewer will ask how the sheaf compares.
Theory:
Build an entity/claim graph over the corpus (the existing wormhole graph from build_edges() is a valid starting point).
Partition it with Leiden community detection, maximizing modularity
Q = \frac{1}{2m}\sum_{i,j}\Big[A_{ij} - \frac{k_i k_j}{2m}\Big]\delta(c_i, c_j)
with Leiden's refinement step guaranteeing well-connected communities, applied hierarchically.

Generate a summary node per community and use those summaries as evidence units for claims that need several pieces of evidence.
Head-to-head experiment: for each multi-evidence claim, compare (a) sheaf verdict from Dirichlet energy E(\mathbf{x}) = \sum_{(u,v)} \|\mathcal{F}_{u \trianglelefteq e}\mathbf{x}_u - \mathcal{F}_{v \trianglelefteq e}\mathbf{x}_v\|^2 over retrieved passages, (b) NLI over community summaries, and (c) NLI over concatenated raw evidence. Report label accuracy and macro-F1 with paired bootstrap.
Datasets: full FEVER dev (multi-sentence evidence subset) and HoVer, which is built for 2–4-hop claim verification and is the natural stress test for set-level consistency.
Impact on LTHCM: Adds a community layer over wormhole.py's graph and an alternative evidence builder in pipeline.py. It also forces sheaf.py to be evaluated honestly: report the pure sheaf (learned restriction maps, no NLI overwrite) separately from any NLI-grounded variant, so the geometry's own contribution is visible. Either outcome is publishable: sheaf wins means a real mechanism; GraphRAG wins means a clear, citable negative result.
Implementation complexity: 🟡 Medium-High. Leiden via igraph/leidenalg. Summaries need a generator (cacheable). HoVer/FEVER loaders must follow the new no-silent-fallback rule (allow_synthetic_fallback=False).
Reference: Edge et al., From Local to Global: A Graph RAG Approach to Query-Focused Summarization (Microsoft Research, 2024); Traag, Waltman & van Eck, From Louvain to Leiden: guaranteeing well-connected communities (Scientific Reports, 2019); Sarthi et al., RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval (ICLR 2024); Jiang et al., HoVer: A Dataset for Many-Hop Fact Extraction and Claim Verification (Findings of EMNLP 2020); Thorne et al., FEVER: a Large-scale Dataset for Fact Extraction and VERification (NAACL 2018); Hansen & Ghrist, Toward a Spectral Theory of Cellular Sheaves (J. Applied and Computational Topology, 2019).
R5. 🧲 Contrastive Claim-Pair Fine-Tuning (Sheaf as Verifier, Not NLI Replacement)
Problem it solves: On VitaminC (test, first 1,000 pairs) the NLI-grounded sheaf C2 scores 49.2% 3-way vs 51.7% for raw DeBERTa NLI. The pure untrained sheaf C1 scores 39.1%, below the 49.5% majority class, and never predicts NEI (F1 = 0.0). Off-the-shelf MiniLM embeddings place "X rose 5%" and "X fell 5%" almost on top of each other, so no restriction map over those stalks can separate support from refute. The information has to be put into the embedding first.
Theory: Fine-tune the encoder f_\theta on (claim, supporting evidence, refuting evidence) triplets with an InfoNCE objective that uses the contrastive refutation as a hard negative, as in supervised SimCSE with NLI contradictions:
\mathcal{L} = -\log \frac{e^{\,\text{sim}(c_i, e_i^{+})/\tau}}{\sum_{j=1}^{B}\Big(e^{\,\text{sim}(c_i, e_j^{+})/\tau} + e^{\,\text{sim}(c_i, e_j^{-})/\tau}\Big)}
VitaminC's contrastive construction (minimal factual edits that flip the label) supplies exactly these pairs. Then train the sheaf's restriction maps \mathcal{F} in CellularSheaf on top of the frozen, fine-tuned stalks with a margin loss on Dirichlet energy:

\mathcal{L}_{\text{sheaf}} = \max\big(0,\; m + E(\text{support}) - E(\text{refute})\big)
so the sheaf becomes a calibrated verifier over semantically meaningful stalks rather than a relabelled NLI score.

Evaluation protocol: train on VitaminC train only; evaluate on VitaminC test and FEVER dev. Arms: raw NLI (B), fine-tuned encoder + cosine threshold, fine-tuned encoder + trained sheaf (no NLI), and fine-tuned sheaf + NLI. The headline significance test must be against raw NLI (B), not the symmetric rule B2 (30.2%), which is currently the only comparison in vitaminc_eval.py.
Impact on LTHCM: Upgrades the stalk space for sheaf.py and, as a side effect, every embedding-consuming module (fisher_rao.py, wormhole.py, titans_memory.py). Gives the paper an honest claim, "sheaf verification over contrastively trained stalks", instead of one reviewers can reduce to "it's NLI". Removes the need for the sheaf.py NLI energy overwrite in the pure-geometry arm.
Implementation complexity: 🟡 Medium. sentence-transformers MultipleNegativesRankingLoss with hard negatives covers the encoder step. Restriction-map training is a few dozen lines of PyTorch. One GPU-hour for MiniLM-scale models.
Reference: Gao, Yao & Chen, SimCSE: Simple Contrastive Learning of Sentence Embeddings (EMNLP 2021); Reimers & Gurevych, Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks (EMNLP 2019); Henderson et al., Efficient Natural Language Response Suggestion for Smart Reply (2017, multiple-negatives ranking loss); Schuster, Fisch & Barzilay, Get Your Vitamin C! Robust Fact Verification with Contrastive Evidence (NAACL 2021); Oord, Li & Vinyals, Representation Learning with Contrastive Predictive Coding (2018, InfoNCE); Bodnar et al., Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs (NeurIPS 2022) for learned restriction maps.
Summary: Priority Matrix
#	Addition	Replaces	Addresses	Measured Gap It Targets	Difficulty	Impact	Status
R1	Learned Edge Scorer	#15	Edge recall / wormhole quality	Bridge Joint@2 32.9% → oracle 47.8%	🟢 Low-Med	⭐⭐⭐⭐⭐	⏳ Next Up
R2	ColBERTv2 / Cross-Encoder Rerank	#14	Precision + geometry ablation	SciFact tie (0.6413 vs 0.6451), oracle no headroom	🟢 Low	⭐⭐⭐⭐	⏳ Planned
R3	HyDE + Query Decomposition	bonus	Query-side multi-hop recall	Bridge R@2 60.7%; 2-step collapses on comparison	🟡 Med	⭐⭐⭐⭐	⏳ Planned
R4	GraphRAG Communities vs Sheaf	#13	Multi-evidence consistency (lever 3)	C2 = NLI in disguise; no set-level baseline	🟠 Med-High	⭐⭐⭐⭐	⏳ Planned
R5	Contrastive Claim-Pair Fine-Tuning	#11	Claim-level semantics for sheaf stalks	C2 49.2% < NLI 51.7%; C1 39.1% < majority	🟡 Med	⭐⭐⭐⭐⭐	⏳ Planned
Recommended Implementation Order
Phase R-A (Close the measured multi-hop gap):
  ├── R1  Learned Edge Scorer                ← train-split labels only; target bridge Joint@2 > 33.7% (2-step)
  └── R2  Cross-Encoder Rerank (+ ablation)  ← 2x2 grid proves or disproves geometry beyond a strong reranker

Phase R-B (Fix recall and semantics at the source):
  ├── R3  HyDE + Decomposition (type-routed) ← compounds with R1: better hop-1 seed → better bridges
  └── R5  Contrastive Fine-Tuning            ← gives sheaf.py stalks that actually encode support vs refute

Phase R-C (Head-to-head paper experiment):
  └── R4  GraphRAG Communities vs Sheaf      ← full FEVER + HoVer; publishable either way it lands
Gate for every item: real dataset, allow_synthetic_fallback=False, the unchanged random and oracle arms, a paired bootstrap against the strongest baseline (not the weakest), and an exact run-conditions line (dataset, split, N, seed, HEAD) next to every number.

Key Architectural Insight
The unifying theme of the replacement track: every addition earns its place on a real benchmark against the strongest available baseline. The geometric core (wormholes, sheaves, Fisher-Rao) stays, but it now sits on top of learned edges, strong reranking, recall-oriented queries and semantically trained stalks, so any gain it shows is a gain beyond the standard toolkit and not a stand-in for it.

The oracle numbers already prove the headroom exists: 47.8% vs 32.9% on bridge, 81.8% vs 52.0% on comparison. The replacement track is about capturing that headroom with mechanisms a reviewer cannot dismiss as metaphor.