"""
Synthetic Scientific IR Demo (SciFact-Style Smoke Test)
======================================================
NOTE: This is a fast synthetic demonstration fixture (35 passages, 8 queries)
illustrating Multi-Facet POVM, Stochastic Resonance, and Fisher-Rao reranking.
For full-scale community evaluation on real BeIR/scifact (5,183 docs, 300 queries),
see lthcm/benchmarks/real/ir/scifact_eval.py.
"""

import time
import math
from typing import List, Dict, Any, Tuple
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from lthcm.core.superposition import SuperpositionMemoryStore
from lthcm.core.fisher_rao import FisherRaoMetric
from lthcm.core.stochastic_resonance import StochasticResonanceQueryLayer
from lthcm.core.wormhole import WormholeGraphIndex
from lthcm.core.wormhole_builder import build_entity_wormholes


class BM25Baseline:
    """Standard Okapi BM25 implementation for lexical retrieval baseline."""

    def __init__(self, corpus: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_ids = [d["id"] for d in corpus]
        self.doc_len = [len(d["text"].lower().split()) for d in corpus]
        self.avg_doc_len = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 1.0

        # Term frequencies & document frequencies
        self.doc_freqs: Dict[str, int] = {}
        self.doc_tfs: List[Dict[str, int]] = []
        for d in corpus:
            tokens = d["text"].lower().split()
            tf_map: Dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1
            self.doc_tfs.append(tf_map)
            for t in tf_map:
                self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1

        self.N = len(corpus)

    def score(self, query: str) -> List[Tuple[str, float]]:
        q_tokens = query.lower().split()
        scores = []
        for i, doc_id in enumerate(self.doc_ids):
            score = 0.0
            doc_len = self.doc_len[i]
            tf_map = self.doc_tfs[i]
            for t in q_tokens:
                if t in tf_map:
                    tf = tf_map[t]
                    df = self.doc_freqs.get(t, 0)
                    idf = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))
                    num = tf * (self.k1 + 1.0)
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                    score += idf * (num / denom)
            scores.append((doc_id, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


class SyntheticSciFactDemo:
    """
    Synthetic Scientific IR demonstration fixture.
    Compares:
      1. Lexical BM25
      2. Flat Dense Bi-Encoder Cosine
      3. LTHCM Topological Memory (Multi-Facet POVM + Stochastic Resonance + Fisher-Rao)
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder = SentenceTransformer(model_name)
        self.d_model = self.encoder.get_sentence_embedding_dimension()

    def generate_scifact_split(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
        """
        Creates a benchmark corpus of 35 scientific passages and 8 queries with graded qrels.
        Relevance levels: 2 = Highly Relevant, 1 = Partially Relevant, 0 = Irrelevant.
        """
        corpus = [
            # Biomarkers & Oncology
            {"id": "doc_01", "text": "Circulating tumor DNA ctDNA enables non-invasive detection of minimal residual disease in colorectal cancer patients post-resection."},
            {"id": "doc_02", "text": "EGFR T790M gatekeeper mutation confers secondary acquired resistance to first-generation tyrosine kinase inhibitors in lung adenocarcinoma."},
            {"id": "doc_03", "text": "Osimertinib selectively overcomes EGFR T790M resistance while sparing wild-type epidermal growth factor receptors."},
            {"id": "doc_04", "text": "PD-L1 expression levels on tumor-infiltrating lymphocytes correlate with therapeutic response to immune checkpoint inhibition."},
            {"id": "doc_05", "text": "Mismatch repair deficiency and high microsatellite instability predict durable objective response to pembrolizumab across solid tumors."},
            {"id": "doc_06", "text": "KRAS G12C covalent inhibitors lock the mutant GTPase in an inactive GDP-bound state, arresting downstream MAPK signaling."},
            {"id": "doc_07", "text": "BRCA1 and BRCA2 pathogenic germline mutations induce homologous recombination deficiency, conferring synthetic lethality to PARP inhibitors."},
            {"id": "doc_08", "text": "Targeted degradation of oncogenic proteins via cereblon-directed proteolysis targeting chimeras PROTACs."},
            {"id": "doc_09", "text": "CAR-T cell therapy targeting CD19 achieves durable remissions in relapsed refractory diffuse large B-cell lymphoma."},
            {"id": "doc_10", "text": "Telomerase reverse transcriptase TERT promoter mutations drive continuous replicative immortality in human glioblastoma multiforme."},

            # Quantum Physics & Nanotechnology
            {"id": "doc_11", "text": "Superconducting transmon qubits achieve coherence times exceeding 100 microseconds via planar capacitor geometries."},
            {"id": "doc_12", "text": "Topological quantum error correction codes like surface codes suppress logical error rates below fault-tolerant thresholds."},
            {"id": "doc_13", "text": "Nitrogen-vacancy centers in diamond enable nanoscale magnetic field sensing under ambient temperature and pressure conditions."},
            {"id": "doc_14", "text": "Twisted bilayer graphene at the magic angle 1.1 degrees exhibits flat electronic bands and unconventional superconductivity."},
            {"id": "doc_15", "text": "Silicon photonic integrated circuits route entangled photon pairs with minimal insertion loss and ultra-low crosstalk."},
            {"id": "doc_16", "text": "Majorana zero modes at the interface of topological insulators and s-wave superconductors provide non-Abelian braiding statistics."},
            {"id": "doc_17", "text": "Quantum key distribution protocols based on BB84 guarantee information-theoretic security via photon polarization states."},
            {"id": "doc_18", "text": "Cavity quantum electrodynamics QED couples single Rydberg atoms to high-Q optical resonators with strong coupling constants."},

            # Climate Science & Geophysics
            {"id": "doc_19", "text": "Atlantic Meridional Overturning Circulation AMOC slowing correlates with North Atlantic sea surface temperature cold blobs."},
            {"id": "doc_20", "text": "Arctic sea ice extent decline amplifies polar warming through positive surface albedo feedback mechanisms."},
            {"id": "doc_21", "text": "Stratospheric aerosol injection with sulfur dioxide mimics volcanic cooling effects to temporarily mitigate radiative forcing."},
            {"id": "doc_22", "text": "Paleoclimate ice core isotope analysis confirms atmospheric carbon dioxide concentrations reached 420 ppm."},
            {"id": "doc_23", "text": "Subduction zone megathrust earthquakes generate devastating tsunamis via abrupt vertical seafloor displacement."},
            {"id": "doc_24", "text": "Permafrost degradation releases ancient methane reservoirs, accelerating terrestrial carbon feedback loops."},

            # Computational Biology & Genomics
            {"id": "doc_25", "text": "Transformer neural networks predict three-dimensional protein structure conformations directly from primary amino acid sequences."},
            {"id": "doc_26", "text": "Single-cell RNA sequencing reveals hidden cellular heterogeneity and lineage trajectories during embryonic organogenesis."},
            {"id": "doc_27", "text": "CRISPR-Cas9 base editors introduce precise single-nucleotide transitions without creating double-strand DNA breaks."},
            {"id": "doc_28", "text": "Cryo-electron microscopy resolves atomic-resolution membrane protein complexes in near-native lipid nanodiscs."},
            {"id": "doc_29", "text": "Spatial transcriptomics maps gene expression topographies onto histological tissue sections with cellular resolution."},
            {"id": "doc_30", "text": "Deep learning models predict RNA splicing alterations caused by non-coding pathogenic genetic variants."},

            # Distractor pass-throughs
            {"id": "doc_31", "text": "Modern microservices architectures utilize distributed consensus algorithms for synchronized state machine replication."},
            {"id": "doc_32", "text": "Automated driving systems fuse lidar, radar, and camera sensor streams using extended Kalman filters."},
            {"id": "doc_33", "text": "High-throughput vertical farming systems optimize red-blue LED wavelength ratios to maximize photosynthesis efficiency."},
            {"id": "doc_34", "text": "Deep reinforcement learning agents master complex multi-agent games through self-play and value iteration."},
            {"id": "doc_35", "text": "Solid-state lithium batteries replace flammable liquid electrolytes with ceramic sulfides to increase energy density."}
        ]

        queries = [
            {"id": "q_01", "text": "How do targeted therapies circumvent acquired T790M resistance in non-small cell lung carcinoma?"},
            {"id": "q_02", "text": "What physical platforms provide non-Abelian statistics for topological quantum computation?"},
            {"id": "q_03", "text": "What role does circulating tumor DNA play in assessing minimal residual disease in oncology?"},
            {"id": "q_04", "text": "What feedback mechanisms accelerate warming due to Arctic sea ice loss?"},
            {"id": "q_05", "text": "How do deep neural networks predict 3D biomolecular protein structures from sequences?"},
            {"id": "q_06", "text": "What causes acquired resistance in EGFR-mutant lung cancers treated with early generation TKIs?"},
            {"id": "q_07", "text": "What are the advantages of CRISPR base editing compared to canonical Cas9 nuclease?"},
            {"id": "q_08", "text": "How do twisted bilayer graphene structures produce superconductivity at the magic angle?"}
        ]

        # Graded relevance judgments (qrels)
        qrels = {
            "q_01": {"doc_03": 2, "doc_02": 1},
            "q_02": {"doc_16": 2, "doc_12": 1},
            "q_03": {"doc_01": 2, "doc_04": 1},
            "q_04": {"doc_20": 2, "doc_24": 1},
            "q_05": {"doc_25": 2, "doc_28": 1},
            "q_06": {"doc_02": 2, "doc_03": 1},
            "q_07": {"doc_27": 2, "doc_30": 1},
            "q_08": {"doc_14": 2, "doc_11": 1}
        }

        return corpus, queries, qrels

    def compute_dcg(self, relevances: List[int], k: int) -> float:
        dcg = 0.0
        for i in range(min(k, len(relevances))):
            rel = relevances[i]
            dcg += (2.0 ** rel - 1.0) / math.log2(i + 2.0)
        return dcg

    def compute_ndcg(self, ranked_ids: List[str], ground_truth: Dict[str, int], k: int = 10) -> float:
        actual_rels = [ground_truth.get(did, 0) for did in ranked_ids]
        dcg = self.compute_dcg(actual_rels, k)

        ideal_rels = sorted(list(ground_truth.values()), reverse=True)
        idcg = self.compute_dcg(ideal_rels, k)

        return (dcg / idcg) if idcg > 0.0 else 0.0

    def compute_recall(self, ranked_ids: List[str], ground_truth: Dict[str, int], k: int = 10) -> float:
        relevant_ids = {did for did, rel in ground_truth.items() if rel > 0}
        if not relevant_ids:
            return 1.0
        retrieved_relevant = set(ranked_ids[:k]) & relevant_ids
        return len(retrieved_relevant) / len(relevant_ids)

    def compute_map(self, ranked_ids: List[str], ground_truth: Dict[str, int], k: int = 100) -> float:
        relevant_ids = {did for did, rel in ground_truth.items() if rel > 0}
        if not relevant_ids:
            return 1.0
        hits = 0
        sum_prec = 0.0
        for i, did in enumerate(ranked_ids[:k]):
            if did in relevant_ids:
                hits += 1
                sum_prec += hits / (i + 1)
        return sum_prec / len(relevant_ids)

    def compute_mrr(self, ranked_ids: List[str], ground_truth: Dict[str, int], k: int = 10) -> float:
        relevant_ids = {did for did, rel in ground_truth.items() if rel > 0}
        if not relevant_ids:
            return 1.0
        for i, did in enumerate(ranked_ids[:k]):
            if did in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0

    def compute_metrics(self, ranked_ids: List[str], ground_truth: Dict[str, int]) -> Dict[str, float]:
        return {
            "ndcg@10": self.compute_ndcg(ranked_ids, ground_truth, k=10),
            "recall@10": self.compute_recall(ranked_ids, ground_truth, k=10),
            "recall@100": self.compute_recall(ranked_ids, ground_truth, k=100),
            "map@100": self.compute_map(ranked_ids, ground_truth, k=100),
            "mrr@10": self.compute_mrr(ranked_ids, ground_truth, k=10),
        }

    @torch.no_grad()
    def run_benchmark(self) -> Dict[str, Any]:
        """
        Executes synthetic SciFact-style evaluation comparing:
          1. Lexical BM25
          2. Dense Bi-Encoder Cosine
          3. LTHCM Topological Memory
        """
        corpus, queries, qrels = self.generate_scifact_split()

        # Build BM25
        bm25 = BM25Baseline(corpus)

        # Encode corpus
        corpus_texts = [d["text"] for d in corpus]
        corpus_embs = self.encoder.encode(corpus_texts, convert_to_tensor=True, normalize_embeddings=True).clone().to(self.device)

        # Build LTHCM Store with Multi-Facet POVM & Wormhole Relational Graph
        lthcm_store = SuperpositionMemoryStore(d_model=self.d_model, num_facets=4).to(self.device)
        fisher_rao = FisherRaoMetric(d_model=self.d_model).to(self.device)
        stochastic_res = StochasticResonanceQueryLayer(d_model=self.d_model).to(self.device)
        wormhole_index = WormholeGraphIndex(d_model=self.d_model)

        for i, item in enumerate(corpus):
            base_vec = corpus_embs[i]
            # Multi-facet POVM construction
            f1 = base_vec
            f2 = F.normalize(base_vec + 0.12 * torch.roll(base_vec, shifts=self.d_model // 4), p=2, dim=0)
            f3 = F.normalize(base_vec + 0.12 * torch.roll(base_vec, shifts=self.d_model // 2), p=2, dim=0)
            f4 = F.normalize(base_vec + 0.12 * torch.roll(base_vec, shifts=3 * self.d_model // 4), p=2, dim=0)
            facets = torch.stack([f1, f2, f3, f4])
            lthcm_store.add_document(doc_id=item["id"], content=item["text"], facet_vectors=facets)
            wormhole_index.add_node(item["id"], base_vec.cpu(), item["text"])

        # Wire relational bridges across documents using automatic entity extraction
        # uniformly across all corpus documents with zero gold label supervision.
        build_entity_wormholes(wormhole_index, corpus, permeability=0.85)

        # Evaluate systems across all queries
        bm25_metrics_list = []
        dense_metrics_list = []
        lthcm_metrics_list = []

        start_time = time.time()
        for q in queries:
            qid = q["id"]
            q_text = q["text"]
            gt = qrels[qid]

            # 1. BM25
            bm25_scores = bm25.score(q_text)
            ranked_bm25 = [did for did, s in bm25_scores]
            bm25_metrics_list.append(self.compute_metrics(ranked_bm25, gt))

            # 2. Dense Bi-Encoder
            q_emb = self.encoder.encode([q_text], convert_to_tensor=True, normalize_embeddings=True).clone().to(self.device)[0]

            sims = torch.matmul(corpus_embs, q_emb)
            ranked_dense_indices = torch.argsort(sims, descending=True).cpu().tolist()
            ranked_dense = [corpus[idx]["id"] for idx in ranked_dense_indices]
            dense_metrics_list.append(self.compute_metrics(ranked_dense, gt))

            # 3. LTHCM Topological Memory
            enhanced_q = stochastic_res(q_emb.unsqueeze(0), noise_scale=0.04, num_samples=5)[0]

            # Retrieve through Multi-Facet POVM
            povm_results = lthcm_store.query(enhanced_q, top_k=15)

            # Rerank through Fisher-Rao manifold metric
            reranked_pool = []
            for item in povm_results:
                did = item["doc_id"]
                doc_idx = next(i for i, d in enumerate(corpus) if d["id"] == did)
                doc_vec = corpus_embs[doc_idx]
                fr_dist = fisher_rao(enhanced_q.unsqueeze(0), doc_vec.unsqueeze(0)).item()
                fr_sim = 1.0 / (1.0 + fr_dist)
                combined_score = 0.55 * item["score"] + 0.45 * fr_sim
                reranked_pool.append((did, combined_score))

            reranked_pool.sort(key=lambda x: x[1], reverse=True)
            top_seed_did = reranked_pool[0][0]

            # Query Wormhole Graph for topological bridges
            bridges = wormhole_index.get_bridges(top_seed_did, min_permeability=0.6)
            bridge_docs = [b["target"] for b in bridges]

            final_ranked_ids = []
            # Place top seed
            final_ranked_ids.append(top_seed_did)

            # Inject top relational bridges if relevant to query
            for b_id in bridge_docs:
                if b_id not in final_ranked_ids:
                    b_idx = next(i for i, d in enumerate(corpus) if d["id"] == b_id)
                    b_emb = corpus_embs[b_idx]
                    b_sim = torch.dot(enhanced_q, b_emb).item()
                    if b_sim > 0.45:
                        final_ranked_ids.append(b_id)

            # Append remainder of reranked pool
            for did, _ in reranked_pool:
                if did not in final_ranked_ids:
                    final_ranked_ids.append(did)

            # Append remainder of corpus
            for d in corpus:
                if d["id"] not in final_ranked_ids:
                    final_ranked_ids.append(d["id"])

            lthcm_metrics_list.append(self.compute_metrics(final_ranked_ids, gt))

        elapsed_ms = (time.time() - start_time) * 1000.0

        def average_metrics(m_list: List[Dict[str, float]]) -> Dict[str, float]:
            keys = m_list[0].keys()
            return {k: round(sum(m[k] for m in m_list) / len(m_list), 4) for k in keys}

        avg_bm25 = average_metrics(bm25_metrics_list)
        avg_dense = average_metrics(dense_metrics_list)
        avg_lthcm = average_metrics(lthcm_metrics_list)

        results = {
            "num_passages": len(corpus),
            "corpus_size": len(corpus),
            "num_queries": len(queries),
            "elapsed_ms": round(elapsed_ms, 2),
            "bm25": avg_bm25,
            "dense": avg_dense,
            "lthcm": avg_lthcm,
            "bm25_baseline": avg_bm25,
            "dense_biencoder_baseline": avg_dense,
            "lthcm_topological": avg_lthcm,
            "gain_over_dense": {
                "ndcg@10_delta": round(avg_lthcm["ndcg@10"] - avg_dense["ndcg@10"], 4),
                "recall@10_delta": round(avg_lthcm["recall@10"] - avg_dense["recall@10"], 4),
                "mrr@10_delta": round(avg_lthcm["mrr@10"] - avg_dense["mrr@10"], 4),
            },
        }

        return results


# Backward compatibility alias
BEIRBenchmark = SyntheticSciFactDemo
