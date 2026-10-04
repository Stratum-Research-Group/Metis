"""
Synthetic Multi-Hop Reasoning Demo (Smoke Test)
==============================================
NOTE: This is a fast synthetic demonstration fixture (10 cases, 30 distractors)
illustrating the Bidirectional Re-Wormholing Loop and Spectral Chebyshev Refinement.
For full-scale community evaluation, see lthcm/benchmarks/real/multihop/.
"""

import time
import math
from typing import List, Dict, Any, Tuple
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from lthcm.core.wormhole import WormholeGraphIndex
from lthcm.core.spectral_wormhole import SpectralWormholeRefiner
from lthcm.core.wormhole_builder import build_entity_wormholes


class SyntheticMultiHopDemo:
    """
    Evaluates multi-hop relational retrieval across 10 contrastive 2-hop reasoning cases
    embedded within 30 high-confusion distractors (50 total documents).
    Compares:
      1. Flat Dense Bi-Encoder Cosine Retrieval (Single-shot)
      2. Flat Dense 2-Step Iterative Retrieval (Retrieve top-1, append to query, retrieve top-2)
      3. LTHCM Topological Wormhole Graph Retrieval (Corpus-mined entity bridges)
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder = SentenceTransformer(model_name)
        self.d_model = self.encoder.get_embedding_dimension()

    def generate_multihop_dataset(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Creates 10 multi-hop reasoning pairs requiring bridging across disparate facts,
        plus 30 high-confusion distractors sharing terminology.
        """
        multihop_cases = [
            {
                "case_id": "01",
                "query": "Which prestigious national award did the inventor of the Kyber-Lattice post-quantum encryption protocol win in 2024?",
                "doc_hop1": {
                    "id": "hop1_kyber",
                    "text": "The Kyber-Lattice post-quantum cryptographic protocol was developed in 2021 by computer scientist Dr. Nadia Rostova at the Zurich Quantum Institute.",
                },
                "doc_hop2": {
                    "id": "hop2_nadia_medal",
                    "text": "Dr. Nadia Rostova was awarded the prestigious Turing Gold Medal in 2024 for groundbreaking contributions to post-quantum lattice security.",
                },
            },
            {
                "case_id": "02",
                "query": "What clinical trial endpoint was achieved by the therapeutic target discovered by biotech Lumicor in Phase II?",
                "doc_hop1": {
                    "id": "hop1_lumicor",
                    "text": "Biotech startup Lumicor discovered that nuclear receptor ROR-gamma inverse agonism reverses chronic dermal inflammation in autoimmune models.",
                },
                "doc_hop2": {
                    "id": "hop2_ror_trial",
                    "text": "In the Phase II clinical trial of ROR-gamma inhibitor LC-882, patients achieved a 75% reduction in Psoriasis Area and Severity Index (PASI-75).",
                },
            },
            {
                "case_id": "03",
                "query": "What launch payload capacity does the primary rocket of the space consortium led by aerospace firm Stellaris possess?",
                "doc_hop1": {
                    "id": "hop1_stellaris",
                    "text": "Aerospace manufacturer Stellaris operates as lead contractor for the European heavy-lift rocket architecture known as Aeolus-V.",
                },
                "doc_hop2": {
                    "id": "hop2_aeolus_rocket",
                    "text": "The Aeolus-V two-stage cryogenic rocket delivers a verified payload capacity of 21.5 metric tons to Geostationary Transfer Orbit (GTO).",
                },
            },
            {
                "case_id": "04",
                "query": "Which German optics supplier manufactures the numerical aperture projection mirrors used in TSMC's Fab-21 EUV photolithography scanners?",
                "doc_hop1": {
                    "id": "hop1_fab21",
                    "text": "TSMC Fab-21 in Phoenix Arizona produces 2nm nanosheet transistors utilizing high-NA TwinScan EXE:5200 extreme ultraviolet lithography systems.",
                },
                "doc_hop2": {
                    "id": "hop2_twinscan_mirror",
                    "text": "The TwinScan EXE:5200 lithography scanner incorporates 0.55 numerical aperture anamorphic projection optics manufactured exclusively by Carl Zeiss SMT.",
                },
            },
            {
                "case_id": "05",
                "query": "What mathematical conjecture was proved by the mathematician who parameterized the Weil-Petersson metric on the moduli space of Calabi-Yau threefolds?",
                "doc_hop1": {
                    "id": "hop1_moduli",
                    "text": "Professor Armand Beauchamp established the explicit Kähler potential for the Weil-Petersson metric on the moduli space of Calabi-Yau threefolds in 2018.",
                },
                "doc_hop2": {
                    "id": "hop2_beauchamp_conjecture",
                    "text": "In 2022, Professor Armand Beauchamp settled the Donaldson-Thomas Finiteness Conjecture for non-compact symplectic 6-manifolds.",
                },
            },
            {
                "case_id": "06",
                "query": "What laser wavelength is utilized by the autonomous LiDAR sensor equipped on Centurion Defense vehicle platforms?",
                "doc_hop1": {
                    "id": "hop1_centurion",
                    "text": "Centurion Defense tactical autonomous ground platforms are equipped with solid-state long-range Velodyne-Opal-9 scanning LiDAR units.",
                },
                "doc_hop2": {
                    "id": "hop2_opal_wavelength",
                    "text": "The Velodyne-Opal-9 solid-state LiDAR operates at a 1550nm eye-safe laser wavelength to penetrate heavy atmospheric aerosol obscurants.",
                },
            },
            {
                "case_id": "07",
                "query": "What human genomic chromosome locus contains the engineered guide RNA cleavage site designed by Prime Editing vector PE-Max?",
                "doc_hop1": {
                    "id": "hop1_grna",
                    "text": "The synthetic Prime Editing therapeutic vector PE-Max targets an off-target-free genomic site within the human ZNF-842 regulatory locus.",
                },
                "doc_hop2": {
                    "id": "hop2_znf_locus",
                    "text": "The ZNF-842 zinc-finger transcriptional factor gene resides at chromosomal locus 19q13.43 in human embryonic kidney cells.",
                },
            },
            {
                "case_id": "08",
                "query": "What electrolyte solvent composition was engineered to stabilize the solid electrolyte interphase of PyroVolt lithium-metal batteries?",
                "doc_hop1": {
                    "id": "hop1_pyrovolt",
                    "text": "PyroVolt next-generation 450 Wh/kg lithium-metal pouch cells utilize an ultra-concentrated fluorinated EC-DMC electrolyte formulation.",
                },
                "doc_hop2": {
                    "id": "hop2_blend_salt",
                    "text": "The fluorinated EC-DMC formulation blends 1.2M LiFSI in fluoroethylene carbonate and dimethyl carbonate in a 1:2 volume ratio.",
                },
            },
            {
                "case_id": "09",
                "query": "Which matrix decomposition theorem underpins the compiler optimization in QubitMaster quantum circuit synthesis?",
                "doc_hop1": {
                    "id": "hop1_qubitmaster",
                    "text": "The QubitMaster optimizing quantum compiler applies Shen-Vandersluis unitarization to minimize multiqubit CNOT circuit depth.",
                },
                "doc_hop2": {
                    "id": "hop2_compiler_decomp",
                    "text": "The Shen-Vandersluis unitarization algorithm relies on the Cosine-Sine CS matrix decomposition of unitary 2^n block operators.",
                },
            },
            {
                "case_id": "10",
                "query": "What consensus finality latency was measured in the testnet deployment of the Paxos-DAG distributed ledger protocol?",
                "doc_hop1": {
                    "id": "hop1_dag",
                    "text": "The Paxos-DAG asynchronous consensus ledger combines directed acyclic graph mempools with pipelined leader verification.",
                },
                "doc_hop2": {
                    "id": "hop2_dag_latency",
                    "text": "In globally distributed testnet benchmark runs across 128 validator nodes, Paxos-DAG recorded a median consensus finality latency of 380 milliseconds.",
                },
            },
        ]

        distractors = [
            "Lattice-based cryptography relies on the hardness of Learning With Errors (LWE) and Ring-LWE mathematical problems.",
            "The Turing Award is widely regarded as the highest distinction in computer science and the Nobel Prize of computing.",
            "Janus kinase inhibitors such as tofacitinib modulate inflammatory cytokine signaling through STAT phosphorylation.",
            "The Ariane 6 expendable launch system is designed to provide European autonomous access to low-Earth and geostationary orbits.",
            "Extreme ultraviolet optics operate under high vacuum utilizing ultra-smooth multi-layer molybdenum-silicon mirrors.",
            "Calabi-Yau manifolds are compact Kähler manifolds with vanishing first Chern class and trivial canonical bundle.",
            "Carl Zeiss SMT manufactures high-precision optics for industrial optical fabrication and lithography systems worldwide.",
            "Moduli spaces of algebraic curves and Riemann surfaces have symplectic structures governed by Weil-Petersson Kähler forms.",
            "Near-infrared LiDAR sensors at 1550nm allow higher laser pulse energy while maintaining Class 1 eye safety.",
            "Chromosome 19 contains the highest gene density of any human chromosome, enriched for zinc finger clusters.",
            "Lithium hexafluorophosphate LiPF6 is the standard commercial electrolyte salt despite thermal instability risks.",
            "Quantum circuit synthesis translates abstract Hilbert space operators into hardware-native quantum gate sequences.",
            "Post-quantum cryptographic migration requires hybrid key exchange algorithms combining classical elliptic curves.",
            "The Nobel Prize in Chemistry has recognized multiple cryo-electron microscopy and gene editing breakthroughs.",
            "High-throughput screening assays test hundreds of thousands of compounds against kinase binding targets.",
            "Asynchronous Byzantine fault tolerance maintains transaction liveness during malicious network partitioning.",
            "Falcon 9 and Falcon Heavy utilize reusable first-stage boosters landing autonomously on drone ships at sea.",
            "Extreme ultraviolet lithography operates at 13.5 nanometer wavelength utilizing tin plasma laser sources.",
            "Algebraic geometry investigates the geometric loci of zeros of multivariate polynomial equations.",
            "Time-of-flight depth cameras measure phase shift differences between emitted infrared light pulses and reflections.",
            "Guide RNA off-target cleavage can be mitigated using high-fidelity Cas9 variants with engineered proofreading.",
            "Solid-state lithium batteries replace flammable organic liquid electrolytes with ceramic sulfide solid conductors.",
            "Clifford group quantum gates can be simulated efficiently in polynomial time according to the Gottesman-Knill theorem.",
            "The RSA public-key cryptosystem factors large composite integers using modular exponentiation arithmetic.",
            "Field effect transistors have evolved from planar MOSFETs to FinFETs and nanosheet gate-all-around architectures.",
            "Microservice architectures decouple monolithic software systems into independently deployable API endpoints.",
            "Atmospheric re-entry vehicles experience aerodynamic shock heating requiring carbon-phenolic thermal protection.",
            "Phase-change materials absorb and release thermal energy during solid-liquid latent heat phase transitions.",
            "Markov decision processes formulate reinforcement learning agents maximizing cumulative discounted reward signals.",
            "Non-small cell lung cancer therapies frequently target activating mutations in the EGFR kinase domain.",
        ]

        return multihop_cases, distractors

    @torch.no_grad()
    def run_benchmark(self) -> Dict[str, Any]:
        """
        Executes multi-hop evaluation comparing:
          1. Flat Dense Bi-Encoder Cosine Retrieval (1-shot)
          2. Flat Dense 2-Step Iterative Retrieval (query expansion)
          3. LTHCM Topological Wormhole Graph Retrieval (corpus-mined entity bridges)
        """
        multihop_cases, distractors = self.generate_multihop_dataset()

        # Build corpus of all documents: hop1 docs, hop2 docs, and distractors
        # Strictly without case_id or label metadata on nodes
        all_docs = []
        hop1_ids = set()
        hop2_ids = set()

        for case in multihop_cases:
            h1 = case["doc_hop1"]
            h2 = case["doc_hop2"]
            all_docs.append({"id": h1["id"], "text": h1["text"]})
            all_docs.append({"id": h2["id"], "text": h2["text"]})
            hop1_ids.add(h1["id"])
            hop2_ids.add(h2["id"])

        for i, text in enumerate(distractors):
            all_docs.append({"id": f"distractor_{i:03d}", "text": text})

        # Encode corpus
        corpus_texts = [d["text"] for d in all_docs]
        corpus_embs = self.encoder.encode(corpus_texts, convert_to_tensor=True, normalize_embeddings=True).clone().to(self.device)

        # Build LTHCM Wormhole Graph Index & Spectral Chebyshev Refiner
        wormhole_index = WormholeGraphIndex(d_model=self.d_model)
        spectral_refiner = SpectralWormholeRefiner(d_model=self.d_model, K_order=3).to(self.device)

        for i, doc in enumerate(all_docs):
            wormhole_index.add_node(
                node_id=doc["id"],
                embedding=corpus_embs[i].cpu(),
                content=doc["text"],
            )

        # Wire relational bridges across documents using automatic entity extraction
        # uniformly across all documents (including distractors) with zero gold label supervision.
        build_entity_wormholes(wormhole_index, all_docs, permeability=0.98)

        # Metrics trackers
        flat_hop1_r1_hits = 0
        flat_hop1_r2_hits = 0
        flat_hop2_hits = 0
        flat_both_hits = 0
        flat_ndcg_list = []

        twostep_hop1_r1_hits = 0
        twostep_hop1_r2_hits = 0
        twostep_hop2_hits = 0
        twostep_both_hits = 0
        twostep_ndcg_list = []

        lthcm_hop1_r1_hits = 0
        lthcm_hop1_r2_hits = 0
        lthcm_hop2_hits = 0
        lthcm_both_hits = 0
        lthcm_ndcg_list = []

        idcg = (2.0 / 1.0) + (1.0 / 1.58496)  # ideal ranks 1 and 2: rel / log2(rank+1)
        start_time = time.time()

        for case in multihop_cases:
            query_text = case["query"]
            h1_id = case["doc_hop1"]["id"]
            h2_id = case["doc_hop2"]["id"]

            q_emb = self.encoder.encode(query_text, convert_to_tensor=True, normalize_embeddings=True).clone().to(self.device)

            # --- 1. Flat Dense Single-Shot Retrieval ---
            flat_similarities = torch.matmul(corpus_embs, q_emb)
            ranked_indices = torch.argsort(flat_similarities, descending=True).cpu().tolist()
            top10_flat = [all_docs[idx]["id"] for idx in ranked_indices[:10]]
            top2_flat = top10_flat[:2]

            h1_hit_r1_flat = (top10_flat[0] == h1_id)
            h1_hit_r2_flat = (h1_id in top2_flat)
            h2_hit_flat = (h2_id in top2_flat)
            both_hit_flat = (h1_hit_r2_flat and h2_hit_flat)
            if h1_hit_r1_flat: flat_hop1_r1_hits += 1
            if h1_hit_r2_flat: flat_hop1_r2_hits += 1
            if h2_hit_flat: flat_hop2_hits += 1
            if both_hit_flat: flat_both_hits += 1

            dcg_flat = 0.0
            for r_idx, doc_id in enumerate(top10_flat):
                rank = r_idx + 1
                rel = 2.0 if doc_id == h2_id else (1.0 if doc_id == h1_id else 0.0)
                if rel > 0:
                    dcg_flat += rel / math.log2(rank + 1)
            flat_ndcg_list.append(dcg_flat / idcg)

            # --- 2. Flat Dense 2-Step Iterative Retrieval ---
            step1_top_idx = ranked_indices[0]
            step1_doc = all_docs[step1_top_idx]
            expanded_q_text = f"{query_text} {step1_doc['text']}"
            q_step2_emb = self.encoder.encode(expanded_q_text, convert_to_tensor=True, normalize_embeddings=True).clone().to(self.device)
            sims_step2 = torch.matmul(corpus_embs, q_step2_emb)
            ranked_step2_indices = torch.argsort(sims_step2, descending=True).cpu().tolist()
            # Select best candidate not equal to step 1
            step2_candidates = [all_docs[i]["id"] for i in ranked_step2_indices if i != step1_top_idx]
            top2_twostep = [step1_doc["id"], step2_candidates[0]]
            top10_twostep = top2_twostep + [c for c in step2_candidates[1:9]]

            h1_hit_r1_2step = (top10_twostep[0] == h1_id)
            h1_hit_r2_2step = (h1_id in top2_twostep)
            h2_hit_2step = (h2_id in top2_twostep)
            both_hit_2step = (h1_hit_r2_2step and h2_hit_2step)
            if h1_hit_r1_2step: twostep_hop1_r1_hits += 1
            if h1_hit_r2_2step: twostep_hop1_r2_hits += 1
            if h2_hit_2step: twostep_hop2_hits += 1
            if both_hit_2step: twostep_both_hits += 1

            dcg_2step = 0.0
            for r_idx, doc_id in enumerate(top10_twostep):
                rank = r_idx + 1
                rel = 2.0 if doc_id == h2_id else (1.0 if doc_id == h1_id else 0.0)
                if rel > 0:
                    dcg_2step += rel / math.log2(rank + 1)
            twostep_ndcg_list.append(dcg_2step / idcg)

            # --- 3. LTHCM Topological Wormhole Graph Retrieval ---
            wh_results = wormhole_index.query(q_emb, top_k=10, max_hops=2)
            candidate_nids = [res["node_id"] for res in wh_results]

            if len(candidate_nids) > 1:
                spec_res = spectral_refiner.refine_candidates(
                    query=q_emb,
                    wormhole_index=wormhole_index,
                    candidate_ids=candidate_nids,
                )
                refined_dict = {r["node_id"]: r["refined_score"] for r in spec_res.get("results", [])}
                for r in wh_results:
                    if r["node_id"] in refined_dict:
                        r["score"] = 0.5 * r["score"] + 0.5 * refined_dict[r["node_id"]]
                wh_results.sort(key=lambda x: x["score"], reverse=True)

            active_seed = wh_results[0]["node_id"]
            active_bridges = [n for n in wormhole_index.graph.successors(active_seed) if n != active_seed]

            if len(active_bridges) == 0:
                for alt in wh_results[1:5]:
                    alt_id = alt["node_id"]
                    alt_bridges = [n for n in wormhole_index.graph.successors(alt_id) if n != alt_id]
                    if len(alt_bridges) > 0:
                        active_seed = alt_id
                        active_bridges = alt_bridges
                        break

            if len(active_bridges) > 0:
                wh_dict = {r["node_id"]: r["score"] for r in wh_results}
                active_bridges.sort(key=lambda b: wh_dict.get(b, -1.0), reverse=True)
                top2_lthcm = [active_seed, active_bridges[0]]
            else:
                top2_lthcm = [wh_results[0]["node_id"], wh_results[1]["node_id"]]

            top10_lthcm = top2_lthcm + [r["node_id"] for r in wh_results if r["node_id"] not in top2_lthcm][:8]

            h1_hit_r1_lthcm = (top10_lthcm[0] == h1_id)
            h1_hit_r2_lthcm = (h1_id in top2_lthcm)
            h2_hit_lthcm = (h2_id in top2_lthcm)
            both_hit_lthcm = (h1_hit_r2_lthcm and h2_hit_lthcm)

            if h1_hit_r1_lthcm: lthcm_hop1_r1_hits += 1
            if h1_hit_r2_lthcm: lthcm_hop1_r2_hits += 1
            if h2_hit_lthcm: lthcm_hop2_hits += 1
            if both_hit_lthcm: lthcm_both_hits += 1

            dcg_lthcm = 0.0
            for r_idx, doc_id in enumerate(top10_lthcm):
                rank = r_idx + 1
                rel = 2.0 if doc_id == h2_id else (1.0 if doc_id == h1_id else 0.0)
                if rel > 0:
                    dcg_lthcm += rel / math.log2(rank + 1)
            lthcm_ndcg_list.append(dcg_lthcm / idcg)

        total_cases = len(multihop_cases)
        elapsed_ms = (time.time() - start_time) * 1000.0

        return {
            "num_cases": total_cases,
            "corpus_size": len(all_docs),
            "elapsed_ms": elapsed_ms,
            "flat_dense_baseline": {
                "hop1_recall@1": float(flat_hop1_r1_hits / total_cases),
                "hop1_recall@2": float(flat_hop1_r2_hits / total_cases),
                "hop2_recall@2": float(flat_hop2_hits / total_cases),
                "joint_multihop_recall@2": float(flat_both_hits / total_cases),
                "ndcg@10": float(sum(flat_ndcg_list) / len(flat_ndcg_list)),
            },
            "flat_iterative_2step_baseline": {
                "hop1_recall@1": float(twostep_hop1_r1_hits / total_cases),
                "hop1_recall@2": float(twostep_hop1_r2_hits / total_cases),
                "hop2_recall@2": float(twostep_hop2_hits / total_cases),
                "joint_multihop_recall@2": float(twostep_both_hits / total_cases),
                "ndcg@10": float(sum(twostep_ndcg_list) / len(twostep_ndcg_list)),
            },
            "lthcm_wormhole": {
                "hop1_recall@1": float(lthcm_hop1_r1_hits / total_cases),
                "hop1_recall@2": float(lthcm_hop1_r2_hits / total_cases),
                "hop2_recall@2": float(lthcm_hop2_hits / total_cases),
                "joint_multihop_recall@2": float(lthcm_both_hits / total_cases),
                "ndcg@10": float(sum(lthcm_ndcg_list) / len(lthcm_ndcg_list)),
            },
        }


# Backward-compatible alias
HotpotQABenchmark = SyntheticMultiHopDemo
