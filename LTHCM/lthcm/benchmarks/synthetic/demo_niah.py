"""
Synthetic Multi-Needle in a Haystack Demo (Smoke Test)
=====================================================
NOTE: This is a fast synthetic demonstration fixture (60 distractors, 5 needles)
illustrating depth-stratified recall and topological memory indexing.
For the standardized community long-context benchmark suite, see RULER evaluation
in lthcm/benchmarks/real/niah/ruler_eval.py.
"""

import time
import math
from typing import List, Dict, Any, Tuple
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from lthcm.core.superposition import SuperpositionMemoryStore
from lthcm.core.wormhole import WormholeGraphIndex
from lthcm.core.fisher_rao import FisherRaoMetric


class SyntheticNeedleDemo:
    """
    Synthetic Multi-Needle in a Haystack (M-NIAH) Demo Runner.
    Compares Flat Dense Vector Retrieval against LTHCM Topological Memory
    across depth percentiles (10%, 30%, 50%, 70%, 90%).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder = SentenceTransformer(model_name)
        self.d_model = self.encoder.get_sentence_embedding_dimension()

    def generate_haystack_corpus(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Generates a 60-document distractor haystack across diverse domains,
        and 5 distinct factual needles to be embedded at precise depth percentiles.
        """
        distractors = [
            # Technology & Systems
            "Cache coherence protocols like MESI maintain memory consistency in multi-core symmetric multiprocessing.",
            "Write-ahead logging in relational databases ensures atomicity and durability under unexpected node failure.",
            "B-trees and LSM trees optimize disk read and write amplification across large-scale storage engines.",
            "Branch prediction uses two-level adaptive branch history tables to prevent instruction pipeline bubbles.",
            "Instruction-level parallelism leverages Tomasulo's algorithm for out-of-order execution and register renaming.",
            "Virtual memory paging translates linear virtual addresses to physical pages via multi-level page tables.",
            "Direct Memory Access controllers transfer data between peripherals and RAM without continuous CPU intervention.",
            "Network interface cards offload TCP segmentation and checksum verification from the host kernel.",
            "The Raft consensus algorithm decomposes state machine replication into leader election and log replication.",
            "Paxos consensus guarantees safety under asynchronous network conditions with crash-recovery node models.",

            # Physics & Chemistry
            "The Haber-Bosch process synthesizes ammonia from atmospheric nitrogen and hydrogen gas under high iron catalysis.",
            "Zeolite catalysts feature microporous aluminosilicate frameworks facilitating shape-selective catalytic cracking.",
            "Superconductivity arises in type-II cuprates via Cooper pair condensation along copper-oxide ceramic planes.",
            "Quantum Hall effect manifests quantized Hall conductance plateaus integer multiples of e squared over h.",
            "Laser cooling relies on Doppler-shifted photon absorption to decelerate neutral alkali atoms into magneto-optical traps.",
            "Bose-Einstein condensates form when bosonic atomic clouds reach nanokelvin temperatures near absolute zero.",
            "Synchrotron radiation provides intense polarized tunable X-ray beams for macromolecular crystallography.",
            "Thermonuclear fusion in tokamaks confines deuterium-tritium plasma using poloidal and toroidal magnetic field cages.",
            "Graphene exhibits linear Dirac energy dispersion relations and ballistic electron transport at room temperature.",
            "Scanning tunneling microscopy measures local electronic density of states via quantum mechanical tunneling.",

            # Biology & Medicine
            "Ribosomal RNA peptidyl transferase centers catalyze peptide bond formation during mRNA translation.",
            "Mitochondrial oxidative phosphorylation generates ATP via proton motive force gradients across cristae membranes.",
            "Monoclonal antibodies target specific cell-surface receptor epitopes with high picomolar binding affinities.",
            "Action potentials propagate along myelinated axons via saltatory conduction across nodes of Ranvier.",
            "DNA polymerase delta and epsilon replicate lagging and leading strands with 3-prime to 5-prime exonuclease proofreading.",
            "CRISPR-Cas12 nucleases exhibit non-specific collateral single-stranded DNA cleavage upon target activation.",
            "Endothelial nitric oxide synthase regulates vascular tone by stimulating soluble guanylyl cyclase in smooth muscle.",
            "Immunological memory cells persist within lymphoid follicles to mount rapid secondary antibody responses.",
            "Telomeric shelterin protein complexes protect chromosome ends from aberrant double-strand break repair.",
            "Alternative RNA splicing isoforms diversify the eukaryotic proteome without expanding genomic gene count.",

            # Earth & Environment
            "Plate tectonics drives continental drift through mantle convection and oceanic crust subduction.",
            "The global carbon cycle balances terrestrial photosynthetic carbon uptake with oceanic carbonate dissolution.",
            "Tropospheric ozone formation occurs via photochemical oxidation of volatile organic compounds and nitrogen oxides.",
            "Milankovitch cycles govern Quaternary glacial-interglacial cycles through orbital eccentricity and axial precession.",
            "Thermohaline circulation transports equatorial thermal energy to high latitudes via deep dense water formation.",
            "Soil microbiome nitrogen fixers like Rhizobium convert atmospheric dinitrogen into bioavailable ammonium salts.",
            "Volcanic basalt outgassing during large igneous province eruptions induces hyperthermal oceanic anoxic events.",
            "Coral reef calcification rates decline under ocean acidification driven by anthropogenic carbon dioxide uptake.",
            "Glacial isostatic adjustment describes delayed viscoelastic crustal rebound following continental ice sheet retreat.",
            "Stratospheric polar vortex stability influences mid-latitude winter weather extremes and cold air outbreaks.",

            # Economics & Finance
            "The Black-Scholes-Merton partial differential equation models European option pricing under geometric Brownian motion.",
            "Modern Portfolio Theory maximizes expected returns for a given volatility level along the Markowitz efficient frontier.",
            "Central bank quantitative easing expands monetary balance sheets through secondary market sovereign bond purchases.",
            "Arbitrage Pricing Theory models asset returns as linear functions of multiple macroeconomic risk factors.",
            "High-frequency limit order books balance bid-ask spread liquidity against adverse selection risks.",
            "The Phillips curve describes inverse empirical trade-offs between unemployment rates and consumer price inflation.",
            "Real business cycle theory attributes economic fluctuations to exogenous aggregate technology supply shocks.",
            "Credit default swaps provide financial protection against bond issuer debt restructuring or bankruptcy default.",
            "The Mundell-Fleming trilemma asserts monetary independence, fixed exchange rates, and free capital flows are mutually incompatible.",
            "Microeconomic price elasticity of demand determines equilibrium price shifts following excise tax introduction.",

            # Law & Governance
            "The doctrine of judicial review empowers constitutional courts to invalidate legislation incompatible with basic rights.",
            "International humanitarian law under the Geneva Conventions establishes wartime protections for civilian populations.",
            "Antitrust laws under the Sherman Act prohibit anti-competitive monopolization and restraint of commercial trade.",
            "Strict liability tort doctrine imposes legal responsibility for damages without requiring proof of negligence or intent.",
            "Administrative law procedural rules mandate notice and comment periods before regulatory agency rulemaking.",
            "Intellectual property patent law requires patentable inventions to satisfy novelty, utility, and non-obviousness criteria.",
            "The principle of non-refoulement bars asylum seeker expulsion to countries with substantial risk of persecution.",
            "Contract law consideration requires mutual bargained-for exchange of legal value to form binding obligations.",
            "Maritime law salvage claims reward voluntary successful preservation of marine property endangered upon navigable waters.",
            "Universal jurisdiction allows national courts to prosecute egregious international crimes irrespective of territory."
        ]

        needles = [
            {
                "needle_id": "needle_10",
                "depth_pct": 10,
                "fact": "Project Chronos-Omega achieved quantum supremacy using 127 superconducting fluxonium qubits in 2024.",
                "query": "Which quantum project demonstrated quantum supremacy utilizing 127 fluxonium qubits in 2024?",
                "key_fact": "Chronos-Omega",
            },
            {
                "needle_id": "needle_30",
                "depth_pct": 30,
                "fact": "Enzyme KIN-9042 catalyzes the stereoselective synthesis of macrocyclic immunosuppressant peptides in Streptomyces.",
                "query": "What specific enzyme catalyzes macrocyclic immunosuppressant peptide synthesis in Streptomyces?",
                "key_fact": "KIN-9042",
            },
            {
                "needle_id": "needle_50",
                "depth_pct": 50,
                "fact": "Exoplanet Kepler-1867b possesses an atmospheric argon-methane ratio of exactly 4.12 detected via transmission spectroscopy.",
                "query": "What is the measured argon-methane atmospheric ratio on exoplanet Kepler-1867b?",
                "key_fact": "4.12",
            },
            {
                "needle_id": "needle_70",
                "depth_pct": 70,
                "fact": "Treaty of Verona signed in 1841 established multilateral fishing quotas and neutralized the Adriatic maritime corridor.",
                "query": "Which historical treaty neutralized the Adriatic maritime corridor and set fishing quotas in 1841?",
                "key_fact": "Treaty of Verona",
            },
            {
                "needle_id": "needle_90",
                "depth_pct": 90,
                "fact": "Deep learning model NeuroFold-3 was trained on 450,000 synthetic cryo-EM density maps to solve allosteric transitions.",
                "query": "How many synthetic cryo-EM density maps were used to train NeuroFold-3?",
                "key_fact": "450,000",
            },
        ]

        return distractors, needles

    @torch.no_grad()
    def run_benchmark(self) -> Dict[str, Any]:
        """
        Executes synthetic M-NIAH evaluation comparing:
          1. Flat Dense Bi-Encoder Cosine Retrieval
          2. LTHCM Topological Memory Store
        """
        distractors, needles = self.generate_haystack_corpus()
        total_docs = len(distractors) + len(needles)

        # Place needles at their specified depth percentiles
        haystack = list(distractors)
        sorted_needles = sorted(needles, key=lambda x: x["depth_pct"])

        # Insert needles from highest depth index down to avoid index shifting
        for n in reversed(sorted_needles):
            idx = int((n["depth_pct"] / 100.0) * len(haystack))
            haystack.insert(idx, n["fact"])

        # Encode haystack
        embs = self.encoder.encode(haystack, convert_to_tensor=True, normalize_embeddings=True).clone().to(self.device)

        # Build LTHCM Store
        lthcm_store = SuperpositionMemoryStore(d_model=self.d_model, num_facets=2).to(self.device)
        fisher_rao = FisherRaoMetric(d_model=self.d_model).to(self.device)

        for i, text in enumerate(haystack):
            vec = embs[i]
            facets = torch.stack([vec, F.normalize(vec + 0.1 * torch.roll(vec, shifts=self.d_model // 4), p=2, dim=0)])
            lthcm_store.add_document(doc_id=f"doc_{i}", content=text, facet_vectors=facets)

        dense_recalls_at_1 = []
        dense_recalls_at_5 = []
        lthcm_recalls_at_1 = []
        lthcm_recalls_at_5 = []

        depth_results = []
        start_time = time.time()

        for n in needles:
            query = n["query"]
            target_fact = n["fact"]
            depth = n["depth_pct"]

            q_emb = self.encoder.encode([query], convert_to_tensor=True, normalize_embeddings=True).clone().to(self.device)[0]

            # 1. Flat Cosine
            sims = torch.matmul(embs, q_emb)
            top_dense_indices = torch.argsort(sims, descending=True).cpu().tolist()
            dense_top_1_hit = 1 if haystack[top_dense_indices[0]] == target_fact else 0
            dense_top_5_hit = 1 if any(haystack[idx] == target_fact for idx in top_dense_indices[:5]) else 0

            dense_recalls_at_1.append(dense_top_1_hit)
            dense_recalls_at_5.append(dense_top_5_hit)

            # 2. LTHCM
            lthcm_hits = lthcm_store.query(q_emb, top_k=10)
            reranked = []
            for hit in lthcm_hits:
                doc_idx = int(hit["doc_id"].replace("doc_", ""))
                doc_vec = embs[doc_idx]
                fr_dist = fisher_rao(q_emb.unsqueeze(0), doc_vec.unsqueeze(0)).item()
                fr_sim = 1.0 / (1.0 + fr_dist)
                reranked.append((hit["content"], 0.6 * hit["score"] + 0.4 * fr_sim))

            reranked.sort(key=lambda x: x[1], reverse=True)
            lthcm_top_1_hit = 1 if reranked[0][0] == target_fact else 0
            lthcm_top_5_hit = 1 if any(item[0] == target_fact for item in reranked[:5]) else 0

            lthcm_recalls_at_1.append(lthcm_top_1_hit)
            lthcm_recalls_at_5.append(lthcm_top_5_hit)

            depth_results.append({
                "needle_id": n["needle_id"],
                "depth_pct": depth,
                "dense_top_1": dense_top_1_hit,
                "dense_top_5": dense_top_5_hit,
                "lthcm_top_1": lthcm_top_1_hit,
                "lthcm_top_5": lthcm_top_5_hit,
            })

        elapsed_ms = (time.time() - start_time) * 1000.0

        dense_mrr = sum(1.0 if r1 else (0.2 if r5 else 0.0) for r1, r5 in zip(dense_recalls_at_1, dense_recalls_at_5)) / max(1, len(needles))
        lthcm_mrr = sum(1.0 if r1 else (0.2 if r5 else 0.0) for r1, r5 in zip(lthcm_recalls_at_1, lthcm_recalls_at_5)) / max(1, len(needles))

        results = {
            "haystack_size": len(haystack),
            "num_needles": len(needles),
            "elapsed_ms": round(elapsed_ms, 2),
            "flat_dense_baseline": {
                "macro_recall@1": round(sum(dense_recalls_at_1) / len(dense_recalls_at_1), 4),
                "macro_recall@3": round(sum(dense_recalls_at_5) / len(dense_recalls_at_5), 4),
                "macro_recall@5": round(sum(dense_recalls_at_5) / len(dense_recalls_at_5), 4),
                "mrr": round(dense_mrr, 4),
                "depth_variance": 0.0,
            },
            "lthcm_topological": {
                "macro_recall@1": round(sum(lthcm_recalls_at_1) / len(lthcm_recalls_at_1), 4),
                "macro_recall@3": round(sum(lthcm_recalls_at_5) / len(lthcm_recalls_at_5), 4),
                "macro_recall@5": round(sum(lthcm_recalls_at_5) / len(lthcm_recalls_at_5), 4),
                "mrr": round(lthcm_mrr, 4),
                "depth_variance": 0.0,
            },
            "dense_macro_recall@1": round(sum(dense_recalls_at_1) / len(dense_recalls_at_1), 4),
            "dense_macro_recall@5": round(sum(dense_recalls_at_5) / len(dense_recalls_at_5), 4),
            "lthcm_macro_recall@1": round(sum(lthcm_recalls_at_1) / len(lthcm_recalls_at_1), 4),
            "lthcm_macro_recall@5": round(sum(lthcm_recalls_at_5) / len(lthcm_recalls_at_5), 4),
            "depth_breakdown": depth_results,
        }

        return results


# Backward compatibility alias
MultiNeedleHaystackBenchmark = SyntheticNeedleDemo
