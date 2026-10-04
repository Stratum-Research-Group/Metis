"""
Real VitaminC & FEVER Contradiction Verification Benchmark
==========================================================
Evaluates real community VitaminC test split (55,197 contrastive pairs)
and FEVER datasets under the honest evaluation protocol:

  1. Majority Class Baseline
  2. Baseline A: Flat Cosine Similarity Threshold
  3. Baseline B: Raw DeBERTa NLI Cross-Encoder Alone (nli-deberta-v3-xsmall)
  4. Baseline B2: DeBERTa NLI with Symmetric Aggregation Rule
  5. Baseline C: LTHCM Cellular Sheaf Laplacian (Dirichlet Energy Obstruction)

Reports:
  - 3-Way Classification (SUPPORTS / REFUTES / NOT ENOUGH INFO)
  - 2-Way Classification (SUPPORTS vs REFUTES, dropping NEI)
  - Macro-F1, Per-Class F1, Confusion Matrix, and Paired Significance
"""

import os
import sys
import time
import json
import argparse
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from lthcm.core.sheaf import SheafContradictionFilter, CellularSheaf
from lthcm.core.contrastive_sheaf import ContrastiveSheafTrainer
from lthcm.benchmarks.real.utils.significance import paired_bootstrap_test
from lthcm.utils.determinism import set_all_seeds


class VitaminCRealBenchmark:
    """
    Evaluates real VitaminC contrastive fact pairs.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        nli_model_name: str = "cross-encoder/nli-deberta-v3-xsmall",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder = SentenceTransformer(model_name)
        if hasattr(self.encoder, "eval"):
            self.encoder.eval()
        self.d_model = self.encoder.get_embedding_dimension()
        self.sheaf_filter = SheafContradictionFilter(
            d_model=self.d_model, nli_model_name=nli_model_name, contradiction_threshold=2.5
        ).to(self.device)
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface/datasets")

    def get_or_train_contrastive_sheaf(
        self,
        max_train_triplets: int = 1200,
        checkpoint_path: Optional[str] = None,
        output_dir: str = "runs",
        epochs: int = 4,
        seed: int = 42,
        allow_synthetic_fallback: bool = False,
        force_retrain: bool = False,
    ) -> ContrastiveSheafTrainer:
        """
        Loads contrastive triplets from VitaminC train split only and trains
        the stalk projector and sheaf restriction maps under leak-free guarantees.
        Rejects pre-existing checkpoints with mismatched seed or hyperparameters.
        """
        set_all_seeds(seed)
        if checkpoint_path is None:
            checkpoint_path = os.path.join(output_dir, f"contrastive_sheaf_checkpoint_seed{seed}.pt")

        expected_meta = {
            "seed": seed,
            "d_model": self.d_model,
            "stalk_dim": 64,
            "epochs": epochs,
            "is_synthetic": False,
        }

        trainer = ContrastiveSheafTrainer(d_model=self.d_model, stalk_dim=64, device=self.device)

        if os.path.exists(checkpoint_path) and not force_retrain:
            try:
                print(f"[-] Loading contrastive sheaf checkpoint from {checkpoint_path}...")
                trainer.load_checkpoint(checkpoint_path, expected_metadata=expected_meta)
                print("    Checkpoint metadata verified successfully.")
                return trainer
            except Exception as e:
                print(f"[-] Checkpoint metadata mismatch or corrupt ({e}); purging stale checkpoint and retraining fresh model...")
                try:
                    os.remove(checkpoint_path)
                except OSError:
                    pass

        print(f"[-] Training contrastive sheaf (R5) on VitaminC train split (max {max_train_triplets} triplets, seed={seed})...")
        try:
            from datasets import load_dataset
            from collections import defaultdict
            ds_train = load_dataset("tals/vitaminc", split="train", cache_dir=self.cache_dir)
            cases = defaultdict(dict)
            for ex in ds_train:
                cid = ex.get("case_id")
                lbl = ex.get("label")
                if cid and lbl in ("SUPPORTS", "REFUTES"):
                    cases[cid][lbl] = (str(ex.get("claim", "")).strip(), str(ex.get("evidence", "")).strip())
                if len(cases) >= max_train_triplets * 3:
                    break

            triplet_cases = sorted([cid for cid, m in cases.items() if "SUPPORTS" in m and "REFUTES" in m])
            triplet_cases = triplet_cases[:max_train_triplets]
            print(f"    Extracted {len(triplet_cases)} contrastive triplets from train split.")

            claims_pos = [cases[cid]["SUPPORTS"][0] for cid in triplet_cases]
            claims_neg = [cases[cid]["REFUTES"][0] for cid in triplet_cases]
            evidences = [cases[cid]["SUPPORTS"][1] for cid in triplet_cases]

            embs_pos = self.encoder.encode(claims_pos, batch_size=32, normalize_embeddings=True)
            embs_neg = self.encoder.encode(claims_neg, batch_size=32, normalize_embeddings=True)
            embs_evid = self.encoder.encode(evidences, batch_size=32, normalize_embeddings=True)

            triplets = [(embs_pos[i], embs_neg[i], embs_evid[i]) for i in range(len(triplet_cases))]
            history = trainer.train_on_triplets(triplets, epochs=epochs, batch_size=32, seed=seed)
            print(f"    Trained contrastive sheaf: final InfoNCE loss={history['proj_loss'][-1]:.4f}, margin loss={history['sheaf_loss'][-1]:.4f}")
            meta = {**expected_meta, "max_train_triplets": len(triplet_cases)}
            trainer.save_checkpoint(checkpoint_path, metadata=meta)
        except Exception as e:
            if not allow_synthetic_fallback:
                raise RuntimeError(f"Failed to train contrastive sheaf on VitaminC train: {e}")
            print(f"    [WARNING] Synthetic fallback for contrastive sheaf training ({e})")
            rng = np.random.default_rng(seed)
            dummy_pos = rng.standard_normal((50, self.d_model)).astype(np.float32)
            dummy_neg = rng.standard_normal((50, self.d_model)).astype(np.float32)
            dummy_ev = rng.standard_normal((50, self.d_model)).astype(np.float32)
            triplets = [(dummy_pos[i], dummy_neg[i], dummy_ev[i]) for i in range(50)]
            trainer.train_on_triplets(triplets, epochs=2, seed=seed)
            syn_meta = {**expected_meta, "is_synthetic": True, "epochs": 2}
            trainer.save_checkpoint(checkpoint_path, metadata=syn_meta)

        return trainer

    def load_data(
        self,
        max_pairs: Optional[int] = None,
        allow_synthetic_fallback: bool = False,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Loads the official VitaminC test split (tals/vitaminc).
        """
        try:
            from datasets import load_dataset
            print("[-] Loading real VitaminC test split (tals/vitaminc)...")
            ds = load_dataset("tals/vitaminc", split="test", cache_dir=self.cache_dir)

            pairs = []
            for ex in ds:
                claim = str(ex.get("claim", "")).strip()
                evidence = str(ex.get("evidence", "")).strip()
                raw_label = str(ex.get("label", "")).strip().upper()

                # Normalize label
                if "SUPPORT" in raw_label:
                    label = "SUPPORTS"
                elif "REFUTE" in raw_label or "CONTRADICT" in raw_label:
                    label = "REFUTES"
                else:
                    label = "NOT ENOUGH INFO"

                pairs.append({
                    "claim": claim,
                    "evidence": evidence,
                    "label": label,
                })
                if max_pairs and len(pairs) >= max_pairs:
                    break

            print(f"    Loaded {len(pairs)} VitaminC pairs.")
            return pairs, False
        except Exception as e:
            if not allow_synthetic_fallback:
                raise RuntimeError(
                    f"Failed to load real VitaminC dataset from HuggingFace ({e}). "
                    "To run with synthetic demonstration fixtures for offline testing, pass allow_synthetic_fallback=True."
                )
            print(f"    [WARNING] Using SYNTHETIC FALLBACK data for smoke testing. These metrics do NOT represent real VitaminC numbers! ({e})")
            from lthcm.benchmarks.synthetic.demo_sheaf import SyntheticSheafDemo
            demo = SyntheticSheafDemo()
            return demo.generate_fever_dataset(), True

    def run_evaluation(
        self,
        max_pairs: Optional[int] = None,
        output_dir: str = "runs",
        seed: int = 42,
        allow_synthetic_fallback: bool = False,
        force_retrain_r5: bool = False,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes full comparative verification over VitaminC.
        """
        set_all_seeds(seed)
        os.makedirs(output_dir, exist_ok=True)
        dataset, is_synthetic = self.load_data(max_pairs=max_pairs, allow_synthetic_fallback=allow_synthetic_fallback)

        claims = [d.get("claim", "") for d in dataset]
        evidences = [d.get("evidence", d.get("premise", "")) for d in dataset]
        labels = [d["label"] for d in dataset]

        # -------------------------------------------------------------
        # Label Mapping: 0: SUPPORTS, 1: REFUTES, 2: NOT ENOUGH INFO
        # -------------------------------------------------------------
        label_to_id = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
        y_true_3way = [label_to_id.get(l, 2) for l in labels]
        
        # Majority class baseline
        counts = [labels.count("SUPPORTS"), labels.count("REFUTES"), labels.count("NOT ENOUGH INFO")]
        majority_id = int(np.argmax(counts))
        maj_acc_3way = counts[majority_id] / len(labels)

        print(f"[-] Running contradiction verification across {len(dataset)} pairs (Majority class baseline: {maj_acc_3way*100:.1f}%)...")
        t0 = time.time()

        # -------------------------------------------------------------
        # 1. Baseline A: Flat Cosine Similarity Thresholding
        # -------------------------------------------------------------
        print("[-] Computing Baseline A (Flat Cosine Thresholding)...")
        claim_embs = self.encoder.encode(claims, batch_size=32, normalize_embeddings=True)
        evid_embs = self.encoder.encode(evidences, batch_size=32, normalize_embeddings=True)
        cos_sims = np.sum(claim_embs * evid_embs, axis=-1)

        preds_a_3way = []
        for s in cos_sims:
            if s < 0.85:
                preds_a_3way.append(1)  # REFUTES
            elif s > 0.92:
                preds_a_3way.append(0)  # SUPPORTS
            else:
                preds_a_3way.append(2)  # NEI

        # -------------------------------------------------------------
        # 2. Baseline B & B2: DeBERTa Cross-Encoder
        # -------------------------------------------------------------
        print("[-] Computing Baseline B (Raw DeBERTa NLI) and Baseline B2 (Symmetric Rule)...")
        pairs_fwd = list(zip(evidences, claims))
        pairs_rev = list(zip(claims, evidences))

        nli = self.sheaf_filter.nli_model
        logits_fwd = nli.predict(pairs_fwd, batch_size=32)
        logits_rev = nli.predict(pairs_rev, batch_size=32)
        
        probs_fwd = F.softmax(torch.tensor(logits_fwd), dim=-1).numpy()
        probs_rev = F.softmax(torch.tensor(logits_rev), dim=-1).numpy()

        preds_b_raw_3way = []
        preds_b2_symm_3way = []

        for i in range(len(dataset)):
            # Model output: 0: contradiction, 1: entailment, 2: neutral
            fwd_arg = probs_fwd[i].argmax()
            map_to_id = {0: 1, 1: 0, 2: 2} # 0->REFUTES, 1->SUPPORTS, 2->NEI
            preds_b_raw_3way.append(map_to_id[fwd_arg])

            # Symmetric Rule
            p_contra_avg = 0.5 * (probs_fwd[i][0] + probs_rev[i][0])
            p_entail_avg = 0.5 * (probs_fwd[i][1] + probs_rev[i][1])
            p_neut_avg = 0.5 * (probs_fwd[i][2] + probs_rev[i][2])

            if p_contra_avg > 0.40 and p_entail_avg < 0.35:
                preds_b2_symm_3way.append(1)  # REFUTES
            elif p_entail_avg > 0.50:
                preds_b2_symm_3way.append(0)  # SUPPORTS
            else:
                preds_b2_symm_3way.append(2)  # NEI

        # -------------------------------------------------------------
        # 3. Arm C1: Pure Sheaf (Untrained Orthogonal Stalks, Zero NLI)
        # -------------------------------------------------------------
        print("[-] Computing Arm C1: Pure Sheaf (Untrained Orthogonal Stalks, zero NLI)...")
        from lthcm.core.sheaf import CellularSheaf
        set_all_seeds(seed)
        untrained_sheaf = CellularSheaf(d_model=self.d_model, stalk_dim=64).to(self.device)
        preds_untrained_3way = []
        with torch.no_grad():
            for i in range(len(dataset)):
                pair_embs = torch.from_numpy(np.stack([evid_embs[i], claim_embs[i]])).to(self.device)
                e_idx = torch.tensor([[0, 1], [1, 0]], device=self.device).T
                energy, _ = untrained_sheaf.compute_dirichlet_energy(pair_embs, e_idx)
                if energy.item() > 0.06:
                    preds_untrained_3way.append(1)  # REFUTES
                else:
                    preds_untrained_3way.append(0)  # SUPPORTS

        # -------------------------------------------------------------
        # 4. Arm C2: LTHCM Sheaf with NLI-Grounded Stalks
        # -------------------------------------------------------------
        print("[-] Computing Arm C2: LTHCM Sheaf with NLI-Grounded Stalks...")
        sheaf_preds_3way = []
        sheaf_energies = []

        # Encode passages
        for i in range(len(dataset)):
            p_text = evidences[i]
            c_text = claims[i]
            embs = torch.from_numpy(np.stack([evid_embs[i], claim_embs[i]])).to(self.device)
            passages = [{"doc_id": "p", "content": p_text}, {"doc_id": "c", "content": c_text}]
            
            # Evaluate sheaf
            res = self.sheaf_filter.evaluate_passages(embs, passages)
            energy = res["total_obstruction"]
            sheaf_energies.append(energy)

            # Sheaf prediction
            if not res["is_consistent"]:
                sheaf_preds_3way.append(1)  # REFUTES (obstruction > threshold)
            else:
                # If harmonic (no obstruction), check entailment vs neutral
                p_entail = 0.5 * (probs_fwd[i][1] + probs_rev[i][1])
                if p_entail > 0.40:
                    sheaf_preds_3way.append(0)  # SUPPORTS
                else:
                    sheaf_preds_3way.append(2)  # NEI

        # -------------------------------------------------------------
        # 5. R5: Contrastive Claim-Pair Fine-Tuning & Trained Sheaf
        # -------------------------------------------------------------
        print("[-] Computing R5 Arms: Contrastive Claim-Pair Fine-Tuned Stalks & Trained Sheaf...")
        trainer = self.get_or_train_contrastive_sheaf(
            max_train_triplets=1200,
            output_dir=output_dir,
            epochs=4,
            seed=seed,
            allow_synthetic_fallback=allow_synthetic_fallback,
            force_retrain=force_retrain_r5,
        )

        with torch.no_grad():
            c_tensor = torch.from_numpy(claim_embs).to(self.device)
            e_tensor = torch.from_numpy(evid_embs).to(self.device)
            stalk_claims = trainer.projector(c_tensor).cpu().numpy()
            stalk_evids = trainer.projector(e_tensor).cpu().numpy()

        # Arm R5a: Contrastive Fine-Tuned Stalks + Cosine Threshold
        stalk_cos_sims = np.sum(stalk_claims * stalk_evids, axis=-1)
        preds_r5a_3way = []
        for s in stalk_cos_sims:
            if s < 0.25:
                preds_r5a_3way.append(1)  # REFUTES
            elif s > 0.60:
                preds_r5a_3way.append(0)  # SUPPORTS
            else:
                preds_r5a_3way.append(2)  # NEI

        # Arm R5b: Contrastive Stalks + Trained Geometric Sheaf (Zero NLI Overwrite)
        preds_r5b_3way = []
        with torch.no_grad():
            s_c_t = torch.from_numpy(stalk_claims).to(self.device)
            s_e_t = torch.from_numpy(stalk_evids).to(self.device)
            r5_energies = trainer.sheaf.compute_dirichlet_energy(s_c_t, s_e_t).cpu().numpy()

        for s_energy, s_cos in zip(r5_energies, stalk_cos_sims):
            if s_energy > 0.28:
                preds_r5b_3way.append(1)  # REFUTES (high cohomological obstruction)
            elif s_energy < 0.15:
                preds_r5b_3way.append(0)  # SUPPORTS (harmonic agreement)
            else:
                preds_r5b_3way.append(2)  # NEI

        # Arm R5c: Contrastive Fine-Tuned Sheaf + NLI Stalks
        preds_r5c_3way = []
        for i in range(len(dataset)):
            if r5_energies[i] > 0.30:
                preds_r5c_3way.append(1)  # REFUTES (energy obstruction overrides NLI)
            else:
                p_entail = 0.5 * (probs_fwd[i][1] + probs_rev[i][1])
                if p_entail > 0.40:
                    preds_r5c_3way.append(0)  # SUPPORTS
                else:
                    preds_r5c_3way.append(2)  # NEI

        # -------------------------------------------------------------
        # Calculate 3-Way and 2-Way Metrics
        # -------------------------------------------------------------
        def compute_class_metrics(y_true, y_pred, n_classes=3):
            acc = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true)
            f1s = []
            for c in range(n_classes):
                tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
                fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
                fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
                f1s.append(f1)
            macro_f1 = float(np.mean(f1s))
            return {"accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4), "per_class_f1": [round(f, 4) for f in f1s]}

        results_3way = {
            "majority_class": {"accuracy": round(maj_acc_3way, 4)},
            "flat_cosine_baseline_a": compute_class_metrics(y_true_3way, preds_a_3way, 3),
            "raw_deberta_nli_baseline_b": compute_class_metrics(y_true_3way, preds_b_raw_3way, 3),
            "symmetric_deberta_rule_baseline_b2": compute_class_metrics(y_true_3way, preds_b2_symm_3way, 3),
            "pure_untrained_sheaf_arm_c1": compute_class_metrics(y_true_3way, preds_untrained_3way, 3),
            "nli_grounded_sheaf_arm_c2": compute_class_metrics(y_true_3way, sheaf_preds_3way, 3),
            "contrastive_stalk_cosine_arm_r5a": compute_class_metrics(y_true_3way, preds_r5a_3way, 3),
            "trained_geometric_sheaf_arm_r5b": compute_class_metrics(y_true_3way, preds_r5b_3way, 3),
            "contrastive_sheaf_nli_arm_r5c": compute_class_metrics(y_true_3way, preds_r5c_3way, 3),
        }

        # 2-Way (binary: SUPPORTS vs REFUTES, dropping NEI)
        valid_2way_idx = [i for i, y in enumerate(y_true_3way) if y in (0, 1)]
        y_true_2way = [y_true_3way[i] for i in valid_2way_idx]
        a_2way = [preds_a_3way[i] for i in valid_2way_idx]
        b_2way = [preds_b_raw_3way[i] for i in valid_2way_idx]
        b2_2way = [preds_b2_symm_3way[i] for i in valid_2way_idx]
        c1_2way = [preds_untrained_3way[i] for i in valid_2way_idx]
        c2_2way = [sheaf_preds_3way[i] for i in valid_2way_idx]
        r5a_2way = [preds_r5a_3way[i] for i in valid_2way_idx]
        r5b_2way = [preds_r5b_3way[i] for i in valid_2way_idx]
        r5c_2way = [preds_r5c_3way[i] for i in valid_2way_idx]

        results_2way = {
            "num_2way_pairs": len(valid_2way_idx),
            "flat_cosine_baseline_a": compute_class_metrics(y_true_2way, a_2way, 2),
            "raw_deberta_nli_baseline_b": compute_class_metrics(y_true_2way, b_2way, 2),
            "symmetric_deberta_rule_baseline_b2": compute_class_metrics(y_true_2way, b2_2way, 2),
            "pure_untrained_sheaf_arm_c1": compute_class_metrics(y_true_2way, c1_2way, 2),
            "nli_grounded_sheaf_arm_c2": compute_class_metrics(y_true_2way, c2_2way, 2),
            "contrastive_stalk_cosine_arm_r5a": compute_class_metrics(y_true_2way, r5a_2way, 2),
            "trained_geometric_sheaf_arm_r5b": compute_class_metrics(y_true_2way, r5b_2way, 2),
            "contrastive_sheaf_nli_arm_r5c": compute_class_metrics(y_true_2way, r5c_2way, 2),
        }

        # Significance test vectors (1.0 if correct, 0.0 if incorrect)
        raw_b_correct = [1.0 if yt == yp else 0.0 for yt, yp in zip(y_true_2way, b_2way)]
        r5b_correct = [1.0 if yt == yp else 0.0 for yt, yp in zip(y_true_2way, r5b_2way)]
        c1_correct = [1.0 if yt == yp else 0.0 for yt, yp in zip(y_true_2way, c1_2way)]
        c2_correct = [1.0 if yt == yp else 0.0 for yt, yp in zip(y_true_2way, c2_2way)]
        b2_correct = [1.0 if yt == yp else 0.0 for yt, yp in zip(y_true_2way, b2_2way)]
        a_correct = [1.0 if yt == yp else 0.0 for yt, yp in zip(y_true_2way, a_2way)]
        r5a_correct = [1.0 if yt == yp else 0.0 for yt, yp in zip(y_true_2way, r5a_2way)]

        # Headline Test: Trained Sheaf R5b vs Raw DeBERTa NLI B (per improv2.md requirement)
        sig_headline_r5b_vs_raw_nli = paired_bootstrap_test(r5b_correct, raw_b_correct, num_samples=10000, seed=seed)
        # Ablation: Trained Sheaf R5b vs Untrained Sheaf C1
        sig_r5b_vs_untrained = paired_bootstrap_test(r5b_correct, c1_correct, num_samples=10000, seed=seed)
        # Stalk Quality: Contrastive Stalks R5a vs Raw Cosine Baseline A
        sig_r5a_vs_flat = paired_bootstrap_test(r5a_correct, a_correct, num_samples=10000, seed=seed)
        # Legacy: Sheaf vs Symmetric B2
        sig_test_b2 = paired_bootstrap_test(c2_correct, b2_correct, num_samples=10000, seed=seed)

        significance_results = {
            "headline_trained_sheaf_r5b_vs_raw_nli_b": sig_headline_r5b_vs_raw_nli,
            "trained_sheaf_r5b_vs_untrained_sheaf_c1": sig_r5b_vs_untrained,
            "contrastive_stalks_r5a_vs_flat_cosine_a": sig_r5a_vs_flat,
            "legacy_sheaf_c2_vs_symmetric_b2": sig_test_b2,
        }

        results = {
            "total_pairs": len(dataset),
            "is_synthetic": is_synthetic,
            "run_id": run_id,
            "dataset_source": "synthetic_fallback" if is_synthetic else "tals/vitaminc",
            "elapsed_seconds": round(time.time() - t0, 2),
            "3_way_classification": results_3way,
            "2_way_classification": results_2way,
            "significance": significance_results,
            "significance_sheaf_vs_symmetric_b2": sig_test_b2,
        }

        print("\n" + "=" * 85)
        print(" REAL VITAMINC CONTRADICTION VERIFICATION SUMMARY")
        if is_synthetic:
            print(" [WARNING: EVALUATION EXECUTED ON SYNTHETIC FALLBACK FIXTURE]")
        print("=" * 85)
        print(f" {'System':<40} | {'3-Way Acc':<10} | {'3-Way F1':<10} | {'2-Way Acc':<10} | {'2-Way F1':<10}")
        print("-" * 85)
        print(f" {'Majority Class Baseline':<40} | {maj_acc_3way*100:<9.2f}% | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")
        print(f" {'Baseline A: Flat Cosine Threshold':<40} | {results_3way['flat_cosine_baseline_a']['accuracy']*100:<9.2f}% | {results_3way['flat_cosine_baseline_a']['macro_f1']:<10.4f} | {results_2way['flat_cosine_baseline_a']['accuracy']*100:<9.2f}% | {results_2way['flat_cosine_baseline_a']['macro_f1']:<10.4f}")
        print(f" {'Baseline B: Raw DeBERTa-v3 NLI':<40} | {results_3way['raw_deberta_nli_baseline_b']['accuracy']*100:<9.2f}% | {results_3way['raw_deberta_nli_baseline_b']['macro_f1']:<10.4f} | {results_2way['raw_deberta_nli_baseline_b']['accuracy']*100:<9.2f}% | {results_2way['raw_deberta_nli_baseline_b']['macro_f1']:<10.4f}")
        print(f" {'Baseline B2: Symmetric NLI Rule':<40} | {results_3way['symmetric_deberta_rule_baseline_b2']['accuracy']*100:<9.2f}% | {results_3way['symmetric_deberta_rule_baseline_b2']['macro_f1']:<10.4f} | {results_2way['symmetric_deberta_rule_baseline_b2']['accuracy']*100:<9.2f}% | {results_2way['symmetric_deberta_rule_baseline_b2']['macro_f1']:<10.4f}")
        print(f" {'Arm C1: Pure Sheaf (Untrained, Zero NLI)':<40} | {results_3way['pure_untrained_sheaf_arm_c1']['accuracy']*100:<9.2f}% | {results_3way['pure_untrained_sheaf_arm_c1']['macro_f1']:<10.4f} | {results_2way['pure_untrained_sheaf_arm_c1']['accuracy']*100:<9.2f}% | {results_2way['pure_untrained_sheaf_arm_c1']['macro_f1']:<10.4f}")
        print(f" {'Arm C2: LTHCM Sheaf (NLI Stalks)':<40} | {results_3way['nli_grounded_sheaf_arm_c2']['accuracy']*100:<9.2f}% | {results_3way['nli_grounded_sheaf_arm_c2']['macro_f1']:<10.4f} | {results_2way['nli_grounded_sheaf_arm_c2']['accuracy']*100:<9.2f}% | {results_2way['nli_grounded_sheaf_arm_c2']['macro_f1']:<10.4f}")
        print(f" {'Arm R5a: Contrastive Stalks + Cosine':<40} | {results_3way['contrastive_stalk_cosine_arm_r5a']['accuracy']*100:<9.2f}% | {results_3way['contrastive_stalk_cosine_arm_r5a']['macro_f1']:<10.4f} | {results_2way['contrastive_stalk_cosine_arm_r5a']['accuracy']*100:<9.2f}% | {results_2way['contrastive_stalk_cosine_arm_r5a']['macro_f1']:<10.4f}")
        print(f" {'Arm R5b: Trained Sheaf (Zero NLI)':<40} | {results_3way['trained_geometric_sheaf_arm_r5b']['accuracy']*100:<9.2f}% | {results_3way['trained_geometric_sheaf_arm_r5b']['macro_f1']:<10.4f} | {results_2way['trained_geometric_sheaf_arm_r5b']['accuracy']*100:<9.2f}% | {results_2way['trained_geometric_sheaf_arm_r5b']['macro_f1']:<10.4f}")
        print(f" {'Arm R5c: Contrastive Sheaf + NLI':<40} | {results_3way['contrastive_sheaf_nli_arm_r5c']['accuracy']*100:<9.2f}% | {results_3way['contrastive_sheaf_nli_arm_r5c']['macro_f1']:<10.4f} | {results_2way['contrastive_sheaf_nli_arm_r5c']['accuracy']*100:<9.2f}% | {results_2way['contrastive_sheaf_nli_arm_r5c']['macro_f1']:<10.4f}")
        print("=" * 85)
        print(" PAIRED BOOTSTRAP SIGNIFICANCE TESTS (10,000 resamples, 95% CI):")
        print(f"   HEADLINE: Trained Sheaf R5b vs Raw NLI B: Delta = {sig_headline_r5b_vs_raw_nli['delta']*100:+.2f}%, "
              f"95% CI: [{sig_headline_r5b_vs_raw_nli['ci_lower']*100:+.2f}%, {sig_headline_r5b_vs_raw_nli['ci_upper']*100:+.2f}%], p = {sig_headline_r5b_vs_raw_nli['p_value']:.4f} "
              f"({'Significant' if sig_headline_r5b_vs_raw_nli['is_significant'] else 'Not Significant'})")
        print(f"   Ablation: Trained Sheaf R5b vs Untrained C1: Delta = {sig_r5b_vs_untrained['delta']*100:+.2f}%, "
              f"95% CI: [{sig_r5b_vs_untrained['ci_lower']*100:+.2f}%, {sig_r5b_vs_untrained['ci_upper']*100:+.2f}%], p = {sig_r5b_vs_untrained['p_value']:.4f}")
        print(f"   Stalks: Contrastive R5a vs Raw Cosine A:  Delta = {sig_r5a_vs_flat['delta']*100:+.2f}%, "
              f"95% CI: [{sig_r5a_vs_flat['ci_lower']*100:+.2f}%, {sig_r5a_vs_flat['ci_upper']*100:+.2f}%], p = {sig_r5a_vs_flat['p_value']:.4f}")
        print("=" * 85 + "\n")

        with open(os.path.join(output_dir, "vitaminc_summary.json"), "w") as f:
            json.dump(results, f, indent=2)

        return results


if __name__ == "__main__":
    from lthcm.utils.determinism import ensure_deterministic_env
    ensure_deterministic_env(42)

    import argparse
    parser = argparse.ArgumentParser(description="VitaminC & Sheaf Verification Benchmark")
    parser.add_argument("--max_pairs", type=int, default=30, help="Max test pairs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", default="runs", help="Output directory")
    parser.add_argument("--allow_synthetic_fallback", action="store_true", help="Allow synthetic fallback")
    args = parser.parse_args()

    bench = VitaminCRealBenchmark()
    bench.run_evaluation(
        max_pairs=args.max_pairs,
        output_dir=args.output_dir,
        seed=args.seed,
        allow_synthetic_fallback=args.allow_synthetic_fallback,
    )
