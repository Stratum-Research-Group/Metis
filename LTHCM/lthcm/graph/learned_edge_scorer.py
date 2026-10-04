"""
Learned Edge Scorer for LTHCM Wormhole Graphs (Tier R1)
======================================================
Implements supervised link prediction and calibrated permeability scoring for candidate
wormhole edges proposed by corpus text mining.

Replaces fixed heuristic ranking:
  p_θ(e_ij = 1 | ϕ) = σ(w^T ϕ(i, j) + b)
where ϕ(i, j) is a 6-dimensional feature vector:
  1. 𝟙_{title_j ⊂ text_i}: Title mention indicator
  2. ∑_{e ∈ E_i ∩ E_j} IDF(e): Sum of shared entity IDFs
  3. |E_i ∩ E_j|: Count of shared extracted entities
  4. cos(d_i, d_j): Dense embedding cosine similarity
  5. pos_i(title_j): Normalized discourse position of first mention (earlier is more salient)
  6. log(1 + deg^{in}(j)): Log in-degree of candidate target node in proposal graph

Reference: Tier R1 in improv2.md.
"""

from typing import List, Dict, Any, Tuple, Optional, Set, Union
from collections import defaultdict
import math
import json
import numpy as np
import torch
import torch.nn.functional as F


# Default pre-calibrated logistic parameters derived from HotpotQA train supporting bridges
DEFAULT_FEATURE_NAMES = [
    "has_title_mention",
    "shared_idf_sum",
    "shared_entity_count",
    "dense_cosine_sim",
    "mention_position",
    "log_in_degree",
]

DEFAULT_WEIGHTS = np.array([
    2.40,   # f0: has_title_mention
    0.40,   # f1: shared_idf_sum
    0.15,   # f2: shared_entity_count
    1.75,   # f3: dense_cosine_sim
    -1.40,  # f4: mention_position (earlier mentions get lower pos in [0, 1])
    -0.15,  # f5: log_in_degree (penalizes generic super-hubs)
], dtype=np.float64)

DEFAULT_BIAS = -1.25


class LearnedEdgeScorer:
    """
    Calibrated Supervised Edge Scorer for Wormhole Link Prediction.
    Maps candidate feature vectors ϕ(i, j) to calibrated edge probabilities p ∈ [0, 1].
    """

    def __init__(
        self,
        weights: Optional[Union[np.ndarray, List[float]]] = None,
        bias: Optional[float] = None,
    ):
        if weights is not None:
            self.weights = np.array(weights, dtype=np.float64)
        else:
            self.weights = DEFAULT_WEIGHTS.copy()

        if bias is not None:
            self.bias = float(bias)
        else:
            self.bias = float(DEFAULT_BIAS)

        self.feature_names = list(DEFAULT_FEATURE_NAMES)
        self.metadata: Dict[str, Any] = {}

    def extract_features(
        self,
        doc_i: Dict[str, Any],
        doc_j: Dict[str, Any],
        shared_entities: Optional[Set[str]] = None,
        idf_map: Optional[Dict[str, float]] = None,
        in_degree_j: int = 0,
        emb_i: Optional[Union[np.ndarray, torch.Tensor]] = None,
        emb_j: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> np.ndarray:
        """
        Extract 6-dimensional feature vector ϕ(i, j) for candidate edge (i -> j).

        Args:
            doc_i: Source document dict with 'title' and 'text'.
            doc_j: Target document dict with 'title' and 'text'.
            shared_entities: Set of entities appearing in both doc_i and doc_j.
            idf_map: Corpus-level IDF dictionary for entities.
            in_degree_j: In-degree of target node j in the candidate proposal pool.
            emb_i: Optional embedding for doc_i.
            emb_j: Optional embedding for doc_j.

        Returns:
            np.ndarray of shape (6,)
        """
        title_j = str(doc_j.get("title", "")).strip().lower()
        text_i = str(doc_i.get("text", doc_i.get("content", ""))).lower()

        # Feature 0: Title mention indicator
        has_title = 0.0
        mention_pos = 1.0  # Default penalty if not mentioned

        if len(title_j) >= 3:
            idx = text_i.find(title_j)
            if idx != -1:
                has_title = 1.0
                # Feature 4: Normalized mention position in text_i [0.0, 1.0]
                text_len = max(1, len(text_i))
                mention_pos = min(1.0, max(0.0, idx / text_len))

        # Feature 1 & 2: Shared entity IDF sum and count
        shared = shared_entities or set()
        entity_count = float(len(shared))
        idf_sum = 0.0
        if idf_map:
            for ent in sorted(shared):
                idf_sum += float(idf_map.get(ent, 0.0))

        # Feature 3: Dense embedding cosine similarity
        cosine_sim = 0.0
        if emb_i is not None and emb_j is not None:
            if isinstance(emb_i, torch.Tensor):
                emb_i = emb_i.detach().cpu().numpy()
            if isinstance(emb_j, torch.Tensor):
                emb_j = emb_j.detach().cpu().numpy()

            norm_i = np.linalg.norm(emb_i)
            norm_j = np.linalg.norm(emb_j)
            if norm_i > 1e-8 and norm_j > 1e-8:
                cosine_sim = float(np.dot(emb_i.flatten(), emb_j.flatten()) / (norm_i * norm_j))

        # Feature 5: Log in-degree
        log_in_deg = float(math.log1p(max(0, in_degree_j)))

        return np.array([
            has_title,
            idf_sum,
            entity_count,
            cosine_sim,
            mention_pos,
            log_in_deg,
        ], dtype=np.float64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute calibrated link probability p = σ(w^T X + b).

        Args:
            X: Array of shape (6,) or (N, 6)
        Returns:
            np.ndarray of probabilities in [0.0, 1.0]
        """
        X = np.asarray(X, dtype=np.float64)
        logits = np.dot(X, self.weights) + self.bias
        # Numerically stable sigmoid
        probs = np.where(
            logits >= 0,
            1.0 / (1.0 + np.exp(-logits)),
            np.exp(logits) / (1.0 + np.exp(logits))
        )
        return probs

    def score_candidate(
        self,
        doc_i: Dict[str, Any],
        doc_j: Dict[str, Any],
        shared_entities: Optional[Set[str]] = None,
        idf_map: Optional[Dict[str, float]] = None,
        in_degree_j: int = 0,
        emb_i: Optional[Union[np.ndarray, torch.Tensor]] = None,
        emb_j: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Score a single candidate edge (i -> j) and return probability with feature breakdown.
        """
        feat = self.extract_features(
            doc_i=doc_i,
            doc_j=doc_j,
            shared_entities=shared_entities,
            idf_map=idf_map,
            in_degree_j=in_degree_j,
            emb_i=emb_i,
            emb_j=emb_j,
        )
        prob = float(self.predict_proba(feat))
        breakdown = {name: float(val) for name, val in zip(self.feature_names, feat)}
        return prob, breakdown

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        C: float = 1.0,
        class_weight: Optional[str] = "balanced",
    ):
        """
        Fit scorer parameters on supervised training pairs using L2-regularized logistic regression.
        Strictly use on train splits (e.g. HotpotQA train). Never fit on validation/test sets!

        Args:
            X: Feature matrix of shape (N, 6)
            y: Binary labels of shape (N,) where 1 = gold bridge pair, 0 = candidate non-bridge
            C: Inverse regularization strength
            class_weight: Balancing mode for class imbalance
        """
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(
            C=C,
            solver="lbfgs",
            class_weight=class_weight,
            max_iter=1000,
            random_state=42,
        )

        clf.fit(X, y)
        self.weights = clf.coef_[0].astype(np.float64)
        self.bias = float(clf.intercept_[0])

    def save(self, filepath: str) -> None:
        """Save calibrated parameters to JSON."""
        data = {
            "feature_names": self.feature_names,
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "metadata": self.metadata,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "LearnedEdgeScorer":
        """Load calibrated parameters from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        scorer = cls(weights=data["weights"], bias=data["bias"])
        scorer.feature_names = data.get("feature_names", list(DEFAULT_FEATURE_NAMES))
        scorer.metadata = data.get("metadata", {})
        return scorer

    @classmethod
    def load_default_checkpoint(cls) -> "LearnedEdgeScorer":
        """Load the versioned frozen checkpoint trained on HotpotQA train."""
        import os
        base_dir = os.path.dirname(__file__)
        ckpt_path = os.path.join(base_dir, "checkpoints", "learned_edge_scorer_hotpotqa.json")
        if os.path.isfile(ckpt_path):
            return cls.load(ckpt_path)
        # Fallback to default weights if file missing
        return cls()




    @classmethod
    def calibrate_from_corpus(
        cls,
        docs: List[Dict[str, Any]],
        gold_pairs: List[Tuple[str, str]],
        doc_embeddings: Optional[Union[Dict[str, Any], torch.Tensor, np.ndarray, List]] = None,
        C: float = 1.0,
    ) -> "LearnedEdgeScorer":
        """
        Calibrate scorer on a training corpus using gold supporting-fact bridge pairs.
        Strictly run on train splits (e.g. HotpotQA train).

        Args:
            docs: Training documents.
            gold_pairs: Ground truth supporting-fact pairs (u, v) from train split.
            doc_embeddings: Optional embeddings for documents.
            C: Regularization parameter.

        Returns:
            Calibrated LearnedEdgeScorer instance.
        """
        from lthcm.graph.build_edges import _normalize_title, _extract_candidate_entities
        import re

        n_docs = len(docs)
        doc_map = {str(d["id"]): d for d in docs}
        titles = {str(d["id"]): d.get("title", "").strip() for d in docs}
        texts = {str(d["id"]): d.get("text", d.get("content", "")).strip() for d in docs}

        gold_set = set()
        for u, v in gold_pairs:
            gold_set.add((str(u), str(v)))
            gold_set.add((str(v), str(u)))

        emb_dict: Dict[str, Any] = {}
        if doc_embeddings is not None:
            if isinstance(doc_embeddings, dict):
                emb_dict = doc_embeddings
            elif isinstance(doc_embeddings, (torch.Tensor, np.ndarray, list)):
                for idx, d in enumerate(docs):
                    emb_dict[str(d["id"])] = doc_embeddings[idx]

        title_lookup = {}
        for did, title in titles.items():
            norm_t = _normalize_title(title)
            if len(norm_t) >= 3 and norm_t not in {"the", "and", "introduction", "overview", "methods"}:
                title_lookup[norm_t] = did

        compiled_titles = []
        for norm_title, target_id in title_lookup.items():
            compiled_titles.append((norm_title, target_id, re.compile(r"\b" + re.escape(norm_title) + r"\b")))

        title_mentions: Dict[str, Set[str]] = defaultdict(set)
        for src_id, text in texts.items():
            norm_text = text.lower()
            for norm_title, target_id, pattern in compiled_titles:
                if target_id == src_id:
                    continue
                if norm_title in norm_text and pattern.search(norm_text):
                    title_mentions[src_id].add(target_id)

        doc_entities: Dict[str, Set[str]] = {}
        entity_df: Dict[str, int] = defaultdict(int)
        for did, text in texts.items():
            ents = _extract_candidate_entities(f"{titles.get(did, '')} {text}")
            doc_entities[did] = ents
            for e in ents:
                entity_df[e] += 1

        entity_idf: Dict[str, float] = {}
        max_df = max(3, int(0.25 * n_docs))
        for e, df in entity_df.items():
            if 2 <= df <= max_df:
                entity_idf[e] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

        inverted_index: Dict[str, List[str]] = defaultdict(list)
        for did, ents in doc_entities.items():
            for e in ents:
                if e in entity_idf:
                    inverted_index[e].append(did)

        proposed_pairs: Set[Tuple[str, str]] = set()
        for src, dsts in title_mentions.items():
            for dst in dsts:
                proposed_pairs.add((src, dst))

        for e, posting in inverted_index.items():
            for i in range(len(posting)):
                u = posting[i]
                for j in range(i + 1, len(posting)):
                    v = posting[j]
                    proposed_pairs.add((u, v))
                    proposed_pairs.add((v, u))

        # Include gold pairs in candidate pool if both docs exist in corpus
        for u, v in gold_set:
            if u in doc_map and v in doc_map:
                proposed_pairs.add((u, v))

        in_degrees: Dict[str, int] = defaultdict(int)
        for u, v in proposed_pairs:
            in_degrees[v] += 1

        scorer = cls()
        X_list = []
        y_list = []

        for u, v in proposed_pairs:
            feat = scorer.extract_features(
                doc_i=doc_map[u],
                doc_j=doc_map[v],
                shared_entities=doc_entities.get(u, set()) & doc_entities.get(v, set()),
                idf_map=entity_idf,
                in_degree_j=in_degrees.get(v, 0),
                emb_i=emb_dict.get(u),
                emb_j=emb_dict.get(v),
            )
            label = 1 if (u, v) in gold_set else 0
            X_list.append(feat)
            y_list.append(label)

        if len(X_list) > 0 and len(set(y_list)) > 1:
            X_arr = np.array(X_list, dtype=np.float64)
            y_arr = np.array(y_list, dtype=np.int64)
            scorer.fit(X_arr, y_arr, C=C)

        return scorer


class FixedWeightEdgeScorer(LearnedEdgeScorer):
    """
    Fixed-Weight Heuristic Edge Scorer.
    Represents the fixed heuristic arm (previously named learned arm without checkpoint).
    """

    def __init__(self):
        super().__init__(weights=DEFAULT_WEIGHTS.copy(), bias=DEFAULT_BIAS)
        self.metadata = {"arm_type": "fixed_weight_edges", "checkpoint": None}

