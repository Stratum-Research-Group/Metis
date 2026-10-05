import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
from kybernetes.contracts.turn import Citation


class LTHCMAdapter:
    """
    Adapter for Long-Term Hierarchical Context Memory (LTHCM) and Stoma index.
    Always specifies an explicit `as_of` epoch timestamp to avoid temporal leakage.
    When operating without a configured external index, explicitly marks citations
    with is_simulated=True.
    When operating with a configured memory_root, validates and loads the real index
    snapshot, failing closed on missing, empty, or incompatible indexes.
    """

    def __init__(self, memory_root: Optional[str] = None):
        self.memory_root = memory_root
        self.is_simulated = True
        self.generation_id: Optional[str] = None
        self._real_passages: List[Dict[str, Any]] = []

        if self.memory_root is not None:
            self._load_real_index(self.memory_root)
        else:
            self._local_corpus = self._init_local_corpus()

    def _load_real_index(self, root_str: str) -> None:
        root = Path(root_str)
        if not root.exists():
            raise FileNotFoundError(f"Memory root path does not exist: {root_str}")
        if not root.is_dir():
            raise NotADirectoryError(f"Memory root is not a directory: {root_str}")

        files = list(root.iterdir())
        if not files:
            raise ValueError(f"Memory root directory is empty: {root_str}")

        current_pointer = root / "current.json"
        target_file = None
        data = None

        if current_pointer.is_file():
            try:
                with open(current_pointer, "r", encoding="utf-8") as f:
                    ptr_data = json.load(f)
            except Exception as e:
                raise ValueError(f"Corrupted current.json pointer in memory root {root_str}: {e}") from e

            gen_val = ptr_data.get("generation")
            if gen_val is None:
                raise ValueError(f"Corrupted generation pointer in {root_str}/current.json: missing generation number")
            try:
                gen_int = int(gen_val)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid generation ID '{gen_val}' in {root_str}/current.json: {e}") from e

            self.generation_id = f"gen-{gen_int}"
            gen_dir = root / "generations" / f"gen_{gen_int:06d}"
            if not gen_dir.is_dir():
                gen_dir = root / "generations" / f"gen_{gen_int}"
            if not gen_dir.is_dir():
                raise FileNotFoundError(f"Referenced generation directory does not exist: {gen_dir}")

            # Verify manifest hash if present in pointer
            expected_man_hash = ptr_data.get("manifest_hash")
            man_file = gen_dir / "manifest.json"
            if expected_man_hash and man_file.is_file():
                import hashlib
                actual_man_hash = hashlib.sha256(man_file.read_bytes()).hexdigest()
                if actual_man_hash != expected_man_hash:
                    raise ValueError(f"Manifest hash mismatch for generation {gen_int}: expected {expected_man_hash}, got {actual_man_hash}")

            # Locate passages file in generation directory
            if (gen_dir / "passages_export.json").is_file():
                target_file = gen_dir / "passages_export.json"
            elif (gen_dir / "passages.json").is_file():
                target_file = gen_dir / "passages.json"
            elif man_file.is_file():
                target_file = man_file
            else:
                raise ValueError(f"Generation directory {gen_dir} contains no passages_export.json or manifest.json")

            try:
                with open(target_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                raise ValueError(f"Corrupted passages file in {target_file}: {e}") from e
        else:
            # Fallback to direct root files: manifest.json, index.json, passages.json
            manifest_file = root / "manifest.json"
            index_file = root / "index.json"
            passages_file = root / "passages.json"

            if manifest_file.is_file():
                target_file = manifest_file
            elif index_file.is_file():
                target_file = index_file
            elif passages_file.is_file():
                target_file = passages_file
            else:
                raise ValueError(f"Incompatible index in memory root {root_str}: missing current.json, manifest.json, or index.json")

            try:
                with open(target_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                raise ValueError(f"Corrupted index in memory root {root_str}: {e}") from e

            # Extract generation identity if available
            gen_file = root / "current_generation.json"
            if gen_file.is_file():
                try:
                    with open(gen_file, "r", encoding="utf-8") as gf:
                        gen_data = json.load(gf)
                        gen_raw = gen_data.get("generation", gen_data.get("generation_id"))
                        if gen_raw is None:
                            raise ValueError("Missing generation identifier in current_generation.json")
                        self.generation_id = f"gen-{gen_raw}" if str(gen_raw).isdigit() else str(gen_raw)
                except Exception as e:
                    raise ValueError(f"Corrupt current_generation.json in {root_str}: {e}") from e
            elif isinstance(data, dict):
                gen_raw = data.get("manifest_version", data.get("generation_id", data.get("generation")))
                self.generation_id = f"gen-{gen_raw}" if (gen_raw is not None and str(gen_raw).isdigit()) else (str(gen_raw) if gen_raw is not None else None)
            else:
                self.generation_id = None

        # Extract passages: check list first to prevent AttributeError: 'list' object has no attribute 'get'
        passages_data = None
        if isinstance(data, list):
            passages_data = data
        elif isinstance(data, dict):
            passages_data = data.get("passages")
            if passages_data is None and "text" in data:
                passages_data = [data]
        else:
            raise ValueError(f"Unrecognized index structure in {target_file}")

        if not passages_data:
            raise ValueError(f"Memory index has 0 passages: {root_str}")

        loaded: List[Dict[str, Any]] = []
        if isinstance(passages_data, dict):
            for pid, pdata in passages_data.items():
                text_content = pdata.get("text", pdata.get("chunk_text", ""))
                created = pdata.get("ingested_at", pdata.get("created_epoch"))
                loaded.append({
                    "passage_id": pid,
                    "source_id": pdata.get("source_id", "unknown_source"),
                    "source_title": pdata.get("source_title", pdata.get("rel_path", "Document")),
                    "text": text_content,
                    "keywords": pdata.get("keywords", text_content.lower().split()),
                    "created_epoch": float(created) if created is not None else None,
                })
        elif isinstance(passages_data, list):
            for item in passages_data:
                text_content = item.get("text", item.get("chunk_text", ""))
                created = item.get("ingested_at", item.get("created_epoch"))
                loaded.append({
                    "passage_id": item.get("passage_id", f"p_{len(loaded)}"),
                    "source_id": item.get("source_id", "unknown_source"),
                    "source_title": item.get("source_title", item.get("rel_path", "Document")),
                    "text": text_content,
                    "keywords": item.get("keywords", text_content.lower().split()),
                    "created_epoch": float(created) if created is not None else None,
                })

        if not loaded or any(not p["text"].strip() for p in loaded):
            raise ValueError(f"Corrupted passage records without text in memory root: {root_str}")

        self._real_passages = loaded
        self.is_simulated = False
        self.capability = "lexical"

    def _init_local_corpus(self) -> List[Dict[str, Any]]:
        return [
            {
                "passage_id": "pass_metis_arch_01",
                "source_id": "doc_metis_overview",
                "source_title": "Metis Ecosystem Architecture",
                "text": "Kybernetes serves as the orchestrator and steersman of the Metis local AI stack, coordinating Hestia, LTHCM, Dianoia, Phylax, and Krisis behind a unified Prosopon interface.",
                "keywords": ["kybernetes", "metis", "orchestrator", "prosopon", "hestia"],
                "created_epoch": 1700000000.0,
            },
            {
                "passage_id": "pass_phylax_sec_01",
                "source_id": "doc_phylax_specs",
                "source_title": "Phylax Security Gateway Specification",
                "text": "Phylax enforces capability authorization at the point of action with immutable SHA-256 hash-chained audit trails and one-use approval digests.",
                "keywords": ["phylax", "security", "approval", "audit", "capability"],
                "created_epoch": 1705000000.0,
            },
            {
                "passage_id": "pass_dianoia_walk_01",
                "source_id": "doc_dianoia_deliberation",
                "source_title": "Dianoia Graph Walk Deliberation",
                "text": "Dianoia conducts multi-step graph deliberation across connected evidence nodes, falling back cleanly to flat passage reranking when 0 edges are traversed.",
                "keywords": ["dianoia", "graph", "deliberation", "multi-hop", "walk", "edges"],
                "created_epoch": 1710000000.0,
            },
            {
                "passage_id": "pass_krisis_gate_01",
                "source_id": "doc_krisis_evaluation",
                "source_title": "Krisis Answer Selection & Scoring",
                "text": "Krisis scores answer candidates using 6-decimal-place half-even rounding, exact candidate deduplication, and strict eligibility gates.",
                "keywords": ["krisis", "scoring", "gates", "selection", "evaluation"],
                "created_epoch": 1715000000.0,
            },
        ]

    def retrieve(
        self,
        query: str,
        as_of: Optional[float] = None,
        top_k: int = 3,
    ) -> List[Citation]:
        """
        Retrieve relevant memory passages subject to explicit temporal cut-off `as_of`.
        as_of: POSIX epoch timestamp in seconds. Passages created after as_of are excluded.
        """
        current_time = as_of if as_of is not None else time.time()
        q_tokens = set(query.lower().split())

        corpus = self._real_passages if not self.is_simulated else self._local_corpus

        scored_passages = []
        for item in corpus:
            if as_of is not None:
                if item["created_epoch"] is None or item["created_epoch"] > as_of:
                    continue
            elif item["created_epoch"] is not None and item["created_epoch"] > current_time:
                continue

            matches = sum(1 for kw in item["keywords"] if kw in query.lower())
            token_overlap = sum(1 for tok in q_tokens if tok in item["text"].lower())
            total_score = matches * 0.4 + token_overlap * 0.1

            if total_score > 0:
                scored_passages.append((total_score, item))

        scored_passages.sort(key=lambda x: x[0], reverse=True)
        results: List[Citation] = []
        for idx, (score, item) in enumerate(scored_passages[:top_k], 1):
            results.append(
                Citation(
                    citation_id=f"[{idx}]",
                    passage_id=item["passage_id"],
                    source_id=item["source_id"],
                    source_title=item["source_title"],
                    text=item["text"],
                    score=round(float(score), 3),
                    is_simulated=self.is_simulated,
                )
            )

        return results
