"""Logger for Anadrome continuation trees and decision records."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from dianoia.anadrome.contracts import ReplayTreeRecord

DEFAULT_REPLAY_TREES_PATH = os.path.expanduser("~/.dianoia/replay_trees.jsonl")


class AnadromeLogger:
    """Manages reading and appending decision-tree records in append-only JSONL format."""

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path or DEFAULT_REPLAY_TREES_PATH

    def append_tree(self, tree: ReplayTreeRecord) -> str:
        """Append a completed replay tree to the JSONL log file."""
        path = Path(self.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = tree.to_dict()
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
        return str(path)

    def iter_trees(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all recorded replay trees."""
        path = Path(self.log_path)
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def load_all_trees(self) -> List[Dict[str, Any]]:
        """Load all recorded trees into memory."""
        return list(self.iter_trees())
