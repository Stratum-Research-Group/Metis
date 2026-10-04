"""
Stoma File Watcher and Directory Synchronizer.
Provides event debouncing, path validation, and background reconciliation.
Implements transactional move/rename handling without stale IDs or resurrection (Finding 2).
Wires retries with exponential backoff for moves, deletions, and ingests, with visible dead-letter states.
Caps diagnostic memory while keeping exact failure counts visible.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from stoma.indexer import StomaIndexer

logger = logging.getLogger(__name__)


class StomaWatchEventHandler(FileSystemEventHandler):
    """
    Watches for file modification, creation, deletion, and rename events.
    Debounces bursts of write events and enforces path safety boundaries.
    """

    MAX_RECORDED_ERRORS: int = 50
    MAX_PENDING: int = 1000
    MAX_RETRIES: int = 1000
    MAX_DEAD_LETTERS: int = 500

    def __init__(self, indexer: StomaIndexer, debounce_sec: float = 0.5):
        super().__init__()
        self.indexer = indexer
        self.config = indexer.config
        self.debounce_sec = debounce_sec
        self._pending_changes: Dict[Path, float] = {}  # Path -> scheduled process epoch time
        self._pending_deletions: Dict[Path, float] = {}  # Path -> scheduled process epoch time
        self._pending_moves: Dict[Tuple[Path, Path], float] = {}  # (src, dest) -> due epoch time

        self._retries: Dict[Path, int] = {}
        self._deletion_retries: Dict[Path, int] = {}
        self._move_retries: Dict[Tuple[Path, Path], int] = {}

        self._dead_letter_ingests: List[Dict[str, Any]] = []
        self._dead_letter_deletions: List[Dict[str, Any]] = []
        self._dead_letter_moves: List[Dict[str, Any]] = []

        self.total_scheduled_changes: int = 0
        self.dropped_scheduled_changes: int = 0
        self.total_scheduled_deletions: int = 0
        self.dropped_scheduled_deletions: int = 0
        self.total_scheduled_moves: int = 0
        self.dropped_scheduled_moves: int = 0

        self.total_dead_letters: int = 0
        self.dropped_dead_letters: int = 0

        self.event_errors: List[Dict[str, Any]] = []
        self.total_event_errors: int = 0
        self.dropped_event_errors: int = 0

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._process_events_loop, daemon=True)
        self._worker_thread.start()

    def record_error(self, path: Path, op: str, error: str) -> None:
        """Record an operation failure with capped diagnostic memory."""
        with self._lock:
            self.total_event_errors += 1
            entry = {
                "path": str(path),
                "op": op,
                "error": error,
                "timestamp": time.time(),
            }
            if len(self.event_errors) < self.MAX_RECORDED_ERRORS:
                self.event_errors.append(entry)
            else:
                self.dropped_event_errors += 1

    def get_dead_letters(self) -> Dict[str, Any]:
        """Return visible dead-letter states for exhausted operations."""
        with self._lock:
            return {
                "ingests": list(self._dead_letter_ingests),
                "deletions": list(self._dead_letter_deletions),
                "moves": list(self._dead_letter_moves),
                "total_dead_letters": (
                    len(self._dead_letter_ingests)
                    + len(self._dead_letter_deletions)
                    + len(self._dead_letter_moves)
                ),
                "dropped_dead_letters": self.dropped_dead_letters,
            }

    def get_queue_stats(self) -> Dict[str, Any]:
        """Return counts and statistics for all retained watcher structures."""
        with self._lock:
            return {
                "pending_changes_count": len(self._pending_changes),
                "pending_deletions_count": len(self._pending_deletions),
                "pending_moves_count": len(self._pending_moves),
                "total_scheduled_changes": self.total_scheduled_changes,
                "dropped_scheduled_changes": self.dropped_scheduled_changes,
                "total_scheduled_deletions": self.total_scheduled_deletions,
                "dropped_scheduled_deletions": self.dropped_scheduled_deletions,
                "total_scheduled_moves": self.total_scheduled_moves,
                "dropped_scheduled_moves": self.dropped_scheduled_moves,
                "dead_letter_ingests_count": len(self._dead_letter_ingests),
                "dead_letter_deletions_count": len(self._dead_letter_deletions),
                "dead_letter_moves_count": len(self._dead_letter_moves),
                "total_dead_letters": self.total_dead_letters,
                "dropped_dead_letters": self.dropped_dead_letters,
                "total_event_errors": self.total_event_errors,
                "dropped_event_errors": self.dropped_event_errors,
            }

    def get_overflow_summary(self) -> Dict[str, Any]:
        """Return counts of all items dropped due to explicit capacity bounds."""
        with self._lock:
            return {
                "dropped_scheduled_changes": self.dropped_scheduled_changes,
                "dropped_scheduled_deletions": self.dropped_scheduled_deletions,
                "dropped_scheduled_moves": self.dropped_scheduled_moves,
                "dropped_dead_letters": self.dropped_dead_letters,
                "dropped_event_errors": self.dropped_event_errors,
                "total_dropped": (
                    self.dropped_scheduled_changes
                    + self.dropped_scheduled_deletions
                    + self.dropped_scheduled_moves
                    + self.dropped_dead_letters
                    + self.dropped_event_errors
                ),
            }

    def stop(self) -> None:
        """Signal worker thread to stop."""
        self._stop_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._schedule_change(Path(event.src_path))

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._schedule_change(Path(event.src_path))

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._schedule_deletion(Path(event.src_path))

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        src = Path(event.src_path)
        dest = Path(event.dest_path)

        # Cancel any pending events for src and dest so old path cannot be resurrected
        with self._lock:
            self._pending_changes.pop(src, None)
            self._pending_deletions.pop(src, None)
            self._pending_changes.pop(dest, None)
            self._pending_deletions.pop(dest, None)

        logger.info("Watchdog detected move event: %s -> %s", src, dest)
        self._schedule_move(src, dest)

    def _schedule_move(self, src: Path, dest: Path, delay: Optional[float] = None) -> None:
        """Schedule a move event preserving (src, dest) move identity with explicit bound."""
        wait = delay if delay is not None else self.debounce_sec
        due_time = time.time() + wait
        with self._lock:
            self.total_scheduled_moves += 1
            if len(self._pending_moves) >= self.MAX_PENDING and (src, dest) not in self._pending_moves:
                self.dropped_scheduled_moves += 1
                logger.warning("Pending moves queue full (cap=%d); dropping schedule for %s -> %s", self.MAX_PENDING, src, dest)
            else:
                self._pending_moves[(src, dest)] = due_time

    def _schedule_move_retry(self, src: Path, dest: Path, err: str) -> None:
        """Retry a move event with bounded exponential backoff (up to 3 retries)."""
        with self._lock:
            count = self._move_retries.get((src, dest), 0)
            if count < 3:
                self._move_retries[(src, dest)] = count + 1
                backoff = self.debounce_sec * (2 ** count)
                logger.warning("Scheduling move retry %d for %s -> %s in %.2fs", count + 1, src, dest, backoff)
                self._pending_moves[(src, dest)] = time.time() + backoff
            else:
                logger.error("Max retries exceeded for move %s -> %s; entering dead-letter state.", src, dest)
                self._move_retries.pop((src, dest), None)
                self.total_dead_letters += 1
                if len(self._dead_letter_moves) < self.MAX_DEAD_LETTERS:
                    self._dead_letter_moves.append({
                        "src": str(src),
                        "dest": str(dest),
                        "error": err,
                        "exhausted_at": time.time(),
                    })
                else:
                    self.dropped_dead_letters += 1

    def _schedule_change(self, path: Path, delay: Optional[float] = None) -> None:
        """Debounce write events for a path with explicit bound."""
        try:
            if not self.config.is_path_allowed(path):
                return
        except Exception:
            return

        wait = delay if delay is not None else self.debounce_sec
        due_time = time.time() + wait
        with self._lock:
            # Preserve active unresolved moves: do not overwrite a pending move destination
            is_move_dest = any(dest == path for (_, dest) in self._pending_moves) or any(dest == path for (_, dest) in self._move_retries)
            if is_move_dest:
                logger.info("Path %s is destination of an unresolved move; preserving move identity.", path)
                return

            self._pending_deletions.pop(path, None)
            self.total_scheduled_changes += 1
            if len(self._pending_changes) >= self.MAX_PENDING and path not in self._pending_changes:
                self.dropped_scheduled_changes += 1
                logger.warning("Pending changes queue full (cap=%d); dropping schedule for %s", self.MAX_PENDING, path)
            else:
                self._pending_changes[path] = due_time

    def _schedule_retry(self, path: Path, err: str = "") -> None:
        """Retry an ingest event with bounded exponential backoff (up to 3 retries)."""
        with self._lock:
            count = self._retries.get(path, 0)
            if count < 3:
                self._retries[path] = count + 1
                backoff = self.debounce_sec * (2 ** count)
                logger.warning("Scheduling retry %d for %s in %.2fs", count + 1, path, backoff)
                self._pending_changes[path] = time.time() + backoff
            else:
                logger.error("Max retries exceeded for %s; entering dead-letter state.", path)
                self._retries.pop(path, None)
                self.total_dead_letters += 1
                if len(self._dead_letter_ingests) < self.MAX_DEAD_LETTERS:
                    self._dead_letter_ingests.append({
                        "path": str(path),
                        "error": err,
                        "exhausted_at": time.time(),
                    })
                else:
                    self.dropped_dead_letters += 1

    def _schedule_deletion(self, path: Path, delay: Optional[float] = None) -> None:
        """Schedule a file deletion with due timestamp and explicit capacity bound."""
        try:
            if not self.config.is_path_allowed(path):
                return
        except Exception:
            return

        wait = delay if delay is not None else self.debounce_sec
        due_time = time.time() + wait
        with self._lock:
            self._pending_changes.pop(path, None)
            self._retries.pop(path, None)
            self.total_scheduled_deletions += 1
            if len(self._pending_deletions) >= self.MAX_PENDING and path not in self._pending_deletions:
                self.dropped_scheduled_deletions += 1
                logger.warning("Pending deletions queue full (cap=%d); dropping schedule for %s", self.MAX_PENDING, path)
            else:
                self._pending_deletions[path] = due_time

    def _schedule_deletion_retry(self, path: Path, err: str) -> None:
        """Retry a deletion event with bounded backoff and due timestamp."""
        with self._lock:
            count = self._deletion_retries.get(path, 0)
            if count < 3:
                self._deletion_retries[path] = count + 1
                backoff = self.debounce_sec * (2 ** count)
                logger.warning("Scheduling deletion retry %d for %s in %.2fs", count + 1, path, backoff)
                self._pending_deletions[path] = time.time() + backoff
            else:
                logger.error("Max retries exceeded for deletion %s; entering dead-letter state.", path)
                self._deletion_retries.pop(path, None)
                self.total_dead_letters += 1
                if len(self._dead_letter_deletions) < self.MAX_DEAD_LETTERS:
                    self._dead_letter_deletions.append({
                        "path": str(path),
                        "error": err,
                        "exhausted_at": time.time(),
                    })
                else:
                    self.dropped_dead_letters += 1

    def _process_events_loop(self) -> None:
        """Background loop to process debounced events safely with error boundaries."""
        while not self._stop_event.is_set():
            time.sleep(0.05)
            now = time.time()

            due_changes: List[Path] = []
            due_deletions: List[Path] = []
            due_moves: List[Tuple[Path, Path]] = []

            with self._lock:
                for path, process_time in list(self._pending_changes.items()):
                    if now >= process_time:
                        due_changes.append(path)
                        del self._pending_changes[path]

                for path, process_time in list(self._pending_deletions.items()):
                    if now >= process_time:
                        due_deletions.append(path)
                        del self._pending_deletions[path]

                for (src, dest), process_time in list(self._pending_moves.items()):
                    if now >= process_time:
                        due_moves.append((src, dest))
                        del self._pending_moves[(src, dest)]

            # 1. Process moves safely preserving move identity
            for (m_src, m_dest) in due_moves:
                try:
                    rep = self.indexer.move_file(m_src, m_dest)
                    if rep.status == "failed":
                        err = rep.error_message or "Move failed"
                        logger.error("Failed move event %s -> %s: %s", m_src, m_dest, err)
                        self.record_error(m_dest, "move", err)
                        self._schedule_move_retry(m_src, m_dest, err)
                    else:
                        with self._lock:
                            self._move_retries.pop((m_src, m_dest), None)
                except Exception as e:
                    logger.error("Unhandled exception moving %s -> %s: %s", m_src, m_dest, e)
                    self.record_error(m_dest, "move", str(e))
                    self._schedule_move_retry(m_src, m_dest, str(e))

            # 2. Process deletions safely distinguishing false/no-op from actual success
            for del_path in due_deletions:
                try:
                    deleted = self.indexer.delete_file(del_path)
                    with self._lock:
                        self._deletion_retries.pop(del_path, None)
                    if not deleted:
                        logger.info("Deletion no-op for %s: path not indexed in manifest.", del_path)
                except Exception as e:
                    logger.error("Error processing deletion for %s: %s", del_path, e)
                    self.record_error(del_path, "delete", str(e))
                    self._schedule_deletion_retry(del_path, str(e))

            # 3. Process changes safely
            for chg_path in due_changes:
                if not chg_path.is_file():
                    continue

                # Guard against partial writes: ensure file size is stable
                try:
                    s1 = chg_path.stat().st_size
                    time.sleep(0.05)
                    s2 = chg_path.stat().st_size
                    if s1 != s2:
                        # Still being written, re-schedule
                        self._schedule_change(chg_path)
                        continue
                except OSError:
                    continue

                try:
                    rep = self.indexer.ingest_file(chg_path)
                    if rep.status == "failed":
                        err = rep.error_message or "Ingestion failed"
                        logger.warning("Ingestion failed for %s: %s", chg_path, err)
                        self.record_error(chg_path, "ingest", err)
                        self._schedule_retry(chg_path, err)
                    elif rep.status == "skipped_unsupported":
                        # Permanent outcome: do not retry
                        with self._lock:
                            self._retries.pop(chg_path, None)
                    else:
                        with self._lock:
                            self._retries.pop(chg_path, None)
                except Exception as e:
                    logger.error("Unhandled exception ingesting %s: %s", chg_path, e)
                    self.record_error(chg_path, "ingest", str(e))
                    self._schedule_retry(chg_path, str(e))


class StomaWatcher:
    """Manager for directory watching and full reconciliation."""

    MAX_FAILED_DETAILS: int = 50

    def __init__(self, indexer: StomaIndexer, debounce_sec: float = 0.5):
        self.indexer = indexer
        self.config = indexer.config
        self.debounce_sec = debounce_sec
        self.observer = Observer()
        self.handler = StomaWatchEventHandler(indexer, debounce_sec=debounce_sec)
        self._is_watching = False

    def start(self, watch_paths: Optional[List[Path]] = None) -> None:
        """Start watching paths."""
        targets = watch_paths or self.config.allowed_roots
        for target in targets:
            target = target.resolve()
            if target.is_dir() and self.config.is_path_allowed(target):
                logger.info("Watching directory: %s", target)
                self.observer.schedule(self.handler, str(target), recursive=True)

        self.observer.start()
        self._is_watching = True

    def stop(self) -> None:
        """Stop watching."""
        if self._is_watching:
            self.handler.stop()
            self.observer.stop()
            self.observer.join(timeout=2.0)
            self._is_watching = False

    def reconcile(self) -> Dict[str, Any]:
        """
        Perform a full manual reconciliation:
        1. Ingest/verify all existing files across allowed roots with visible counts.
        2. Preserve valid existing passages if file extraction fails.
        3. Detect deleted files whose paths no longer exist on disk, and clean them.
        4. Cap diagnostic memory for failure details while accurately counting all failures.
        """
        logger.info("Starting manual full reconciliation...")
        disk_source_ids: Set[str] = set()
        total_scanned = 0
        files_indexed = 0
        files_unchanged = 0
        files_failed = 0
        files_skipped_unsupported = 0
        failed_details: List[Dict[str, Any]] = []
        failed_details_dropped = 0

        # Scan disks
        for root in self.config.allowed_roots:
            if not root.is_dir():
                continue
            for r_str, dirs, files in os.walk(root, followlinks=False):
                dirs[:] = [
                    d for d in dirs
                    if d not in self.config.forbidden_dir_names
                    and self.config.is_path_allowed(Path(r_str) / d)
                ]
                for f in files:
                    fpath = Path(r_str) / f
                    if self.config.is_path_allowed(fpath):
                        loc = self.config.locate_path(fpath)
                        if loc:
                            sid = f"{loc[0]}:{loc[2]}"
                            # Add to disk_source_ids so physical presence prevents erroneous deletion
                            disk_source_ids.add(sid)
                            total_scanned += 1
                            try:
                                rep = self.indexer.ingest_file(fpath)
                                if rep.status == "indexed":
                                    files_indexed += 1
                                elif rep.status == "skipped_unchanged":
                                    files_unchanged += 1
                                elif rep.status == "skipped_unsupported":
                                    files_skipped_unsupported += 1
                                elif rep.status == "failed":
                                    files_failed += 1
                                    detail = {
                                        "path": str(fpath),
                                        "source_id": sid,
                                        "error": rep.error_message or "Ingestion failed",
                                    }
                                    if len(failed_details) < self.MAX_FAILED_DETAILS:
                                        failed_details.append(detail)
                                    else:
                                        failed_details_dropped += 1
                            except Exception as e:
                                files_failed += 1
                                detail = {
                                    "path": str(fpath),
                                    "source_id": sid,
                                    "error": str(e),
                                }
                                if len(failed_details) < self.MAX_FAILED_DETAILS:
                                    failed_details.append(detail)
                                else:
                                    failed_details_dropped += 1

        # Check for deleted files currently in manifest
        deleted_count = 0
        manifest_source_ids = set(self.indexer.manifest.sources.keys())
        missing_from_disk = manifest_source_ids - disk_source_ids

        for missing_sid in missing_from_disk:
            try:
                self.indexer.delete_file(missing_sid)
                deleted_count += 1
            except Exception as e:
                logger.error("Failed to delete missing file %s: %s", missing_sid, e)

        return {
            "total_files_scanned": total_scanned,
            "files_indexed": files_indexed,
            "files_unchanged": files_unchanged,
            "files_skipped_unsupported": files_skipped_unsupported,
            "files_failed": files_failed,
            "failed_details": failed_details,
            "failed_details_dropped": failed_details_dropped,
            "missing_files_cleaned": deleted_count,
            "active_manifest_sources": len(self.indexer.manifest.sources),
            "active_manifest_passages": len(self.indexer.manifest.passages),
        }

    def get_queue_stats(self) -> Dict[str, Any]:
        """Return counts and statistics for all retained watcher structures."""
        return self.handler.get_queue_stats()

    def get_overflow_summary(self) -> Dict[str, Any]:
        """Return counts of all items dropped due to explicit capacity bounds."""
        return self.handler.get_overflow_summary()

    def reconcile_overflowed_work(self) -> Dict[str, Any]:
        """
        Visible recovery path after capacity overflow:
        Runs full directory reconciliation to detect and ingest any dropped changes.
        """
        return self.reconcile()

