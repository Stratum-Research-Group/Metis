"""Tests for Dokime CLI entry point and subcommands."""

import json
from pathlib import Path
import pytest

from dokime.cli import main
from dokime.manifest import sha256_file


def test_cli_tasks(capsys):
    ret = main(["tasks"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Discovered tasks:" in captured.out
    assert "arc_easy_25" in captured.out


def test_cli_run_help():
    with pytest.raises(SystemExit) as exc_info:
        main(["run", "--help"])
    assert exc_info.value.code == 0


def test_cli_compare_help():
    with pytest.raises(SystemExit) as exc_info:
        main(["compare", "--help"])
    assert exc_info.value.code == 0


def test_cli_compare_execution(tmp_path: Path, capsys):
    runs_dir = tmp_path / "runs"
    # Create two runs
    for run_name, score in [("run_1", 1.0), ("run_2", 0.0)]:
        rdir = runs_dir / run_name
        rdir.mkdir(parents=True, exist_ok=True)
        with open(rdir / "results.jsonl", "w", encoding="utf-8") as f:
            for i in range(10):
                f.write(
                    json.dumps(
                        {
                            "run_id": run_name,
                            "task_id": "arc_easy_25",
                            "query_id": f"q_{i}",
                            "model_id": "test_m",
                            "expected": "A",
                            "score": score,
                        }
                    )
                    + "\n"
                )

        with open(rdir / "manifest.json", "w", encoding="utf-8") as mf:
            json.dump(
                {
                    "run_id": run_name,
                    "model_ids": ["test_m"],
                    "task_ids": ["arc_easy_25"],
                    "checksums": {
                        "dataset_arc_easy_25": "common_hash_123",
                        "results_jsonl": sha256_file(rdir / "results.jsonl"),
                    },
                },
                mf,
                indent=2,
            )

    ret = main(["compare", "run_1", "run_2", "--model", "test_m", "--output-dir", str(runs_dir)])
    assert ret == 0
    captured = capsys.readouterr()
    assert "== Dokime Regression Compare ==" in captured.out
    assert "[REGRESSION]" in captured.out
    assert "Paired Successes:  10" in captured.out


def test_format_helpers():
    from dokime.cli import format_stat_value, format_interval

    assert format_stat_value(None) == "N/A"
    assert format_stat_value(0.0, sign=False) == "0.0000"
    assert format_stat_value(0.0, sign=True) == "+0.0000"
    assert format_stat_value(0.00001, sign=True) == "+0.00001"
    assert format_stat_value(1e-9, sign=True) == "+1.0000e-09"
    assert format_stat_value(-1e-9, sign=True) == "-1.0000e-09"

    assert format_interval(None, 0.5) == "[N/A, N/A]"
    assert format_interval(0.0, 0.0, sign=True) == "[+0.0000, +0.0000]"
    assert format_interval(0.0, 0.0, sign=False) == "[0.0000, 0.0000]"
    assert format_interval(0.00001, 0.00001, sign=True) == "[+0.00001, +0.00001]"
    assert format_interval(1e-9, 1e-9, sign=True) == "[+1.0000e-09, +1.0000e-09]"
    assert format_interval(-1e-9, -1e-9, sign=True) == "[-1.0000e-09, -1.0000e-09]"
    assert format_interval(-1e-9, 1e-9, sign=True) == "[-1.0000e-09, +1.0000e-09]"
    assert format_interval(0.0, 1e-9, sign=True) == "[+0.0000, +1.0000e-09]"

