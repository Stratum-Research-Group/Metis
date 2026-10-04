"""Unit tests for baseline ladder models and runner."""

from pathlib import Path
import pytest

from paideia.data.router_loader import load_router_dataset
from paideia.data.splitter import split_grouped_dataset, derive_question_family_key
from paideia.baselines.majority import MajorityBaseline
from paideia.baselines.regex_router import RegexRouterBaseline
from paideia.baselines.lexical import LexicalLogisticBaseline
from paideia.baselines.embedding import SmallEmbeddingBaseline
from paideia.baselines.ladder import run_baseline_ladder

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_baseline_models():
    fixture_path = FIXTURES_DIR / "lthcm_sample.json"
    records, _ = load_router_dataset(fixture_path)

    # 1. Majority
    maj = MajorityBaseline().fit(records)
    pred_maj = maj.predict(records)
    assert len(pred_maj) == len(records)
    assert all(p == "comparison" for p in pred_maj)  # 3 comp vs 2 bridge

    # 2. Regex
    regex = RegexRouterBaseline().fit(records)
    pred_regex = regex.predict(records)
    assert len(pred_regex) == len(records)
    # Question 1: "Were Scott Derrickson and Ed Wood of the same nationality?" -> comparison
    assert pred_regex[0] == "comparison"
    # Question 2: "What government position was held by..." -> bridge
    assert pred_regex[1] == "bridge"

    # 3. Lexical
    lex = LexicalLogisticBaseline().fit(records)
    pred_lex = lex.predict(records)
    assert len(pred_lex) == len(records)

    # 4. Embedding
    emb = SmallEmbeddingBaseline().fit(records)
    pred_emb = emb.predict(records)
    assert len(pred_emb) == len(records)


def test_baseline_ladder_runner():
    fixture_path = FIXTURES_DIR / "lthcm_sample.json"
    records, _ = load_router_dataset(fixture_path)

    # Duplicate records to have enough samples for train/eval
    augmented = records * 4
    splits = split_grouped_dataset(
        augmented,
        group_key_fn=lambda r: derive_question_family_key(r.question),
        train_ratio=0.60,
        dev_ratio=0.20,
        test_ratio=0.20,
        seed=42,
    )

    report = run_baseline_ladder(
        train_items=splits.train,
        eval_items=splits.dev,
        target_extractor=lambda r: r.gold_type,
        task_name="test_ladder",
    )

    assert len(report.results) == 4
    names = [r["name"] for r in report.results]
    assert "majority_baseline" in names
    assert "lthcm_regex_router" in names
    assert "lexical_logistic_baseline" in names
    assert "small_embedding_baseline" in names

    table_str = report.to_table()
    assert "Model" in table_str
    assert "Accuracy" in table_str
