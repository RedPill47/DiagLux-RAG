"""End-to-end runner tests with MockClient on a tiny synthetic dataset.

Covers: preds rows matching the contract schema, the config sidecar,
resume behavior, unparseable accounting, retrieval (rag) context wiring,
and the random baseline. No network, no real LLM calls.
"""

import json

import pytest

from answering_testutils import (
    make_questions,
    read_jsonl,
    write_chunks,
    write_questions,
    write_rankings,
    write_texts_dir,
)
from diaglux.answering.clients import MockClient
from diaglux.answering.prompts import prompt_template_hash
from diaglux.answering.runner import RunConfig, compute_config_id, run

# Contract keys for one preds row (docs/CONTRACTS.md).
PRED_KEYS = {
    "question_id", "system", "setting", "k", "model", "context_chunk_ids",
    "raw_output", "parsed_letter", "parse_status", "semantic_choice",
    "is_correct", "timestamp",
}

# Gold letters of the fixture questions, in file order:
# text1_q00 -> B, text1_q01 -> A, text2_q00 -> D, text2_q01 -> C.


@pytest.fixture()
def workspace(tmp_path):
    questions_path = tmp_path / "questions.jsonl"
    questions = write_questions(questions_path)
    texts_dir = write_texts_dir(tmp_path / "Texts")
    chunks_path = tmp_path / "corpus_chunks_overlap.jsonl"
    write_chunks(chunks_path)
    rankings_path = tmp_path / "rankings_text_restricted_bm25_overlap.jsonl"
    write_rankings(rankings_path, questions)
    return {
        "questions": questions,
        "questions_path": questions_path,
        "texts_dir": texts_dir,
        "chunks_path": chunks_path,
        "rankings_path": rankings_path,
        "out_dir": tmp_path / "runs",
    }


def oracle_config(ws):
    return RunConfig(
        system="oracle",
        provider="mock",
        model="mock-model",
        setting="none",
        k=None,
        questions_path=str(ws["questions_path"]),
        texts_dir=str(ws["texts_dir"]),
    )


def test_oracle_run_end_to_end(workspace):
    ws = workspace
    # q00: exact correct; q01: extracted correct; q02: unparseable;
    # q03: extracted but wrong (gold C, answers D -> distractor_span).
    client = MockClient(
        outputs=["B", "Answer: A", "Ech weess et net.", "**D**"]
    )
    summary = run(oracle_config(ws), client=client, out_dir=ws["out_dir"])

    assert summary["n_questions"] == 4
    assert summary["n_new"] == 4
    assert summary["n_skipped_resumed"] == 0
    assert summary["n_correct_new"] == 2
    assert summary["n_unparseable_new"] == 1
    assert summary["accuracy_new"] == pytest.approx(0.5)

    rows = read_jsonl(ws["out_dir"] / f"preds_{summary['config_id']}.jsonl")
    assert [r["question_id"] for r in rows] == [
        q["question_id"] for q in ws["questions"]
    ]
    for row in rows:
        assert set(row) == PRED_KEYS
        assert row["system"] == "oracle"
        assert row["setting"] == "none"
        assert row["k"] is None
        assert row["model"] == "mock-model"
        assert row["timestamp"].endswith("Z")

    by_qid = {r["question_id"]: r for r in rows}
    assert by_qid["text1_q00"]["parsed_letter"] == "B"
    assert by_qid["text1_q00"]["parse_status"] == "exact"
    assert by_qid["text1_q00"]["semantic_choice"] == "correct"
    assert by_qid["text1_q00"]["is_correct"] is True
    assert by_qid["text1_q00"]["context_chunk_ids"] == ["text1_full"]

    assert by_qid["text1_q01"]["parse_status"] == "extracted"
    assert by_qid["text1_q01"]["is_correct"] is True

    bad = by_qid["text2_q00"]
    assert bad["parsed_letter"] is None
    assert bad["parse_status"] == "unparseable"
    assert bad["semantic_choice"] is None
    assert bad["is_correct"] is False
    assert bad["raw_output"] == "Ech weess et net."

    wrong = by_qid["text2_q01"]
    assert wrong["parsed_letter"] == "D"
    assert wrong["semantic_choice"] == "distractor_span"  # via permutation
    assert wrong["is_correct"] is False

    # Each prompt was the open-book variant with the full clean body.
    assert len(client.calls) == 4
    for prompt in client.calls:
        assert "Use only the provided context." in prompt
        assert "Context: " in prompt


def test_config_sidecar_written(workspace):
    ws = workspace
    client = MockClient(default="A")
    summary = run(oracle_config(ws), client=client, out_dir=ws["out_dir"])
    sidecar_path = ws["out_dir"] / f"preds_{summary['config_id']}.config.json"
    assert sidecar_path.exists()
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["config_id"] == summary["config_id"]
    config = sidecar["config"]
    assert config["system"] == "oracle"
    assert config["model"] == "mock-model"
    assert config["seed"] == 13
    assert config["prompt_template_hash"] == prompt_template_hash()
    assert sidecar["client"]["client"] == "MockClient"
    # config_id is the documented short hash of the canonical config.
    assert summary["config_id"] == compute_config_id(oracle_config(ws))


def test_resume_skips_all_completed_questions(workspace):
    ws = workspace
    first = run(
        oracle_config(ws),
        client=MockClient(outputs=["B", "A", "D", "C"]),
        out_dir=ws["out_dir"],
    )
    preds_path = ws["out_dir"] / f"preds_{first['config_id']}.jsonl"
    before = preds_path.read_text(encoding="utf-8")

    # Identical config, different client outputs: everything must be skipped
    # and the preds file must be byte-identical.
    second_client = MockClient(default="Z?")
    second = run(oracle_config(ws), client=second_client, out_dir=ws["out_dir"])
    assert second["config_id"] == first["config_id"]
    assert second["n_new"] == 0
    assert second["n_skipped_resumed"] == 4
    assert second["n_unparseable_new"] == 0
    assert second["accuracy_new"] is None
    assert second_client.calls == []
    assert preds_path.read_text(encoding="utf-8") == before
    assert len(read_jsonl(preds_path)) == 4


def test_resume_fills_only_the_gap(workspace):
    ws = workspace
    first = run(
        oracle_config(ws),
        client=MockClient(default="A"),
        out_dir=ws["out_dir"],
        limit=2,
    )
    assert first["n_new"] == 2
    second = run(
        oracle_config(ws), client=MockClient(default="B"), out_dir=ws["out_dir"]
    )
    assert second["config_id"] == first["config_id"]
    assert second["n_new"] == 2
    assert second["n_skipped_resumed"] == 2
    rows = read_jsonl(ws["out_dir"] / f"preds_{first['config_id']}.jsonl")
    assert len(rows) == 4
    assert len({r["question_id"] for r in rows}) == 4


def test_all_unparseable_counted(workspace):
    ws = workspace
    client = MockClient(default="Dat ass eng Iddi.")  # never parses
    summary = run(oracle_config(ws), client=client, out_dir=ws["out_dir"])
    assert summary["n_unparseable_new"] == 4
    assert summary["n_correct_new"] == 0
    rows = read_jsonl(ws["out_dir"] / f"preds_{summary['config_id']}.jsonl")
    for row in rows:
        assert row["parsed_letter"] is None
        assert row["parse_status"] == "unparseable"
        assert row["semantic_choice"] is None
        assert row["is_correct"] is False


def test_rag_run_resolves_system_and_uses_topk_context(workspace):
    ws = workspace
    config = RunConfig(
        system="rag",
        provider="mock",
        model="mock-model",
        k=2,
        questions_path=str(ws["questions_path"]),
        rankings_path=str(ws["rankings_path"]),
        chunks_path=str(ws["chunks_path"]),
    )
    client = MockClient(default="A")
    summary = run(config, client=client, out_dir=ws["out_dir"])

    # 'rag' placeholder resolved from the rankings file metadata.
    assert summary["system"] == "bm25"
    assert summary["setting"] == "text_restricted"

    rows = read_jsonl(ws["out_dir"] / f"preds_{summary['config_id']}.jsonl")
    for row in rows:
        assert row["system"] == "bm25"
        assert row["setting"] == "text_restricted"
        assert row["k"] == 2
        tid = row["question_id"].split("_")[0]
        # Rank order, not file order: rank 1 = c002, rank 2 = c001.
        assert row["context_chunk_ids"] == [
            f"{tid}_overlap_c002", f"{tid}_overlap_c001",
        ]
    # The prompt context concatenates the top-k chunk texts in rank order.
    assert "Context: Chunk 2 vum text1.\n\nChunk 1 vum text1.\n" in client.calls[0]


def test_rag_run_requires_paths_and_k(workspace):
    ws = workspace
    base = dict(
        system="rag",
        questions_path=str(ws["questions_path"]),
    )
    with pytest.raises(ValueError, match="rankings_path and chunks_path"):
        run(RunConfig(**base), client=MockClient(), out_dir=ws["out_dir"])
    with pytest.raises(ValueError, match="need k"):
        run(
            RunConfig(
                **base,
                rankings_path=str(ws["rankings_path"]),
                chunks_path=str(ws["chunks_path"]),
            ),
            client=MockClient(),
            out_dir=ws["out_dir"],
        )


def test_closed_book_run_omits_context(workspace):
    ws = workspace
    config = RunConfig(
        system="closed_book",
        model="mock-model",
        setting="none",
        questions_path=str(ws["questions_path"]),
    )
    client = MockClient(default="A")
    run(config, client=client, out_dir=ws["out_dir"])
    assert len(client.calls) == 4
    for prompt in client.calls:
        assert "Context:" not in prompt
        assert "Use only the provided context." not in prompt


def test_non_random_system_requires_client(workspace):
    ws = workspace
    with pytest.raises(ValueError, match="requires an LLM client"):
        run(oracle_config(ws), client=None, out_dir=ws["out_dir"])


def test_random_system_is_deterministic_and_needs_no_client(workspace):
    ws = workspace
    config = RunConfig(
        system="random",
        model="none",
        setting="none",
        questions_path=str(ws["questions_path"]),
        seed=13,
    )
    run(config, client=None, out_dir=ws["out_dir"] / "a")
    run(config, client=None, out_dir=ws["out_dir"] / "b")
    cid = compute_config_id(config)
    rows_a = read_jsonl(ws["out_dir"] / "a" / f"preds_{cid}.jsonl")
    rows_b = read_jsonl(ws["out_dir"] / "b" / f"preds_{cid}.jsonl")
    letters_a = [r["parsed_letter"] for r in rows_a]
    letters_b = [r["parsed_letter"] for r in rows_b]
    assert letters_a == letters_b
    for row in rows_a:
        assert row["parsed_letter"] in list("ABCD")
        assert row["parse_status"] == "exact"
        assert row["raw_output"] == row["parsed_letter"]


def test_runner_output_passes_analysis_schema_validation(workspace):
    """Cross-module contract check: the analysis loader accepts runner output."""
    from diaglux.analysis.loading import load_preds_file

    ws = workspace
    client = MockClient(outputs=["B", "Answer: A", "garbage", "**D**"])
    summary = run(oracle_config(ws), client=client, out_dir=ws["out_dir"])
    df = load_preds_file(ws["out_dir"] / f"preds_{summary['config_id']}.jsonl")
    assert len(df) == 4
    assert df.attrs["config"]["config_id"] == summary["config_id"]
