"""Runner resilience: transient retries, terminal abort, error rows.

A single bad API call must not silently wedge a run (the bug that stalled a
real variance repeat). The runner retries transient errors, records a persistent
transient failure as a per-question ``error`` row and continues, and aborts the
whole run cleanly on an unrecoverable quota/auth error (progress preserved).
No network: a scripted fake client raises on cue.
"""
import json

import pytest

from answering_testutils import make_questions, read_jsonl, write_questions
from diaglux.answering.clients import LLMClient
from diaglux.answering.runner import RunConfig, run


class ScriptedClient(LLMClient):
    """Yields canned outputs; an entry that is an Exception is raised instead.

    A tuple ``(exc, n)`` raises ``exc`` n times then falls through to the next
    output, to exercise the retry path.
    """

    def __init__(self, script):
        self.script = list(script)
        self.i = 0
        self.raise_left = 0
        self.pending = None

    def complete(self, prompt: str) -> str:
        if self.raise_left > 0:
            self.raise_left -= 1
            raise self.pending
        item = self.script[self.i]
        self.i += 1
        if isinstance(item, tuple):  # (exc, n): raise n times then continue
            exc, n = item
            self.pending = exc
            self.raise_left = n
            return self.complete(prompt)
        if isinstance(item, Exception):
            raise item
        return item

    def describe(self):
        return {"client": "ScriptedClient"}


def _config(tmp_path):
    qp = tmp_path / "questions.jsonl"
    write_questions(qp)  # 4 questions: text1_q00/q01, text2_q00/q01
    return RunConfig(system="closed_book", provider="openai", model="m",
                     questions_path=str(qp)), tmp_path / "runs"


def test_transient_error_retried_then_succeeds(tmp_path):
    cfg, out = _config(tmp_path)
    # q00: fail twice (transient) then "B"; rest answer cleanly.
    client = ScriptedClient([(RuntimeError("503 service unavailable"), 2), "B", "A", "C", "D"])
    summary = run(cfg, client=client, out_dir=out, progress=False)
    assert summary["n_new"] == 4
    assert summary["n_error_new"] == 0  # recovered via retry
    assert summary["terminated_early"] is False


def test_persistent_transient_logs_error_row_and_continues(tmp_path):
    cfg, out = _config(tmp_path)
    # q00 always fails transiently; q01..q03 answer. One error row, run finishes.
    client = ScriptedClient([RuntimeError("timeout"), "A", "C", "D"])
    summary = run(cfg, client=client, out_dir=out, progress=False)
    assert summary["n_new"] == 4
    assert summary["n_error_new"] == 1
    assert summary["terminated_early"] is False
    rows = read_jsonl(out / f"preds_{summary['config_id']}.jsonl")
    err = [r for r in rows if r["parse_status"] == "error"]
    assert len(err) == 1
    assert err[0]["parsed_letter"] is None
    assert err[0]["semantic_choice"] is None
    assert err[0]["is_correct"] is False
    assert err[0]["raw_output"].startswith("<<error:")


def test_terminal_quota_error_aborts_cleanly_and_resumable(tmp_path):
    cfg, out = _config(tmp_path)
    # q00 answers; q01 hits insufficient_quota -> abort before q02/q03.
    client = ScriptedClient(["B", RuntimeError("Error code: 429 - insufficient_quota"), "C", "D"])
    summary = run(cfg, client=client, out_dir=out, progress=False)
    assert summary["terminated_early"] is True
    assert summary["n_new"] == 1  # only q00 written
    rows = read_jsonl(out / f"preds_{summary['config_id']}.jsonl")
    assert [r["question_id"] for r in rows] == ["text1_q00"]

    # Resume with a healthy client fills the remaining three.
    client2 = ScriptedClient(["A", "C", "D"])
    summary2 = run(cfg, client=client2, out_dir=out, progress=False)
    assert summary2["n_skipped_resumed"] == 1
    assert summary2["n_new"] == 3
    assert summary2["terminated_early"] is False
