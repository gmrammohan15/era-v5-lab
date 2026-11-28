"""build_gsm.py: last-question extraction, answer checks, dedup and the hash split. Made-up fixtures, not data."""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_gsm as bg  # noqa: E402
from math_check import score  # noqa: E402

parametrize = pytest.mark.parametrize
SHOTS = ("Question: Ann has 2 pens and buys 3. How many pens?\nAnswer:She has 2 + 3 = 5. So the answer is 5.\n\n"
         "Question: Bo had 9 cups and broke 4. How many are left?\nAnswer:9 - 4 = 5. So the answer is 5.\n\n")


def row(problem: str, answer="7", shots=SHOTS, tail=""):
    return {"messages": [{"role": "user", "content": f"{shots}Question: {problem}{tail}"}], "ground_truth": answer}


def test_keeps_only_the_last_question():
    rec = bg.gsm_record(4, row("Cy walks 3 km then 4 km. How far?"))
    assert rec["problem"] == "Cy walks 3 km then 4 km. How far?" and rec["answer"] == "7" and rec["id"] == "gsm-4"
    assert "Answer" not in rec["problem"] and rec["licence_table_set"] == "rl-math-rlvr-gsm"


@parametrize("tail", ["\nAnswer:", "\nAnswer: ", "  \n"])
def test_trailing_answer_cue_or_space_is_accepted(tail):
    assert bg.gsm_record(0, row("How many?", tail=tail))["problem"] == "How many?"


def test_zero_shot_prompt_works():
    assert bg.gsm_record(0, row("A multi\nline problem?", shots=""))["problem"] == "A multi\nline problem?"


@parametrize("answer, want", [("1,104", "1104"), ("-3", "-3"), ("2.5", "2.5"), (" 12 ", "12")])
def test_numeric_answers_normalised(answer, want):
    assert bg.gsm_record(0, row("Q?", answer=answer))["answer"] == want


@parametrize("bad_row, msg", [
    (row("Q?", answer=""), "not a number"),
    (row("Q?", answer="seven"), "not a number"),
    (row("Q?", answer="7; rm -rf /"), "not a number"),
    ({"messages": [], "ground_truth": "7"}, "one user message"),
    ({"messages": [{"role": "user", "content": "Q?"}] * 2, "ground_truth": "7"}, "one user message"),
    ({"ground_truth": "7"}, "one user message"),
    ({"messages": [{"role": "user", "content": "no question marker"}], "ground_truth": "7"}, "Question"),
    ({"messages": [{"role": "user", "content": SHOTS}], "ground_truth": "7"}, "Question"),  # last block answered
    ({"messages": [{"role": "user", "content": "Question:   "}], "ground_truth": "7"}, "Question"),
    (row("x" * (bg.MAX_PROBLEM_CHARS + 1)), "longer than"),
])
def test_bad_rows_raise(bad_row, msg):
    with pytest.raises(bg.BadRow, match=msg):
        bg.gsm_record(0, bad_row)


def test_build_skips_and_counts_bad_and_duplicate_rows():
    recs, skipped = bg.build([row("A?"), row("B?", answer="x"), row("A?"), row("C?")])
    assert [r["problem"] for r in recs] == ["A?", "C?"] and [r["id"] for r in recs] == ["gsm-0", "gsm-3"]
    assert skipped == {"ground_truth 'x' is not a number": 1, "duplicate problem": 1}


def test_split_is_disjoint_complete_and_order_free():
    recs, _ = bg.build([row(f"P{i}?") for i in range(50)])
    tr, he = bg.split(recs, 10)
    assert len(he) == 10 and not set(tr) & set(he) and set(tr) | set(he) == {r["id"] for r in recs}
    by_problem = {r["id"]: r["problem"] for r in recs}
    recs2, _ = bg.build([row(f"P{i}?") for i in reversed(range(50))])
    _, he2 = bg.split(recs2, 10)
    assert {by_problem[i] for i in he} == {r["problem"] for r in recs2 if r["id"] in set(he2)}


@parametrize("n", [0, -1, 5, 6])
def test_split_size_bounds(n):
    recs, _ = bg.build([row(f"P{i}?") for i in range(5)])
    with pytest.raises(ValueError, match="--heldout"):
        bg.split(recs, n)


def test_records_score_with_math_check():
    pytest.importorskip("math_verify")
    rec = bg.gsm_record(0, row("Q?", answer="1,104"))
    assert score(rec, "so it is \\boxed{1104}")["reward"] == 1.0
    assert score(rec, "so it is \\boxed{1105}")["reward"] == 0.0


def test_exclude_reads_decontam_report_and_plain_ids(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"doc_id": "gsm-1", "flagged": true}\n\ngsm-3\n')
    assert bg.read_exclude(p) == {"gsm-1", "gsm-3"}
    recs, _ = bg.build([row(f"P{i}?") for i in range(5)])
    assert [r["id"] for r in bg.drop_excluded(recs, {"gsm-1", "gsm-3"})] == ["gsm-0", "gsm-2", "gsm-4"]


def test_stale_exclude_list_is_refused():
    recs, _ = bg.build([row(f"P{i}?") for i in range(3)])
    with pytest.raises(ValueError, match="stale"):
        bg.drop_excluded(recs, {"gsm-9"})


def test_cli_end_to_end(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows = [row(f"P{i}?", answer=str(i)) for i in range(12)] + [row("bad", answer="?")]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "t.parquet")
    out = tmp_path / "o"
    (tmp_path / "ex.txt").write_text("gsm-5\n")
    assert bg.main(["--parquet", str(tmp_path / "t.parquet"), "--out", str(out / "t.jsonl"), "--train-ids",
                    str(out / "tr.txt"), "--heldout-ids", str(out / "he.txt"), "--heldout", "3",
                    "--exclude", str(tmp_path / "ex.txt")]) == 0
    recs = [json.loads(x) for x in (out / "t.jsonl").read_text().splitlines()]
    tr, he = (out / "tr.txt").read_text().split(), (out / "he.txt").read_text().split()
    assert len(recs) == 11 and len(he) == 3 and len(tr) == 8 and "gsm-5" not in tr + he
