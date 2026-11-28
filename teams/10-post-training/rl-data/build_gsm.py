"""Turn allenai/RLVR-GSM (train split) into math task records for rl-loop `task.kind: math`, plus a train/held-out split.

    python build_gsm.py --parquet <RLVR-GSM train parquet> --out out/gsm_tasks.jsonl \
        --train-ids out/gsm_train_ids.txt --heldout-ids out/gsm_heldout_ids.txt --heldout 256 \
        [--exclude out/gsm_flagged.jsonl]   # teams/09-evaluation/decontam/decontam_check.py on the records' problems

Source: licence table set `rl-math-rlvr-gsm` (MIT, decision "TSAI reserves"). Each row's user message is an 8-shot
prompt: worked GSM8K examples ("Question: ...\\nAnswer: ..."), then "Question: <problem>" (a trailing "Answer:" is
accepted too). Only the last question is kept,
as the human-written GSM8K problem text, so a rollout prompt is the problem alone (about 900 tokens shorter) and no
team- or model-written instruction is added (TSAI 3 Nov rule). The answer is the row's ground_truth, which
rl-data/math_check.py compares with the completion's last \\boxed{} (or last number) by math-verify.

The held-out split is the rows whose sha256(problem) sorts first: fixed for a given source, independent of row order.
A problem that appears twice keeps its first row only, so the two splits never share a prompt. Records are data and
stay out of the repository (out/ is local).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

META = {"dataset": "allenai/RLVR-GSM", "licence": "MIT", "licence_table_set": "rl-math-rlvr-gsm",
        "generator": "none (GSM8K problems and answers, reformatted)"}
LAST_QUESTION = re.compile(r"(?:\A|\n)Question:[ \t]*((?:(?!\nQuestion:)(?!\nAnswer:).)+?)\s*(?:\nAnswer:\s*)?\Z", re.DOTALL)
ANSWER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
MAX_PROBLEM_CHARS = 4000


class BadRow(ValueError):
    pass


def last_question(prompt: str) -> str:
    """The problem of the final 'Question:' block (the row's own question, after the shots)."""
    m = LAST_QUESTION.search(prompt or "")
    if not m or not m.group(1).strip():
        raise BadRow("no final unanswered 'Question:' block")
    problem = m.group(1).strip()
    if len(problem) > MAX_PROBLEM_CHARS:
        raise BadRow(f"problem longer than {MAX_PROBLEM_CHARS} characters")
    return problem


def gsm_record(i: int, row: dict) -> dict:
    msgs = row.get("messages") or []
    users = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
    if len(users) != 1:
        raise BadRow(f"expected one user message, got {len(users)}")
    answer = str(row.get("ground_truth", "")).strip()
    if not ANSWER.fullmatch(answer):
        raise BadRow(f"ground_truth {answer[:40]!r} is not a number")
    problem = last_question(users[0].get("content", ""))
    return {"id": f"gsm-{i}", "kind": "math", "problem": problem, "answer": answer.replace(",", ""),
            "source": f"{META['dataset']} (train, row {i})", "licence": META["licence"],
            "generator": META["generator"], "licence_table_set": META["licence_table_set"]}


def build(rows: list[dict]) -> tuple[list[dict], dict]:
    out, seen, skipped = [], set(), {}
    for i, row in enumerate(rows):
        try:
            rec = gsm_record(i, row)
        except BadRow as exc:
            skipped[str(exc).split(",")[0]] = skipped.get(str(exc).split(",")[0], 0) + 1
            continue
        if rec["problem"] in seen:
            skipped["duplicate problem"] = skipped.get("duplicate problem", 0) + 1
            continue
        seen.add(rec["problem"])
        out.append(rec)
    return out, skipped


def read_exclude(path: Path) -> set[str]:
    """Ids to drop: decontam_check.py output (one JSON object with doc_id per line) or one plain id per line."""
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            ids.add(json.loads(line)["doc_id"] if line.startswith("{") else line)
    return ids


def drop_excluded(records: list[dict], exclude: set[str]) -> list[dict]:
    unknown = exclude - {r["id"] for r in records}
    if unknown:
        raise ValueError(f"{len(unknown)} excluded ids are not records of this build (stale list?), "
                         f"e.g. {sorted(unknown)[:3]}")
    return [r for r in records if r["id"] not in exclude]


def split(records: list[dict], heldout: int) -> tuple[list[str], list[str]]:
    if not 0 < heldout < len(records):
        raise ValueError(f"--heldout must be between 1 and {len(records) - 1}, got {heldout}")
    order = sorted(records, key=lambda r: hashlib.sha256(r["problem"].encode("utf-8")).hexdigest())
    held = sorted(r["id"] for r in order[:heldout])
    held_set = set(held)
    return [r["id"] for r in records if r["id"] not in held_set], held


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", type=Path, required=True, help="RLVR-GSM data/train-*.parquet")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-ids", type=Path, required=True)
    ap.add_argument("--heldout-ids", type=Path, required=True)
    ap.add_argument("--heldout", type=int, default=256)
    ap.add_argument("--exclude", type=Path, help="ids to drop, e.g. the decontam_check.py report of this build")
    args = ap.parse_args(argv)
    import pyarrow.parquet as pq  # noqa: PLC0415  (only the CLI needs it)
    rows = pq.read_table(args.parquet, columns=["messages", "ground_truth"]).to_pylist()
    records, skipped = build(rows)
    if args.exclude:
        kept = drop_excluded(records, read_exclude(args.exclude))
        skipped["excluded (decontamination)"] = len(records) - len(kept)
        records = kept
    train_ids, held_ids = split(records, args.heldout)
    for p in (args.out, args.train_ids, args.heldout_ids):
        p.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    args.train_ids.write_text("".join(i + "\n" for i in train_ids))
    args.heldout_ids.write_text("".join(i + "\n" for i in held_ids))
    print(json.dumps({"rows": len(rows), "records": len(records), "skipped": skipped, "train": len(train_ids),
                      "heldout": len(held_ids),
                      "parquet_sha256": hashlib.sha256(args.parquet.read_bytes()).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
