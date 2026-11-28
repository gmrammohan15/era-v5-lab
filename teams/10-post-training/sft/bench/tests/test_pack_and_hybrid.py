"""Tests for `train.py --pack` and `--optim hybrid_muon` on a tiny random Llama (CPU).

Fixture rows are code-generated test input, never training data.

    python -m pytest teams/10-post-training/sft/bench/tests/test_pack_and_hybrid.py -q
"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest
import torch

_fixture = pytest.fixture
parametrize = pytest.mark.parametrize

BENCH = Path(__file__).resolve().parents[1]


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


train = _load("g10_bench_train", BENCH / "train.py")
probe = train.probe
VOCAB = 64


def rows_of(lengths, seed=0):
    """Rows with a context prefix (mask 0) and a supervised suffix (mask 1), like a user turn then an assistant turn."""
    rng = random.Random(seed)
    out = []
    for i, n in enumerate(lengths):
        ctx = max(1, n // 3)
        out.append({"id": f"r{i}", "input_ids": [rng.randrange(1, VOCAB) for _ in range(n)],
                    "loss_mask": [0] * ctx + [1] * (n - ctx)})
    return out


def tiny_model(seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=128, tie_word_embeddings=True, bos_token_id=0,
                      eos_token_id=0)
    return LlamaForCausalLM(cfg)


# ---------------------------------------------------------------- packing


def test_pack_bins_first_fit_decreasing_respects_capacity():
    rows = rows_of([9, 3, 7, 2, 5, 4])
    bins = train.pack_bins(rows, 10)
    assert sorted(r["id"] for b in bins for r in b) == sorted(r["id"] for r in rows)  # every row once
    assert all(sum(len(r["input_ids"]) for r in b) <= 10 for b in bins)
    # FFD: 9 | 7+3 | 5+4 | 2 (no 3-bin split exists: 9 fits only with a 1)
    assert [[len(r["input_ids"]) for r in b] for b in bins] == [[9], [7, 3], [5, 4], [2]]


def test_pack_bins_rejects_a_row_longer_than_capacity():
    with pytest.raises(ValueError, match="more than the pack capacity"):
        train.pack_bins(rows_of([4, 11]), 10)


def test_packed_labels_never_cross_a_row_boundary():
    rows = [{"id": "a", "input_ids": [1, 2, 3, 4], "loss_mask": [0, 1, 1, 1]},
            {"id": "b", "input_ids": [5, 6, 7], "loss_mask": [1, 1, 1]}]  # b's first token is marked: still no target
    mb = train.collate_packed([rows], pad_id=0)
    assert mb.labels.tolist() == [[2, 3, 4, probe.IGNORE, 6, 7]]
    assert mb.position_ids.tolist() == [[0, 1, 2, 3, 0, 1, 2]]
    attn = mb.attention_mask[0, 0]
    assert not attn[4:, :4].any() and not attn[:4, 4:].any()  # no attention between the two rows
    assert attn[4:, 4:].equal(torch.tril(torch.ones(3, 3, dtype=torch.bool)))


def test_padding_is_its_own_segment_and_never_attended():
    mb = train.collate_packed([rows_of([6]), rows_of([3], seed=1)], pad_id=0)
    attn = mb.attention_mask[1, 0]
    assert not attn[:3, 3:].any()  # real tokens of the short sequence never see its padding
    assert attn.any(dim=-1).all()  # no all-masked query row (would give NaN in attention)


@parametrize("lengths,micro", [([9, 3, 7, 2, 5, 4], 12), ([12, 12, 1], 12), ([5], 64), ([20, 1, 1, 1], 24)])
def test_packed_loss_and_gradient_equal_unpacked(lengths, micro):
    rows = rows_of(lengths)
    model = tiny_model()
    unpacked = [train.collate(g, 0) for g in train.plan_microbatches(rows, 64)]
    packed = train.pack_microbatches(rows, micro, 0)
    assert all(mb.input_ids.size(0) == 1 for mb in packed)
    assert sum(mb.n_targets for mb in packed) == sum(mb.n_targets for mb in unpacked)
    model.zero_grad()
    lu, cu = train.accumulate(model, unpacked, "cpu")
    gu = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    lp, cp = train.accumulate(model, packed, "cpu")
    gp = [p.grad.clone() for p in model.parameters()]
    assert cu == cp
    assert abs(lu - lp) < 1e-5
    assert max(float((a - b).abs().max()) for a, b in zip(gu, gp)) < 1e-5


def test_packing_cuts_padding_slots():
    rows = rows_of([40, 6, 5, 4, 30, 8, 3, 2])
    unpacked = [train.collate(g, 0) for g in train.plan_microbatches(rows, 64)]
    packed = train.pack_microbatches(rows, 48, 0)
    assert sum(map(train.slot_tokens, packed)) == sum(len(r["input_ids"]) for r in rows)  # zero padding
    assert sum(map(train.slot_tokens, unpacked)) > sum(len(r["input_ids"]) for r in rows)


# ---------------------------------------------------------------- end to end (tiny random Llama on CPU)


@_fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny_ph")
    tiny_model().save_pretrained(d / "base")
    rows = probe.synthetic_rows({"vocab_size": VOCAB, "fixture_rows": 24, "seed": 0, "eos_id": 0, "min_len": 6,
                                 "max_len": 24})
    (d / "train.tok.jsonl").write_text("".join(json.dumps({"id": f"r{i}", "split": "train", **r}) + "\n"
                                               for i, r in enumerate(rows)))
    (d / "tokdir").mkdir()
    return d


def run(d, out, extra=(), steps=6, seed=5):
    micro = [] if "--micro-tokens" in extra else ["--micro-tokens", "64"]
    args = train.parse_args(["--train", str(d / "train.tok.jsonl"), "--model", str(d / "base"), "--revision", "main",
                             "--tokenizer-dir", str(d / "tokdir"), "--method", "full", "--steps", str(steps),
                             "--warmup", "2", "--lr", "1e-3", "--seed", str(seed), "--device", "cpu", *micro,
                             "--max-tokens", "48", "--out", str(d / out), "--log-every", "100", *extra])
    args.revision = None
    return train.train(args)


def test_hybrid_muon_run_puts_matrices_on_muon_and_reports_share(tiny):
    rep = run(tiny, "hyb", ["--optim", "hybrid_muon", "--muon-lr", "2e-3"])
    opt = rep["optimizer"]
    assert opt["kind"] == "hybrid_muon" and opt["muon_params"] > 0 and opt["adamw_params"] > 0
    assert opt["adamw_params"] == 32 * (2 * 2 + 1)  # only the norm gains (2 per layer + final) stay on AdamW
    assert 0 < opt["optim_step_share"] < 1
    assert rep["checks"]["writeback_step1"]["tensor"].endswith("input_layernorm.weight")
    assert all(c["pass"] for c in rep["checks"].values())


def test_hybrid_muon_differs_from_adamw_and_is_deterministic(tiny):
    a = run(tiny, "h1", ["--optim", "hybrid_muon", "--muon-lr", "2e-3"])
    b = run(tiny, "h2", ["--optim", "hybrid_muon", "--muon-lr", "2e-3"])
    w = run(tiny, "w1")
    assert a["saved_state_sha256"] == b["saved_state_sha256"]
    assert a["saved_state_sha256"] != w["saved_state_sha256"]
    assert w["optimizer"]["kind"] == "adamw" and "optim_step_share" in w["optimizer"]


def test_hybrid_muon_crash_and_resume_restores_momentum(tiny, monkeypatch):
    extra = ["--optim", "hybrid_muon", "--muon-lr", "2e-3", "--save-every", "2"]
    whole = run(tiny, "hwhole", extra, steps=7)
    real = train.accumulate
    calls = {"n": 0}

    def dying(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 6:
            raise KeyboardInterrupt("simulated crash")
        return real(*a, **kw)

    monkeypatch.setattr(train, "accumulate", dying)
    with pytest.raises(KeyboardInterrupt):
        run(tiny, "hcrash", extra, steps=7)
    monkeypatch.setattr(train, "accumulate", real)
    st = torch.load(tiny / "hcrash" / "train_state.pt", map_location="cpu", weights_only=False)
    assert any("buf" in s for s in st["opt"]["optimizers"]["muon"]["state"].values())  # Muon momentum is saved
    done = run(tiny, "hcrash", extra, steps=7)
    assert [r["loss"] for r in done["log"]] == [r["loss"] for r in whole["log"]]
    assert done["saved_state_sha256"] == whole["saved_state_sha256"]


def test_pack_run_matches_unpacked_first_loss(tiny):
    p = run(tiny, "pk", ["--pack"], steps=2)
    u = run(tiny, "upk", steps=2)
    assert abs(p["log"][0]["loss"] - u["log"][0]["loss"]) < 1e-5
    assert p["log"][0]["targets"] == u["log"][0]["targets"]
    assert p["packing"]["slot_tokens"] <= u["packing"]["slot_tokens"]
    assert all(c["pass"] for c in p["checks"].values())


@parametrize("optim", [[], ["--optim", "hybrid_muon", "--muon-lr", "2e-3"]], ids=["adamw", "hybrid_muon"])
def test_pack_with_grad_checkpointing_matches_pack(tiny, optim):
    """The Kaggle T4 run needs --grad-checkpointing: same losses with the block-diagonal mask and position ids."""
    a = run(tiny, "pk_gc", ["--pack", "--grad-checkpointing", *optim], steps=3)
    b = run(tiny, "pk_nogc", ["--pack", *optim], steps=3)
    for x, y in zip(a["log"], b["log"]):
        assert abs(x["loss"] - y["loss"]) < 1e-5
    assert all(c["pass"] for c in a["checks"].values())


@parametrize("optim", [[], ["--optim", "hybrid_muon", "--muon-lr", "2e-3"]], ids=["adamw", "hybrid_muon"])
def test_pack_with_grad_checkpointing_matches_pack(tiny, optim):
    """The Kaggle T4 run needs --grad-checkpointing: same losses with the block-diagonal mask and position ids."""
    a = run(tiny, "pk_gc", ["--pack", "--grad-checkpointing", *optim], steps=3)
    b = run(tiny, "pk_nogc", ["--pack", *optim], steps=3)
    for x, y in zip(a["log"], b["log"]):
        assert abs(x["loss"] - y["loss"]) < 1e-5
    assert all(c["pass"] for c in a["checks"].values())


@parametrize("extra,msg", [
    (["--optim", "hybrid_muon"], "needs --muon-lr"),
    (["--optim", "hybrid_muon", "--muon-lr", "0"], "needs --muon-lr"),
    (["--muon-lr", "1e-3"], "only used with --optim hybrid_muon"),
    (["--pack", "--micro-tokens", "32"], "--micro-tokens >= --max-tokens"),
])
def test_bad_optimizer_flags_stop_with_a_clear_error(tiny, extra, msg):
    with pytest.raises(SystemExit, match=msg):
        run(tiny, "bad", extra, steps=1)


def test_hybrid_muon_refuses_non_full_methods(tiny):
    args = train.parse_args(["--train", str(tiny / "train.tok.jsonl"), "--model", str(tiny / "base"), "--method", "lora",
                             "--optim", "hybrid_muon", "--muon-lr", "1e-3", "--out", str(tiny / "x")])
    with pytest.raises(SystemExit, match="--method full only"):
        train.train(args)


def test_resume_key_unchanged_when_new_flags_are_off():
    base = ["--train", "t", "--model", "m", "--out", "o"]
    key = train.resume_key(train.parse_args(base))
    for k in ("optim", "muon_lr", "pack"):
        assert f'"{k}"' not in key
    assert '"pack"' in train.resume_key(train.parse_args(base + ["--pack"]))
    assert '"optim"' in train.resume_key(train.parse_args(base + ["--optim", "hybrid_muon", "--muon-lr", "1e-3"]))
