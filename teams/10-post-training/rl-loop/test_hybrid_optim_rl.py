"""train.optim: hybrid_muon in the RL loop, and task.chat_template_kwargs for math prompts.

Toy task, tiny random OLMoE, CPU, real train.train. Fixtures, not training data.
"""
import copy
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tasks  # noqa: E402
import train  # noqa: E402

parametrize = pytest.mark.parametrize
TIMING = {"times_s", "step_s", "seconds", "rollout_tokens_per_s", "peak_torch_gb", "gpu_used_gb", "gpu_total_gb"}


def cfg_for(out: Path, steps: int = 6, **train_kw) -> dict:
    cfg = yaml.safe_load((HERE / "config" / "smoke_objective_v1.yaml").read_text())
    cfg["out_dir"] = str(out)
    cfg["verbose"] = False
    cfg["train"].update(steps=steps, eval_every=3, state_every=2, state_dir=str(out / "state"), **train_kw)
    return cfg


HYBRID = {"optim": "hybrid_muon", "muon_lr": 0.02}


def rows(out: Path) -> list[dict]:
    return [{k: v for k, v in json.loads(line).items() if k not in TIMING}
            for line in (out / "metrics.jsonl").read_text().splitlines()]


def final_state(out: Path) -> dict:
    latest = json.loads((out / "state" / "LATEST.json").read_text())
    return torch.load(out / "state" / latest["file"], map_location="cpu", weights_only=False)


def tensors(obj, path=""):
    if isinstance(obj, torch.Tensor):
        yield path, obj
    elif isinstance(obj, dict):
        for k in sorted(obj, key=str):
            yield from tensors(obj[k], f"{path}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from tensors(v, f"{path}/{i}")


def test_hybrid_run_reports_split_and_share(tmp_path):
    s = train.train(cfg_for(tmp_path / "h", **HYBRID))
    o = s["optimizer"]
    assert o["kind"] == "hybrid_muon" and o["muon_params"] > 0 and o["adamw_params"] > 0
    assert 0 < o["optim_share_of_train_phase"] < 1 and 0 < o["optim_share_of_step"] < 1


def test_adamw_default_reports_its_share(tmp_path):
    s = train.train(cfg_for(tmp_path / "a"))
    assert s["optimizer"]["kind"] == "adamw" and 0 < s["optimizer"]["optim_share_of_step"] < 1


def test_hybrid_differs_from_adamw(tmp_path):
    a = train.train(cfg_for(tmp_path / "a", steps=3))
    h = train.train(cfg_for(tmp_path / "h", steps=3, **HYBRID))
    assert a["trainable_weight_change"] != h["trainable_weight_change"]


def test_hybrid_stop_and_continue_restores_muon_buffers(tmp_path):
    train.train(cfg_for(tmp_path / "whole", **HYBRID))
    cfg = cfg_for(tmp_path / "split", **HYBRID)
    cfg["train"]["max_steps_this_call"] = 3
    assert train.train(copy.deepcopy(cfg))["status"] == "PAUSED_AT_DEADLINE"
    st = final_state(tmp_path / "split")
    assert any(k.endswith(("/momentum_buffer", "/buf"))
               for k, _ in tensors(st["opt"]["optimizers"]["muon"]["state"]))
    del cfg["train"]["max_steps_this_call"]
    train.train(copy.deepcopy(cfg))
    assert rows(tmp_path / "whole") == rows(tmp_path / "split")
    a, b = final_state(tmp_path / "whole"), final_state(tmp_path / "split")
    ta, tb = dict(tensors(a["opt"])), dict(tensors(b["opt"]))
    assert ta.keys() == tb.keys() and all(torch.equal(ta[k], tb[k]) for k in ta)
    assert all(torch.equal(a["trainable"][n], b["trainable"][n]) for n in a["trainable"])


@parametrize("kw", [{}, HYBRID], ids=["adamw", "hybrid_muon"])
@parametrize("pause", [2, 3, 5])
def test_pause_on_any_step_equals_uninterrupted(tmp_path, kw, pause):
    """eval_every 3: a pause at 3 lands on an evaluation step, whose evaluation draws on the training generator."""
    train.train(cfg_for(tmp_path / "whole", **kw))
    cfg = cfg_for(tmp_path / "split", **kw)
    cfg["train"]["max_steps_this_call"] = pause
    train.train(copy.deepcopy(cfg))
    del cfg["train"]["max_steps_this_call"]
    train.train(copy.deepcopy(cfg))
    assert rows(tmp_path / "whole") == rows(tmp_path / "split")
    ev = [(r["step"], r["name"]) for r in map(json.loads, (tmp_path / "split" / "evals.jsonl").read_text().splitlines())]
    assert len(ev) == len(set(ev))


def test_state_of_adamw_run_is_refused_by_hybrid_run(tmp_path):
    cfg = cfg_for(tmp_path / "x", steps=4)
    cfg["train"]["max_steps_this_call"] = 2
    train.train(copy.deepcopy(cfg))
    cfg["train"].update(HYBRID)
    with pytest.raises(Exception, match="(?i)identity|differ|match"):
        train.train(copy.deepcopy(cfg))


def test_identity_unchanged_without_the_option():
    cfg = cfg_for(Path("/nonexistent"), steps=4)
    named = [("w", torch.zeros(2, 2))]
    before = train.state_identity(cfg, named)
    cfg["train"]["optim"] = "adamw"
    assert train.state_identity(cfg, named) == before
    cfg["train"].update(HYBRID)
    assert train.state_identity(cfg, named) != before


@parametrize("extra, msg", [
    ({"optim": "sgd"}, "must be one of"),
    ({"optim": "hybrid_muon"}, "needs train.muon_lr"),
    ({"optim": "hybrid_muon", "muon_lr": 0}, "needs train.muon_lr"),
    ({"optim": "hybrid_muon", "muon_lr": -1}, "needs train.muon_lr"),
    ({"muon_lr": 0.02}, "only used with"),
])
def test_bad_optimizer_config_is_refused(tmp_path, extra, msg):
    with pytest.raises(ValueError, match=msg):
        train.train(cfg_for(tmp_path / "bad", steps=2, **extra))


@parametrize("dtype", ["float32", "bfloat16"])
def test_save_trainable_dtype(tmp_path, dtype):
    cfg = cfg_for(tmp_path / "s", steps=2, save_trainable=str(tmp_path / "w.pt"), save_trainable_dtype=dtype)
    train.train(cfg)
    w = torch.load(tmp_path / "w.pt")
    assert {v.dtype for v in w.values()} == {getattr(torch, dtype)}


def test_save_trainable_default_stays_bf16(tmp_path):
    train.train(cfg_for(tmp_path / "s", steps=2, save_trainable=str(tmp_path / "w.pt")))
    assert {v.dtype for v in torch.load(tmp_path / "w.pt").values()} == {torch.bfloat16}


@parametrize("bad", ["float16", "fp32", "int8"])
def test_bad_save_dtype_refused_before_training(tmp_path, bad):
    with pytest.raises(ValueError, match="save_trainable_dtype"):
        train.train(cfg_for(tmp_path / "s", steps=2, save_trainable=str(tmp_path / "w.pt"), save_trainable_dtype=bad))
    assert not (tmp_path / "s" / "metrics.jsonl").exists()


def test_hybrid_refused_with_deletion_tuning():
    with pytest.raises(ValueError, match="Deletion Tuning"):
        train.build_optimizer({"lr": 1e-3, **HYBRID}, [("w", torch.nn.Parameter(torch.zeros(2, 2)))], dt=True)


class RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, msgs, add_generation_prompt, tokenize, **kw):
        self.calls.append(kw)
        return [ord(c) for c in msgs[0]["content"]] + [len(kw)]

    def decode(self, ids, skip_special_tokens=True):
        return ""


def write_math(tmp_path: Path) -> dict:
    (tmp_path / "t.jsonl").write_text("\n".join(json.dumps({"id": f"g/{i}", "problem": f"{i}+1?", "answer": str(i + 1)})
                                                for i in range(4)))
    (tmp_path / "tr.txt").write_text("g/0\ng/1\n")
    (tmp_path / "he.txt").write_text("g/2\ng/3\n")
    return {"kind": "math", "tasks": str(tmp_path / "t.jsonl"), "train_ids": str(tmp_path / "tr.txt"),
            "heldout_ids": str(tmp_path / "he.txt"), "instruction_file": None}


def test_math_task_passes_template_kwargs(tmp_path):
    tok = RecordingTokenizer()
    task = train.math_task({**write_math(tmp_path), "chat_template_kwargs": {"effort": "1k"}}, tok)
    assert tok.calls == [{"effort": "1k"}] * 4 and len(task.train) == 2


def test_math_task_without_kwargs_is_unchanged(tmp_path):
    tok = RecordingTokenizer()
    train.math_task(write_math(tmp_path), tok)
    assert tok.calls == [{}] * 4


@parametrize("bad", ["effort=1k", ["1k"], 3])
def test_math_task_refuses_non_mapping_kwargs(tmp_path, bad):
    with pytest.raises(TypeError, match="mapping"):
        train.math_task({**write_math(tmp_path), "chat_template_kwargs": bad}, RecordingTokenizer())


def test_reward_env_task_default_has_no_kwargs():
    tok = RecordingTokenizer()
    tasks.RewardEnvTask(tok, [("a", "x")], [("b", "y")], reward_fn=lambda e, t: (0.0, {}))
    assert tok.calls == [{}, {}]
