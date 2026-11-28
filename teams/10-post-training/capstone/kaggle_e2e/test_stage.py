"""stage.py helpers: splits, row tags, template install, GRPO config, command failure. Made-up fixtures, not data."""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage  # noqa: E402

parametrize = pytest.mark.parametrize


def test_hash_split_is_fixed_and_order_free():
    ids = [f"r{i}" for i in range(100)]
    a = stage.hash_split(ids, 10, "s")
    assert len(a) == 10 and a == stage.hash_split(list(reversed(ids)), 10, "s")
    assert a != stage.hash_split(ids, 10, "other salt")


@parametrize("n", [0, -3, 5, 9])
def test_hash_split_bounds(n):
    with pytest.raises(ValueError, match="n_test"):
        stage.hash_split([f"r{i}" for i in range(5)], n, "s")


def test_tag_rows():
    rows = stage.tag_rows([{"id": "a", "x": 1}, {"id": "b", "x": 2}], {"b"})
    assert rows == [{"id": "a", "x": 1, "split": "train", "cluster": "a"},
                    {"id": "b", "x": 2, "split": "test", "cluster": "b"}]


def test_install_template_keeps_other_keys(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 8192, "chat_template": "old"}))
    stage.install_template(tmp_path)
    cfg = json.loads((tmp_path / "tokenizer_config.json").read_text())
    want = (stage.TEAM / "chat-format" / "chat_template.jinja").read_text(encoding="utf-8")
    assert cfg["model_max_length"] == 8192 and cfg["chat_template"] == want
    assert (tmp_path / "chat_template.jinja").read_text(encoding="utf-8") == want


def test_install_template_without_config(tmp_path):
    stage.install_template(tmp_path)
    assert "chat_template" in json.loads((tmp_path / "tokenizer_config.json").read_text())


@parametrize("optim", ["adamw", "hybrid_muon"])
def test_grpo_config_is_accepted_by_the_rl_loop(tmp_path, optim):
    sys.path.insert(0, str(stage.TEAM / "rl-loop"))
    import train  # noqa: PLC0415

    cfg = stage.grpo_config(tmp_path, tmp_path / "m", tmp_path / "out", optim, 50, 3600)
    t = cfg["train"]
    assert cfg["task"]["chat_template_kwargs"] == {"effort": stage.EFFORT} and t["algo"] == "objective_v1"
    assert "kl_coef" not in t and "tis_clip" not in t and t["max_train_seconds"] == 3600
    assert "deadline_unix" not in t and t["save_trainable_dtype"] == "float32"
    assert cfg["model"]["dense"] is True and t["screen"]["n"] == 64
    assert "screen" not in stage.grpo_config(tmp_path, tmp_path, tmp_path, optim, 2, None, screen=False)["train"]
    _, info = train.build_optimizer(t, [("w", __import__("torch").nn.Parameter(__import__("torch").zeros(4, 4)))])
    assert info["kind"] == optim


def test_grpo_config_math_task_builds(tmp_path):
    sys.path.insert(0, str(stage.TEAM / "rl-loop"))
    import train  # noqa: PLC0415

    prep = tmp_path / "prep"
    prep.mkdir()
    (prep / "gsm_tasks.jsonl").write_text("".join(json.dumps({"id": f"gsm-{i}", "problem": f"{i}?", "answer": str(i)})
                                                  + "\n" for i in range(4)))
    (prep / "gsm_train_ids.txt").write_text("gsm-0\ngsm-1\n")
    (prep / "gsm_heldout_ids.txt").write_text("gsm-2\n")

    class Tok:
        def apply_chat_template(self, msgs, add_generation_prompt, tokenize, **kw):
            assert kw == {"effort": stage.EFFORT}
            return [ord(c) for c in msgs[0]["content"]]

    task = train.math_task(stage.grpo_config(tmp_path, tmp_path, tmp_path, "adamw", 1, None)["task"], Tok())
    assert len(task.train) == 2 and len(task.heldout) == 1


def test_run_raises_with_log_tail(tmp_path):
    log = tmp_path / "x.log"
    assert stage.run([sys.executable, "-c", "print('ok')"], log) >= 0
    with pytest.raises(RuntimeError, match="exit 3"):
        stage.run([sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"], log)
    assert "boom" in log.read_text()


import push  # noqa: E402

SHA = "0123456789abcdef0123456789abcdef01234567"


@parametrize("kernel", ["full", "k0", "k1", "k3", "k4"])
def test_kernel_script_compiles_private_and_clean(kernel):
    script, meta = push.kernel_files(kernel, SHA, push.plans(100, 3600, "/kaggle/working/w/sft_adamw/model")[kernel])
    compile(script, "kernel.py", "exec")
    assert meta["is_private"] is True and meta["enable_gpu"] is True and meta["enable_internet"] is True
    assert meta["id"].startswith(f"{push.USER}/") and SHA in script and push.FORK in script
    low = script.lower()
    assert "token" not in low and "kaggle.json" not in low and "/users/" not in low and "hf_" not in low


@parametrize("sha", ["main", "0123abc", SHA.upper(), SHA + "0", "; rm -rf /"])
def test_kernel_refuses_non_commit_sha(sha):
    with pytest.raises(SystemExit, match="40-hex"):
        push.kernel_files("k0", sha, push.plans(1, 1, "x")["k0"])


def test_parallel_groups_use_both_gpus():
    k1 = push.plans(1, 1, "x")["k1"]
    gpus = [argv[argv.index("--gpu") + 1] for argv in k1["groups"][1]]
    assert gpus == ["0", "1"]


def test_push_dry_run_writes_kernel_folder(tmp_path):
    assert push.main(["--kernel", "k0", "--sha", SHA, "--dir", str(tmp_path), "--dry-run"]) == 0
    meta = json.loads((tmp_path / "k0" / "kernel-metadata.json").read_text())
    assert meta["code_file"] == "kernel.py" and (tmp_path / "k0" / "kernel.py").is_file()


def test_manifest_hashes_files(tmp_path):
    sd = tmp_path / "s"
    sd.mkdir()
    (sd / "a.txt").write_text("hi")
    stage.write_manifest(sd, {"stage": "s"}, [sd / "a.txt", sd / "missing"])
    m = json.loads((sd / "manifest.json").read_text())
    assert list(m["files"]) == ["s/a.txt"] and m["files"]["s/a.txt"]["bytes"] == 2
