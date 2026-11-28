"""Small base-to-chat run (SmolLM2-360M, Lab 5 Team 10 stand-in): one stage per Kaggle kernel session.

    python stage.py fetch --work W                      pinned downloads only (a kernel that links earlier outputs)
    python stage.py prep  --work W                      pinned downloads, decontamination, SFT rows, RL and eval tasks
    python stage.py smoke --work W                      GPU check: tests, 20 SFT updates per optimizer, 2 GRPO steps
    python stage.py sft   --work W --arm adamw|hybrid_muon [--gpu 0]
    python stage.py grpo  --work W --sft-model DIR --optim adamw|hybrid_muon --steps N [--gpu 0] [--max-seconds S]
    python stage.py eval  --work W --model DIR --name NAME [--gpu 0]

Everything a stage reads comes from public, pinned sources (PINS) or from the repository clone; nothing is uploaded
from a laptop. Every stage writes W/<stage>/manifest.json with the sha256 of each input and output it touched.
Data rule (TSAI 3 Nov): SFT rows are human-written (databricks-dolly-15k); RL prompts are human-written GSM8K problems
with no added instruction text; model generations are evaluation output only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEAM = HERE.parents[1]
REPO = TEAM.parents[1]
PY = sys.executable

PINS = {
    "model": ("HuggingFaceTB/SmolLM2-360M", "f8027fd0eaeea54caa13c31d31b9fdc459c38b49", "model"),
    "dolly": ("databricks/databricks-dolly-15k", "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a", "dataset"),
    "gsm": ("allenai/RLVR-GSM", "b14884519de816306cf8ce7fc74547284b9ac548", "dataset"),
}
TOKENIZER_SHA256 = "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"
EFFORT = "1k"  # convert.py gives every Dolly row (no reasoning trace) the smallest effort tier
DOLLY_TEST_ROWS = 200
GSM_HELDOUT = 256
GSM_EVAL = 300
# E9 arm A1 (results/e9_sft_before_rl.md): full fine-tune, FP32, lr 1e-4, 150 updates x 8,000 supervised tokens
SFT_ARGS = ["--method", "full", "--steps", "150", "--batch-target-tokens", "8000", "--lr", "1e-4",
            "--max-tokens", "2048", "--micro-tokens", "4096", "--warmup", "10", "--log-every", "10"]
MUON_LR = "0.002"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(cmd: list[str], log: Path, env: dict | None = None, timeout: float | None = None) -> float:
    """Run one command, its output to `log` (appended), raise on a nonzero exit; returns the wall seconds."""
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})},
                              timeout=timeout, check=False)
    if proc.returncode:
        tail = log.read_text(errors="replace").splitlines()[-30:]
        raise RuntimeError(f"exit {proc.returncode}: {' '.join(cmd[:4])} ...\n" + "\n".join(tail))
    return time.time() - t0


def hash_split(ids: list[str], n_test: int, salt: str) -> set[str]:
    """The n_test ids whose sha256(salt + id) sorts first: a fixed choice for a given id list, not row order."""
    if not 0 < n_test < len(ids):
        raise ValueError(f"n_test must be between 1 and {len(ids) - 1}, got {n_test}")
    return set(sorted(ids, key=lambda i: hashlib.sha256(f"{salt}{i}".encode()).hexdigest())[:n_test])


def tag_rows(rows: list[dict], test_ids: set[str]) -> list[dict]:
    """split train/test for sft/bench (train.py reads split == train; evaluate.py dev/test with a cluster each)."""
    return [{**r, "split": "test" if r["id"] in test_ids else "train", "cluster": r["id"]} for r in rows]


def write_manifest(stage_dir: Path, info: dict, files: list[Path]) -> None:
    info = {**info, "files": {str(p.relative_to(stage_dir.parent)): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                              for p in files if p.is_file()}}
    (stage_dir / "manifest.json").write_text(json.dumps(info, indent=1))


def download(work: Path) -> dict:
    from huggingface_hub import hf_hub_download, snapshot_download  # noqa: PLC0415

    src = work / "src"
    model_dir = Path(snapshot_download(PINS["model"][0], revision=PINS["model"][1], local_dir=src / "model",
                                       allow_patterns=["*.json", "*.safetensors", "*.txt"]))
    dolly = Path(hf_hub_download(PINS["dolly"][0], "databricks-dolly-15k.jsonl", revision=PINS["dolly"][1],
                                 repo_type="dataset", local_dir=src / "dolly"))
    gsm = {s: Path(hf_hub_download(PINS["gsm"][0], f"data/{s}-00000-of-00001.parquet", revision=PINS["gsm"][1],
                                   repo_type="dataset", local_dir=src / "gsm")) for s in ("train", "test")}
    if sha256_file(model_dir / "tokenizer.json") != TOKENIZER_SHA256:
        raise RuntimeError("SmolLM2 tokenizer.json is not the pinned one")
    return {"model": model_dir, "dolly": dolly, "gsm_train": gsm["train"], "gsm_test": gsm["test"]}


def decontam(docs: list[dict], tmp: Path, log: Path) -> set[str]:
    """Team 9 fingerprints v2 (decontam_check.py) and the canary scan; returns the flagged ids."""
    tmp.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs), encoding="utf-8")
    dc = REPO / "teams" / "09-evaluation" / "decontam"
    flagged = tmp.with_suffix(".flagged.jsonl")
    with open(flagged, "w") as out, open(log, "a") as err:
        subprocess.run([PY, str(dc / "decontam_check.py"), str(tmp), "--id-field", "id", "--text-field", "text"],
                       stdout=out, stderr=err, check=True)
    canary = tmp.with_suffix(".canary.jsonl")
    with open(canary, "w") as out, open(log, "a") as err:
        subprocess.run([PY, str(dc / "canary_scan.py"), str(tmp), "--id-field", "id"], stdout=out, stderr=err,
                       check=True)
    ids = {json.loads(x)["doc_id"] for x in flagged.read_text().splitlines() if x.strip()}
    ids |= {json.loads(x).get("doc_id") or json.loads(x).get("id") for x in canary.read_text().splitlines() if x.strip()}
    return ids


def prompt_parity(model_dir: Path, rows: list[dict], template: str, limit: int = 300) -> dict:
    """RL prompts (apply_chat_template with the G10 template, effort EFFORT) must be the SFT rows' prompt ids."""
    from transformers import AutoTokenizer  # noqa: PLC0415

    render = load_module("g10_render", TEAM / "chat-format" / "render.py")
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    bad = 0
    checked = rows[:limit]
    for r in checked:
        msgs = render.parse(r["text"])["messages"]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        ids = tok.apply_chat_template([{"role": "user", "content": user}], chat_template=template,
                                      add_generation_prompt=True, tokenize=True, effort=EFFORT)
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        bad += ids != r["input_ids"][:r["prompt_len"]]
    if bad:
        raise RuntimeError(f"prompt parity: {bad} of {len(checked)} RL prompts differ from the SFT prompt ids")
    return {"checked": len(checked), "mismatch": 0}


def install_template(model_dir: Path) -> None:
    """The saved model must render G10 ChatML v0 (the RL loop and any user call apply_chat_template)."""
    template = (TEAM / "chat-format" / "chat_template.jinja").read_text(encoding="utf-8")
    cfg_path = model_dir / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    cfg["chat_template"] = template
    cfg_path.write_text(json.dumps(cfg, indent=1, ensure_ascii=False))
    (model_dir / "chat_template.jinja").write_text(template, encoding="utf-8")


def stage_prep(args) -> dict:
    work = Path(args.work)
    sd = work / "prep"
    sd.mkdir(parents=True, exist_ok=True)
    log = sd / "prep.log"
    t0 = time.time()
    src = download(work)
    bg = load_module("g10_build_gsm", TEAM / "rl-data" / "build_gsm.py")
    import pyarrow.parquet as pq  # noqa: PLC0415

    # RL pool: GSM8K train problems, decontaminated, hash split into train / held-out
    train_recs, skipped = bg.build(pq.read_table(src["gsm_train"], columns=["messages", "ground_truth"]).to_pylist())
    flagged_gsm = decontam([{"id": r["id"], "text": r["problem"]} for r in train_recs], sd / "dc_gsm.jsonl", log)
    train_recs = bg.drop_excluded(train_recs, flagged_gsm)
    rl_train, rl_held = bg.split(train_recs, GSM_HELDOUT)
    test_recs, _ = bg.build(pq.read_table(src["gsm_test"], columns=["messages", "ground_truth"]).to_pylist())
    test_recs = [{**r, "id": r["id"].replace("gsm-", "gsmtest-")} for r in test_recs]
    eval_ids = sorted(hash_split([r["id"] for r in test_recs], GSM_EVAL, "g10-gsm-eval"))
    tasks = sd / "gsm_tasks.jsonl"
    tasks.write_text("".join(json.dumps(r) + "\n" for r in train_recs + test_recs), encoding="utf-8")
    for name, ids in (("gsm_train_ids.txt", rl_train), ("gsm_heldout_ids.txt", rl_held), ("gsm_eval_ids.txt", eval_ids)):
        (sd / name).write_text("".join(i + "\n" for i in ids))

    # SFT rows: Dolly, decontaminated, through convert.py (G10 ChatML v0) and retok.py (SmolLM2 surrogate ids)
    dolly_rows = [json.loads(x) for x in src["dolly"].read_text(encoding="utf-8").splitlines() if x.strip()]
    docs = [{"id": f"dolly/{i}", "text": "\n\n".join(x for x in (r["instruction"], r["context"], r["response"]) if x)}
            for i, r in enumerate(dolly_rows)]
    flagged_dolly = decontam(docs, sd / "dc_dolly.jsonl", log)
    kept = sd / "dolly_kept.jsonl"
    kept.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for i, r in enumerate(dolly_rows)
                            if f"dolly/{i}" not in flagged_dolly), encoding="utf-8")
    conv, tok = sd / "dolly.conv.jsonl", sd / "dolly.tok.jsonl"
    tokjson = str(src["model"] / "tokenizer.json")
    run([PY, str(TEAM / "sft" / "convert.py"), "--set", "sft-dolly-15k", "--input", str(kept), "--tokenizer", tokjson,
         "--out", str(conv)], log)
    run([PY, str(TEAM / "sft" / "bench" / "retok.py"), "--tokenizer", tokjson, "--in", str(conv), "--out", str(tok),
         "--max-tokens", "2048"], log)
    rows = [json.loads(x) for x in tok.read_text(encoding="utf-8").splitlines() if x.strip()]
    if any(r.get("effort", EFFORT) != EFFORT for r in rows):
        raise RuntimeError(f"a Dolly row has an effort tier other than {EFFORT}")
    test_ids = hash_split([r["id"] for r in rows], DOLLY_TEST_ROWS, "g10-dolly-test")
    tagged = tag_rows(rows, test_ids)
    sft_train, sft_test = sd / "sft_train.tok.jsonl", sd / "sft_test.tok.jsonl"
    sft_train.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in tagged if r["split"] == "train"))
    sft_test.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in tagged if r["split"] == "test"))
    template = (TEAM / "chat-format" / "chat_template.jinja").read_text(encoding="utf-8")
    parity = prompt_parity(src["model"], rows, template)
    info = {"stage": "prep", "pins": PINS, "seconds": round(time.time() - t0, 1),
            "gsm": {"train_rows": len(train_recs) + len(flagged_gsm), "flagged": sorted(flagged_gsm),
                    "rl_train": len(rl_train), "rl_heldout": len(rl_held), "eval": len(eval_ids), "skipped": skipped},
            "dolly": {"rows": len(dolly_rows), "flagged": sorted(flagged_dolly), "sft_rows": len(rows),
                      "sft_train": len(rows) - len(test_ids), "sft_test": len(test_ids)},
            "prompt_parity": parity}
    write_manifest(sd, info, [tasks, sft_train, sft_test, kept, *(sd / n for n in
                   ("gsm_train_ids.txt", "gsm_heldout_ids.txt", "gsm_eval_ids.txt"))])
    for p in (conv, sd / "dc_gsm.jsonl", sd / "dc_dolly.jsonl"):
        p.unlink(missing_ok=True)  # intermediate copies; the manifest holds what training reads
    return info


def gpu_env(gpu: int | None) -> dict:
    return {"CUDA_VISIBLE_DEVICES": str(gpu)} if gpu is not None else {}


def stage_sft(args, steps: int | None = None, tag: str | None = None) -> dict:
    work = Path(args.work)
    prep = work / "prep"
    name = tag or f"sft_{args.arm}"
    sd = work / name
    cmd = [PY, str(TEAM / "sft" / "bench" / "train.py"), "--train", str(prep / "sft_train.tok.jsonl"),
           "--model", str(work / "src" / "model"), "--revision", PINS["model"][1],
           "--tokenizer-dir", str(work / "src" / "model"), "--device", "cuda", "--seed", str(args.seed),
           "--save-every", "50", "--out", str(sd), *SFT_ARGS]
    if steps is not None:
        cmd[cmd.index("--steps") + 1] = str(steps)
    if args.arm == "hybrid_muon":
        cmd += ["--optim", "hybrid_muon", "--muon-lr", args.muon_lr]
    if args.pack:
        cmd += ["--pack"]
    secs = run(cmd, sd / "train.log", env=gpu_env(args.gpu))
    install_template(sd / "model")
    rows = [json.loads(x) for x in (prep / "sft_test.tok.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    template = (TEAM / "chat-format" / "chat_template.jinja").read_text(encoding="utf-8")
    info = {"stage": name, "arm": args.arm, "seconds": round(secs, 1), "cmd": cmd[1:],
            "prompt_parity_saved_model": prompt_parity(sd / "model", rows, template)}
    report = sd / "report.json"
    if report.exists():
        rep = json.loads(report.read_text())
        info.update({k: rep.get(k) for k in ("optimizer", "packing", "first_loss", "last_loss", "train_seconds",
                                             "seconds_per_100_steps", "checks")})
    write_manifest(sd, info, [report, sd / "model" / "model.safetensors"])
    return info


def grpo_config(work: Path, sft_model: Path, out: Path, optim: str, steps: int, max_seconds: float | None) -> dict:
    prep = work / "prep"
    train = {"algo": "objective_v1", "steps": steps, "prompts_per_step": 16, "k": 8, "max_new": 384,
             "temperature": 1.0, "lr": 1.0e-6, "ppo_epochs": 1, "grad_clip": 1.0, "eval_every": 10, "eval_k": 1,
             "sampler": "hf_batch", "micro_rows": 16, "state_dir": str(out / "state"), "state_every": 10,
             "screen": {"n": 64, "k": 8, "min_mixed": 8}, "time_sync": True,
             "save_trainable": str(out / "trainable_fp32.pt"), "save_trainable_dtype": "float32"}
    if optim == "hybrid_muon":
        train.update(optim="hybrid_muon", muon_lr=2.0e-5)
    if max_seconds:  # stops, evaluates and saves where training got to (a deadline would only pause)
        train["max_train_seconds"] = float(max_seconds)
    return {"seed": 1234, "device": "cuda", "out_dir": str(out),
            "model": {"kind": "hf", "path": str(sft_model), "dtype": "float32", "stop_at_eos": True},
            "task": {"kind": "math", "tasks": str(prep / "gsm_tasks.jsonl"), "train_ids": str(prep / "gsm_train_ids.txt"),
                     "heldout_ids": str(prep / "gsm_heldout_ids.txt"), "instruction_file": None,
                     "chat_template_kwargs": {"effort": EFFORT}},
            "train": train}


def stage_grpo(args) -> dict:
    import yaml  # noqa: PLC0415

    work = Path(args.work)
    sd = work / (getattr(args, "tag", None) or f"grpo_{args.optim}")
    sd.mkdir(parents=True, exist_ok=True)
    cfg = grpo_config(work, Path(args.sft_model), sd, args.optim, args.steps, args.max_seconds)
    cfg_path = sd / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    secs = run([PY, str(TEAM / "rl-loop" / "train.py"), "--config", str(cfg_path)], sd / "train.log",
               env=gpu_env(args.gpu))
    summary = sd / "summary.json"
    status = sd / "status.json"
    info = {"stage": sd.name, "seconds": round(secs, 1),
            "result": json.loads((summary if summary.exists() else status).read_text()) if (summary.exists()
                                                                                             or status.exists()) else None}
    files = [cfg_path, summary, sd / "metrics.jsonl"]
    if (sd / "trainable_fp32.pt").exists():
        files.append(build_final_model(Path(args.sft_model), sd / "trainable_fp32.pt", sd / "final_model"))
        (sd / "trainable_fp32.pt").unlink()
    write_manifest(sd, {k: v for k, v in info.items() if k != "result"}, files)
    return info


def build_final_model(sft_model: Path, weights: Path, out: Path) -> Path:
    """SFT model folder + the GRPO run's fp32 trainable weights -> a loadable model folder (G10 template installed)."""
    import shutil  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    model = AutoModelForCausalLM.from_pretrained(str(sft_model), dtype=torch.float32)
    state = torch.load(weights, map_location="cpu")
    params = dict(model.named_parameters())
    missing = [n for n in state if n not in params]
    if missing:
        raise RuntimeError(f"{len(missing)} trained weights are not parameters of the SFT model, e.g. {missing[:3]}")
    with torch.no_grad():
        for n, w in state.items():
            if w.dtype != torch.float32 or w.shape != params[n].shape:
                raise RuntimeError(f"{n}: {w.dtype} {tuple(w.shape)} does not fit {tuple(params[n].shape)} float32")
            params[n].copy_(w)
    model.save_pretrained(str(out), safe_serialization=True)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"):
        if (sft_model / f).is_file():
            shutil.copy(sft_model / f, out / f)
    return out / "model.safetensors"


def gsm_eval(model_dir: Path, prep: Path, out: Path, batch: int = 32, max_new: int = 384) -> dict:
    """Greedy GSM8K-test accuracy (math_check), closed-turn and boxed rates, on the fixed GSM_EVAL problems."""
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    mc = load_module("g10_math_check", TEAM / "rl-data" / "math_check.py")
    by_id = {json.loads(x)["id"]: json.loads(x) for x in (prep / "gsm_tasks.jsonl").read_text().splitlines() if x}
    ids = (prep / "gsm_eval_ids.txt").read_text().split()
    template = (TEAM / "chat-format" / "chat_template.jinja").read_text(encoding="utf-8")
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.float32).cuda().eval()
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    eos = [i for i in {im_end, tok.eos_token_id} if i is not None]
    prompts = [tok.apply_chat_template([{"role": "user", "content": by_id[i]["problem"]}], chat_template=template,
                                       add_generation_prompt=True, tokenize=True, effort=EFFORT) for i in ids]
    prompts = [list(p["input_ids"] if hasattr(p, "keys") else p) for p in prompts]
    order = sorted(range(len(ids)), key=lambda j: len(prompts[j]))
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos[0]
    recs, t0, new_tokens = [None] * len(ids), time.time(), 0
    with torch.no_grad():
        for b in range(0, len(order), batch):
            idx = order[b:b + batch]
            width = max(len(prompts[j]) for j in idx)
            inp = torch.tensor([[pad] * (width - len(prompts[j])) + prompts[j] for j in idx]).cuda()
            att = torch.tensor([[0] * (width - len(prompts[j])) + [1] * len(prompts[j]) for j in idx]).cuda()
            gen = model.generate(input_ids=inp, attention_mask=att, max_new_tokens=max_new, do_sample=False,
                                 eos_token_id=eos, pad_token_id=pad)[:, width:].tolist()
            for j, g in zip(idx, gen):
                stop = next((n for n, t in enumerate(g) if t in eos), None)
                body = g[:stop] if stop is not None else g
                new_tokens += len(body) + (stop is not None)
                text = tok.decode(body, skip_special_tokens=True)
                r = mc.score(by_id[ids[j]], text)
                recs[j] = {"id": ids[j], "reward": r["reward"], "closed": stop is not None,
                           "boxed": "\\boxed{" in text, "tokens": len(body), "text": text}
    secs = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    n = len(recs)
    return {"n": n, "accuracy": round(sum(r["reward"] for r in recs) / n, 4),
            "closed_turn": round(sum(r["closed"] for r in recs) / n, 4),
            "boxed": round(sum(r["boxed"] for r in recs) / n, 4),
            "mean_tokens": round(sum(r["tokens"] for r in recs) / n, 1),
            "seconds": round(secs, 1), "new_tokens_per_s": round(new_tokens / max(secs, 1e-9), 1)}


def stage_eval(args) -> dict:
    work = Path(args.work)
    prep = work / "prep"
    sd = work / f"eval_{args.name}"
    sd.mkdir(parents=True, exist_ok=True)
    model = Path(args.model)
    info = {"stage": sd.name, "model": str(model)}
    env = gpu_env(args.gpu)
    if args.gpu is not None:
        os.environ.update(env)
    info["gsm8k_test"] = gsm_eval(model, prep, sd / "gsm_generations.jsonl")
    rep = sd / "bench_eval.json"
    info["bench_eval_seconds"] = round(run(
        [PY, str(TEAM / "sft" / "bench" / "evaluate.py"), "--model", str(model), "--tokenizer",
         str(model / "tokenizer.json"), "--split-file", str(prep / "sft_test.tok.jsonl"), "--metrics", "bpb,format,forget",
         "--gen-limit", str(DOLLY_TEST_ROWS), "--generations-out", str(sd / "format_generations.jsonl"),
         "--device", "cuda", "--out", str(rep)], sd / "eval.log", env=env), 1)
    info["bench_eval"] = json.loads(rep.read_text())
    (sd / "summary.json").write_text(json.dumps(info, indent=1))
    write_manifest(sd, {"stage": sd.name, "model": str(model)}, [rep, sd / "summary.json"])
    return info


def stage_smoke(args) -> dict:
    """GPU environment, the touched tests on the GPU box, 20 SFT updates per optimizer, 2 GRPO steps."""
    import torch  # noqa: PLC0415

    work = Path(args.work)
    sd = work / "smoke"
    sd.mkdir(parents=True, exist_ok=True)
    log = sd / "smoke.log"
    info = {"stage": "smoke", "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    tok_env = {"G10_BENCH_TOKENIZER": str(work / "src" / "model" / "tokenizer.json")}
    tests = {}
    for folder in ("sft/bench/tests", "rl-loop/test_hybrid_optim_rl.py", "rl-data/test_build_gsm.py"):
        t0 = time.time()
        try:
            run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(TEAM / folder)], log, env=tok_env)
            tests[folder] = {"pass": True, "seconds": round(time.time() - t0, 1)}
        except RuntimeError as exc:
            tests[folder] = {"pass": False, "error": str(exc).splitlines()[0]}
    info["tests"] = tests
    smoke_args = argparse.Namespace(**{**vars(args), "seed": 1, "pack": True})
    info["sft_adamw_20"] = stage_sft(argparse.Namespace(**{**vars(smoke_args), "arm": "adamw", "gpu": 0}),
                                     steps=20, tag="smoke_sft_adamw")
    info["sft_hybrid_20"] = stage_sft(argparse.Namespace(**{**vars(smoke_args), "arm": "hybrid_muon", "gpu": 0}),
                                      steps=20, tag="smoke_sft_hybrid")
    g_args = argparse.Namespace(work=args.work, sft_model=str(work / "smoke_sft_adamw" / "model"), optim="adamw",
                                steps=2, gpu=0, max_seconds=None, tag="smoke_grpo")
    info["grpo_2"] = stage_grpo(g_args)
    (sd / "summary.json").write_text(json.dumps(info, indent=1))
    failed = [k for k, v in tests.items() if not v["pass"]]
    if failed:  # the full run must not start on a box where the touched tests fail
        raise RuntimeError(f"smoke: tests failed on the GPU box: {failed}")
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("fetch", "prep", "smoke", "sft", "grpo", "eval"))
    ap.add_argument("--work", required=True)
    ap.add_argument("--arm", choices=("adamw", "hybrid_muon"), default="adamw")
    ap.add_argument("--muon-lr", default=MUON_LR)
    ap.add_argument("--pack", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--sft-model")
    ap.add_argument("--optim", choices=("adamw", "hybrid_muon"), default="adamw")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--max-seconds", type=float, help="grpo: train.max_train_seconds")
    ap.add_argument("--tag", help="grpo: output folder name (default grpo_<optim>)")
    ap.add_argument("--model")
    ap.add_argument("--name")
    args = ap.parse_args(argv)
    fn = {"fetch": lambda a: {"stage": "fetch", **{k: str(v) for k, v in download(Path(a.work)).items()}},
          "prep": stage_prep, "smoke": stage_smoke, "sft": stage_sft, "grpo": stage_grpo, "eval": stage_eval}
    info = fn[args.stage](args)
    print(json.dumps({"stage": args.stage, "done": True, **{k: v for k, v in info.items() if k in ("seconds", "stage")}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
