"""hf_batch sampler_max_rows: chunked generate calls, unchanged behaviour when unset. Tiny random model, CPU."""
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rollout  # noqa: E402
from policy import hf_policy  # noqa: E402

parametrize = pytest.mark.parametrize


def tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=32, intermediate_size=32, num_hidden_layers=2,
                                        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                                        eos_token_id=None, bos_token_id=None, pad_token_id=None))


def counting_policy():
    pol = hf_policy(tiny_llama())
    calls = []
    gen = pol.model.generate

    def wrapped(*a, **kw):
        calls.append(kw["input_ids"].shape[0])
        return gen(*a, **kw)

    pol.model.generate = wrapped
    return pol, calls


PROMPTS = [[1, 2, 3], [4, 5], [6], [7, 8, 9, 10], [11, 12]]


@parametrize("max_rows, want", [(None, [10]), (10, [10]), (64, [10]), (4, [4, 4, 2]), (2, [2, 2, 2, 2, 2]),
                                (1, [2, 2, 2, 2, 2]), (3, [2, 2, 2, 2, 2])])
def test_rows_per_generate(max_rows, want):
    pol, calls = counting_policy()
    rolls = rollout.HFBatchSampler(pol, 0, max_rows).sample(PROMPTS, 2, 4, 1.0, torch.Generator().manual_seed(1))
    assert calls == want
    assert len(rolls) == len(PROMPTS)
    for p, r in zip(PROMPTS, rolls):
        assert r.seq.shape == (2, len(p) + 4) and r.cmask.shape == (2, 4)
        assert r.seq[:, :len(p)].tolist() == [p, p]


def test_unset_is_one_generate_and_same_draws_as_before():
    a_pol, _ = counting_policy()
    b_pol, _ = counting_policy()
    a = rollout.HFBatchSampler(a_pol, 0).sample(PROMPTS, 2, 4, 1.0, torch.Generator().manual_seed(7))
    b = rollout.HFBatchSampler(b_pol, 0, 10).sample(PROMPTS, 2, 4, 1.0, torch.Generator().manual_seed(7))
    assert all(torch.equal(x.seq, y.seq) for x, y in zip(a, b))


def test_greedy_chunks_equal_one_batch():
    a_pol, _ = counting_policy()
    b_pol, _ = counting_policy()
    a = rollout.HFBatchSampler(a_pol, 0).sample(PROMPTS, 1, 5, 0.0)
    b = rollout.HFBatchSampler(b_pol, 0, 2).sample(PROMPTS, 1, 5, 0.0)
    assert all(torch.equal(x.seq[:, -5:], y.seq[:, -5:]) for x, y in zip(a, b))


@parametrize("bad", [0, -1])
def test_bad_max_rows(bad):
    with pytest.raises(ValueError, match="at least 1"):
        rollout.HFBatchSampler(hf_policy(tiny_llama()), 0, bad)


def test_make_sampler_hf_default_is_unchunked():
    pol = hf_policy(tiny_llama())
    assert rollout.make_sampler("hf_batch", pol, None, 0, max_rows=None).max_rows is None
    assert rollout.make_sampler("hf_batch", pol, None, 0, max_rows=64).max_rows == 64
