# verifies that batching works by running the prefill+decode
# then running batched decode in engine and when temp=0 
# we should expect perect match

#    python verify_batching.py --model Qwen/Qwen3-0.6B --device cpu --max-tokens 16


from __future__ import annotations

import argparse
import sys

import torch

from miniengine.core import Request, SamplingParams
from miniengine.engine import Engine


PROMPTS = [
    "What is the capital of USA?",
    "Write python code to print 'hello'"
]


def make_request(engine, prompt, max_tokens):
    input_ids = engine.tokenize_messages([{"role": "user", "content": prompt}])
    return Request(
        request_id=f"req-{prompt[:8]}", # can set to uuid or smth idk
        input_ids=input_ids,
        sampling_params=SamplingParams(
            max_new_tokens=max_tokens,
            temperature=0.0,  # greedy -> deterministic
        ),
    )


def run_unbatched(engine, prompt, max_tokens):
    req = make_request(engine, prompt, max_tokens)
    tokens= []

    tok = engine.prefill(req)
    req.output_ids.append(tok)
    tokens.append(tok)

    while len(tokens) < max_tokens and not engine.is_stop_token(tok):
        tok = engine.decode_step(req)
        req.output_ids.append(tok)
        tokens.append(tok)

    if req.slot_idx is not None:
        engine.free_slot(req.slot_idx)
    return tokens

# put both the prompts here
def run_batched(engine, prompts, max_tokens):
    reqs = [make_request(engine, p, max_tokens) for p in prompts]
    outputs = [[] for _ in prompts]

    # we still do prefill indiviudally 
    for req, out in zip(reqs, outputs):
        tok = engine.prefill(req)
        req.output_ids.append(tok)
        out.append(tok)

    active = list(range(len(reqs)))
    while active:
        live_reqs = [reqs[i] for i in active]
        new_tokens = engine.batched_decode(live_reqs)

        still_active= []
        for slot, (req_idx, tok) in enumerate(zip(active, new_tokens)):
            reqs[req_idx].output_ids.append(tok)
            outputs[req_idx].append(tok)
            done = (
                len(outputs[req_idx]) >= max_tokens
                or engine.is_stop_token(tok)
            )
            if not done:
                still_active.append(req_idx)
        active = still_active

    for r in reqs:
        if r.slot_idx is not None:
            engine.free_slot(r.slot_idx)
    return outputs


def compare(gold, test):
    all_ok = True
    for i, (g, t) in enumerate(zip(gold, test)):
        match = g == t
        all_ok = all_ok and match
        print(f"  prompt #{i}: {'OK' if match else 'FAIL'}")
        if not match:
            print(f"    gold: {g}")
            print(f"    test: {t}")
    return all_ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--max-tokens", type=int, default=16)
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)

    print(f"loading engine  model={args.model}  device={args.device}  dtype={args.dtype}")
    # works on local we should increase this on VM
    engine = Engine(
        model_path=args.model, dtype=dtype, device=args.device,
        max_running=4, max_seq_len=128,
    )

    # unbatch
    print("baseline")
    gold = [run_unbatched(engine, p, args.max_tokens) for p in PROMPTS]
    for i, (prompt, toks) in enumerate(zip(PROMPTS, gold)):
        text = engine.tokenizer.decode(toks, skip_special_tokens=True)
        print(f"prompt #{i}: {prompt!r}") #repr call
        print(f"tokens: {toks}")
        print(f"text: {text!r}")

    # batch version and compare
    print("batched run")
    if not hasattr(engine, "batched_decode"):
        print("engine.batched_decode is not implemented yet.")
        return 0

    test = run_batched(engine, PROMPTS, args.max_tokens)
    ok_batched = compare(gold, test)
    print(f"{'batched output matches gold' if ok_batched else 'not match'}")

    if hasattr(engine, "mixed_step"):
        all_mixed_ok = True
        for chunk in (1, 4, 16):
            print(f"mixed run, prefill_chunk_size={chunk}")
            test_mixed = run_mixed(engine, PROMPTS, args.max_tokens, chunk)
            ok = compare(gold, test_mixed)
            print(f"  {'mixed output matches gold' if ok else 'not match'}")
            all_mixed_ok = all_mixed_ok and ok
        ok_overall = ok_batched and all_mixed_ok
    else:
        ok_overall = ok_batched

    return 0 if ok_overall else 1


def run_mixed(engine, prompts, max_tokens, prefill_chunk_size):
    reqs = [make_request(engine, p, max_tokens) for p in prompts]
    outputs = [[] for _ in prompts]

    # allocate slots up front; mixed_step assumes they're already there.
    for r in reqs:
        r.slot_idx = engine.alloc_slot(r)
        r.cache_len = 0

    active = list(range(len(reqs)))
    while active:
        ops = []
        for i in active:
            r = reqs[i]
            if r.cache_len < r.num_input_tokens:
                remaining = r.num_input_tokens - r.cache_len
                ops.append((r, min(remaining, prefill_chunk_size)))
            else:
                ops.append((r, 1))

        results = engine.mixed_step(ops)

        new_active = []
        for active_pos, (req_idx, (op, tok)) in enumerate(
            zip(active, zip(ops, results))
        ):
            req = ops[active_pos][0]
            if tok is not None:
                req.output_ids.append(tok)
                outputs[req_idx].append(tok)
                if (
                    len(outputs[req_idx]) >= max_tokens
                    or engine.is_stop_token(tok)
                ):
                    continue
            new_active.append(req_idx)
        active = new_active

    for r in reqs:
        if r.slot_idx is not None:
            engine.free_slot(r.slot_idx)
    return outputs


if __name__ == "__main__":
    sys.exit(main())
