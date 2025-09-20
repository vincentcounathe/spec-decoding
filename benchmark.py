#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark: greedy vs. speculative decoding.

- Loads GPT-2 small (draft) and GPT-2 medium (verifier).
- Runs both methods on a small prompt set and several output lengths.
- Reports latency, tokens/sec, acceptance rates; optionally plots latency vs. length.

Usage:
    python benchmark.py --plot
"""

import argparse
import os
import statistics
from typing import List, Dict, Any
import time

import torch
import matplotlib.pyplot as plt

from spec_decoding import load_bundle, greedy_decode, speculative_decode


DEFAULT_PROMPTS = [
    "Explain diffusion models to a high school student in two sentences.",
    "Write a short paragraph about the importance of scaling laws in machine learning.",
    "Summarize the concept of speculative decoding in large language models.",
]

def quality_proxy(verifier_bundle, text: str) -> float:
    """
    A lightweight 'quality' proxy: average per-token logprob under the verifier model.
    Not a perfect metric, but correlates with plausibility.
    """
    tok = verifier_bundle.tokenizer
    model = verifier_bundle.model
    device = verifier_bundle.device

    enc = tok(text, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(**enc)
        logits = out.logits[:, :-1, :]
        labels = enc["input_ids"][:, 1:]
        logprobs = torch.log_softmax(logits, dim=-1)
        token_ll = logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        avg_ll = token_ll.mean().item()
    return avg_ll


def run_once(draft, verifier, prompt: str, max_new: int, draft_chunk: int, disable_spec: bool = False) -> Dict[str, Any]:
    if disable_spec:
        text, t, ids = greedy_decode(verifier, prompt, max_new)
        q = quality_proxy(verifier, text)
        return {"method": "greedy", "latency": t, "tokens": len(ids), "avg_ll": q, "acc_rate": None, "text": text}
    else:
        text, t, ids, acc = speculative_decode(draft, verifier, prompt, max_new, draft_chunk)
        q = quality_proxy(verifier, text)
        return {"method": "speculative", "latency": t, "tokens": len(ids), "avg_ll": q, "acc_rate": acc, "text": text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", default="gpt2", help="Draft model (e.g., gpt2)")
    parser.add_argument("--verifier", default="gpt2-medium", help="Verifier model (e.g., gpt2-medium)")
    parser.add_argument("--max_new_tokens", type=int, nargs="+", default=[16, 32, 64], help="List of output lengths")
    parser.add_argument("--draft_chunk", type=int, default=8, help="Number of tokens per speculative draft chunk")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per setting to average")
    parser.add_argument("--plot", action="store_true", help="Produce latency plot")
    args = parser.parse_args()

    draft = load_bundle(args.draft)
    verifier = load_bundle(args.verifier)

    results = []

    for L in args.max_new_tokens:
        for prompt in DEFAULT_PROMPTS:
            # Greedy baseline (verifier model)
            for _ in range(args.repeats):
                r = run_once(draft, verifier, prompt, L, args.draft_chunk, disable_spec=True)
                r["max_new"] = L
                r["prompt"] = prompt
                results.append(r)

            # Speculative
            for _ in range(args.repeats):
                r = run_once(draft, verifier, prompt, L, args.draft_chunk, disable_spec=False)
                r["max_new"] = L
                r["prompt"] = prompt
                results.append(r)

    # Aggregate
    def agg(method: str, L: int):
        subset = [r for r in results if r["method"] == method and r["max_new"] == L]
        lat = [r["latency"] for r in subset]
        q = [r["avg_ll"] for r in subset]
        acc = [r["acc_rate"] for r in subset if r["acc_rate"] is not None]
        return {
            "method": method,
            "L": L,
            "latency_mean": statistics.mean(lat) if lat else float("nan"),
            "latency_std": statistics.pstdev(lat) if len(lat) > 1 else 0.0,
            "avg_ll_mean": statistics.mean(q) if q else float("nan"),
            "accept_mean": statistics.mean(acc) if acc else None,
        }

    lines = []
    for L in args.max_new_tokens:
        g = agg("greedy", L)
        s = agg("speculative", L)
        speedup = g["latency_mean"] / s["latency_mean"] if (g["latency_mean"] and s["latency_mean"]) else float("nan")
        lines.append((L, g, s, speedup))

    print("\n=== Benchmark Summary ===")
    for L, g, s, sp in lines:
        print(f"L={L:>3}  | Greedy: {g['latency_mean']:.3f}s  | Spec: {s['latency_mean']:.3f}s  "
              f"| Speedup x{sp:.2f}  | Spec accept {s['accept_mean'] if s['accept_mean'] is not None else '—'}  "
              f"| LL(greedy) {g['avg_ll_mean']:.3f}  | LL(spec) {s['avg_ll_mean']:.3f}")

    if args.plot:
        # Latency vs length
        Ls = args.max_new_tokens
        greedy_means = [agg("greedy", L)["latency_mean"] for L in Ls]
        spec_means = [agg("speculative", L)["latency_mean"] for L in Ls]

        plt.figure(figsize=(6,4))
        plt.plot(Ls, greedy_means, marker="o", label="Greedy (verifier)")
        plt.plot(Ls, spec_means, marker="o", label="Speculative")
        plt.xlabel("Max new tokens")
        plt.ylabel("Latency (s)")
        plt.title("Latency vs Output Length")
        plt.legend()
        plt.tight_layout()
        out_path = "latency_vs_length.png"
        plt.savefig(out_path, dpi=200)
        print(f"\nSaved plot: {out_path}")


if __name__ == "__main__":
    main()
