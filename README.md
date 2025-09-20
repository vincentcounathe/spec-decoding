# spec-decoding — Speculative Decoding Playground

A minimal, readable implementation of **speculative decoding** vs a **greedy baseline** using GPT-2 small (draft) and GPT-2 medium (verifier). Includes benchmarks and an optional plot of latency vs output length.

## Why
Speculative decoding is a hot inference acceleration trick: use a **small draft model** to propose multiple tokens, then **verify** them with a larger model in fewer passes.

This repo is intentionally small and practical for interviews / portfolio.

## Setup

```bash
git clone https://github.com/vincentcounathe/spec-decoding
cd spec-decoding
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
