#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Speculative decoding (toy but faithful) vs. greedy baseline.

- Draft model proposes a chunk of tokens fast (small model).
- Verifier (larger model) checks the proposed tokens with one forward pass using KV-cache.
- If verifier's top-1 matches a proposed token, we accept it.
- On the first mismatch, we take the verifier's top-1 for that step and stop verifying the rest of the draft chunk.
- Repeat until max_new_tokens generated.

This is a simplified, readable implementation intended for educational benchmarking.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time


@dataclass
class ModelBundle:
    name: str
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: str


def load_bundle(model_name: str, device: Optional[str] = None) -> ModelBundle:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model.to(device)
    model.eval()
    return ModelBundle(model_name, model, tok, device)


@torch.inference_mode()
def greedy_decode(bundle: ModelBundle, prompt: str, max_new_tokens: int = 64) -> Tuple[str, float, List[int]]:
    tok = bundle.tokenizer
    model = bundle.model
    device = bundle.device

    inputs = tok(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]

    start = time.perf_counter()
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.eos_token_id,
        use_cache=True,
    )[0]
    elapsed = time.perf_counter() - start

    text = tok.decode(output_ids, skip_special_tokens=True)
    return text, elapsed, output_ids.tolist()


@torch.inference_mode()
def speculative_decode(
    draft: ModelBundle,
    verifier: ModelBundle,
    prompt: str,
    max_new_tokens: int = 64,
    draft_chunk: int = 8,
) -> Tuple[str, float, List[int], float]:
    """
    Simplified speculative decoding using top-1 agreement:
    - Generate up to `draft_chunk` draft tokens (greedy) using the draft model with cache.
    - Run the verifier forward pass incrementally; accept tokens while its top-1 matches.
    - On first mismatch in the chunk, insert verifier's top-1 and end the chunk.
    Returns (text, latency_seconds, token_ids, acceptance_rate)
    """
    dtok = draft.tokenizer
    vtok = verifier.tokenizer
    assert dtok.vocab_size == vtok.vocab_size, "Draft and verifier must share a vocabulary/tokenizer family."

    device = draft.device
    dmodel = draft.model
    vmodel = verifier.model

    # Tokenize prompt with *verifier* tokenizer to keep the decode consistent.
    v_inputs = vtok(prompt, return_tensors="pt").to(device)
    v_input_ids = v_inputs["input_ids"]
    v_attn = v_inputs["attention_mask"]

    # Also encode with draft tok (identical BPE family for GPT-2).
    d_inputs = dtok(prompt, return_tensors="pt").to(device)
    d_input_ids = d_inputs["input_ids"]
    d_attn = d_inputs["attention_mask"]

    # Initialize caches for incremental decoding
    # Verifier first pass to get past_key_values
    v_out = vmodel(input_ids=v_input_ids, attention_mask=v_attn, use_cache=True)
    v_past = v_out.past_key_values

    # draft initial cache
    d_out = dmodel(input_ids=d_input_ids, attention_mask=d_attn, use_cache=True)
    d_past = d_out.past_key_values

    generated = []
    accepted_tokens = 0
    proposed_tokens = 0

    start = time.perf_counter()

    for _ in range(max_new_tokens):
        # 1) DRAFT proposes up to draft_chunk tokens greedily using its cache.
        d_proposals = []
        d_last_token = d_input_ids[:, -1]  # last known input (or last generated)
        d_pv = d_past

        for _k in range(draft_chunk):
            d_out = dmodel(input_ids=d_last_token, use_cache=True, past_key_values=d_pv)
            d_logits = d_out.logits[:, -1, :]
            d_next = torch.argmax(d_logits, dim=-1)
            d_proposals.append(d_next.item())
            proposed_tokens += 1
            d_pv = d_out.past_key_values
            d_last_token = d_next  # feed back greedily for further drafts

        # 2) VERIFIER checks the proposals incrementally using its cache.
        accepted_in_chunk = 0
        v_pv = v_past
        v_last_token = v_input_ids[:, -1]

        for i, proposal in enumerate(d_proposals):
            v_out = vmodel(input_ids=v_last_token, use_cache=True, past_key_values=v_pv)
            v_logits = v_out.logits[:, -1, :]
            v_next = torch.argmax(v_logits, dim=-1).item()
            v_pv = v_out.past_key_values

            if v_next == proposal:
                # Accept the draft token
                accepted_tokens += 1
                accepted_in_chunk += 1
                generated.append(proposal)
                v_last_token = torch.tensor([[proposal]], device=device)
            else:
                # First mismatch: take verifier token and end the chunk
                generated.append(v_next)
                v_last_token = torch.tensor([[v_next]], device=device)
                accepted_in_chunk += 1  # we still produced a valid token
                break

        # Update verifier past cache and input ids with what we actually emitted
        # We have appended accepted_in_chunk tokens (accepted match OR first mismatch)
        # To keep cache consistent, we must replay those tokens through the verifier quickly:
        # But we already advanced cache incrementally above; so just update bookkeeping tensors:
        emitted = generated[-accepted_in_chunk:] if accepted_in_chunk > 0 else []
        if emitted:
            add_ids = torch.tensor([emitted], device=device)
            v_input_ids = torch.cat([v_input_ids, add_ids], dim=1)
            # v_pv is already advanced in the loop; keep it.
            v_past = v_pv
        else:
            # This would only happen if draft_chunk==0 (not the case).
            pass

        # Also advance the draft context by the emitted tokens so its cache stays in sync for next proposals.
        if emitted:
            for t in emitted:
                d_out = dmodel(
                    input_ids=torch.tensor([[t]], device=device),
                    use_cache=True,
                    past_key_values=d_past,
                )
                d_past = d_out.past_key_values
                d_input_ids = torch.cat([d_input_ids, torch.tensor([[t]], device=device)], dim=1)

        # Early stop if EOS emitted
        if len(generated) > 0 and generated[-1] == vtok.eos_token_id:
            break

    elapsed = time.perf_counter() - start
    acc_rate = (accepted_tokens / max(1, proposed_tokens)) if proposed_tokens > 0 else 0.0

    # Decode final text using verifier tokenizer for consistency
    full_ids = torch.cat([v_inputs["input_ids"].to("cpu"), torch.tensor([generated])], dim=1)[0].tolist()
    text = vtok.decode(full_ids, skip_special_tokens=True)

    return text, elapsed, full_ids, acc_rate
