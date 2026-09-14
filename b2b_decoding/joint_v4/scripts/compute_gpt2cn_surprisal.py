#!/usr/bin/env python3
"""
Compute word-level surprisal from a Chinese GPT-2 (GPT2CN) for B2B lexical analysis.

Writes, for each story section:
  <out>/section_XXX/X_word_gpt2cn_surprisal.npy          (n_words × 1)
  <out>/section_XXX/X_word_gpt2cn_surprisal_feature_names.txt
  <out>/section_XXX/X_word_gpt2cn_surprisal_meta.json

Word surprisal = sum of per-token -log p(token | previous) over tokens owned by
that word (same attribution rule as analysisEV/sae_extract_features.py).

Default model: uer/gpt2-chinese-cluecorpussmall (CPU-friendly Chinese GPT-2).
Override with --model for a GPT-2 XL Chinese checkpoint if available.

Example:
  python compute_gpt2cn_surprisal.py \\
    --timing-root /home/haolun52/orcd/pool/extracted_sections_wordlocked_shared/_shared_wordlocked_features \\
    --out-root    /home/haolun52/orcd/pool/extracted_sections_wordlocked_shared/_shared_wordlocked_features \\
    --sections 1 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def map_tokens_to_words(words: list[str], offset_mapping: list[tuple[int, int]]):
    """Map each tokenizer token to the word with maximal character-span overlap."""
    # Rebuild character spans by concatenating words (Chinese: no spaces).
    word_spans: list[tuple[int, int]] = []
    cursor = 0
    for w in words:
        n = len(w)
        word_spans.append((cursor, cursor + n))
        cursor += n

    tokens_per_word: list[list[int]] = [[] for _ in words]
    token_owner = np.full(len(offset_mapping), -1, dtype=int)

    for ti, (a, b) in enumerate(offset_mapping):
        if b <= a:
            continue  # special / empty
        best_i, best_ov = -1, 0
        for wi, (wa, wb) in enumerate(word_spans):
            ov = max(0, min(b, wb) - max(a, wa))
            if ov > best_ov:
                best_ov, best_i = ov, wi
        if best_i >= 0 and best_ov > 0:
            tokens_per_word[best_i].append(ti)
            # First owner wins for surprisal attribution (no double count)
            if token_owner[ti] < 0:
                token_owner[ti] = best_i
    return tokens_per_word, token_owner, cursor


def compute_section_surprisal(
    words: list[str],
    model,
    tokenizer,
    device: str,
    max_length: int,
    stride: int,
) -> np.ndarray:
    import torch

    text = "".join(words)
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"][0]
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"][0].tolist()]
    _, token_owner, _ = map_tokens_to_words(words, offsets)

    n_tok = int(input_ids.shape[0])
    surprisal_tok = np.full(n_tok, np.nan, dtype=np.float64)

    # Sliding windows for long sections
    starts = list(range(0, max(n_tok - 1, 1), stride))
    if not starts:
        starts = [0]
    # Ensure last tokens covered
    if starts[-1] + max_length < n_tok:
        starts.append(max(0, n_tok - max_length))

    model.eval()
    with torch.no_grad():
        for start in starts:
            end = min(start + max_length, n_tok)
            if end - start < 2:
                continue
            chunk = input_ids[start:end].unsqueeze(0).to(device)
            out = model(chunk)
            logprobs = torch.log_softmax(out.logits[0, :-1, :], dim=-1)  # (L-1, V)
            targets = chunk[0, 1:]
            for local_i in range(targets.shape[0]):
                gpos = start + local_i + 1  # position of predicted token
                if not np.isnan(surprisal_tok[gpos]):
                    continue  # already filled by earlier window
                tgt = int(targets[local_i].item())
                surprisal_tok[gpos] = float(-logprobs[local_i, tgt].item())

    # First token has no context → leave NaN → treated as 0 contribution
    word_surp = np.zeros(len(words), dtype=np.float64)
    for ti, wi in enumerate(token_owner):
        if wi < 0:
            continue
        s = surprisal_tok[ti]
        if np.isfinite(s):
            word_surp[wi] += s
    return word_surp.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--timing-root",
        required=True,
        help="Root containing section_XXX/word_timing_relative.csv (or word_timing.csv)",
    )
    ap.add_argument(
        "--out-root",
        required=True,
        help="Root to write X_word_gpt2cn_surprisal.* (often same as timing-root)",
    )
    ap.add_argument("--sections", type=int, nargs="+", default=[1, 2])
    ap.add_argument(
        "--model",
        default="uer/gpt2-chinese-cluecorpussmall",
        help="HuggingFace model id for Chinese GPT-2",
    )
    ap.add_argument("--device", default="cpu", help="cpu | cuda | cuda:0 | …")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--dtype", default="float32", choices=("float32", "float16", "bfloat16"))
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise SystemExit(
            "Need torch + transformers. Example:\n"
            "  python3 -m venv .venv_gpt2 && . .venv_gpt2/bin/activate\n"
            "  pip install torch transformers numpy pandas\n"
            f"Import error: {e}"
        )

    timing_root = Path(args.timing_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {args.model} on {args.device} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype)
    model.to(args.device)
    model.eval()

    for sid in args.sections:
        sec = f"section_{sid:03d}"
        tdir = timing_root / sec
        csv_path = tdir / "word_timing_relative.csv"
        if not csv_path.is_file():
            csv_path = tdir / "word_timing.csv"
        if not csv_path.is_file():
            raise SystemExit(f"Missing timing CSV under {tdir}")
        df = pd.read_csv(csv_path)
        if "word" not in df.columns:
            raise SystemExit(f"{csv_path} missing 'word' column")
        words = [str(w) for w in df["word"].tolist()]
        print(f"[{sec}] n_words={len(words)} computing GPT2CN surprisal …")
        surp = compute_section_surprisal(
            words, model, tokenizer, args.device, args.max_length, args.stride
        )
        assert surp.shape == (len(words),)
        odir = out_root / sec
        odir.mkdir(parents=True, exist_ok=True)
        npy = odir / "X_word_gpt2cn_surprisal.npy"
        np.save(npy, surp.reshape(-1, 1))
        with open(odir / "X_word_gpt2cn_surprisal_feature_names.txt", "w") as f:
            f.write("gpt2cn_surprisal\n")
        meta = {
            "model_name": args.model,
            "device": args.device,
            "max_length": args.max_length,
            "stride": args.stride,
            "dtype": args.dtype,
            "n_words": len(words),
            "surprisal_units": "nats (sum of token -log p)",
            "mean": float(np.mean(surp)),
            "std": float(np.std(surp)),
            "min": float(np.min(surp)),
            "max": float(np.max(surp)),
            "timing_csv": str(csv_path),
        }
        with open(odir / "X_word_gpt2cn_surprisal_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(
            f"  wrote {npy}  mean={meta['mean']:.3f} "
            f"range=[{meta['min']:.3f},{meta['max']:.3f}]"
        )

    print("Done.")


if __name__ == "__main__":
    main()
