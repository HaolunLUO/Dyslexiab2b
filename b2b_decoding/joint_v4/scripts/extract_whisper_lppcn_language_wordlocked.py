#!/usr/bin/env python3
"""
Re-extract Whisper *decoder* word-locked language embeddings from lPPCN WAVs.

Aligned with hassonlab/247-pickling whisper-paper-1
  tfsemb_genemb_whisper_var_win.py:

  language_audio_fused  <-> model_type in {full, full-onset}
      decoder_hidden_states[layer] at the last context token

  language_text_only    <-> model_type == de-only
      zero encoder_outputs + zero cross_attn_head_mask, then same decoder pick

Default windowing matches full-onset (TFS comments):
  clip_end = word_onset + 0.2325 s
  clip_start = max(0, clip_end - 30)

Context tokens = <|startoftranscript|> + tokens of all words with
  clip_start <= onset <= current_onset  (sorted by onset), matching var_win.

Per-word embedding: mean over the current word's trailing subword tokens
(ave_emb), which collapses to the last token when the word is a single piece.

lPPCN is Chinese → use multilingual openai/whisper-tiny (not .en).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


DEFAULT_SHARED = Path(
    os.environ.get(
        "B2B_SHARED_FEATURES_DIR",
        "/orcd/pool/005/haolun52/extracted_sections_wordlocked_shared/_shared_wordlocked_features",
    )
)

SR = 16000
CLIP_LEN_S = 30.0
N_ENCODER_FRAMES = 1500
DEFAULT_CLIP_END_OFFSET_S = 0.2325


def _load_wav_16k(path: Path) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(SR),
        "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"WARNING: ffmpeg load failed ({e}); falling back to soundfile/librosa")

    import soundfile as sf

    audio, orig_sr = sf.read(str(path), always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32, copy=False)
    if int(orig_sr) == SR:
        return audio
    import librosa

    return librosa.resample(audio, orig_sr=int(orig_sr), target_sr=SR).astype(np.float32)


def _clip_bounds(onset_s: float, clip_end_offset_s: float) -> tuple[float, float]:
    clip_end = float(onset_s + clip_end_offset_s)
    clip_start = float(np.max([0.0, clip_end - CLIP_LEN_S]))
    return clip_start, clip_end


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--section-id", type=int, default=1)
    ap.add_argument("--shared-dir", type=str, default=str(DEFAULT_SHARED))
    ap.add_argument("--model-name", type=str, required=True)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--hf-cache-dir", type=str, default=None)
    ap.add_argument("--max-words", type=int, default=50)
    ap.add_argument("--start-word-idx", type=int, default=0)
    ap.add_argument("--clip-end-offset-s", type=float, default=DEFAULT_CLIP_END_OFFSET_S)
    ap.add_argument(
        "--window-mode",
        type=str,
        default="full-onset",
        choices=["full-onset", "full"],
        help="full-onset: clip_end=onset+offset_s; full: clip_end=word offset_relative",
    )
    ap.add_argument(
        "--decoder-layer",
        type=int,
        default=3,
        help="HF decoder_hidden_states index (0=embed). Default 3 (paper-aligned for tiny).",
    )
    ap.add_argument(
        "--probe-all-decoder-layers",
        action="store_true",
        help="Write language_* for every decoder layer index as layer{k}_*.",
    )
    ap.add_argument(
        "--decoder-layers",
        type=str,
        default=None,
        help="Comma-separated decoder_hidden_states indices to extract (overrides --decoder-layer).",
    )
    ap.add_argument(
        "--token-pool",
        type=str,
        default="last",
        choices=["ave", "last"],
        help="ave: mean over current word subwords; last: only final subword token",
    )
    ap.add_argument(
        "--prefix-style",
        type=str,
        default="zh-transcribe",
        choices=["hasson", "sot", "zh-transcribe"],
        help="hasson=tokenize('<|startoftranscript|> '); sot=SOT only; zh-transcribe=SOT+zh+transcribe+notimestamps",
    )
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--stem", type=str, required=True)
    args = ap.parse_args()

    shared = Path(args.shared_dir)
    sec_dir = shared / f"section_{args.section_id:03d}"
    wav_path = sec_dir / f"task-lppCN_section_{args.section_id}.wav"
    timing_csv = sec_dir / "word_timing_relative.csv"
    if not wav_path.is_file():
        raise SystemExit(f"Missing wav: {wav_path}")
    if not timing_csv.is_file():
        raise SystemExit(f"Missing timing csv: {timing_csv}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.hf_cache_dir:
        os.environ["HF_HOME"] = args.hf_cache_dir
        os.environ["TRANSFORMERS_CACHE"] = args.hf_cache_dir

    words: list[str] = []
    onsets: list[float] = []
    offsets: list[float] = []
    with open(timing_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            words.append(row["word"])
            onsets.append(float(row["onset_relative"]))
            offsets.append(float(row["offset_relative"]))

    words_all = words
    onsets_all = np.asarray(onsets, dtype=np.float64)
    offsets_all = np.asarray(offsets, dtype=np.float64)

    i0 = args.start_word_idx
    i1 = min(i0 + args.max_words, len(words_all)) if args.max_words > 0 else len(words_all)
    n_words = i1 - i0
    if n_words <= 0:
        raise SystemExit("No words selected")

    audio = _load_wav_16k(wav_path)
    print(f"Loaded audio: {wav_path.name} sr={SR} len={len(audio) / SR:.2f}s")
    print(f"Using n_words={n_words} rows [{i0}:{i1})")
    print(f"window_mode={args.window_mode} clip_end_offset_s={args.clip_end_offset_s}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {args.model_name} device={device}")

    processor = WhisperProcessor.from_pretrained(args.model_name)
    tokenizer = processor.tokenizer
    # large-v3 may default to float16; keep compute dtype explicit and match inputs to it
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name, torch_dtype=torch.float32)
    model.eval()
    model.to(device)
    param_dtype = next(model.parameters()).dtype

    d_model = int(model.config.d_model)
    n_dec = int(model.config.decoder_layers)
    n_heads = int(model.config.decoder_attention_heads)
    # decoder_hidden_states: [0]=embed, [1..n_dec]=layers
    default_layer = int(args.decoder_layer)
    if args.decoder_layers:
        layers = [int(x) for x in args.decoder_layers.replace(":", ",").split(",") if x.strip() != ""]
        default_layer = layers[0] if default_layer not in layers else default_layer
    elif args.probe_all_decoder_layers:
        layers = list(range(0, n_dec + 1))
    else:
        layers = [default_layer]
    bad = [li for li in layers if li < 0 or li > n_dec]
    if bad:
        raise SystemExit(f"decoder layers {bad} out of range 0..{n_dec}")
    print(f"decoder_layers={n_dec} d_model={d_model} extracting layers={layers}")

    # Pre-tokenize every word once (Hasson uses conversation_df["token"] lists)
    word_token_ids: list[list[int]] = []
    for w in words_all:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 0:
            # empty / punctuation edge case: keep a single unk-ish fallback
            ids = tokenizer.encode(" ", add_special_tokens=False) or [tokenizer.eos_token_id]
        word_token_ids.append(ids)

    # Prefix tokens (Hasson overwrites en/transcribe attempt with SOT+space)
    if args.prefix_style == "hasson":
        prefix_tokens = tokenizer.tokenize("<|startoftranscript|> ")
        sot_ids = [int(x) for x in tokenizer.convert_tokens_to_ids(prefix_tokens)]
    elif args.prefix_style == "sot":
        sot_ids = tokenizer.encode("<|startoftranscript|>", add_special_tokens=False)
    else:
        # multilingual task prefix commonly used by Whisper generate()
        sot_ids = tokenizer.encode(
            "<|startoftranscript|><|zh|><|transcribe|><|notimestamps|>",
            add_special_tokens=False,
        )
    sot_ids = [int(x) for x in sot_ids if x is not None and int(x) >= 0]
    if not sot_ids:
        sot_ids = tokenizer.encode("<|startoftranscript|>", add_special_tokens=False)
    print(f"prefix_style={args.prefix_style} sot_ids={sot_ids} token_pool={args.token_pool}")

    # Outputs: variant -> layer -> matrix
    outs: dict[str, dict[int, np.ndarray]] = {
        "language_audio_fused": {li: np.empty((n_words, d_model), dtype=np.float32) for li in layers},
        "language_text_only": {li: np.empty((n_words, d_model), dtype=np.float32) for li in layers},
    }

    with torch.no_grad():
        for local_i, global_i in enumerate(range(i0, i1)):
            onset = float(onsets_all[global_i])
            if args.window_mode == "full":
                clip_end = float(offsets_all[global_i])
                clip_start = float(np.max([0.0, clip_end - CLIP_LEN_S]))
            else:
                clip_start, clip_end = _clip_bounds(onset, args.clip_end_offset_s)

            a0 = max(0, int(clip_start * SR))
            a1 = max(a0 + 1, min(len(audio), int(clip_end * SR)))
            chunk = audio[a0:a1]

            feats = processor.feature_extractor(
                chunk, sampling_rate=SR, return_tensors="pt"
            ).input_features.to(device=device, dtype=param_dtype)

            # Context: words with onset in [clip_start, current_onset], sorted by onset
            ctx_idxs = [
                j
                for j in range(0, global_i + 1)
                if clip_start <= float(onsets_all[j]) <= onset + 1e-9
            ]
            if not ctx_idxs:
                ctx_idxs = [global_i]
            ctx_idxs = sorted(ctx_idxs, key=lambda j: float(onsets_all[j]))

            ctx_ids: list[int] = list(sot_ids)
            for j in ctx_idxs:
                ctx_ids.extend(word_token_ids[j])
            n_cur = len(word_token_ids[global_i])
            decoder_input_ids = torch.tensor([ctx_ids], dtype=torch.long, device=device)

            # --- full / audio-fused ---
            out_full = model(
                input_features=feats,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
            )
            # --- de-only / text-only ---
            enc_zeros = torch.zeros(1, N_ENCODER_FRAMES, d_model, device=device, dtype=param_dtype)
            cross_mask = torch.zeros(n_dec, n_heads, device=device)
            out_de = model(
                input_features=feats,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                encoder_outputs=(enc_zeros,),
                cross_attn_head_mask=cross_mask,
            )

            for name, out in (
                ("language_audio_fused", out_full),
                ("language_text_only", out_de),
            ):
                dec_hs = out.decoder_hidden_states
                if dec_hs is None:
                    raise RuntimeError(f"No decoder_hidden_states for {name}")
                for li in layers:
                    h = dec_hs[li]  # (1, T, D)
                    # pool over current word's trailing tokens
                    if n_cur <= 0 or n_cur > h.shape[1] or args.token_pool == "last" or n_cur == 1:
                        vec = h[0, -1, :]
                    else:
                        vec = h[0, -n_cur:, :].mean(dim=0)
                    outs[name][li][local_i, :] = vec.detach().to("cpu", dtype=torch.float32).numpy()

            if (local_i + 1) % 10 == 0 or local_i == 0 or local_i + 1 == n_words:
                print(f"  word {local_i + 1}/{n_words} (global {global_i}) ctx_tok={len(ctx_ids)} n_cur={n_cur}")

    def write_variant(stem: str, name: str, X: np.ndarray, layer: int) -> None:
        npy = outdir / f"X_word_{stem}_{name}.npy"
        np.save(npy, X)
        names = [f"{stem}_{name}_{j}" for j in range(X.shape[1])]
        (outdir / f"X_word_{stem}_{name}_feature_names.txt").write_text(
            "\n".join(names) + "\n", encoding="utf-8"
        )
        meta = {
            "tag": args.model_name,
            "variant": name,
            "shape": [int(X.shape[0]), int(X.shape[1])],
            "pipeline": "hasson_247_decoder",
            "model_type": "full-onset" if "fused" in name else "de-only",
            "window_mode": args.window_mode,
            "clip_end_offset_s": args.clip_end_offset_s,
            "decoder_layer": layer,
            "start_word_idx": args.start_word_idx,
            "token_pool": args.token_pool,
            "prefix_style": args.prefix_style,
        }
        (outdir / f"X_word_{stem}_{name}_meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )

    for name, by_layer in outs.items():
        if len(by_layer) > 1:
            for li, X in by_layer.items():
                write_variant(args.stem, f"{name}_layer{li}", X, li)
            write_variant(args.stem, name, by_layer[default_layer], default_layer)
        else:
            write_variant(args.stem, name, by_layer[default_layer], default_layer)

    print("Done.")


if __name__ == "__main__":
    main()
