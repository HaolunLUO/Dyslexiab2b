#!/usr/bin/env python3
"""
Re-extract Whisper *encoder* word-locked embeddings from lPPCN section WAVs.

Aligned with hassonlab/247-pickling `whisper-paper-1`:
  tfsemb_genemb_whisper_var_win.py  (model_type == "en-only")
  tfsemb_genemb_whisper_en_win.py   (acoustic = encoder hidden_states[0])

Paper-style fixed window (TFS comments in var_win):
  clip_end = word_onset + 0.2325 s
  take 10 encoder frames (~20 ms each = 200 ms)
  start_windows:
    if clip_end < 30 s:
      int((((onset - clip_start) * 1000) - 7.5) // 20 + 3)
    else:
      1500 - 10
  Audio is NOT left-padded to 30 s. HF WhisperFeatureExtractor right-pads
  the log-mel to 30 s (3000 frames -> 1500 encoder frames).

Outputs:
  - X_word_{stem}_{acoustic|speech}.npy  shape (n_words, hidden*10)
  - feature_names.txt and minimal _meta.json

lPPCN is Chinese; existing shared matrices match openai/whisper-tiny (multilingual),
not whisper-tiny.en.
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
from transformers import WhisperModel, WhisperProcessor


DEFAULT_SHARED = Path(
    os.environ.get(
        "B2B_SHARED_FEATURES_DIR",
        "/orcd/pool/005/haolun52/extracted_sections_wordlocked_shared/_shared_wordlocked_features",
    )
)

SR = 16000
CLIP_LEN_S = 30.0
N_ENCODER_FRAMES = 1500  # 30 s * 50 Hz mel / 2 (conv stride)
DEFAULT_NUM_WINDOWS = 10  # 200 ms of encoder frames
# TFS / 247-pickling comment: word_onset + 232.5 ms (podcast used 272.5 ms)
DEFAULT_CLIP_END_OFFSET_S = 0.2325


def _load_wav_16k(path: Path) -> np.ndarray:
    """Prefer openai-whisper.load_audio (ffmpeg). Fall back to soundfile+librosa."""
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

    try:
        import soundfile as sf

        audio, orig_sr = sf.read(str(path), always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32, copy=False)
        if int(orig_sr) == SR:
            return audio
        import librosa

        return librosa.resample(audio, orig_sr=int(orig_sr), target_sr=SR).astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to read wav {path}: {e}") from e


def _hasson_start_windows(onset_s: float, clip_start_s: float, clip_end_s: float, num_windows: int) -> int:
    """Frame index of the first concatenated encoder state (var_win en-only)."""
    if clip_end_s < CLIP_LEN_S:
        start = int((((onset_s - clip_start_s) * 1000.0) - 7.5) // 20 + 3)
    else:
        start = N_ENCODER_FRAMES - num_windows
    start = max(0, min(start, N_ENCODER_FRAMES - num_windows))
    return start


def _word_clip_and_windows(
    onset_s: float,
    clip_end_offset_s: float,
    num_windows: int,
) -> tuple[float, float, int]:
    clip_end = float(onset_s + clip_end_offset_s)
    clip_start = float(np.max([0.0, clip_end - CLIP_LEN_S]))
    start_windows = _hasson_start_windows(onset_s, clip_start, clip_end, num_windows)
    return clip_start, clip_end, start_windows


def _concat_encoder_frames(hidden: torch.Tensor, start_windows: np.ndarray, num_windows: int) -> torch.Tensor:
    """hidden: (B, T, D); start_windows: (B,); returns (B, num_windows*D)."""
    bsz, t, d = hidden.shape
    out = hidden.new_empty((bsz, num_windows * d))
    for i in range(bsz):
        s = int(start_windows[i])
        e = s + num_windows
        if e > t:
            raise RuntimeError(f"Need encoder frames [{s}:{e}) but T={t}")
        out[i] = hidden[i, s:e, :].reshape(-1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--section-id", type=int, default=1)
    ap.add_argument("--shared-dir", type=str, default=str(DEFAULT_SHARED))
    ap.add_argument("--model-name", type=str, required=True)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--hf-cache-dir", type=str, default=None)
    ap.add_argument("--max-words", type=int, default=50)
    ap.add_argument("--start-word-idx", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--clip-end-offset-s", type=float, default=DEFAULT_CLIP_END_OFFSET_S)
    ap.add_argument("--num-windows", type=int, default=DEFAULT_NUM_WINDOWS)
    ap.add_argument(
        "--frame-mode",
        type=str,
        default="hasson",
        choices=["hasson", "last"],
        help="hasson: start_windows formula; last: always final num_windows frames of encoder",
    )
    ap.add_argument(
        "--torch-dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Model compute dtype (existing large-v2 arrays look float16-quantized).",
    )
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--stem", type=str, required=True, help="e.g. whispertiny or whisperlargev2")
    ap.add_argument(
        "--save-all-encoder-layers",
        action="store_true",
        help="Also write X_word_{stem}_encoder_layer{k}.npy for k=0..encoder_layers (probe).",
    )
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

    onsets: list[float] = []
    with open(timing_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            onsets.append(float(row["onset_relative"]))

    onsets_all = np.asarray(onsets, dtype=np.float64)
    i0 = args.start_word_idx
    i1 = min(i0 + args.max_words, len(onsets_all)) if args.max_words > 0 else len(onsets_all)
    onsets_s = onsets_all[i0:i1]
    n_words = len(onsets_s)
    if n_words == 0:
        raise SystemExit("No words selected")

    audio = _load_wav_16k(wav_path)
    print(f"Loaded audio: {wav_path.name} sr={SR} len={len(audio) / SR:.2f}s")
    print(f"Using n_words={n_words} rows [{i0}:{i1}) from onset_relative")
    print(
        f"en-only: clip_end=onset+{args.clip_end_offset_s}s "
        f"num_windows={args.num_windows} frame_mode={args.frame_mode} dtype={args.torch_dtype}"
    )

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map[args.torch_dtype]

    print(f"Model: {args.model_name} device={device} torch_dtype={torch_dtype}")
    processor = WhisperProcessor.from_pretrained(args.model_name)
    model = WhisperModel.from_pretrained(args.model_name, torch_dtype=torch_dtype)
    encoder = model.encoder
    encoder.eval()
    encoder.to(device)

    enc_layers = int(getattr(model.config, "encoder_layers", None) or 0)
    if enc_layers <= 0:
        raise SystemExit(f"Could not infer encoder_layers from config for {args.model_name}")
    # hidden_states[0] = conv/embed output; [encoder_layers] = top encoder layer
    acoustic_layer = 0
    speech_layer = enc_layers
    print(f"encoder_layers={enc_layers} acoustic_layer={acoustic_layer} speech_layer={speech_layer}")

    clip_starts = np.empty(n_words, dtype=np.float64)
    clip_ends = np.empty(n_words, dtype=np.float64)
    start_windows = np.empty(n_words, dtype=np.int32)
    for i, t in enumerate(onsets_s.tolist()):
        cs, ce, sw = _word_clip_and_windows(t, args.clip_end_offset_s, args.num_windows)
        clip_starts[i] = cs
        clip_ends[i] = ce
        if args.frame_mode == "last":
            start_windows[i] = N_ENCODER_FRAMES - args.num_windows
        else:
            start_windows[i] = sw

    print(
        f"start_windows: min={int(start_windows.min())} max={int(start_windows.max())} "
        f"n_early_clip(end<30s)={int((clip_ends < CLIP_LEN_S).sum())}"
    )

    X_acoustic: np.ndarray | None = None
    X_speech: np.ndarray | None = None
    X_all_layers: dict[int, np.ndarray] = {}

    with torch.no_grad():
        w_idx = 0
        while w_idx < n_words:
            b1 = min(w_idx + args.batch_size, n_words)
            chunks: list[np.ndarray] = []
            for i in range(w_idx, b1):
                a0 = int(clip_starts[i] * SR)
                a1 = int(clip_ends[i] * SR)
                a0 = max(0, a0)
                a1 = max(a0 + 1, min(len(audio), a1))
                chunks.append(audio[a0:a1])

            inputs = processor.feature_extractor(
                chunks,
                sampling_rate=SR,
                return_tensors="pt",
            )
            input_features = inputs.input_features.to(device=device, dtype=torch_dtype)
            enc_out = encoder(input_features, output_hidden_states=True)
            enc_hidden = enc_out.hidden_states
            if enc_hidden is None:
                raise RuntimeError("Whisper encoder did not return hidden_states")

            sw = start_windows[w_idx:b1]
            emb_ac = _concat_encoder_frames(enc_hidden[acoustic_layer], sw, args.num_windows)
            emb_sp = _concat_encoder_frames(enc_hidden[speech_layer], sw, args.num_windows)
            emb_ac = emb_ac.detach().to("cpu", dtype=torch.float32).numpy()
            emb_sp = emb_sp.detach().to("cpu", dtype=torch.float32).numpy()

            if X_acoustic is None:
                feat_dim_out = int(emb_ac.shape[1])
                X_acoustic = np.empty((n_words, feat_dim_out), dtype=np.float32)
                X_speech = np.empty((n_words, feat_dim_out), dtype=np.float32)
                if args.save_all_encoder_layers:
                    for li in range(len(enc_hidden)):
                        X_all_layers[li] = np.empty((n_words, feat_dim_out), dtype=np.float32)

            bsz = emb_ac.shape[0]
            X_acoustic[w_idx : w_idx + bsz, :] = emb_ac
            X_speech[w_idx : w_idx + bsz, :] = emb_sp
            if args.save_all_encoder_layers:
                for li in range(len(enc_hidden)):
                    layer_emb = _concat_encoder_frames(enc_hidden[li], sw, args.num_windows)
                    X_all_layers[li][w_idx : w_idx + bsz, :] = (
                        layer_emb.detach().to("cpu", dtype=torch.float32).numpy()
                    )
            w_idx += bsz
            print(f"  batch done: upto word {w_idx}/{n_words}")

    def write_variant(stem: str, name: str, X: np.ndarray) -> None:
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
            "pipeline": "hasson_247_en-only",
            "clip_end_offset_s": args.clip_end_offset_s,
            "num_windows": args.num_windows,
            "start_word_idx": args.start_word_idx,
            "acoustic_layer": acoustic_layer,
            "speech_layer": speech_layer,
        }
        (outdir / f"X_word_{stem}_{name}_meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )

    if X_acoustic is None or X_speech is None:
        raise RuntimeError("No features extracted (empty clip batch?)")

    write_variant(args.stem, "acoustic", X_acoustic)
    write_variant(args.stem, "speech", X_speech)
    if args.save_all_encoder_layers:
        for li, X in X_all_layers.items():
            write_variant(args.stem, f"encoder_layer{li}", X)
    print("Done.")


if __name__ == "__main__":
    main()
