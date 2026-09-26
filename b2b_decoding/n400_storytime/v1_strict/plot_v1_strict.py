#!/orcd/software/core/001/pkg/miniforge/25.11.0-0/bin/python3
"""Subject-equal tertile waveforms and low-minus-high topomaps.

Reads fig.npz written by extract_channels.py. Does not refit models.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

POOL = Path("/orcd/pool/005/haolun52")
TRIALS = POOL / "n400_storytime_v1" / "v1_strict" / "trials"
# Compute nodes cannot write /run/user. Default is the pool; copy into the store after.
MEDIA = Path(
    os.environ.get(
        "N400_V1_MEDIA",
        "/orcd/pool/005/haolun52/n400_storytime_v1/v1_strict/figures",
    )
)
CHANNELS = [
    "Fpz", "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8",
    "Fz", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8",
    "FCz", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6",
    "FT7", "FT8", "FT9", "FT10",
    "Cz", "C1", "C2", "C3", "C4", "C5", "C6", "T7", "T8",
    "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "TP7", "TP8", "TP9", "TP10",
    "Pz", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8",
    "POz", "PO3", "PO4", "PO5", "PO6", "PO7", "PO8",
    "Oz", "O1", "O2",
]
PREDS = ("freq", "assoc", "cloze")
PRED_TITLE = {"freq": "Frequency", "assoc": "Association", "cloze": "Cloze probability"}
TERT_COLOR = {0: "#d95f02", 1: "#7570b3", 2: "#1b9e77"}
TERT_NAME = {0: "Low", 1: "Mid", 2: "High"}


def _load():
    paths = sorted(p for p in TRIALS.iterdir() if (p / "fig.npz").is_file())
    if len(paths) != 63:
        raise SystemExit(f"expected 63 fig.npz, found {len(paths)}")
    pack = []
    times = None
    for path in paths:
        z = np.load(path / "fig.npz", allow_pickle=True)
        if times is None:
            times = z["times"].astype(float)
        elif not np.allclose(times, z["times"].astype(float)):
            raise SystemExit(f"time axis differs: {path.name}")
        n = z["wave_n"].astype(int)
        fz = np.full((3, 3, 121), np.nan)
        pz = np.full((3, 3, 121), np.nan)
        for i in range(3):
            for j in range(3):
                if n[i, j] > 0:
                    fz[i, j] = z["fz_sum"][i, j] / n[i, j]
                    pz[i, j] = z["pz_sum"][i, j] / n[i, j]
        topo_n = z["topo_n"].astype(int)
        low = np.full((3, 65), np.nan)
        high = np.full((3, 65), np.nan)
        for i in range(3):
            if topo_n[i, 0] > 0:
                low[i] = z["topo_low_sum"][i] / topo_n[i, 0]
            if topo_n[i, 1] > 0:
                high[i] = z["topo_high_sum"][i] / topo_n[i, 1]
        pack.append(
            {
                "subject": path.name,
                "group": str(z["group"]),
                "fz": fz,
                "pz": pz,
                "n": n,
                "low": low,
                "high": high,
            }
        )
    return times, pack


def _mean_sem(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """stack: subject, tertile, time. NaN subjects are dropped per tertile."""
    mean = np.full(stack.shape[1:], np.nan)
    sem = np.full(stack.shape[1:], np.nan)
    ns = []
    for j in range(stack.shape[1]):
        sl = stack[:, j, :]
        ok = np.isfinite(sl).all(axis=1)
        ns.append(int(ok.sum()))
        if ok.sum() == 0:
            continue
        mean[j] = sl[ok].mean(axis=0)
        if ok.sum() > 1:
            sem[j] = sl[ok].std(axis=0, ddof=1) / np.sqrt(ok.sum())
        else:
            sem[j] = 0.0
    return mean, sem, int(min(ns) if ns else 0)


def _draw_wave(ax, times, mean, sem, title):
    for j in (0, 1, 2):
        if not np.isfinite(mean[j]).all():
            continue
        ax.plot(times, mean[j], color=TERT_COLOR[j], lw=1.4, label=TERT_NAME[j])
        ax.fill_between(times, mean[j] - sem[j], mean[j] + sem[j], color=TERT_COLOR[j], alpha=0.18, lw=0)
    ax.axvline(0, color="0.6", lw=0.6)
    ax.axvspan(0.35, 0.55, color="0.85", zorder=0)
    ax.set_xlim(-0.2, 1.0)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (as stored)")


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    if not path.is_file() or path.stat().st_size < 1000:
        raise SystemExit(f"plot missing or empty: {path}")


def _group_mask(pack, kind: str) -> list[dict]:
    if kind == "TD":
        return [p for p in pack if p["group"] == "TD"]
    if kind == "dyslexia":
        return [p for p in pack if str(p["group"]).startswith("dyslexia")]
    if kind == "nCAP":
        return [p for p in pack if p["group"] == "dyslexia_normal_CAP"]
    if kind == "aCAP":
        return [p for p in pack if p["group"] == "dyslexia_atypical_CAP"]
    raise SystemExit(kind)


def main() -> None:
    times, pack = _load()
    canon = np.round(-0.2 + np.arange(121) / 100.0, 5)
    if times.shape != canon.shape or np.max(np.abs(times - canon)) > 1e-4:
        raise SystemExit("times are not the 100 Hz −0.2…1.0 grid")

    for pi, pred in enumerate(PREDS):
        fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.6), sharex=True, sharey=True)
        for col, kind in enumerate(("TD", "dyslexia")):
            sub = _group_mask(pack, kind)
            fz = np.stack([p["fz"][pi] for p in sub], axis=0)
            pz = np.stack([p["pz"][pi] for p in sub], axis=0)
            for row, (wave, site) in enumerate(((fz, "Fz"), (pz, "Pz"))):
                mean, sem, _n = _mean_sem(wave)
                _draw_wave(axes[row, col], times, mean, sem, f"{kind} · {site} · N={len(sub)}")
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{PRED_TITLE[pred]} tertiles")
        _save(fig, MEDIA / f"waveform_{pred}.png")

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.6), sharex=True, sharey=True)
    for col, kind in enumerate(("nCAP", "aCAP")):
        sub = _group_mask(pack, kind)
        fz = np.stack([p["fz"][2] for p in sub], axis=0)
        pz = np.stack([p["pz"][2] for p in sub], axis=0)
        for row, (wave, site) in enumerate(((fz, "Fz"), (pz, "Pz"))):
            mean, sem, _n = _mean_sem(wave)
            _draw_wave(axes[row, col], times, mean, sem, f"{kind} · {site} · N={len(sub)}")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Cloze probability tertiles · CAP")
    _save(fig, MEDIA / "waveform_cap_cloze.png")

    montage = mne.channels.make_standard_montage("standard_1005")
    missing = [ch for ch in CHANNELS if ch not in montage.ch_names]
    if missing:
        raise SystemExit(f"montage missing {missing}")
    info = mne.create_info(CHANNELS, sfreq=100.0, ch_types="eeg")
    info.set_montage(montage, match_case=True, on_missing="raise")
    maps = []
    labels = []
    for kind in ("TD", "dyslexia"):
        sub = _group_mask(pack, kind)
        for pi, pred in enumerate(PREDS):
            lows = np.stack([p["low"][pi] for p in sub], axis=0)
            highs = np.stack([p["high"][pi] for p in sub], axis=0)
            ok = np.isfinite(lows).all(axis=1) & np.isfinite(highs).all(axis=1)
            if ok.sum() < 3:
                raise SystemExit(f"too few subjects for topo {kind} {pred}")
            diff = lows[ok].mean(axis=0) - highs[ok].mean(axis=0)
            maps.append(diff)
            labels.append(f"{kind}\n{PRED_TITLE[pred]}")
    peak = max(float(np.max(np.abs(m))) for m in maps)
    if not np.isfinite(peak) or peak <= 0:
        raise SystemExit(f"bad topo scale {peak}")
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.8))
    for ax, data, lab in zip(axes.ravel(), maps, labels):
        ret = mne.viz.plot_topomap(
            data, info, axes=ax, show=False, cmap="RdBu_r", vlim=(-peak, peak),
            sensors=True, contours=6, extrapolate="head",
        )
        im = ret[0] if isinstance(ret, tuple) else ret
        ax.set_title(lab, fontsize=9)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="Low − high (as stored)")
    fig.suptitle("350–550 ms")
    _save(fig, MEDIA / "topomap_low_minus_high.png")
    summary = {
        "n_subjects": len(pack),
        "n_td": len(_group_mask(pack, "TD")),
        "n_dyslexia": len(_group_mask(pack, "dyslexia")),
        "n_ncap": len(_group_mask(pack, "nCAP")),
        "n_acap": len(_group_mask(pack, "aCAP")),
        "topo_peak_abs": peak,
        "png": sorted(p.name for p in MEDIA.glob("*.png")),
    }
    (MEDIA / "figure_qc.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
