"""
Audio feature extraction for violence detection.

Reads the labelled video corpus, decodes each video's audio track with PyAV,
and computes per-clip acoustic features that distinguish aggression from
ordinary sound.

Design constraints that shaped this file:
  * PyAV is the only audio library available (no librosa/soundfile/ffmpeg CLI),
    so decoding goes through av.decode() and the raw PCM is used directly.
  * Many videos have NO audio stream. Those are SKIPPED and recorded as such.
    Treating "no audio track" as "non-violence" would poison the dataset:
    absence of audio is not evidence of absence of violence.
  * Features are per-clip so the same group-wise split used for the visual
    model can be reused verbatim, keeping train/val clip identities aligned.

Feature set (chosen to be computable from a raw FFT with numpy only):
  - log-mel-ish band energies (13 bands)      -> spectral shape of the sound
  - spectral centroid / spread / flatness      -> brightness + noisiness
  - zero-crossing rate                          -> harshness (shouts score high)
  - RMS energy and its temporal statistics     -> loudness dynamics
  - spectral flux                               -> sudden onsets (impacts)
  - pitch estimate via autocorrelation         -> scream/shout cues
  - escalation slope and peak-late ratio       -> violence *rises* over time
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "audio"
FEAT_DIR = OUT_DIR / "features"
AUD_DIR = OUT_DIR / "wav"

VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
SR = 22050          # analysis sample rate (16 kHz is enough for speech/scream cues)
N_FFT = 1024
HOP = 256
N_MELS = 13
MAX_SECONDS = 10.0

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Label inference -- mirrors build_dataset.infer_label exactly so the two
# modalities agree on which clip belongs to which class.
# --------------------------------------------------------------------------

VIOLENCE_TOKENS = {
    "violence", "violent", "fight", "fighting", "fights", "punch", "punching",
    "kick", "kicking", "hit", "hitting", "assault", "attack", "beating",
    "riot", "abuse", "aggressive", "aggression", "abnormal", "brawl",
    "shoplifting", "stealing", "vandalism",
}
NONVIOLENCE_TOKENS = {
    "nonviolence", "normal", "calm", "peace", "nofight", "benign",
    "walking", "running", "sleeping", "talking", "playing",
}


def _norm(part: str) -> str:
    return (part.lower().replace(" ", "").replace("_", "")
            .replace("-", "").replace(".", ""))


VIOL_LOOKUP = {_norm(t): 1 for t in VIOLENCE_TOKENS}
NV_LOOKUP = {_norm(t): 0 for t in NONVIOLENCE_TOKENS}


def infer_label(path: Path) -> int | None:
    parts = [p.lower() for p in path.parts]
    for part in reversed(parts[:-1]):
        n = _norm(part)
        if n in NV_LOOKUP:
            return 0
        if n in VIOL_LOOKUP:
            return 1
        for tok in sorted(NONVIOLENCE_TOKENS, key=len, reverse=True):
            if tok in n:
                return 0
        for tok in sorted(VIOLENCE_TOKENS, key=len, reverse=True):
            if tok in n:
                return 1
    return None


# --------------------------------------------------------------------------
# Audio decoding
# --------------------------------------------------------------------------


def decode_audio(path: Path) -> tuple[np.ndarray | None, str]:
    """Return mono float32 PCM at SR, or (None, reason) on failure.

    Skipping silent tracks is deliberate: a near-silent track produces
    meaningless features, and training on it teaches the model nothing.
    """
    try:
        import av
    except Exception as exc:  # pragma: no cover
        return None, f"av unavailable: {exc}"

    try:
        container = av.open(str(path))
    except Exception as exc:
        return None, f"open failed: {type(exc).__name__}"

    try:
        audio_streams = [s for s in container.streams if s.type == "audio"]
        if not audio_streams:
            return None, "no audio stream"
        stream = audio_streams[0]
        stream.thread_type = "AUTO"

        chunks: list[np.ndarray] = []
        total = 0
        max_samples = int(SR * MAX_SECONDS)
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=SR)
        try:
            # PyAV 17 requires the stream positionally; decode(audio=stream)
            # raises TypeError on this version.
            for frame in container.decode(stream):
                if frame.samples == 0:
                    continue
                for rf in resampler.resample(frame):
                    arr = rf.to_ndarray()
                    if arr.size == 0:
                        continue
                    arr = arr.astype(np.float32).reshape(-1) / 32768.0
                    chunks.append(arr)
                    total += arr.size
                    if total >= max_samples:
                        break
                if total >= max_samples:
                    break
        finally:
            try:
                for rf in resampler.resample(None):
                    arr = rf.to_ndarray().astype(np.float32).reshape(-1) / 32768.0
                    if arr.size:
                        chunks.append(arr)
            except Exception:
                pass

        if not chunks:
            return None, "decoder produced no samples"
        audio = np.concatenate(chunks)[:max_samples]

        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak < 1e-4:
            return None, "silent track"
        return audio.astype(np.float32), ""
    except Exception as exc:
        return None, f"decode error: {type(exc).__name__}: {exc}"
    finally:
        try:
            container.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------


def _frames(x: np.ndarray) -> np.ndarray:
    if x.size < N_FFT:
        x = np.pad(x, (0, N_FFT - x.size))
    n = 1 + (x.size - N_FFT) // HOP
    if n <= 0:
        return np.zeros((1, N_FFT), dtype=np.float32)
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx]


def mel_filterbank(n_mels: int, n_fft: int, sr: int) -> np.ndarray:
    """Triangular mel filterbank (Slaney-style, no librosa dependency)."""

    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_pts = np.linspace(hz2mel(20.0), hz2mel(sr / 2), n_mels + 2)
    hz_pts = mel2hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        l, c, r = bins[m], bins[m + 1], bins[m + 2]
        if c == l:
            c = l + 1
        if r == c:
            r = c + 1
        r = min(r, n_fft // 2)
        if l >= r:
            continue
        fb[m, l:c] = (np.arange(l, c) - l) / max(c - l, 1)
        fb[m, c:r] = (r - np.arange(c, r)) / max(r - c, 1)
    return fb


_FB = mel_filterbank(N_MELS, N_FFT, SR)


def compute_features(audio: np.ndarray) -> dict[str, float]:
    """Compute the acoustic feature vector for one clip."""
    x = _frames(audio)
    window = np.hanning(N_FFT).astype(np.float32)
    S = np.abs(np.fft.rfft(x * window, axis=1)) + 1e-10

    # --- band energies (mel) ---
    mel = _FB @ S.T
    mel_db = 20.0 * np.log10(mel + 1e-10)
    band_mean = mel_db.mean(axis=1)
    band_std = mel_db.std(axis=1)

    # --- spectral shape ---
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    spec_sum = S.sum(axis=1) + 1e-10
    centroid = (S * freqs[None, :]).sum(axis=1) / spec_sum
    centroid_mean = float(centroid.mean())
    spread = np.sqrt(((freqs[None, :] - centroid[:, None]) ** 2 * S).sum(axis=1) / spec_sum)
    geo = np.exp(np.log(S + 1e-10).mean(axis=1))
    arith = S.mean(axis=1) + 1e-10
    flatness = float(np.mean(geo / arith))

    # --- zero crossing rate (harshness proxy) ---
    zcr = np.mean(np.abs(np.diff(np.sign(audio))) > 0)
    zcr = float(zcr)

    # --- loudness dynamics ---
    rms = np.sqrt((x ** 2).mean(axis=1))
    rms_db = 20.0 * np.log10(rms + 1e-10)
    rms_mean = float(rms_db.mean())
    rms_std = float(rms_db.std())
    rms_max = float(rms_db.max())
    rms_p95 = float(np.percentile(rms_db, 95))
    crest = float(np.max(rms) / (np.mean(rms) + 1e-10))

    # --- spectral flux (onsets: impacts) ---
    diff = np.diff(S, axis=0)
    flux = np.sum(np.maximum(diff, 0.0), axis=1)
    flux_mean = float(flux.mean())
    flux_peak_ratio = float(np.max(flux) / (np.mean(flux) + 1e-10))
    onset_count = int(np.sum(flux > (np.mean(flux) + 2.0 * np.std(flux))))

    # --- pitch (autocorrelation, screams/shouts are high-pitch) ---
    seg = audio[: SR * 2] if audio.size >= SR else audio
    seg = seg - seg.mean() if seg.size else seg
    if seg.size > 512:
        corr = np.correlate(seg, seg, mode="full")[seg.size - 1:]
        lo, hi = int(SR / 500), min(int(SR / 70), corr.size - 1)
        if hi > lo:
            window_slice = corr[lo:hi]
            lag = int(np.argmax(window_slice)) + lo
            f0 = SR / lag if lag > 0 else 0.0
            f0_conf = float(window_slice.max() / (corr[0] + 1e-10))
        else:
            f0, f0_conf = 0.0, 0.0
    else:
        f0, f0_conf = 0.0, 0.0

    # --- escalation: violence rises over time, conversation does not ---
    n_seg = 8
    seg_len = max(len(rms_db) // n_seg, 1)
    segments = [float(rms_db[i * seg_len:(i + 1) * seg_len].mean()) for i in range(n_seg)]
    segments = segments[:n_seg]
    if len(segments) >= 2:
        t = np.arange(len(segments), dtype=np.float32)
        slope = float(np.polyfit(t, np.array(segments), 1)[0])
        early = float(np.mean(segments[: max(len(segments) // 2, 1)]))
        late = float(np.mean(segments[max(len(segments) // 2, 1):]))
        escalation = late - early
        peak_late_ratio = late / (early + 1e-10)
    else:
        slope = escalation = 0.0
        peak_late_ratio = 1.0

    feats: dict[str, float] = {}
    for i in range(N_MELS):
        feats[f"mel{i}_mean"] = float(band_mean[i])
        feats[f"mel{i}_std"] = float(band_std[i])
    feats.update({
        "centroid": centroid_mean,
        "spread": float(spread.mean()),
        "flatness": flatness,
        "zcr": zcr,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "rms_max": rms_max,
        "rms_p95": rms_p95,
        "crest": crest,
        "flux_mean": flux_mean,
        "flux_peak_ratio": flux_peak_ratio,
        "onset_count": float(onset_count),
        "f0": float(f0),
        "f0_conf": f0_conf,
        "escalation_slope": slope,
        "escalation_delta": escalation,
        "peak_late_ratio": float(peak_late_ratio),
        "duration": float(len(audio) / SR),
    })
    return feats


FEATURE_NAMES: list[str] | None = None


def process_one(video: Path, save_wav: bool) -> dict | None:
    label = infer_label(video)
    if label is None:
        return None
    audio, reason = decode_audio(video)
    if audio is None:
        return {"video": str(video), "label": label, "ok": False, "reason": reason}

    feats = compute_features(audio)
    if not np.all(np.isfinite(list(feats.values()))):
        return {"video": str(video), "label": label, "ok": False, "reason": "non-finite features"}

    clip_id = f"{video.parent.name[:24]}_{video.stem[:48]}"
    if save_wav:
        try:
            import wave

            AUD_DIR.mkdir(parents=True, exist_ok=True)
            pcm = np.clip(audio, -1.0, 1.0)
            pcm16 = (pcm * 32767).astype(np.int16)
            with wave.open(str(AUD_DIR / f"{clip_id}.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SR)
                w.writeframes(pcm16.tobytes())
        except Exception:
            pass

    feats["label"] = float(label)
    feats["clip_id"] = clip_id
    feats["video"] = str(video)
    return {"ok": True, "clip_id": clip_id, "features": feats, "label": label,
            "video": str(video)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) // 2))
    ap.add_argument("--max-clips", type=int, default=100000)
    ap.add_argument("--save-wav", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    random.seed(args.seed)
    FEAT_DIR.mkdir(parents=True, exist_ok=True)

    roots: list[Path] = []
    if args.sources:
        roots = [Path(s) for s in args.sources]
    else:
        raw = ROOT / "data_raw"
        if raw.exists():
            roots += sorted([p for p in raw.iterdir() if p.is_dir()])
        for legacy in ("Violence", "NonViolence"):
            if (ROOT / legacy).exists():
                roots.append(ROOT / legacy)

    videos: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        found = [p for p in r.rglob("*")
                 if p.is_file() and p.suffix.lower() in VIDEO_EXT]
        for p in found:
            k = str(p.resolve()).lower()
            if k not in seen:
                seen.add(k)
                videos.append(p)
    random.shuffle(videos)
    videos = videos[: args.max_clips]
    log(f"candidate videos: {len(videos)} from {len(roots)} roots")

    results: list[dict] = []
    skipped: dict[str, int] = {}
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, v, args.save_wav): v for v in videos}
        for fut in as_completed(futs):
            v = futs[fut]
            try:
                r = fut.result()
            except Exception as exc:
                r = {"video": str(v), "ok": False, "reason": f"worker {type(exc).__name__}"}
            done += 1
            if r is None:
                skipped["unlabelled"] = skipped.get("unlabelled", 0) + 1
            elif r.get("ok"):
                results.append(r)
            else:
                key = r.get("reason", "unknown").split(":")[0]
                skipped[key] = skipped.get(key, 0) + 1
            if done % 100 == 0:
                rate = done / max(time.time() - t0, 1e-6)
                log(f"  {done}/{len(videos)} ({rate:.1f}/s) ok={len(results)} skipped={len(skipped)}")

    # persist
    out_json = FEAT_DIR / "audio_features.json"
    out_json.write_text(json.dumps(results, indent=1), encoding="utf-8")

    # CSV for quick inspection
    if results:
        names = sorted({k for r in results for k in r["features"]})
        csv = FEAT_DIR / "audio_features.csv"
        with open(csv, "w", encoding="utf-8", newline="") as fh:
            fh.write(",".join(names) + "\n")
            for r in results:
                fh.write(",".join(str(r["features"].get(n, "")) for n in names) + "\n")

    n_v = sum(1 for r in results if r["label"] == 1)
    n_nv = len(results) - n_v
    summary = {
        "videos_scanned": len(videos),
        "clips_with_audio": len(results),
        "violence": n_v,
        "nonviolence": n_nv,
        "skipped": skipped,
        "n_features": len(results[0]["features"]) if results else 0,
    }
    (FEAT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"DONE with_audio={len(results)} V={n_v} NV={n_nv}")
    log(f"skipped: {skipped}")
    log(f"-> {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
