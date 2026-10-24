"""Streaming RTF benchmark for the MLX backend (end to end through ModelWrapperMLX).

    python benchmarks/bench_mlx.py --audio ~/.cache/hikari_mlx_ref/jfk.flac --window 100 --seconds 90
"""

import argparse
import time

import librosa
import numpy as np

CHUNK = 1280  # 80 ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="sbintuitions/hikari-medium")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=60.0, help="stream length (audio is tiled to reach it)")
    ap.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    ap.add_argument("--tgt-lang", default="Japanese")
    ap.add_argument("--profile", action="store_true", help="sync between stages for a per-stage breakdown")
    ap.add_argument("--stride", type=int, default=0, help="approximate stride-refresh mode with K steps (0 = exact)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--report-every", type=int, default=0, help="print mean ms/step every N steps (thermal drift)")
    args = ap.parse_args()

    from hikari.server.model_wrapper_mlx import ModelWrapperMLX

    audio, _ = librosa.load(args.audio, sr=16000, mono=True)
    audio = audio.astype(np.float32)
    n = int(args.seconds * 16000)
    audio = np.tile(audio, int(np.ceil(n / len(audio))))[:n]
    chunks = [audio[i : i + CHUNK] for i in range(0, len(audio) - CHUNK + 1, CHUNK)]

    w = ModelWrapperMLX(checkpoint=args.checkpoint)
    w.apply_settings(
        {"task": args.task, "tgt_lang": args.tgt_lang, "baseline_wait_penalty": 0.0, "decoder_context": args.window,
         "refresh_stride": args.stride}
    )
    # warm up (kernel compilation) on the first chunks, then restart the stream
    for c in chunks[: 3 + args.warmup]:
        w.get_one_token(c)
    w.reset()
    w.apply_settings(
        {"task": args.task, "tgt_lang": args.tgt_lang, "baseline_wait_penalty": 0.0, "decoder_context": args.window,
         "refresh_stride": args.stride}
    )
    w.profile = args.profile

    times, text = [], []
    for c in chunks:
        t0 = time.perf_counter()
        word, _ = w.get_one_token(c)
        times.append(time.perf_counter() - t0)
        if args.report_every and len(times) % args.report_every == 0:
            r = np.array(times[-args.report_every :])
            print(f"  [{len(times)*0.08:6.0f}s of audio] mean {r.mean()*1e3:6.1f} ms  p95 {np.percentile(r,95)*1e3:6.1f}  RTF {r.mean()/0.08:4.2f}", flush=True)
        if word:
            text.append(word)
    times = np.array(times[2:])  # first two chunks only buffer audio
    step_ms = 80.0
    n_win = args.window  # steps until the window is full
    early, late = times[:n_win], times[n_win:]

    def rep(name, t):
        if len(t) == 0:
            return
        print(
            f"{name:14s} n={len(t):4d}  mean {t.mean()*1e3:7.1f} ms  p50 {np.percentile(t,50)*1e3:7.1f}  "
            f"p95 {np.percentile(t,95)*1e3:7.1f}  max {t.max()*1e3:7.1f}  RTF mean {t.mean()*1e3/step_ms:5.2f}  "
            f"RTF p95 {np.percentile(t,95)*1e3/step_ms:5.2f}"
        )

    print(f"window={args.window} tokens ({args.window*0.08:.1f}s) total stream {args.seconds:.0f}s")
    rep("all", times)
    rep("window filling", early)
    rep("steady state", late)
    if args.profile:
        for k, v in w.timings.items():
            v = np.array(v[args.warmup:]) * 1e3
            print(f"  {k:10s} mean {v.mean():7.1f} ms  p95 {np.percentile(v,95):7.1f}")
    if args.stride:
        print(f"forced (staleness) refreshes: {w.session.forced_refreshes}")
    print("text:", "".join(text)[:300])


if __name__ == "__main__":
    main()
