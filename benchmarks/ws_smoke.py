"""End-to-end smoke test: stream an audio file to a running hikari-server in real time (80 ms chunks).

    hikari-server --backend mlx --port 4440 --checkpoint sbintuitions/hikari-medium &
    python benchmarks/ws_smoke.py --audio jfk.flac --port 4440 --window 75 --stride 16
"""

import argparse
import asyncio
import base64
import json
import time

import librosa
import numpy as np
import websockets

CHUNK = 1280


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--port", type=int, default=4440)
    ap.add_argument("--window", type=int, default=75)
    ap.add_argument("--stride", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=30)
    args = ap.parse_args()

    audio, _ = librosa.load(args.audio, sr=16000, mono=True)
    audio = np.tile(audio, int(np.ceil(args.seconds * 16000 / len(audio))))[: int(args.seconds * 16000)]
    pcm = (audio * 32768).astype(np.int16)
    chunks = [pcm[i : i + CHUNK] for i in range(0, len(pcm) - CHUNK + 1, CHUNK)]

    async with websockets.connect(f"ws://127.0.0.1:{args.port}") as ws:
        control = {"task": "transcribe", "tgt_lang": "English", "baseline_wait_penalty": 0.0,
                   "decoder_context": args.window, "refresh_stride": args.stride}
        await ws.send(json.dumps({"text": json.dumps({"control": control}), "audio": None, "timestamp": 0}))
        await asyncio.sleep(1.0)

        sent_at, lat, words, qmax = {}, [], [], 0

        async def receiver():
            nonlocal qmax
            async for msg in ws:
                d = json.loads(msg)
                now = time.time()
                if d["timestamp"] in sent_at:
                    lat.append(now - sent_at[d["timestamp"]])
                qmax = max(qmax, d["telemetry"]["queue_in"])
                if d["text"]:
                    words.append(d["text"])

        rt = asyncio.create_task(receiver())
        t0 = time.time()
        for i, c in enumerate(chunks):
            await asyncio.sleep(max(0, t0 + i * 0.08 - time.time()))
            sent_at[i] = time.time()
            await ws.send(json.dumps({"text": "", "audio": base64.b64encode(c.tobytes()).decode(), "timestamp": i}))
        await asyncio.sleep(2.0)
        rt.cancel()
    lat = np.array(lat[3:])
    print(f"chunks sent {len(chunks)}  replies {len(lat)+3}  round-trip latency: median {np.median(lat)*1e3:.0f} ms  "
          f"p95 {np.percentile(lat,95)*1e3:.0f} ms  max {lat.max()*1e3:.0f} ms   max incoming queue depth {qmax}")
    print("text:", "".join(words)[:200])


asyncio.run(main())
