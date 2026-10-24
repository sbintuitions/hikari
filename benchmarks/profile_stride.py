"""Per-stage timing (with syncs) of non-refresh steps vs refresh steps in stride mode."""
import os, sys, time
import numpy as np, mlx.core as mx
from hikari.mlx.engine import HikariMLX
from hikari.mlx.mel import log_mel
import librosa

W, K = int(sys.argv[1]), int(sys.argv[2])
audio, _ = librosa.load(os.path.expanduser("~/.cache/hikari_mlx_ref/jfk.flac"), sr=16000)
audio = np.tile(audio.astype(np.float32), 6)
CH = 1280
e = HikariMLX(); m = e.model
dil = e.cfg.time_dilation
ids = [50258, 50259, 50359, 50363]
rng = np.random.default_rng(0)
rows = {"mel": [], "stale": [], "enc": [], "xkv": [], "dec": [], "n": []}
kvs = cross = mel_used = None; n_frames = 0; ws = 0
for j in range(0, 360):
    c = j + 3; t = c * CH
    refresh = False
    if len(ids) > W:
        keep = W - K; del ids[4:4 + len(ids) - keep]; refresh = True
    if refresh or kvs is None: ws = max(0, t - (len(ids) - 1) * CH)
    pos = len(ids) - 1; F = dil * pos
    t0 = time.perf_counter()
    mel = log_mel(audio[ws:t], pad_to=2 * F + 2, dtype=e.dtype); mx.eval(mel); t1 = time.perf_counter()
    nc = max(0, n_frames - 4)
    full = refresh or kvs is None
    t2 = t1
    if not full:
        full = False
    if full:
        enc, kvs = m.encoder.forward_capture(mel, F); mx.eval(enc, kvs); t3 = time.perf_counter()
        cross = m.decoder.cross_kv(enc); mx.eval(cross); t4 = time.perf_counter()
    else:
        enc_new, kvs = m.encoder.forward_incremental(mel, nc, F, kvs); mx.eval(enc_new, kvs); t3 = time.perf_counter()
        new = m.decoder.cross_kv(enc_new)
        cross = [(mx.concatenate([a[:, :, :nc], b], 2), mx.concatenate([c_[:, :, :nc], d], 2)) for (a, c_), (b, d) in zip(cross, new)]
        mx.eval(cross); t4 = time.perf_counter()
    n_frames = F
    ids_mx = mx.array(np.asarray(ids, dtype=np.int32)[None])
    row = m.decoder.head(m.decoder(ids_mx, cross_kv=cross)[:, pos]); mx.eval(row); t5 = time.perf_counter()
    if j > 10:
        key = "refresh" if full else "incr"
        rows.setdefault(key, []).append((t1 - t0, t3 - t2, t4 - t3, t5 - t4, t5 - t0, len(ids)))
    ids.append(int(rng.integers(300, 3000)))
for key in ("incr", "refresh"):
    r = np.array(rows.get(key, []))
    if len(r) == 0: continue
    ms = r[:, :5].mean(0) * 1e3
    print(f"{key:8s} n={len(r):3d}  mel {ms[0]:5.1f}  encoder {ms[1]:6.1f}  cross-kv {ms[2]:5.1f}  decoder+head {ms[3]:6.1f}  total {ms[4]:6.1f} ms")
