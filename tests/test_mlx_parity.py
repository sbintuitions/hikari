"""MLX vs torch-CPU-fp32 parity. Needs the golden dumps from benchmarks/make_reference.py:

    python benchmarks/make_reference.py --audio jfk.flac --window 100 --out ~/.cache/hikari_mlx_ref/jfk_w100.npz
"""

import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from hikari.mlx.engine import HikariMLX  # noqa: E402
from hikari.mlx.mel import log_mel  # noqa: E402

REF_DIR = os.path.expanduser(os.environ.get("HIKARI_REF_DIR", "~/.cache/hikari_mlx_ref"))
CHUNK = 1280
LOGIT_ATOL = 1.0  # fp16 MLX vs fp32 torch after removing the per-row offset (see _centered_err); typical 0.1, one outlier step ~0.75 (the same step is within 8e-3 in fp32)
TIE_MARGIN = 0.5  # top-1 may differ only when the reference's top-2 gap is below this


@pytest.fixture(scope="module")
def engine():
    return HikariMLX("sbintuitions/hikari-medium")


def _ref(name):
    path = os.path.join(REF_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"missing golden reference {path}")
    return np.load(path)


def _centered_err(row, ref_row):
    """fp16 error is dominated by a near-constant shift of the whole logit row, which does not change the
    argmax or any logit margin; compare the shift-invariant part."""
    d = row - ref_row
    return float(np.abs(d - d.mean()).max())


def _window_audio(ref, j):
    W = int(ref["window"])
    n = (j + 3) * CHUNK
    return ref["audio"][max(0, n - W * CHUNK) : n]


@pytest.mark.parametrize("name", ["jfk_w100.npz", "jfk_w375.npz"])
def test_logits_teacher_forced(engine, name):
    ref = _ref(name)
    W = int(ref["window"])
    n_steps = len(ref["pos"])
    mismatches, worst = 0, 0.0
    for j in range(n_steps):
        ids = [int(t) for t in ref["ids"][j][: ref["n_ids"][j]]]
        row = engine.logits_row(_window_audio(ref, j), ids, W, int(ref["pos"][j]))
        r = ref["logits"][j]
        worst = max(worst, _centered_err(row, r))
        if row.argmax() != r.argmax():
            top2 = np.sort(r)[-2:]
            assert top2[1] - top2[0] < TIE_MARGIN, f"step {j}: argmax differs with a clear margin"
            mismatches += 1
    assert worst < LOGIT_ATOL, f"max centered |logit diff| {worst}"
    assert mismatches <= max(1, int(0.005 * n_steps)), f"{mismatches} near-tie argmax flips"


def test_encoder_cosine_fp16(engine):
    """fp16 has a few precision-sensitive frames (torch fp16 shows the same ones), so test it statistically."""
    ref = _ref("jfk_w100.npz")
    for j in [int(s) for s in ref["enc_step"]]:
        mel = log_mel(_window_audio(ref, j), dtype=engine.dtype)
        e = np.array(engine.model.encoder(mel)[0].astype(mx.float32))
        r = ref[f"enc_{j}"]
        n_real = min(1500, 4 * (j + 3) - 2)  # frames fed by real audio; later frames are padding noise
        cos = (e * r).sum(-1) / (np.linalg.norm(e, axis=-1) * np.linalg.norm(r, axis=-1) + 1e-9)
        cos = cos[:n_real]
        assert np.median(cos) > 0.9999 and (cos < 0.999).mean() < 0.02 and cos.min() > 0.8, f"step {j}"


def test_fp32_is_exact():
    """In fp32 the MLX port reproduces the torch reference to ~1e-3 (encoder cosine, logits), also across the
    steps where the context window starts sliding (W=100: steps >= 97)."""
    ref = _ref("jfk_w100.npz")
    eng = HikariMLX("sbintuitions/hikari-medium", dtype=mx.float32)
    W = int(ref["window"])
    for j in [0, 50, 96, 97, 98, 120]:
        if j >= len(ref["pos"]):
            continue
        ids = [int(t) for t in ref["ids"][j][: ref["n_ids"][j]]]
        row = eng.logits_row(_window_audio(ref, j), ids, W, int(ref["pos"][j]))
        assert np.abs(row - ref["logits"][j]).max() < 5e-3, f"step {j}"
    for j in [int(s) for s in ref["enc_step"]][:3]:
        e = np.array(eng.model.encoder(log_mel(_window_audio(ref, j), dtype=mx.float32))[0])
        r = ref[f"enc_{j}"]
        cos = (e * r).sum(-1) / (np.linalg.norm(e, axis=-1) * np.linalg.norm(r, axis=-1) + 1e-9)
        assert cos.min() > 0.99999, f"enc step {j}"


def test_fully_masked_cross_attn_row_is_finite(engine):
    ids = mx.array(np.zeros((1, 8), dtype=np.int32))
    enc = mx.zeros((1, 32, engine.cfg.d_model), dtype=engine.dtype)
    h = engine.model.decoder(ids, enc)
    assert bool(mx.all(mx.isfinite(h)).item())


def test_wrapper_stream_matches_reference():
    """End to end (VAD + wait penalty + MLX): the emitted token stream follows the torch reference."""
    ref = _ref("jfk_w100.npz")
    from hikari.server.model_wrapper_mlx import ModelWrapperMLX

    w = ModelWrapperMLX(checkpoint="sbintuitions/hikari-medium")
    w.apply_settings(
        {"task": str(ref["task"]), "tgt_lang": "English", "baseline_wait_penalty": 0.0, "decoder_context": int(ref["window"])}
    )
    audio = ref["audio"]
    got = []
    for c in range(len(audio) // CHUNK):
        j_before = w.j
        w.get_one_token(audio[c * CHUNK : (c + 1) * CHUNK])
        if w.j != j_before:
            got.append(w.decoder_input_ids_lst[-1])
    want = [int(t) for t in ref["token"]][: len(got)]
    assert len(got) == len(want) and len(got) > 100
    agree = np.mean([a == b for a, b in zip(got, want)])
    assert agree > 0.95, f"token agreement {agree:.3f}"


def test_stride_mode_matches_fresh_recompute(engine):
    """Approximate stride-refresh mode: incremental encoding must stay close to re-encoding the same window
    from scratch (refresh steps are identical by construction; the rest differ only by cached-frame staleness)."""
    from hikari.mlx.stride import StrideSession

    ref = _ref("jfk_w100.npz")
    audio, W, K = ref["audio"], 50, 8
    sess = StrideSession(engine)
    ids = [50258, 50259, 50359, 50363]
    ws, errs, flips, n_refresh = 0, [], 0, 0
    for j, tok in enumerate(ref["token"]):
        t = (j + 3) * CHUNK
        refresh = False
        if len(ids) > W:
            del ids[4 : 4 + len(ids) - max(5, W - K)]
            refresh = True
            n_refresh += 1
        if refresh or sess.kvs is None:
            ws = max(0, t - (len(ids) - 1) * CHUNK)
        got = sess.logits_row(audio[ws:t], ids, refresh)
        want = engine.logits_row(audio[ws:t], ids, W, len(ids) - 1)
        errs.append((refresh, _centered_err(got, want)))
        flips += int(got.argmax() != want.argmax())
        ids.append(int(tok))
    assert n_refresh >= 5
    assert max(e for r, e in errs if r) < 1e-2  # a refresh is an exact full re-encode
    non = np.array([e for r, e in errs if not r])
    assert np.median(non) < 0.2 and non.max() < 2.0, f"median {np.median(non)}, max {non.max()}"
    assert flips <= 3, f"{flips} argmax flips"
