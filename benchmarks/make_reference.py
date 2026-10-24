"""Golden reference: run the original torch model (CPU, fp32) in the server's streaming loop.

The loop mirrors ``ModelWrapper.get_one_token`` with wait/repetition penalties disabled and plain greedy
decoding, and dumps, for every step, the token history, the logits row used for the decision, and (for a
few steps) the encoder output. Used by tests/test_mlx_parity.py.

    python benchmarks/make_reference.py --audio jfk.flac --window 100 --out ~/.cache/hikari_mlx_ref/jfk_w100.npz
"""

import argparse
import copy
import time

import numpy as np
import librosa
import torch
import torch.nn.functional as F
import whisper

from hikari.models.configuration_hikari import HikariConfig
from hikari.models.hikari_model import HikariForConditionalGeneration

CHUNK = 1280  # 80 ms @ 16 kHz
PROMPTS = {
    "transcribe": [50258, 50259, 50359, 50363],
    "translate-ja": [50258, 50266, 50358, 50363],
    "translate-de": [50258, 50261, 50358, 50363],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="sbintuitions/hikari-medium")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--task", default="transcribe", choices=PROMPTS)
    ap.add_argument("--max-steps", type=int, default=10_000)
    ap.add_argument("--enc-every", type=int, default=25, help="store encoder output every N steps")
    ap.add_argument("--wait-boost", type=float, default=0.3)
    ap.add_argument("--wait-decay", type=float, default=0.3)
    ap.add_argument("--baseline-wait", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = HikariConfig.from_pretrained(args.checkpoint)
    cfg.use_cache = False
    model = HikariForConditionalGeneration.from_pretrained(
        args.checkpoint, config=cfg, attn_implementation="sdpa"
    ).float().eval()  # fp32 on CPU
    W = args.window
    model._effective_window = W
    dil = cfg.decoder_time_dilation

    audio, _ = librosa.load(args.audio, sr=16000, mono=True)
    audio = audio.astype(np.float32)
    n_chunks = len(audio) // CHUNK

    # same Silero VAD + wait-penalty policy as ModelWrapper.get_one_token
    vad, _ = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True)
    wp = args.baseline_wait
    ids = copy.deepcopy(PROMPTS[args.task])
    rec = {k: [] for k in ("ids", "n_ids", "pos", "logits", "token", "enc_step", "wp", "speech")}
    enc_out = {}
    j = -1
    t0 = time.time()
    for c in range(1, n_chunks + 1):
        audio_np = audio[: c * CHUNK]
        if len(audio_np) < 160 * 2 * dil * 3:
            continue
        j += 1
        pos = min(W - 1, j + 3)
        win = audio_np[max(0, len(audio_np) - W * 160 * 2 * dil) :]
        with torch.no_grad():
            mel = whisper.log_mel_spectrogram(torch.from_numpy(win))
            mel = F.pad(mel.unsqueeze(0), (0, 3000 - mel.shape[-1]))
            enc = model.model.encoder(mel)
            if len(ids) > W:
                ids.pop(4)
            dec_ids = F.pad(torch.tensor(ids).unsqueeze(0), (0, W - len(ids)))
            out = model(
                encoder_outputs=enc,
                decoder_input_ids=dec_ids,
                positions=torch.arange(W).unsqueeze(0),
                past_key_values=None,
                use_cache=False,
                return_dict=True,
            )
            row = out.logits[0, pos].float().numpy()
        with torch.no_grad():
            speech = vad(torch.from_numpy(win[-512:]), 16000).item() > 0.8
        pen = row.copy()
        pen[93] -= wp
        tok = int(pen.argmax())
        if tok == 93:
            if speech and len(ids) + 1 > 10 and all(t == 93 for t in (ids + [tok])[-10:]):
                wp += args.wait_boost
        else:
            wp -= args.wait_decay * (wp - args.baseline_wait)
        rec["ids"].append(np.pad(np.array(ids), (0, W - len(ids))))
        rec["n_ids"].append(len(ids))
        rec["pos"].append(pos)
        rec["logits"].append(row)
        rec["token"].append(tok)
        rec["wp"].append(wp)
        rec["speech"].append(speech)
        if j % args.enc_every == 0:
            enc_out[f"enc_{j}"] = enc.last_hidden_state[0].numpy()
            rec["enc_step"].append(j)
        ids.append(tok)
        print(f"step {j} chunk {c} pos {pos} tok {tok} wp {wp:.2f} speech {speech} ({time.time() - t0:.0f}s)", flush=True)
        if j + 1 >= args.max_steps:
            break

    np.savez_compressed(
        args.out,
        audio=audio[: (c) * CHUNK],
        window=W,
        task=args.task,
        **{k: np.array(v) for k, v in rec.items()},
        **enc_out,
    )
    print("saved", args.out)


if __name__ == "__main__":
    main()
