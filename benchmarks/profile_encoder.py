"""Time encoder pieces (convs, layers) and the decoder for a W=100-like step, using real weights."""
import sys, time
import numpy as np, mlx.core as mx
from hikari.mlx.engine import HikariMLX
from hikari.mlx.model import gelu

e = HikariMLX()
m = e.model
F = int(sys.argv[1]) if len(sys.argv) > 1 else 396
mel = mx.random.normal((1, 2 * F + 2, 80)).astype(mx.float16)
mx.eval(mel)

def bench(name, fn, reps=10):
    mx.eval(fn())
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t)
    print(f"{name:34s} median {np.median(ts)*1e3:7.2f} ms  min {np.min(ts)*1e3:7.2f}")
    return fn

enc = m.encoder
bench("conv1+gelu", lambda: gelu(enc.conv1(mel)))
c1 = gelu(enc.conv1(mel)); mx.eval(c1)
bench("conv2+gelu", lambda: gelu(enc.conv2(c1)))
x = (gelu(enc.conv2(c1))[:, :F] + enc.embed_positions.weight[:F]); mx.eval(x)
def layers(x):
    for l in enc.layers: x = l(x)
    return x
bench("24 encoder layers", lambda: layers(x))
bench("encoder total (1 eval)", lambda: enc(mel, n_out=F))
out = enc(mel, n_out=F); mx.eval(out)
ids = mx.array(np.zeros((1, 100), dtype=np.int32))
bench("decoder (100 tok, F frames)", lambda: m.decoder(ids, out))
# cross-attn K/V projection alone for all 24 layers
bench("cross K/V proj x24", lambda: [(l.encoder_attn.k_proj(out), l.encoder_attn.v_proj(out)) for l in m.decoder.layers])
h = m.decoder(ids, out); mx.eval(h)
bench("head row", lambda: m.decoder.head(h[:, 99]))
