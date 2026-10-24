"""Sustained fp16 matmul throughput of this GPU: python benchmarks/matmul_bench.py M K N"""
import sys
import time

import mlx.core as mx

m, k, n = (int(x) for x in sys.argv[1:4]) if len(sys.argv) > 3 else (1500, 1024, 4096)
a = mx.random.normal((m, k)).astype(mx.float16)
w = mx.random.normal((k, n)).astype(mx.float16)
mx.eval(a, w)
for rnd in range(6):
    t = time.perf_counter()
    mx.eval([a @ w for _ in range(40)])
    dt = (time.perf_counter() - t) / 40
    print(f"round {rnd}: {2 * m * k * n / dt / 1e12:5.2f} TFLOP/s")
