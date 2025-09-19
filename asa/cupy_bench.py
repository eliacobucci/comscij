#!/usr/bin/env python3
import time
try:
    import cupy as cp
except Exception as e:
    print("CuPy not available:", e)
    raise SystemExit(1)

def bench_cupy(n=4096, iters=5):
    A = cp.random.randn(n, n, dtype=cp.float32)
    B = cp.random.randn(n, n, dtype=cp.float32)
    cp.dot(A,B); cp.cuda.Stream.null.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.time()
        C = A.dot(B)
        cp.cuda.Stream.null.synchronize()
        times.append(time.time() - t0)
    avg = sum(times)/len(times)
    gflops = (2*n**3)/avg/1e9
    print(f"Avg time: {avg:.3f}s  ~ {gflops:.1f} GFLOP/s")
    return avg

if __name__ == "__main__":
    bench_cupy()
