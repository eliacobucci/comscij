#!/usr/bin/env python3
import time, torch

def bench_torch(n=4096, iters=5):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}")
    A = torch.randn(n, n, device=dev, dtype=torch.float32)
    B = torch.randn(n, n, device=dev, dtype=torch.float32)
    for _ in range(2):
        C = A @ B
        if dev == "cuda":
            torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.time()
        C = A @ B
        if dev == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
    avg = sum(times)/len(times)
    gflops = (2*n**3)/avg/1e9
    print(f"Avg time: {avg:.3f}s  ~ {gflops:.1f} GFLOP/s")
    if dev == "cpu":
        print("⚠ Running on CPU. Check CUDA install / torch build.")
    return avg

if __name__ == "__main__":
    bench_torch()
