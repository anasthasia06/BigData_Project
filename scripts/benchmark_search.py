#!/usr/bin/env python3
import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx


def run_one(base_url: str, path: str, params: dict, timeout: float) -> float:
    start = time.perf_counter()
    resp = httpx.get(f"{base_url}{path}", params=params, timeout=timeout)
    resp.raise_for_status()
    return (time.perf_counter() - start) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark direct de /search")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--q", default="action")
    parser.add_argument("--genres", default="RPG")
    parser.add_argument("--min-positive-ratio", type=float, default=0.3)
    parser.add_argument("--max-price", type=float, default=46.0)
    parser.add_argument("--size", type=int, default=10)
    args = parser.parse_args()

    params = {
        "q": args.q,
        "genres": args.genres,
        "min_positive_ratio": args.min_positive_ratio,
        "max_price": args.max_price,
        "size": args.size,
    }

    before = httpx.get(f"{args.base_url}/metrics-json", timeout=args.timeout).json()

    latencies = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(run_one, args.base_url, "/search", params, args.timeout)
            for _ in range(args.requests)
        ]
        for fut in as_completed(futures):
            latencies.append(fut.result())

    after = httpx.get(f"{args.base_url}/metrics-json", timeout=args.timeout).json()

    latencies.sort()
    p95_index = int(0.95 * (len(latencies) - 1))

    print("=== SEARCH BENCHMARK ===")
    print(f"base_url: {args.base_url}")
    print(f"requests: {args.requests}")
    print(f"concurrency: {args.concurrency}")
    print(f"avg_ms: {statistics.mean(latencies):.2f}")
    print(f"median_ms: {statistics.median(latencies):.2f}")
    print(f"p95_ms: {latencies[p95_index]:.2f}")
    print(f"min_ms: {latencies[0]:.2f}")
    print(f"max_ms: {latencies[-1]:.2f}")
    print("---")
    print(f"memory_rss_before: {before.get('memory_rss_bytes')}")
    print(f"memory_rss_after: {after.get('memory_rss_bytes')}")
    print(f"search_cache_after: {after.get('search_cache')}")
    print(f"recommend_cache_after: {after.get('recommend_cache')}")
    print(f"latency_summary_after: {after.get('latency')}")


if __name__ == "__main__":
    main()
