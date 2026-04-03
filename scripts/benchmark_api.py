#!/usr/bin/env python3
import argparse
import json
import statistics
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _call(url: str, timeout: float = 10.0):
    start = time.perf_counter()
    req = Request(url)
    with urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, payload


def _summary(values):
    if not values:
        return {"count": 0, "avg_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None}
    ordered = sorted(values)
    p50 = ordered[int(0.50 * (len(ordered) - 1))]
    p95 = ordered[int(0.95 * (len(ordered) - 1))]
    return {
        "count": len(values),
        "avg_ms": round(statistics.mean(values), 3),
        "p50_ms": round(p50, 3),
        "p95_ms": round(p95, 3),
        "max_ms": round(max(values), 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark simple des endpoints API")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", default="docs/performance_report.json")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    output_path = Path(args.output)

    search_lat = []
    reco_lat = []

    for _ in range(args.iterations):
        search_params = urlencode({"q": "action", "size": 10, "min_positive_ratio": 0.5, "max_price": 30})
        reco_params = urlencode({"n": 10, "genre": "Action"})

        try:
            elapsed, _ = _call(f"{base}/search?{search_params}")
            search_lat.append(elapsed)
        except (HTTPError, URLError):
            pass

        try:
            elapsed, _ = _call(f"{base}/recommend?{reco_params}")
            reco_lat.append(elapsed)
        except (HTTPError, URLError):
            pass

    report = {
        "base_url": base,
        "iterations": args.iterations,
        "search": _summary(search_lat),
        "recommend": _summary(reco_lat),
        "generated_at_epoch": int(time.time()),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
