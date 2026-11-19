#!/usr/bin/env python3
"""Aggregate Attentracker's numerical monitor_window logs (no frame data)."""

import argparse
from collections import defaultdict
from pathlib import Path
import re

FIELD = re.compile(r'\b([a-zA-Z_][a-zA-Z_0-9]*)=("[^"]*"|[^\s]+)')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()

    grouped = defaultdict(lambda: {"frames": 0, "elapsed": 0.0, "inference_ms": 0.0,
                                  "pipeline_ms": 0.0, "windows": 0})
    for path in args.logs:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if 'event="monitor_window"' not in line:
                    continue
                fields = {name: raw.strip('"') for name, raw in FIELD.findall(line)}
                try:
                    frames = int(fields["frames"])
                    elapsed = float(fields["elapsed_s"])
                    key = (
                        fields.get("model_input_side", "unknown"),
                        fields.get("preview_enabled", "unknown"),
                        fields.get("target_interval_ms", "unknown"),
                    )
                    if frames <= 0 or elapsed <= 0:
                        continue
                    group = grouped[key]
                    group["frames"] += frames
                    group["elapsed"] += elapsed
                    group["inference_ms"] += float(fields["inference_avg_ms"]) * frames
                    group["pipeline_ms"] += float(fields["pipeline_avg_ms"]) * frames
                    group["windows"] += 1
                except (KeyError, ValueError):
                    continue

    if not grouped:
        parser.exit(1, "No usable monitor_window observations found in supplied logs.\n")
    print(f'{"Input":>7} {"Preview":>8} {"Period":>8} {"Frames":>8} '
          f'{"Windows":>7} {"FPS":>8} {"Infer ms":>10} {"Pipe ms":>10}')
    for (side, preview, interval), g in sorted(grouped.items()):
        n = g["frames"]
        print(f'{side:>7} {preview:>8} {interval:>8} {n:>8} '
              f'{g["windows"]:>7} {n/g["elapsed"]:>8.2f} '
              f'{g["inference_ms"]/n:>10.1f} {g["pipeline_ms"]/n:>10.1f}')


if __name__ == "__main__":
    main()
