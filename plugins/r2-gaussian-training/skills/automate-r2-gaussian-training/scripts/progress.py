#!/usr/bin/env python3
"""Print one R2-Gaussian TensorBoard training progress snapshot."""

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def latest(ea, tag):
    values = ea.Scalars(tag) if tag in ea.Tags().get("scalars", []) else []
    return values[-1] if values else None


def duration(seconds):
    if seconds is None:
        return "?"
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}h{minutes:02d}m" if hours else f"{minutes:d}m{secs:02d}s"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--total-iterations", required=True, type=int)
    args = parser.parse_args()

    accumulator = EventAccumulator(str(args.model_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    loss = latest(accumulator, "train/loss_total")
    points = latest(accumulator, "train/total_points")
    timing = accumulator.Scalars("train/iter_time") if "train/iter_time" in accumulator.Tags().get("scalars", []) else []
    current = max((item.step for item in (loss, points) if item is not None), default=0)
    total = max(1, args.total_iterations)
    fraction = min(1.0, current / total)
    filled = round(fraction * 20)
    bar = "█" * filled + "░" * (20 - filled)

    elapsed = None
    eta = None
    if len(timing) >= 2:
        elapsed = timing[-1].wall_time - timing[0].wall_time
        window = timing[-min(100, len(timing)):]
        step_span = window[-1].step - window[0].step
        wall_span = window[-1].wall_time - window[0].wall_time
        if step_span > 0 and wall_span > 0:
            eta = (total - current) * wall_span / step_span

    loss_text = f"{loss.value:.5g}" if loss else "?"
    points_text = f"{int(points.value):,}" if points else "?"
    print(
        f"[{bar}] {current:,}/{total:,} {fraction * 100:5.2f}% | "
        f"loss {loss_text} | points {points_text} | elapsed {duration(elapsed)} | "
        f"ETA ~{duration(eta)}"
    )


if __name__ == "__main__":
    main()
