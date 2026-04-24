#!/usr/bin/env python3
"""Compare browser and MuJoCo PHP trace JSONL files.

This is a lightweight trace-diff tool for the PHPParkour migration work.
It focuses on the signals that most often reveal environment mismatches:
commands, depth summaries / latent, gravity, angular velocity, joint state,
and previous-action feedback.

Example:
    python tools/compare_php_traces.py \
      --browser web-record/php_parkour_trace_browser.jsonl \
      --mujoco logs/php_parkour_trace_mujoco_20260424_102245_752371.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def load_jsonl(path: Path) -> List[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def cmd_index(rec: dict) -> int:
    cmd = rec.get("cmd15")
    if not cmd:
        return -1
    for i, value in enumerate(cmd):
        if float(value) == 1.0:
            return i
    return -1


def mean_abs_diff(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(abs(float(x) - float(y)) for x, y in zip(a, b)) / len(a)


def max_abs_diff(a: Sequence[float], b: Sequence[float]) -> float:
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def depth_summary_distance(a: dict, b: dict) -> float:
    keys = ("min", "mean", "center_mean")
    score = 0.0
    used = 0
    for key in keys:
        if a and b and key in a and key in b:
            score += abs(float(a[key]) - float(b[key]))
            used += 1
    return score / used if used else float("inf")


def choose_offset(
    browser: Sequence[dict],
    mujoco: Sequence[dict],
    search: int,
    samples: int,
) -> Tuple[int, float]:
    best_offset = 0
    best_score = float("inf")
    for offset in range(-search, search + 1):
        total = 0.0
        used = 0
        for i in range(samples):
            bi = i
            mi = i + offset
            if bi < 0 or mi < 0 or bi >= len(browser) or mi >= len(mujoco):
                continue
            total += depth_summary_distance(
                browser[bi].get("depth_stats"),
                mujoco[mi].get("depth_stats"),
            )
            if cmd_index(browser[bi]) != cmd_index(mujoco[mi]):
                total += 0.25
            used += 1
        if used == 0:
            continue
        score = total / used
        if score < best_score:
            best_score = score
            best_offset = offset
    return best_offset, best_score


def aligned_pairs(
    browser: Sequence[dict],
    mujoco: Sequence[dict],
    offset: int,
    limit: int | None,
) -> List[Tuple[dict, dict]]:
    pairs = []
    count = 0
    for bi, brec in enumerate(browser):
        mi = bi + offset
        if mi < 0 or mi >= len(mujoco):
            continue
        pairs.append((brec, mujoco[mi]))
        count += 1
        if limit is not None and count >= limit:
            break
    return pairs


def summarize_slot(
    pairs: Sequence[Tuple[dict, dict]],
    slot_name: str,
) -> Tuple[float, float, int]:
    values = []
    maxima = []
    for brec, mrec in pairs:
        bslots = brec.get("obs_slots", {})
        mslots = mrec.get("obs_slots", {})
        if slot_name not in bslots or slot_name not in mslots:
            continue
        b = bslots[slot_name]
        m = mslots[slot_name]
        if len(b) != len(m):
            continue
        values.append(mean_abs_diff(b, m))
        maxima.append(max_abs_diff(b, m))
    if not values:
        return float("nan"), float("nan"), 0
    return sum(values) / len(values), max(maxima), len(values)


def summarize_depth_latent(
    pairs: Sequence[Tuple[dict, dict]],
) -> Tuple[float, float, int]:
    values = []
    maxima = []
    for brec, mrec in pairs:
        b = brec.get("depth_latent")
        m = mrec.get("depth_latent")
        if not b or not m or len(b) != len(m):
            continue
        values.append(mean_abs_diff(b, m))
        maxima.append(max_abs_diff(b, m))
    if not values:
        return float("nan"), float("nan"), 0
    return sum(values) / len(values), max(maxima), len(values)


def summarize_commands(
    pairs: Sequence[Tuple[dict, dict]],
) -> Tuple[float, Dict[str, int]]:
    mismatches = 0
    details: Dict[str, int] = {}
    for brec, mrec in pairs:
        b = cmd_index(brec)
        m = cmd_index(mrec)
        if b != m:
            mismatches += 1
            key = f"{b}->{m}"
            details[key] = details.get(key, 0) + 1
    frac = mismatches / len(pairs) if pairs else float("nan")
    return frac, details


def summarize_depth_stats(
    pairs: Sequence[Tuple[dict, dict]],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    keys = ("min", "mean", "center_mean", "top_mean", "bottom_mean")
    for key in keys:
        values = []
        for brec, mrec in pairs:
            b = brec.get("depth_stats")
            m = mrec.get("depth_stats")
            if not b or not m or key not in b or key not in m:
                continue
            values.append(abs(float(b[key]) - float(m[key])))
        if values:
            out[key] = sum(values) / len(values)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", type=Path, required=True)
    parser.add_argument("--mujoco", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument(
        "--offset",
        type=int,
        default=None,
                        help="Explicit mujoco tick offset relative to browser.")
    parser.add_argument("--offset-search", type=int, default=60,
                        help="Search window for automatic offset selection.")
    parser.add_argument("--offset-samples", type=int, default=80,
                        help="Number of early frames used for auto alignment.")
    args = parser.parse_args()

    browser = load_jsonl(args.browser)
    mujoco = load_jsonl(args.mujoco)
    if not browser:
        raise SystemExit(f"No records found in {args.browser}")
    if not mujoco:
        raise SystemExit(f"No records found in {args.mujoco}")

    if args.offset is None:
        offset, score = choose_offset(
            browser, mujoco, args.offset_search, args.offset_samples)
        print(f"auto_offset={offset}  alignment_score={score:.6f}")
    else:
        offset = args.offset
        print(f"manual_offset={offset}")

    pairs = aligned_pairs(browser, mujoco, offset, args.limit)
    print(f"aligned_pairs={len(pairs)}  limit={args.limit}")

    cmd_mismatch_frac, cmd_details = summarize_commands(pairs)
    print(f"command_mismatch_frac={cmd_mismatch_frac:.6f}")
    if cmd_details:
        top = sorted(
            cmd_details.items(), key=lambda kv: kv[1], reverse=True)[:8]
        print("command_mismatches_top=", top)

    print("depth_stats_mae=", summarize_depth_stats(pairs))

    for slot in (
        "robot_anchor_projected_gravity",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "actions",
    ):
        mean_mae, max_mae, count = summarize_slot(pairs, slot)
        print(
            f"{slot}: mean_abs={mean_mae:.6f}  max_abs={max_mae:.6f}  "
            f"count={count}"
        )

    latent_mean, latent_max, latent_count = summarize_depth_latent(pairs)
    print(
        f"depth_latent: mean_abs={latent_mean:.6f}  max_abs={latent_max:.6f}  "
        f"count={latent_count}"
    )


if __name__ == "__main__":
    main()
