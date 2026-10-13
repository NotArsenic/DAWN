#!/usr/bin/env python3
"""Measure inference computational cost (GMACs) and runtime latency per component."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dawn.flow.loaders import load_config  # noqa: E402
from dawn.flow.render import RenderSettings, render  # noqa: E402
from dawn.flow.sampler import sample_parts  # noqa: E402
from dawn.flow.tokens import TokenLayout  # noqa: E402
from dawn.flow.trainer import build_models  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def sync(device: torch.device) -> None:
    """Synchronize compute device streams for accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def count_macs(fn) -> float:
    """Measure GMACs for a single invocation of fn."""
    from torch.utils.flop_counter import FlopCounterMode

    with FlopCounterMode(display=False) as counter:
        fn()
    return counter.get_total_flops() / 2 / 1e9


def time_ms(fn, device: torch.device, warmup: int, trials: int) -> dict[str, float]:
    """Measure median, p90, and minimum execution time in milliseconds."""
    for _ in range(warmup):
        fn()
    sync(device)
    times = []
    for _ in range(trials):
        start = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - start) * 1000.0)
    times.sort()
    return {
        "ms_median": statistics.median(times),
        "ms_p90": times[min(len(times) - 1, int(0.9 * (len(times) - 1)))],
        "ms_min": times[0],
    }


@torch.no_grad()
def benchmark(cfg, bundle, device, args) -> list[dict]:
    """Benchmark execution time and GMACs for each model component."""
    layout = bundle["layout"]
    encoder, velocity = bundle["encoder"], bundle["velocity"]
    size = bundle["input_size"]
    settings = bundle["settings"]
    b = args.batch_size
    thermal = torch.rand(b, 1, *size, device=device)
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and args.precision == "bf16"
        else nullcontext()
    )

    def encode():
        with autocast:
            return encoder.encode(thermal, return_attn=not args.no_attn)

    encoded = encode()
    mean = torch.zeros(layout.dim, device=device)
    std = torch.ones(layout.dim, device=device)

    def sample():
        with autocast:
            return sample_parts(
                velocity,
                encoded["patch_tokens"],
                layout,
                mean,
                std,
                attn_bias=encoded.get("attn_bias"),
                thumb=encoded.get("thumb"),
                method=args.method,
                nfe=args.nfe,
                num_samples=args.samples,
            )

    parts = sample()
    one = {k: v[0] for k, v in parts.items()}

    def do_render():
        return render(one, thermal, size, settings)

    stages = [
        ("encoder (DINOv3 + LoRA, once)", encode),
        (f"velocity x {args.nfe} NFE x {args.samples} sample(s)", sample),
        (f"render (slice + {settings.upsample} upsample + Lab)", do_render),
    ]
    rows = []
    for name, fn in stages:
        row = {"component": name}
        row |= time_ms(fn, device, args.warmup, args.trials)
        try:
            row["gmacs"] = count_macs(fn)
        except Exception as exc:
            logger.warning("MAC count failed for %s: %s", name, exc)
            row["gmacs"] = float("nan")
        rows.append({k: (v / b if k.startswith(("ms_", "gmacs")) else v) for k, v in row.items()})

    total = {"component": "TOTAL per frame"}
    for k in ("ms_median", "ms_p90", "ms_min", "gmacs"):
        total[k] = sum(r[k] for r in rows)
    rows.append(total)
    return rows


def table(rows: list[dict]) -> str:
    """Format benchmark results as a Markdown table."""
    head = f"| {'component':<42} | {'GMACs':>8} | {'ms':>8} | {'p90 ms':>8} | {'FPS':>7} |"
    out = [
        head,
        "|" + "-" * 44 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 9 + "|",
    ]
    for r in rows:
        fps = 1000.0 / r["ms_median"] if r["ms_median"] > 0 else float("inf")
        out.append(
            f"| {r['component']:<42} | {r['gmacs']:>8.2f} | {r['ms_median']:>8.2f} | "
            f"{r['ms_p90']:>8.2f} | {fps:>7.1f} |"
        )
    return "\n".join(out)


def main() -> None:
    """Run inference latency and compute benchmark."""
    parser = argparse.ArgumentParser(description="Dawn inference cost, per component")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Pin the layout/resolution of a run. Random weights otherwise: "
        "cost does not depend on weight values",
    )
    parser.add_argument("--config", type=str, default="configs/flow.yaml")
    parser.add_argument("--data-config", type=str, default="configs/data.yaml")
    parser.add_argument("--override", nargs="*", default=[], help="Extra YAML files, in order")
    parser.add_argument("--device", type=str, default="cpu", help="cpu | cuda | mps")
    parser.add_argument("--batch-size", type=int, default=1, help="Frames per call")
    parser.add_argument("--nfe", type=int, default=2, help="Velocity evaluations")
    parser.add_argument("--method", choices=("euler", "midpoint"), default="euler")
    parser.add_argument("--samples", type=int, default=1, help="K, for the uncertainty path")
    parser.add_argument(
        "--precision",
        choices=("bf16", "fp32"),
        default="bf16",
        help="bf16 autocast is CUDA only; CPU and MPS run fp32 regardless",
    )
    parser.add_argument(
        "--no-attn",
        action="store_true",
        help="Skip the DINO attention bias, to price what capturing it costs",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--output", type=str, default=None, help="Write the table as JSON here")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.checkpoint:
        from dawn.flow.inference import load_checkpoint

        ck = load_checkpoint(args.checkpoint, device, use_ema=True)
        cfg = ck.config
        bundle = {
            "encoder": ck.encoder,
            "velocity": ck.velocity,
            "layout": ck.layout,
            "input_size": ck.input_size,
            "settings": ck.settings,
        }
        logger.info("checkpoint %s, step %d", args.checkpoint, ck.step)
    else:
        cfg = load_config(args.config, args.data_config, args.override)
        layout = TokenLayout.from_config(cfg)
        encoder, velocity, _ = build_models(cfg, layout, device)
        bundle = {
            "encoder": encoder.eval(),
            "velocity": velocity.eval(),
            "layout": layout,
            "input_size": tuple(cfg["encoder"]["input_size"]),
            "settings": RenderSettings.from_config(cfg),
        }
        logger.info("random weights (cost does not depend on them)")

    logger.info(
        "%s, %s, batch %d, %s NFE %d, K=%d, attention bias %s",
        device,
        bundle["input_size"],
        args.batch_size,
        args.method,
        args.nfe,
        args.samples,
        "off" if args.no_attn else "on",
    )
    if device.type == "cpu":
        logger.warning("CPU timings do not predict GPU ones; the GMACs column still does")

    rows = benchmark(cfg, bundle, device, args)
    print()
    print(table(rows))
    print()
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(
                {
                    "device": str(device),
                    "input_size": list(bundle["input_size"]),
                    "nfe": args.nfe,
                    "samples": args.samples,
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
