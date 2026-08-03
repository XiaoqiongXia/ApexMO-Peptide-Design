"""Generate exact-length AMP candidates from a trained checkpoint."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import torch

from .flow_matching import generate_token_ids
from .model import AMPDDiT, AMPDDiTConfig
from .tokenizer import PeptideTokenizer
from .tracking import seed_everything


def load_checkpoint_model(
    checkpoint_path: Path,
    *,
    device: torch.device,
    use_ema: bool = True,
) -> tuple[AMPDDiT, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = AMPDDiT(AMPDDiTConfig.from_mapping(checkpoint["config"]["model"])).to(device)
    model.load_state_dict(checkpoint["model"])
    if use_ema:
        shadow = checkpoint["ema"]["shadow"]
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in shadow:
                    parameter.copy_(shadow[name].to(device))
    model.eval()
    return model, checkpoint


@torch.no_grad()
def generate_candidates(
    model: AMPDDiT,
    *,
    lengths: Sequence[int],
    checkpoint_path: Path,
    steps: int,
    seed: int,
    device: torch.device,
    batch_size: int = 512,
    path_exponent: float = 1.0,
    time_epsilon: float = 1e-3,
    max_length: int | None = None,
) -> pd.DataFrame:
    seed_everything(seed)
    tokenizer = PeptideTokenizer()
    rows = []
    for start in range(0, len(lengths), batch_size):
        batch_lengths = list(lengths[start : start + batch_size])
        token_ids = generate_token_ids(
            model,
            lengths=batch_lengths,
            steps=steps,
            device=device,
            path_exponent=path_exponent,
            time_epsilon=time_epsilon,
            max_length=max_length,
        )
        for length, tokens in zip(batch_lengths, token_ids, strict=True):
            sequence = tokenizer.decode(tokens)
            if len(sequence) != length:
                raise RuntimeError("Generated sequence length does not match condition")
            rows.append(
                {
                    "sequence": sequence,
                    "length": length,
                    "seed": seed,
                    "checkpoint": str(checkpoint_path.resolve()),
                    "sampling_steps": steps,
                }
            )
    return pd.DataFrame(rows)


def _parse_lengths(value: str) -> list[int]:
    lengths = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not lengths or any(length < 5 or length > 64 for length in lengths):
        raise argparse.ArgumentTypeError("lengths must be comma-separated integers in [5, 64]")
    return lengths


def allocate_length_distribution(frame: pd.DataFrame, total: int) -> list[int]:
    """Allocate an exact sample total using validation length frequencies."""

    if total <= 0:
        raise ValueError("total must be positive")
    if "length" not in frame:
        raise ValueError("Length-distribution CSV must contain a length column")
    lengths = pd.to_numeric(frame["length"], errors="raise").astype(int)
    if lengths.empty or not lengths.between(5, 64).all():
        raise ValueError("Length-distribution values must be in [5, 64]")
    counts = lengths.value_counts().sort_index()
    expected = counts / counts.sum() * total
    allocated = expected.map(math.floor).astype(int)
    remainder = total - int(allocated.sum())
    fractional = (expected - allocated).sort_values(ascending=False, kind="mergesort")
    for length in fractional.index[:remainder]:
        allocated.loc[length] += 1
    return [
        int(length) for length, count in allocated.sort_index().items() for _ in range(int(count))
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    lengths = parser.add_mutually_exclusive_group(required=True)
    lengths.add_argument("--lengths", type=_parse_lengths)
    lengths.add_argument("--length-distribution", type=Path)
    parser.add_argument("--num-per-length", type=int, default=100)
    parser.add_argument(
        "--total",
        type=int,
        help="Exact total when --length-distribution is used.",
    )
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-weights", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    model, checkpoint = load_checkpoint_model(
        args.checkpoint,
        device=device,
        use_ema=not args.raw_weights,
    )
    if args.length_distribution is not None:
        if args.total is None:
            raise ValueError("--total is required with --length-distribution")
        requested = allocate_length_distribution(pd.read_csv(args.length_distribution), args.total)
    else:
        if args.total is not None:
            raise ValueError("--total is only valid with --length-distribution")
        if args.num_per_length <= 0:
            raise ValueError("--num-per-length must be positive")
        requested = [length for length in args.lengths for _ in range(args.num_per_length)]
    candidates = generate_candidates(
        model,
        lengths=requested,
        checkpoint_path=args.checkpoint,
        steps=args.steps,
        seed=args.seed,
        device=device,
        batch_size=args.batch_size,
        path_exponent=float(checkpoint["config"]["flow"]["exponent"]),
        time_epsilon=float(checkpoint["config"]["generation"]["time_epsilon"]),
        max_length=int(checkpoint["config"]["model"]["max_length"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.output, index=False)
    print(f"Wrote {len(candidates)} candidates to {args.output}")


if __name__ == "__main__":
    main()
