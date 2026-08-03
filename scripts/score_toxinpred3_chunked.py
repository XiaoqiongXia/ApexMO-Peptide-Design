#!/usr/bin/env python3
"""Run the released ToxinPred3 CLI in bounded FASTA chunks and merge outputs."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


def records(path: Path):
    header = None
    seq = []
    for line in path.read_text().splitlines():
        if line.startswith('>'):
            if header is not None:
                yield header, ''.join(seq)
            header, seq = line, []
        elif line.strip():
            seq.append(line.strip())
    if header is not None:
        yield header, ''.join(seq)


def pad_singleton(chunk):
    """Work around the released CLI's 1-D feature matrix for one sequence."""

    if len(chunk) != 1:
        return chunk
    return [*chunk, ('>__AMP_DESIGN_PADDING__', chunk[0][1])]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--runtime-dir', type=Path, required=True)
    p.add_argument('--script', type=Path, required=True)
    p.add_argument('--python', default='python')
    p.add_argument('--chunk-size', type=int, default=1000)
    p.add_argument('--threshold', type=float, default=0.38)
    p.add_argument('--chunks-dir', type=Path, required=True)
    args = p.parse_args()
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    args.chunks_dir = args.chunks_dir.resolve()
    rows = list(records(args.input))
    args.chunks_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    with tempfile.TemporaryDirectory(prefix='toxinpred3_chunks_') as tmp:
        tmp = Path(tmp)
        for start in range(0, len(rows), args.chunk_size):
            chunk = rows[start:start + args.chunk_size]
            # The released ToxinPred3 CLI uses np.loadtxt and crashes on a
            # one-record batch because its feature matrix becomes 1-D. Pad
            # only the inference batch, then discard the padding prediction.
            inference_chunk = pad_singleton(chunk)
            out = args.chunks_dir / f'chunk_{start:06d}.csv'
            output_paths.append(out)
            if out.exists():
                frame = pd.read_csv(out)
                if len(frame) == len(chunk):
                    print(f'skipping validated {start + len(chunk)}/{len(rows)}', flush=True)
                    continue
                out.unlink()
            fasta = tmp / f'chunk_{start:06d}.fasta'
            fasta.write_text('\n'.join(f'{h}\n{s}' for h, s in inference_chunk) + '\n')
            log = args.chunks_dir / f'chunk_{start:06d}.log'
            subprocess.run([
                args.python, str(args.script), '-i', str(fasta), '-o', str(out),
                '-m', '2', '-t', str(args.threshold), '-d', '2',
            ], cwd=args.runtime_dir, check=True, stdout=log.open('w'), stderr=subprocess.STDOUT)
            frame = pd.read_csv(out)
            if len(frame) != len(inference_chunk):
                raise RuntimeError(
                    f'chunk cardinality mismatch at {start}: '
                    f'{len(frame)} != {len(inference_chunk)}'
                )
            if len(inference_chunk) != len(chunk):
                frame = frame.iloc[:len(chunk)].copy()
                frame.to_csv(out, index=False)
            print(f'completed {min(start + len(chunk), len(rows))}/{len(rows)}', flush=True)
    outputs = [pd.read_csv(path) for path in output_paths]
    merged = pd.concat(outputs, ignore_index=True)
    if len(merged) != len(rows) or merged.iloc[:, 0].nunique() != len(rows):
        raise RuntimeError('merged output cardinality or identifier uniqueness failed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f'wrote {args.output} ({len(merged)} rows)')


if __name__ == '__main__':
    main()
