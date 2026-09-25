#!/usr/bin/env python3
"""Run the released HemoPI2 Model 4 with a batch-invariant MERCI parser.

The upstream parser treats an empty negative-motif coverage section as one
negative hit and only parses the final coverage block.  This wrapper leaves
the released ESM2 model and MERCI executable unchanged, but replaces only the
parser functions in an isolated copy of the script.  All motif-coverage
summary lines are collected and missing sequences receive zero hits.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


def _fixed_functions() -> str:
    return r'''
def _parse_merci_summary(path):
    import re
    counts = {}
    pattern = re.compile(r"^([A-Za-z0-9]+)\s+\((\d+)\s+motifs match\)\s*$")
    with open(path) as handle:
        for line in handle:
            match = pattern.match(line.strip())
            if match:
                counts[match.group(1).lstrip('>')] = int(match.group(2))
    return counts

def _merci_write(wd, merci_file, merci_processed, name, hit_column, positive):
    import numpy as np
    import pandas as pd
    ids = [str(x).lstrip('>') for x in list(pd.DataFrame(name)[0])]
    counts = _parse_merci_summary(f"{wd}/{merci_file}")
    values = [int(counts.get(seqid, 0)) for seqid in ids]
    labels = [('Hemolytic' if positive else 'Non-Hemolytic') if n > 0
              else ('Non-Hemolytic' if positive else 'Hemolytic') for n in values]
    pd.DataFrame({'SeqID': ids, hit_column: values, 'Prediction': labels}).to_csv(
        f"{wd}/{merci_processed}", index=False)

def MERCI_Processor_p(wd, merci_file, merci_processed, name):
    _merci_write(wd, merci_file, merci_processed, name, 'PHits', True)

def Merci_after_processing_p(wd, merci_processed, final_merci_p):
    import pandas as pd
    df = pd.read_csv(f"{wd}/{merci_processed}")
    df['MERCI Score Pos'] = (df['PHits'] > 0).astype(float) * 0.5
    df[['SeqID', 'MERCI Score Pos']].to_csv(f"{wd}/{final_merci_p}", index=False)

def MERCI_Processor_n(wd, merci_file, merci_processed, name):
    _merci_write(wd, merci_file, merci_processed, name, 'NHits', False)

def Merci_after_processing_n(wd, merci_processed, final_merci_n):
    import pandas as pd
    df = pd.read_csv(f"{wd}/{merci_processed}")
    df['MERCI Score Neg'] = -(df['NHits'] > 0).astype(float) * 0.5
    df[['SeqID', 'MERCI Score Neg']].to_csv(f"{wd}/{final_merci_n}", index=False)
'''


def patch_script(source: Path, destination: Path) -> None:
    text = source.read_text()
    start = text.index("def MERCI_Processor_p(")
    end = text.index("def hybrid(", start)
    destination.write_text(text[:start] + _fixed_functions() + "\n\n" + text[end:])


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--official-script', type=Path, required=True)
    parser.add_argument('--runtime-dir', type=Path, required=True)
    args, forwarded = parser.parse_known_args()
    if not forwarded:
        raise SystemExit('forward official HemoPI2 arguments after wrapper options')
    with tempfile.TemporaryDirectory(prefix='hemopi2_fixed_') as tmp:
        patched = Path(tmp) / 'hemopi2_classification.py'
        patch_script(args.official_script, patched)
        subprocess.run(['python', str(patched), *forwarded], cwd=args.runtime_dir, check=True)


if __name__ == '__main__':
    main()
