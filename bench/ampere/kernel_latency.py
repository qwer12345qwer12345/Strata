#!/usr/bin/env python3
"""Compare SM86 Gyro expert kernels on identical GPU-resident fixtures."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess


def run(exe, mode):
    env = {key: value for key, value in os.environ.items() if not key.startswith('STRATA_')}
    env.update(STRATA_TQ_AMPERE_ROWS={'single': '0', 'pair': '2'}[mode])
    trial = subprocess.run([exe, '--ampere-latency'], env=env, check=True,
                           text=True, capture_output=True)
    if ('TQ Ampere paired-row GU enabled' in trial.stderr) != (mode == 'pair'):
        raise RuntimeError(f'{mode}: requested CUDA path did not run')
    cases = {}
    for line in trial.stdout.splitlines():
        tag, gu_type, groups, entries, phase, checksum, *measurements = line.split()
        shape = (int(gu_type), int(groups), int(entries), phase)
        samples = [float(value) for value in measurements]
        if (tag != 'LATENCY' or shape in cases or len(samples) != 200
                or not all(x > 0 and math.isfinite(x) for x in samples)):
            raise RuntimeError(f'{mode}: invalid kernel samples')
        cases[shape] = (checksum, samples)
    if len(cases) != 80:
        raise RuntimeError(f'{mode}: incomplete shape grid')
    return cases


def percentile(samples, fraction):
    ordered = sorted(samples)
    return ordered[math.ceil(len(ordered) * fraction) - 1]


def compare(exe, rounds):
    order = (['single', 'pair', 'pair', 'single', 'pair', 'single', 'single', 'pair'] * rounds)
    trials = []
    expected = None
    for mode in order:
        cases = run(exe, mode)
        checksums = {shape: record[0] for shape, record in cases.items()}
        if expected is None:
            expected = checksums
        elif checksums != expected:
            raise RuntimeError(f'{mode}: output hashes differ on identical fixtures')
        trials.append(cases)
    results = []
    for shape, checksum in expected.items():
        gu_type, groups, entries, phase = shape
        per_run = []
        for mode, trial in zip(order, trials):
            _, samples = trial[shape]
            per_run.append({'mode': mode, 'p50_us': statistics.median(samples),
                            'p95_us': percentile(samples, .95), 'p99_us': percentile(samples, .99)})
        # Compare matched adjacent processes to reduce clock and thermal drift.
        deltas = {}
        for field in ('p50_us', 'p95_us', 'p99_us'):
            differences = []
            for a, b in zip(per_run[::2], per_run[1::2]):
                single, pair = (a, b) if a['mode'] == 'single' else (b, a)
                differences.append(round(pair[field] - single[field], 3))
            deltas[field] = differences
        results.append({'gu_type': gu_type, 'down_type': 146, 'groups': groups,
                        'entries': entries or 'mixed_0_to_4', 'phase': phase,
                        'output_fnv1a': checksum, 'runs': per_run, 'pair_minus_single_us': deltas})
    return {'order': order, 'samples_per_run': 200, 'shape_results': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable', type=Path)
    parser.add_argument('--rounds', type=int, default=2, help='number of ABBA + BAAB blocks')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error('--rounds must be positive')
    args.output.write_text(json.dumps(compare(args.executable.resolve(), args.rounds), indent=2) + '\n')


if __name__ == '__main__':
    main()
