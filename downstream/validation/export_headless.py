#!/usr/bin/env python3
"""Export numeric measurements from the synthetic headless suite."""
import argparse
import json
from pathlib import Path

CASE_FIELDS = ('case', 'target_tokens', 'first_content_ms', 'wall_ms', 'finish',
               'verify_rounds', 'verify_width_mean', 'tokens_per_round', 'ms_per_round',
               'expert_file_reads', 'expert_mb_accounted', 'output_sha256')
TIMING_FIELDS = ('prompt_n', 'prompt_ms', 'predicted_n', 'predicted_ms', 'predicted_per_second',
                 'draft_n', 'draft_n_accepted', 'cache_n')
STREAM_FIELDS = ('content_events', 'gap_p50_ms', 'gap_p90_ms', 'gap_p95_ms', 'gap_p99_ms', 'gap_max_ms')


def export(source):
    report = json.loads(source.read_text())
    if report.get('rows_order') != ['pair', 'single', 'single', 'pair'] or report.get('stream_latency') is not True:
        raise ValueError('expected the four-arm synthetic streaming suite')
    output = {'engine': '0.1.60 plus local kernels and PR5500', 'prior': .5,
              'order': report['rows_order'], 'runs': []}
    for trial in report['runs']:
        if trial['phase'] not in ('pair', 'single', 'gyro-m-32k'):
            raise ValueError('unexpected experiment arm')
        run = {'mode': trial['phase'], 'status': trial['status'], 'cases': []}
        run['memory'] = {key: value for key, value in trial.get('memory', {}).items()
                         if key in ('min_available_mib', 'max_new_swap_mib') and type(value) is int}
        if trial['phase'] == 'gyro-m-32k' and trial['status'] == 'failed':
            run['failure'] = 'Stopped at the configured host memory or time limit.'
        for case in trial['cases']:
            if case['case'] not in ('warmup', 'long_output', 'long_context', 'second_output'):
                raise ValueError('unexpected synthetic workload')
            row = {key: case[key] for key in CASE_FIELDS if key in case}
            for field, names in [('timings', TIMING_FIELDS), ('stream', STREAM_FIELDS)]:
                row[field] = {key: case[field][key] for key in names
                              if key in case.get(field, {}) and type(case[field][key]) in (int, float)}
            run['cases'].append(row)
        output['runs'].append(run)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(export(args.source), indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
