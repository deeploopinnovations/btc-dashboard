#!/usr/bin/env python3
"""
Export model/research/ledger.json to CSV format.

Converts the machine-readable experiment record to a tabular format
with columns: id, topic, question, rule, verdict, result, supersedes, superseded_by

Usage:
    python export_registry.py --input ledger.json --output registry.csv
"""

import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any


def export_ledger_to_csv(input_path: Path, output_path: Path) -> int:
    """
    Read ledger.json and export to CSV.

    Args:
        input_path: Path to ledger.json
        output_path: Path to output CSV file

    Returns:
        Number of experiments exported
    """
    with open(input_path, 'r') as f:
        data = json.load(f)

    experiments = data.get('experiments', [])

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['id', 'topic', 'question', 'rule', 'verdict', 'result', 'supersedes', 'superseded_by'],
            extrasaction='ignore'
        )
        writer.writeheader()

        for exp in experiments:
            row = {
                'id': exp.get('id', ''),
                'topic': exp.get('topic', ''),
                'question': exp.get('question', ''),
                'rule': exp.get('rule', ''),
                'verdict': exp.get('verdict', ''),
                'result': exp.get('result', ''),
                'supersedes': ','.join(exp.get('supersedes', [])) if exp.get('supersedes') else '',
                'superseded_by': ','.join(exp.get('superseded_by', [])) if exp.get('superseded_by') else '',
            }
            writer.writerow(row)

    return len(experiments)


def main():
    parser = argparse.ArgumentParser(
        description='Export ledger.json to CSV format.'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path(__file__).parent / 'ledger.json',
        help='Path to input ledger.json (default: ledger.json in same directory)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path(__file__).parent.parent.parent / 'EXPERIMENT_REGISTRY.csv',
        help='Path to output CSV file (default: EXPERIMENT_REGISTRY.csv at repo root)'
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=__import__('sys').stderr)
        return 1

    try:
        count = export_ledger_to_csv(args.input, args.output)
        print(f"Exported {count} experiments to {args.output}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=__import__('sys').stderr)
        return 1


if __name__ == '__main__':
    exit(main())
