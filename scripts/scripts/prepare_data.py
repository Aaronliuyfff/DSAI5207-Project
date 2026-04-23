from __future__ import annotations

import argparse
from pathlib import Path

from ecommerce_strategy_ft.dataset_builders import build_processed_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_processed_datasets(args.raw_root, args.processed_root)
    for name, count in stats.items():
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
