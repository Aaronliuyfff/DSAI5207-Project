from __future__ import annotations

import argparse
from pathlib import Path

from ecommerce_strategy_ft.trainer_utils import merge_adapter, run_sft
from ecommerce_strategy_ft.utils import load_yaml, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--merge", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    set_seed(config.get("seed", 42))
    run_sft(config)
    if args.merge:
        merge_adapter(
            base_model_path=config["model_name_or_path"],
            adapter_path=str(Path(config["output_dir"]) / "final_adapter"),
            output_path=str(Path(config["output_dir"]) / "final_merged"),
        )


if __name__ == "__main__":
    main()
