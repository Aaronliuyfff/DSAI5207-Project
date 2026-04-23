from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

from ecommerce_strategy_ft.trainer_utils import build_dataset, load_tokenizer
from ecommerce_strategy_ft.utils import load_yaml, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--merge", action="store_true")
    return parser.parse_args()


def build_lora_model(model: torch.nn.Module, config: dict) -> torch.nn.Module:
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config["target_modules"],
    )
    return get_peft_model(model, lora_config)


def run_sft_mps(config: dict) -> None:
    tokenizer = load_tokenizer(config["model_name_or_path"])
    dataset = build_dataset(config["train_file"], tokenizer)
    max_train_samples = config.get("max_train_samples")
    if max_train_samples is not None:
        max_train_samples = int(max_train_samples)
        if max_train_samples > 0 and len(dataset) > max_train_samples:
            dataset = dataset.select(range(max_train_samples))

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    model = build_lora_model(model, config)
    model.to("mps")

    training_args = SFTConfig(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        max_steps=config.get("max_steps", -1),
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        warmup_ratio=config["warmup_ratio"],
        bf16=False,
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
        optim="adamw_torch",
        dataset_text_field="text",
        max_length=config["max_seq_length"],
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    final_adapter = Path(config["output_dir"]) / "final_adapter"
    trainer.model.save_pretrained(final_adapter)
    tokenizer.save_pretrained(final_adapter)


def merge_adapter(base_model_path: str, adapter_path: str, output_path: str) -> None:
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    merged = PeftModel.from_pretrained(model, adapter_path)
    merged = merged.merge_and_unload()
    merged.save_pretrained(output_path)

    tokenizer = load_tokenizer(adapter_path)
    tokenizer.save_pretrained(output_path)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    set_seed(config.get("seed", 42))
    run_sft_mps(config)
    if args.merge:
        merge_adapter(
            base_model_path=config["model_name_or_path"],
            adapter_path=str(Path(config["output_dir"]) / "final_adapter"),
            output_path=str(Path(config["output_dir"]) / "final_merged"),
        )


if __name__ == "__main__":
    main()
