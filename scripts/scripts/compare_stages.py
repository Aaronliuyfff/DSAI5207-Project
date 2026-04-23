from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "你是电商客服策略模型。根据历史对话给出 JSON 格式的 intent、slots、action、response、need_handoff。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--base-model-stage1", required=True)
    parser.add_argument("--adapter-stage1", required=True)
    parser.add_argument("--base-model-stage2", required=True)
    parser.add_argument("--adapter-stage2", required=True)
    parser.add_argument("--base-model-stage3", required=True)
    parser.add_argument("--adapter-stage3", required=True)
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_messages(history: list[str], user_text: str) -> list[dict[str, str]]:
    content = "\n".join(f"用户: {turn}" for turn in history)
    user_payload = {"history": content, "current_user": user_text, "task_type": "eval"}
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def generate_reply(
    tokenizer: Any,
    model: Any,
    history: list[str],
    user_text: str,
    max_new_tokens: int,
) -> str:
    messages = build_messages(history, user_text)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def run_stage(
    base_model: str,
    adapter_path: str,
    cases: list[dict[str, Any]],
    max_new_tokens: int,
    device: str,
) -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=None,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.to(device)
    results = []
    for case in cases:
        history = case.get("history", [])
        turns = case.get("turns", [])
        stage_outputs = []
        for user_text in turns:
            reply = generate_reply(tokenizer, model, history, user_text, max_new_tokens)
            stage_outputs.append({"user": user_text, "assistant": reply})
            history = history + [user_text, reply]
        results.append({"id": case.get("id"), "outputs": stage_outputs})
    return results


def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases)

    all_results = {
        "stage1": run_stage(
            args.base_model_stage1,
            args.adapter_stage1,
            cases,
            args.max_new_tokens,
            args.device,
        ),
        "stage2": run_stage(
            args.base_model_stage2,
            args.adapter_stage2,
            cases,
            args.max_new_tokens,
            args.device,
        ),
        "stage3": run_stage(
            args.base_model_stage3,
            args.adapter_stage3,
            cases,
            args.max_new_tokens,
            args.device,
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(all_results, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
