from __future__ import annotations

import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "你是电商客服策略模型。根据历史对话给出 JSON 格式的 intent、slots、action、response、need_handoff。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path)

    history: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Input empty line to exit.")
    while True:
        user_text = input("user> ").strip()
        if not user_text:
            break
        history.append({"role": "user", "content": user_text})
        prompt = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"assistant> {generated}")
        history.append({"role": "assistant", "content": generated})
        try:
            parsed = json.loads(generated)
            if "response" in parsed:
                print(f"response> {parsed['response']}")
        except json.JSONDecodeError:
            pass


if __name__ == "__main__":
    main()
