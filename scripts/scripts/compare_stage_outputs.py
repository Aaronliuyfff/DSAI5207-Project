from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "你是电商客服策略模型。"
    "请根据历史对话、当前用户输入输出 JSON。"
    "JSON 必须包含 intent、slots、action、response、need_handoff。"
)
REQUIRED_KEYS = {"intent", "slots", "action", "response", "need_handoff"}
DEFAULT_STAGE_CONFIGS = {
    "stage1": {
        "model_path": "outputs/stage1_domain_mps/final_merged",
        "adapter_path": None,
    },
    "stage2": {
        "model_path": "outputs/stage2_policy_mps/final_merged",
        "adapter_path": None,
    },
    "stage3": {
        "model_path": "outputs/stage2_policy_mps/final_merged",
        "adapter_path": "outputs/stage3_align_mps/final_adapter",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare stage1/stage2/stage3 outputs on the same fixed test suite."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("data/eval/model_test_cases.json"),
        help="JSON file containing evaluation cases.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory used to save comparison outputs.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage1-model", default=DEFAULT_STAGE_CONFIGS["stage1"]["model_path"])
    parser.add_argument("--stage1-adapter", default=DEFAULT_STAGE_CONFIGS["stage1"]["adapter_path"])
    parser.add_argument("--stage2-model", default=DEFAULT_STAGE_CONFIGS["stage2"]["model_path"])
    parser.add_argument("--stage2-adapter", default=DEFAULT_STAGE_CONFIGS["stage2"]["adapter_path"])
    parser.add_argument("--stage3-model", default=DEFAULT_STAGE_CONFIGS["stage3"]["model_path"])
    parser.add_argument("--stage3-adapter", default=DEFAULT_STAGE_CONFIGS["stage3"]["adapter_path"])
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_messages(history: list[str], current_user: str) -> list[dict[str, str]]:
    lines = []
    for index, turn in enumerate(history):
        speaker = "用户" if index % 2 == 0 else "客服"
        lines.append(f"{speaker}: {turn}")
    user_payload = {
        "history": "\n".join(lines),
        "current_user": current_user,
        "task_type": "eval",
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def resolve_dtype(device: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    return torch.float16


def resolve_tokenizer_source(model_path: str, adapter_path: str | None) -> str:
    candidates = []
    if adapter_path:
        candidates.append(Path(adapter_path))
    model_dir = Path(model_path)
    candidates.append(model_dir)
    sibling_adapter_dir = model_dir.parent / "final_adapter"
    if sibling_adapter_dir != model_dir:
        candidates.append(sibling_adapter_dir)

    for candidate in candidates:
        if (candidate / "tokenizer.json").exists() or (candidate / "tokenizer_config.json").exists():
            return str(candidate)
    return adapter_path or model_path


def has_model_weights(model_path: str) -> bool:
    model_dir = Path(model_path)
    weight_names = [
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ]
    return any((model_dir / name).exists() for name in weight_names)


def resolve_model_source(model_path: str, adapter_path: str | None) -> tuple[str, str | None]:
    if has_model_weights(model_path):
        return model_path, adapter_path
    if adapter_path:
        adapter_config_path = Path(adapter_path) / "adapter_config.json"
    else:
        adapter_config_path = Path(model_path).parent / "final_adapter" / "adapter_config.json"
        sibling_adapter = str(Path(model_path).parent / "final_adapter")
        if adapter_config_path.exists():
            with adapter_config_path.open("r", encoding="utf-8") as fh:
                config = json.load(fh)
            base_model_path = config.get("base_model_name_or_path")
            if base_model_path:
                return base_model_path, sibling_adapter
    return model_path, adapter_path


def prepare_tokenizer_dir(tokenizer_source: str) -> str:
    source = Path(tokenizer_source)
    config_path = source / "tokenizer_config.json"
    if not config_path.exists():
        return tokenizer_source

    with config_path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)

    extra_special_tokens = config.get("extra_special_tokens")
    if not isinstance(extra_special_tokens, list):
        return tokenizer_source

    temp_dir = Path(tempfile.mkdtemp(prefix="tokenizer_fix_"))
    for item in source.iterdir():
        if item.is_file():
            shutil.copy2(item, temp_dir / item.name)

    config.pop("extra_special_tokens", None)
    with (temp_dir / "tokenizer_config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2)
    return str(temp_dir)


def load_model_and_tokenizer(model_path: str, adapter_path: str | None, device: str) -> tuple[Any, Any]:
    model_path, adapter_path = resolve_model_source(model_path, adapter_path)
    tokenizer_source = resolve_tokenizer_source(model_path, adapter_path)
    tokenizer_source = prepare_tokenizer_dir(tokenizer_source)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=resolve_dtype(device),
        trust_remote_code=True,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.to(device)
    model.eval()
    return tokenizer, model


def generate_text(
    tokenizer: Any,
    model: Any,
    history: list[str],
    current_user: str,
    max_new_tokens: int,
) -> str:
    messages = build_messages(history, current_user)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def extract_json_candidate(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                snippet = text[start:index + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    return None
    return None


def evaluate_case(case: dict[str, Any], generated_text: str) -> dict[str, Any]:
    parsed = extract_json_candidate(generated_text)
    expected = case.get("expected", {})
    result: dict[str, Any] = {
        "id": case["id"],
        "generated_text": generated_text,
        "parsed_output": parsed,
        "json_valid": parsed is not None,
        "schema_complete": False,
        "checks": {},
    }
    if parsed is None:
        return result

    result["schema_complete"] = REQUIRED_KEYS.issubset(parsed.keys())
    checks: dict[str, Any] = {}
    if "intent" in expected:
        checks["intent_match"] = parsed.get("intent") == expected["intent"]
    if "action" in expected:
        checks["action_match"] = parsed.get("action") == expected["action"]
    if "need_handoff" in expected:
        checks["handoff_match"] = parsed.get("need_handoff") == expected["need_handoff"]

    expected_slot_keys = expected.get("slot_keys", [])
    if expected_slot_keys:
        slots = parsed.get("slots", {})
        if not isinstance(slots, dict):
            checks["slot_keys_match"] = False
        else:
            checks["slot_keys_match"] = all(key in slots for key in expected_slot_keys)
    else:
        checks["slot_keys_match"] = True

    result["checks"] = checks
    return result


def build_stage_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    json_valid = sum(1 for row in results if row["json_valid"])
    schema_complete = sum(1 for row in results if row["schema_complete"])

    def score(check_name: str) -> dict[str, Any]:
        available = [row for row in results if check_name in row["checks"]]
        passed = sum(1 for row in available if row["checks"][check_name])
        rate = round(passed / len(available), 4) if available else 0.0
        return {"passed": passed, "total": len(available), "rate": rate}

    return {
        "total_cases": total,
        "json_valid": {
            "passed": json_valid,
            "total": total,
            "rate": round(json_valid / total, 4) if total else 0.0,
        },
        "schema_complete": {
            "passed": schema_complete,
            "total": total,
            "rate": round(schema_complete / total, 4) if total else 0.0,
        },
        "intent_match": score("intent_match"),
        "action_match": score("action_match"),
        "handoff_match": score("handoff_match"),
        "slot_keys_match": score("slot_keys_match"),
    }


def build_case_comparison(
    cases: list[dict[str, Any]],
    stage_results: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    comparisons = []
    for case_index, case in enumerate(cases):
        item = {
            "id": case["id"],
            "history": case.get("history", []),
            "current_user": case.get("current_user", ""),
            "expected": case.get("expected", {}),
            "stages": {},
        }
        for stage_name, results in stage_results.items():
            item["stages"][stage_name] = results[case_index]
        comparisons.append(item)
    return comparisons


def print_stage_summary(stage_name: str, summary: dict[str, Any]) -> None:
    print(f"=== {stage_name} ===")
    print(
        f"json_valid      : {summary['json_valid']['passed']}/{summary['json_valid']['total']}"
        f" ({summary['json_valid']['rate']:.2%})"
    )
    print(
        f"schema_complete : {summary['schema_complete']['passed']}/{summary['schema_complete']['total']}"
        f" ({summary['schema_complete']['rate']:.2%})"
    )
    for key in ["intent_match", "action_match", "handoff_match", "slot_keys_match"]:
        entry = summary[key]
        print(f"{key:16}: {entry['passed']}/{entry['total']} ({entry['rate']:.2%})")


def default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/stage_comparisons") / timestamp


def build_stage_configs(args: argparse.Namespace) -> dict[str, dict[str, str | None]]:
    return {
        "stage1": {"model_path": args.stage1_model, "adapter_path": args.stage1_adapter},
        "stage2": {"model_path": args.stage2_model, "adapter_path": args.stage2_adapter},
        "stage3": {"model_path": args.stage3_model, "adapter_path": args.stage3_adapter},
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    cases = load_cases(args.cases)
    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_configs = build_stage_configs(args)
    metadata = {
        "cases_path": str(args.cases),
        "device": args.device,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stages": stage_configs,
    }

    stage_results: dict[str, list[dict[str, Any]]] = {}
    stage_summaries: dict[str, dict[str, Any]] = {}

    for stage_name, config in stage_configs.items():
        print(f"Loading {stage_name} from {config['model_path']}")
        tokenizer, model = load_model_and_tokenizer(
            model_path=str(config["model_path"]),
            adapter_path=config["adapter_path"],
            device=args.device,
        )

        results = []
        for index, case in enumerate(cases, start=1):
            generated_text = generate_text(
                tokenizer=tokenizer,
                model=model,
                history=case.get("history", []),
                current_user=case.get("current_user", ""),
                max_new_tokens=args.max_new_tokens,
            )
            result = evaluate_case(case, generated_text)
            results.append(result)
            print(f"[{stage_name}][{index}/{len(cases)}] {case['id']} done")

        stage_results[stage_name] = results
        stage_summaries[stage_name] = build_stage_summary(results)
        print_stage_summary(stage_name, stage_summaries[stage_name])

        del model
        if args.device in {"mps", "cuda"}:
            if args.device == "cuda":
                torch.cuda.empty_cache()
            else:
                if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()

    case_comparisons = build_case_comparison(cases, stage_results)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump({"metadata": metadata, "stage_summaries": stage_summaries}, fh, ensure_ascii=False, indent=2)
    with (output_dir / "results.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "metadata": metadata,
                "stage_results": stage_results,
                "case_comparisons": case_comparisons,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved summary to: {output_dir / 'summary.json'}")
    print(f"Saved detailed results to: {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
