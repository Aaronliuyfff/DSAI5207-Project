from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _as_list(value: Any) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _zip_messages(fields: dict[str, Any]) -> list[dict[str, Any]]:
    contents = _as_list(fields.get("content") or fields.get("utterance") or fields.get("text"))
    roles = _as_list(fields.get("role") or fields.get("speaker") or ["user"] * len(contents))
    dialog_acts = _as_list(fields.get("dialog_act") or fields.get("dialogue_act") or [])
    messages: list[dict[str, Any]] = []
    for idx, content in enumerate(contents):
        if content is None:
            continue
        role = roles[idx] if idx < len(roles) else "user"
        message: dict[str, Any] = {"content": content, "role": role}
        if idx < len(dialog_acts):
            message["dialog_act"] = dialog_acts[idx]
        messages.append(message)
    return messages


def normalize_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    if "messages" in record:
        messages = record["messages"]
        if isinstance(messages, list):
            return [
                {
                    "content": msg.get("content") or msg.get("text") or msg.get("utterance") or "",
                    "role": msg.get("role") or msg.get("speaker") or "user",
                    **({"dialog_act": msg["dialog_act"]} if "dialog_act" in msg else {}),
                }
                for msg in messages
                if isinstance(msg, dict)
            ]
        if isinstance(messages, dict):
            return _zip_messages(messages)
    if "dialogue" in record:
        return [
            {
                "content": msg.get("content") or msg.get("text") or msg.get("utterance") or "",
                "role": msg.get("role") or msg.get("speaker") or "user",
                **({"dialog_act": msg["dialog_act"]} if "dialog_act" in msg else {}),
            }
            for msg in record["dialogue"]
            if isinstance(msg, dict)
        ]
    if "log" in record:
        return [
            {
                "content": msg.get("text") or msg.get("content") or "",
                "role": msg.get("role") or msg.get("speaker") or "user",
                **({"dialog_act": msg["dialog_act"]} if "dialog_act" in msg else {}),
            }
            for msg in record["log"]
            if isinstance(msg, dict)
        ]
    return []


def download_file(url: str, dest: Path, overwrite: bool) -> Path:
    if dest.exists() and not overwrite:
        return dest
    _ensure_dir(dest.parent)
    urllib.request.urlretrieve(url, dest)
    return dest


def dump_jsonl(path: Path, rows: Iterable[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Use --overwrite to replace.")
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_crosswoz_messages(zip_path: Path) -> Iterable[list[dict[str, Any]]]:
    with zipfile.ZipFile(zip_path) as zf:
        json_name = next(name for name in zf.namelist() if name.endswith(".json"))
        with zf.open(json_name) as fh:
            payload = json.load(fh)
    for dialog in payload.values():
        messages = dialog.get("messages", [])
        if messages:
            yield messages


def export_crosswoz(output_root: Path, overwrite: bool) -> None:
    base = "https://github.com/thu-coai/CrossWOZ/raw/master/data/crosswoz"
    zip_paths = []
    for split in ["train", "val", "test"]:
        zip_paths.append(
            download_file(f"{base}/{split}.json.zip", output_root / "_downloads" / f"crosswoz_{split}.zip", overwrite)
        )
    rows = []
    for zip_path in zip_paths:
        for messages in iter_crosswoz_messages(zip_path):
            if len(messages) < 2:
                continue
            rows.append({"messages": messages})
    dump_jsonl(output_root / "crosswoz" / "crosswoz.jsonl", rows, overwrite)


def iter_risawoz_messages(zip_path: Path) -> Iterable[list[dict[str, Any]]]:
    with zipfile.ZipFile(zip_path) as zf:
        json_names = [
            name
            for name in zf.namelist()
            if name.endswith(("/train.json", "/val.json", "/test.json"))
        ]
        for json_name in json_names:
            with zf.open(json_name) as fh:
                payload = json.load(fh)
            for dialog in payload:
                if not isinstance(dialog, dict):
                    continue
                turns = dialog.get("dialogue", [])
                messages: list[dict[str, Any]] = []
                for turn in turns:
                    user_text = turn.get("user_utterance") or ""
                    sys_text = turn.get("system_utterance") or ""
                    user_actions = turn.get("user_actions")
                    sys_actions = turn.get("system_actions")
                    if user_text:
                        msg = {"content": user_text, "role": "user"}
                        if user_actions is not None:
                            msg["dialog_act"] = user_actions
                        messages.append(msg)
                    if sys_text:
                        msg = {"content": sys_text, "role": "sys"}
                        if sys_actions is not None:
                            msg["dialog_act"] = sys_actions
                        messages.append(msg)
                if messages:
                    yield messages


def export_risawoz(output_root: Path, overwrite: bool) -> None:
    url = (
        "https://github.com/terryqj0107/RiSAWOZ/raw/master/"
        "RiSAWOZ-data/task3-data-E2E-Context-to-text.zip"
    )
    zip_path = download_file(url, output_root / "_downloads" / "risawoz_task3.zip", overwrite)
    rows = []
    for messages in iter_risawoz_messages(zip_path):
        if len(messages) < 2:
            continue
        rows.append({"messages": messages})
    dump_jsonl(output_root / "risawoz" / "risawoz.jsonl", rows, overwrite)


def main() -> None:
    args = parse_args()
    export_crosswoz(args.output_root, args.overwrite)
    export_risawoz(args.output_root, args.overwrite)


if __name__ == "__main__":
    main()
