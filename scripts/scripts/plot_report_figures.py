from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "outputs/stage_comparisons/20260423_110022/summary.json"
RESULTS_PATH = ROOT / "outputs/stage_comparisons/20260423_110022/results.json"
STAGE2_STATE = ROOT / "outputs/stage2_policy_mps/checkpoint-5000/trainer_state.json"
STAGE3_STATE = ROOT / "outputs/stage3_align_mps/checkpoint-5000/trainer_state.json"
STAGE3_LOW_STATE = ROOT / "outputs/stage3_align_mps_low_resource/checkpoint-1000/trainer_state.json"
STAGE1_DATA = ROOT / "data/processed/stage1_domain.jsonl"
STAGE2_DATA = ROOT / "data/processed/stage2_policy.jsonl"
STAGE3_DATA = ROOT / "data/processed/stage3_align.jsonl"
OUTPUT_DIR = ROOT / "outputs/figures/report"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_stage_metrics(summary_data: dict) -> None:
    rows = []
    for stage_name, stage_summary in summary_data["stage_summaries"].items():
        for metric in [
            "json_valid",
            "schema_complete",
            "intent_match",
            "action_match",
            "handoff_match",
            "slot_keys_match",
        ]:
            rows.append(
                {
                    "stage": stage_name,
                    "metric": metric,
                    "rate": stage_summary[metric]["rate"] * 100,
                }
            )
    df = pd.DataFrame(rows)
    order = ["json_valid", "schema_complete", "intent_match", "action_match", "handoff_match", "slot_keys_match"]
    plt.figure(figsize=(10.5, 5.6))
    sns.barplot(data=df, x="metric", y="rate", hue="stage", order=order, palette="Set2")
    plt.title("Three-Stage Evaluation Metrics")
    plt.ylabel("Rate (%)")
    plt.xlabel("")
    plt.ylim(0, 110)
    plt.xticks(rotation=20, ha="right")
    savefig(OUTPUT_DIR / "01_stage_metrics.png")


def plot_case_heatmap(results_data: dict) -> None:
    rows = []
    for case in results_data["case_comparisons"]:
        case_id = case["id"]
        for stage_name, stage_result in case["stages"].items():
            checks = stage_result.get("checks", {})
            valid = int(stage_result.get("json_valid", False))
            score = valid + sum(int(value) for value in checks.values())
            rows.append({"case_id": case_id, "stage": stage_name, "score": score})
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="case_id", columns="stage", values="score").fillna(0)
    plt.figure(figsize=(6.6, 4.8))
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", cbar_kws={"label": "Case Score"})
    plt.title("Per-Case Stage Comparison")
    plt.xlabel("")
    plt.ylabel("")
    savefig(OUTPUT_DIR / "02_case_heatmap.png")


def trainer_history(path: Path, label: str) -> pd.DataFrame:
    data = load_json(path)
    history = pd.DataFrame(data.get("log_history", []))
    history["run"] = label
    return history


def plot_training_loss() -> None:
    df = pd.concat(
        [
            trainer_history(STAGE2_STATE, "stage2_policy"),
            trainer_history(STAGE3_STATE, "stage3_align"),
            trainer_history(STAGE3_LOW_STATE, "stage3_low_resource"),
        ],
        ignore_index=True,
    )
    plt.figure(figsize=(8.8, 5.2))
    sns.lineplot(data=df, x="step", y="loss", hue="run", linewidth=2.0)
    plt.title("Training Loss Curves")
    plt.ylabel("Loss")
    plt.xlabel("Step")
    savefig(OUTPUT_DIR / "03_training_loss.png")


def plot_training_accuracy() -> None:
    df = pd.concat(
        [
            trainer_history(STAGE2_STATE, "stage2_policy"),
            trainer_history(STAGE3_STATE, "stage3_align"),
            trainer_history(STAGE3_LOW_STATE, "stage3_low_resource"),
        ],
        ignore_index=True,
    )
    plt.figure(figsize=(8.8, 5.2))
    sns.lineplot(data=df, x="step", y="mean_token_accuracy", hue="run", linewidth=2.0)
    plt.title("Training Token Accuracy Curves")
    plt.ylabel("Mean Token Accuracy")
    plt.xlabel("Step")
    plt.ylim(0.0, 1.0)
    savefig(OUTPUT_DIR / "04_training_accuracy.png")


def plot_training_stability() -> None:
    df = pd.concat(
        [
            trainer_history(STAGE2_STATE, "stage2_policy"),
            trainer_history(STAGE3_STATE, "stage3_align"),
        ],
        ignore_index=True,
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    sns.lineplot(data=df, x="step", y="grad_norm", hue="run", linewidth=1.8, ax=axes[0])
    axes[0].set_title("Gradient Norm")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Grad Norm")
    sns.lineplot(data=df, x="step", y="learning_rate", hue="run", linewidth=1.8, ax=axes[1])
    axes[1].set_title("Learning Rate")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Learning Rate")
    handles, labels = axes[1].get_legend_handles_labels()
    if axes[0].legend_:
        axes[0].legend_.remove()
    if axes[1].legend_:
        axes[1].legend_.remove()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(OUTPUT_DIR / "05_training_stability.png", dpi=220, bbox_inches="tight")
    plt.close()


def plot_data_composition_and_actions() -> None:
    stage1_rows = load_jsonl(STAGE1_DATA)
    stage2_rows = load_jsonl(STAGE2_DATA)
    stage3_rows = load_jsonl(STAGE3_DATA)

    dataset_counts = (
        pd.DataFrame(stage3_rows)["dataset"]
        .value_counts()
        .rename_axis("dataset")
        .reset_index(name="count")
    )
    action_counts = (
        pd.DataFrame([row["target"] for row in stage3_rows])["action"]
        .value_counts()
        .head(10)
        .rename_axis("action")
        .reset_index(name="count")
    )
    stage_sizes = pd.DataFrame(
        [
            {"stage": "stage1_domain", "count": len(stage1_rows)},
            {"stage": "stage2_policy", "count": len(stage2_rows)},
            {"stage": "stage3_align", "count": len(stage3_rows)},
        ]
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    sns.barplot(data=stage_sizes, x="stage", y="count", palette="Blues_d", ax=axes[0])
    axes[0].set_title("Processed Samples per Stage")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(data=dataset_counts, x="dataset", y="count", palette="Greens_d", ax=axes[1])
    axes[1].set_title("Stage 3 Dataset Composition")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=20)

    sns.barplot(data=action_counts, x="action", y="count", palette="Oranges_d", ax=axes[2])
    axes[2].set_title("Top Action Labels in Stage 3")
    axes[2].set_xlabel("")
    axes[2].tick_params(axis="x", rotation=55)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_data_distribution.png", dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    summary_data = load_json(SUMMARY_PATH)
    results_data = load_json(RESULTS_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_stage_metrics(summary_data)
    plot_case_heatmap(results_data)
    plot_training_loss()
    plot_training_accuracy()
    plot_training_stability()
    plot_data_composition_and_actions()

    print(f"Saved figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
