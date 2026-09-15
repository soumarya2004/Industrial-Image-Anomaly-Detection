from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from src.evaluation.experiment_log import load_all_experiments

#maps a substring of experiment_name -> a human-readable group label.
GROUPS={
    "ablation_classifier_finetune_": "Classifier: fine-tune mode",
    "ablation_classifier_aug_": "Classifier: augmentation on/off",
    "ablation_autoencoder_latent": "Autoencoder: latent dimension",
    "ablation_autoencoder_loss_": "Autoencoder: reconstruction loss",
}


def _extract_variant(experiment_name: str, prefix: str)->str:
    return experiment_name.replace(prefix, "")


def build_ablation_table(experiments_dir: str|Path="results/experiments")->pd.DataFrame:
    records=load_all_experiments(experiments_dir)
    rows=[]
    for r in records:
        name=r["experiment_name"]
        if not name.startswith("ablation_"):
            continue
        group_label, prefix = None, None
        for p, label in GROUPS.items():
            if name.startswith(p):
                group_label, prefix = label, p
                break
        if group_label is None:
            continue  # unrecognized ablation naming — skip rather than guess

        test=r["metrics"].get("test", {})
        rows.append(
            {
                "group": group_label,
                "variant": _extract_variant(name, prefix),
                "experiment_name": name,
                "timestamp": r["timestamp_utc"],
                "f1": test.get("f1"),
                "roc_auc": test.get("roc_auc"),
                "pr_auc": test.get("pr_auc"),
                "training_time_sec": r["metrics"].get("training_time_sec"),
            }
        )
    df=pd.DataFrame(rows)
    if df.empty:
        return df
    #keep only the latest run per experiment_name.
    df=df.sort_values("timestamp").groupby("experiment_name", as_index=False).last()
    return df


def plot_ablation_group(df: pd.DataFrame, group_label: str, output_dir: str|Path)->Path|None:
    subset=df[df["group"]==group_label]
    if subset.empty:
        return None

    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax=plt.subplots(figsize=(7, 4.5))
    x=range(len(subset))
    width=0.35
    ax.bar([i-width/2 for i in x], subset["f1"], width, label="Test F1")
    ax.bar([i+width/2 for i in x], subset["roc_auc"], width, label="Test ROC-AUC")
    ax.set_xticks(list(x))
    ax.set_xticklabels(subset["variant"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title(group_label)
    ax.legend()
    plt.tight_layout()

    safe_name=group_label.lower().replace(" ", "_").replace(":", "").replace("/", "_")
    out_path=output_dir/f"ablation_{safe_name}.png"
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main()->None:
    df=build_ablation_table()
    if df.empty:
        print(
            "No ablation experiments found in results/experiments/. "
            "Run ./scripts/run_ablations.sh first."
        )
        return

    print(df.to_string(index=False))
    print()

    for group_label in df["group"].unique():
        out_path=plot_ablation_group(df, group_label, output_dir="results/plots")
        print(f"Saved plot for '{group_label}' -> {out_path}")

    csv_path=Path("results/experiments/ablation_summary_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nFull table saved to {csv_path} — paste into README section 12.")


if __name__=="__main__":
    main()
