from __future__ import annotations
import argparse
from src.data.dataset import build_samples
from src.evaluation.error_analysis import (
    analyze_autoencoder_errors,
    analyze_classifier_errors,
    save_error_analysis_summary,
    summarize_defect_area_vs_failure,
)
from src.inference import load_autoencoder, load_classifier
from src.utils import get_device, load_config


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--classifier_config", type=str, default="configs/classifier.yaml")
    parser.add_argument("--autoencoder_config", type=str, default="configs/autoencoder.yaml")
    parser.add_argument("--classifier_checkpoint", type=str, default="models/classifier/best_model.pt")
    parser.add_argument("--autoencoder_checkpoint", type=str, default="models/autoencoder/best_model.pt")
    parser.add_argument(
        "--autoencoder_threshold", type=float, required=True,
        help="The threshold selected during autoencoder training (see its logged experiment JSON). "
             "Must be the SAME value already used for the final reported test metrics.",
    )
    parser.add_argument("--output_dir", type=str, default="results/examples")
    args=parser.parse_args()
    device=get_device(prefer_cuda=True)
    clf_cfg=load_config(args.classifier_config)
    print("=== Loading models ===")
    classifier, clf_ckpt_cfg = load_classifier(args.classifier_checkpoint, device)
    autoencoder, ae_ckpt_cfg = load_autoencoder(args.autoencoder_checkpoint, device)

    #use the exact same labeled_test samples already used for final reporting
    #in train_classifier.py / train_autoencoder.py (same seed, same val_fraction).
    test_samples=build_samples(
        clf_cfg["data"]["root"], clf_cfg["data"]["category"], "labeled_test",
        seed=clf_cfg["experiment"]["seed"], val_fraction=clf_cfg["data"]["val_fraction"],
    )
    print(f"Analyzing {len(test_samples)} held-out test images (touched once already for final metrics).\n")

    print("=== Classifier error analysis ===")
    clf_results=analyze_classifier_errors(
        classifier, clf_ckpt_cfg, device, test_samples, output_dir=f"{args.output_dir}/classifier"
    )
    print("Counts:", clf_results["counts"])
    clf_area_corr=summarize_defect_area_vs_failure(clf_results["records"])
    print("Defect-area vs. failure correlation:", clf_area_corr)

    print("\n=== Autoencoder error analysis ===")
    ae_results=analyze_autoencoder_errors(
        autoencoder, ae_ckpt_cfg, device, test_samples, args.autoencoder_threshold,
        output_dir=f"{args.output_dir}/autoencoder",
    )
    print("Counts:", ae_results["counts"])
    ae_area_corr=summarize_defect_area_vs_failure(ae_results["records"])
    print("Defect-area vs. failure correlation:", ae_area_corr)

    #breakdown by defect subtype -- which specific defect categories does each
    #model struggle with most, in absolute counts of false negatives.
    def _fn_by_defect_type(records: list[dict]) -> dict:
        counts: dict[str, int] = {}
        for r in records:
            if r["error_type"]=="false_negative":
                counts[r["defect_type"]]=counts.get(r["defect_type"], 0) + 1
        return counts

    clf_fn_by_type=_fn_by_defect_type(clf_results["records"])
    ae_fn_by_type=_fn_by_defect_type(ae_results["records"])
    print(f"\nClassifier false negatives by defect subtype: {clf_fn_by_type}")
    print(f"Autoencoder false negatives by defect subtype: {ae_fn_by_type}")

    summary={
        "classifier": {
            "counts": clf_results["counts"],
            "defect_area_correlation": clf_area_corr,
            "false_negatives_by_defect_type": clf_fn_by_type,
            "records": clf_results["records"],
        },
        "autoencoder": {
            "counts": ae_results["counts"],
            "defect_area_correlation": ae_area_corr,
            "false_negatives_by_defect_type": ae_fn_by_type,
            "records": ae_results["records"],
        },
    }
    out_path=f"{args.output_dir}/error_analysis_summary.json"
    save_error_analysis_summary(summary, out_path)
    print(f"\nFull summary saved to: {out_path}")
    print(f"Visual artifacts saved under: {args.output_dir}/classifier/ and {args.output_dir}/autoencoder/")


if __name__=="__main__":
    main()
