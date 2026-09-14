from __future__ import annotations
import argparse
import time
import numpy as np
from src.data.dataset import build_samples
from src.evaluation.experiment_log import log_experiment
from src.evaluation.metrics import (
    anomaly_detection_report_dict,
    classification_report_dict,
    select_threshold_by_f1,
)
from src.models.baseline import ClassicalFeatureClassifier, MeanImageBaseline
from src.utils import load_config, set_seed


def run_mean_image_baseline(cfg: dict) -> dict:
    data_cfg=cfg["data"]
    seed=cfg["experiment"]["seed"]
    image_size=128  # matches the autoencoder's default working resolution

    ae_train_samples=build_samples(data_cfg["root"], data_cfg["category"], "ae_train", seed=seed)
    labeled_val_samples=build_samples(
        data_cfg["root"], data_cfg["category"], "labeled_val", seed=seed, val_fraction=0.3
    )
    labeled_test_samples=build_samples(
        data_cfg["root"], data_cfg["category"], "labeled_test", seed=seed, val_fraction=0.3
    )

    start=time.time()
    model=MeanImageBaseline(image_size=image_size)
    model.fit([s.image_path for s in ae_train_samples])

    val_scores=model.score_batch([s.image_path for s in labeled_val_samples])
    val_labels=np.array([s.label for s in labeled_val_samples])
    threshold=select_threshold_by_f1(val_labels, val_scores)

    test_scores=model.score_batch([s.image_path for s in labeled_test_samples])
    test_labels=np.array([s.label for s in labeled_test_samples])
    training_time_sec=time.time()-start

    val_report=anomaly_detection_report_dict(val_labels, val_scores, threshold)
    test_report=anomaly_detection_report_dict(test_labels, test_scores, threshold)

    return{
        "baseline_name": "mean_image_distance",
        "compares_to": "autoencoder",
        "training_time_sec": training_time_sec,
        "selected_threshold": threshold,
        "validation": val_report,
        "test": test_report,
    }


def run_classical_feature_baseline(cfg: dict)->dict:
    data_cfg=cfg["data"]
    seed=cfg["experiment"]["seed"]
    image_size=128

    labeled_val_samples=build_samples(
        data_cfg["root"], data_cfg["category"], "labeled_val", seed=seed, val_fraction=data_cfg["val_fraction"]
    )
    labeled_test_samples=build_samples(
        data_cfg["root"], data_cfg["category"], "labeled_test", seed=seed, val_fraction=data_cfg["val_fraction"]
    )

    n=len(labeled_val_samples)
    rng=np.random.default_rng(seed+1)  # identical derived seed to train_classifier.py
    indices=rng.permutation(n)
    split_point=int(n*0.7)
    train_idx, val_idx = indices[:split_point], indices[split_point:]

    train_samples=[labeled_val_samples[i] for i in train_idx]
    val_samples=[labeled_val_samples[i] for i in val_idx]

    start=time.time()
    model=ClassicalFeatureClassifier(image_size=image_size)
    model.fit(
        [s.image_path for s in train_samples],
        [s.label for s in train_samples],
    )
    training_time_sec=time.time()-start

    val_labels=np.array([s.label for s in val_samples])
    val_scores=model.predict_proba([s.image_path for s in val_samples])
    val_preds=(val_scores>=0.5).astype(int)
    val_report=classification_report_dict(val_labels, val_preds, val_scores)

    test_labels=np.array([s.label for s in labeled_test_samples])
    test_scores=model.predict_proba([s.image_path for s in labeled_test_samples])
    test_preds=(test_scores>=0.5).astype(int)
    test_report=classification_report_dict(test_labels, test_preds, test_scores)

    return{
        "baseline_name": "hog_color_hist_logreg",
        "compares_to": "classifier",
        "training_time_sec": training_time_sec,
        "validation": val_report,
        "test": test_report,
    }


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/classifier.yaml")
    args=parser.parse_args()

    cfg=load_config(args.config)
    set_seed(cfg["experiment"]["seed"])

    print("=== Baseline 1: Mean-image distance (control for autoencoder) ===")
    mean_image_results=run_mean_image_baseline(cfg)
    print(mean_image_results["test"])
    log_experiment("baseline_mean_image_distance", cfg, mean_image_results)

    print("\n=== Baseline 2: HOG + color histogram + logistic regression (control for classifier) ===")
    classical_results=run_classical_feature_baseline(cfg)
    print(classical_results["test"])
    log_experiment("baseline_hog_color_logreg", cfg, classical_results)

    print("\nBoth baselines logged to results/experiments/. "
          "Add their test metrics to the README results table alongside the deep models.")


if __name__=="__main__":
    main()
