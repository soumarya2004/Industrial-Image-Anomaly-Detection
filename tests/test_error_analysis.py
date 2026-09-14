from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from src.data.dataset import Sample
from src.evaluation.error_analysis import (
    analyze_autoencoder_errors,
    analyze_classifier_errors,
    save_error_analysis_summary,
    summarize_defect_area_vs_failure,
)
from src.models.autoencoder import ConvAutoencoder
from src.models.classifier import DefectClassifier


def _save_random_image(path: Path, size=(64, 64), seed=None):
    rng=np.random.default_rng(seed)
    arr=(rng.random((*size, 3))*255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _save_mask(path: Path, coverage_fraction: float, size=(64, 64)):
    mask=np.zeros(size, dtype=np.uint8)
    n_pixels=int(coverage_fraction * size[0] * size[1])
    flat=mask.reshape(-1)
    flat[:n_pixels]=255
    Image.fromarray(flat.reshape(size)).save(path)


def _make_samples(tmp_path: Path)->list[Sample]:
    samples=[]
    for i in range(3):
        p= tmp_path / f"good_{i}.png"
        _save_random_image(p, seed=i)
        samples.append(Sample(image_path=p, label=0, defect_type="good", mask_path=None))
    for i in range(3):
        p= tmp_path / f"defect_{i}.png"
        _save_random_image(p, seed=100 + i)
        mask_p= tmp_path / f"defect_{i}_mask.png"
        _save_mask(mask_p, coverage_fraction=0.05 + i * 0.1)
        samples.append(
            Sample(image_path=p, label=1, defect_type="broken_large", mask_path=mask_p)
        )
    return samples


class TestAnalyzeClassifierErrors:
    def _make_cfg(self, image_size=64):
        return {"data": {"image_size": image_size}}

    def test_returns_expected_structure_and_saves_artifacts_for_errors(self, tmp_path):
        samples=_make_samples(tmp_path)
        model=DefectClassifier(architecture="resnet18", pretrained=False, finetune_mode="frozen")
        model.eval()
        device=torch.device("cpu")
        out_dir= tmp_path / "examples"
        results=analyze_classifier_errors(model, self._make_cfg(), device, samples, out_dir)
        assert set(results["counts"].keys())=={"false_positive", "false_negative", "low_margin_correct"}
        assert len(results["records"])==len(samples)
        for record in results["records"]:
            if record["error_type"] is not None:
                assert record["saved_visualization_path"] is not None
                assert Path(record["saved_visualization_path"]).exists()
            else:
                assert record["saved_visualization_path"] is None

    def test_mask_coverage_recorded_only_for_defective_samples(self, tmp_path):
        samples=_make_samples(tmp_path)
        model=DefectClassifier(architecture="resnet18", pretrained=False, finetune_mode="frozen")
        model.eval()
        device=torch.device("cpu")
        results=analyze_classifier_errors(model, self._make_cfg(), device, samples, tmp_path / "out")
        for record in results["records"]:
            if record["true_label"]==0:
                assert record["mask_coverage"] is None
            else:
                assert record["mask_coverage"] is not None
                assert 0.0<=record["mask_coverage"]<=1.0


class TestAnalyzeAutoencoderErrors:
    def _make_cfg(self, image_size=32):
        return {"data": {"image_size": image_size}}

    def test_returns_expected_structure_and_saves_artifacts_for_errors(self, tmp_path):
        samples=_make_samples(tmp_path)
        model=ConvAutoencoder(image_size=32, latent_dim=16, base_channels=8, num_downsample_blocks=3)
        model.eval()
        device=torch.device("cpu")
        out_dir=tmp_path / "examples"
        results = analyze_autoencoder_errors(
            model, self._make_cfg(), device, samples, threshold=0.0001, output_dir=out_dir
        )

        assert set(results["counts"].keys())=={"false_positive", "false_negative", "low_margin_correct"}
        assert len(results["records"])==len(samples)
        for record in results["records"]:
            if record["error_type"] is not None:
                assert Path(record["saved_visualization_path"]).exists()


class TestSummarizeDefectAreaVsFailure:
    def test_computes_means_when_data_present(self):
        records=[
            {"error_type": "false_negative", "true_label": 1, "mask_coverage": 0.05},
            {"error_type": None, "true_label": 1, "mask_coverage": 0.30},
            {"error_type": None, "true_label": 1, "mask_coverage": 0.40},
            {"error_type": None, "true_label": 0, "mask_coverage": None},
        ]
        summary=summarize_defect_area_vs_failure(records)
        assert summary["n_false_negatives_with_mask"] == 1
        assert summary["n_true_positives_with_mask"] == 2
        assert summary["mean_defect_area_fraction_false_negatives"] == 0.05
        assert abs(summary["mean_defect_area_fraction_true_positives"] - 0.35) < 1e-9
        assert summary["difference"] > 0  # true positives had larger average defect area

    def test_returns_nan_when_no_false_negatives(self):
        records=[{"error_type": None, "true_label": 1, "mask_coverage": 0.2}]
        summary=summarize_defect_area_vs_failure(records)
        assert summary["n_false_negatives_with_mask"] == 0
        assert np.isnan(summary["mean_defect_area_fraction_false_negatives"])
        assert np.isnan(summary["difference"])


class TestSaveErrorAnalysisSummary:
    def test_writes_valid_json(self, tmp_path):
        summary={"a": 1, "b": {"c": [1, 2, 3]}}
        out_path= tmp_path / "nested" / "summary.json"
        save_error_analysis_summary(summary, out_path)

        assert out_path.exists()
        import json
        with open(out_path) as f:
            loaded= json.load(f)
        assert loaded== summary
