from __future__ import annotations
from pathlib import Path
from src.evaluation.experiment_log import log_experiment
from src.evaluation.summarize_ablations import build_ablation_table, plot_ablation_group


def _log_fake_ablation(
    tmp_path: Path, experiment_name: str, f1: float, roc_auc: float, pr_auc: float = 0.9
):
    log_experiment(
        experiment_name=experiment_name,
        config={"experiment": {"name": experiment_name}},
        metrics={
            "test": {"f1": f1, "roc_auc": roc_auc, "pr_auc": pr_auc},
            "training_time_sec": 12.3,
        },
        output_dir=tmp_path,
    )


class TestBuildAblationTable:
    def test_empty_directory_returns_empty_dataframe(self, tmp_path):
        df=build_ablation_table(tmp_path)
        assert df.empty

    def test_non_ablation_experiments_are_excluded(self, tmp_path):
        _log_fake_ablation(tmp_path, "classifier_resnet18_bottle_v1", f1=0.9, roc_auc=0.95)
        df=build_ablation_table(tmp_path)
        assert df.empty

    def test_recognized_ablation_groups_are_parsed(self, tmp_path):
        _log_fake_ablation(tmp_path, "ablation_classifier_finetune_frozen", f1=0.80, roc_auc=0.85)
        _log_fake_ablation(tmp_path, "ablation_classifier_finetune_partial", f1=0.90, roc_auc=0.93)
        _log_fake_ablation(tmp_path, "ablation_classifier_finetune_full", f1=0.75, roc_auc=0.80)

        df=build_ablation_table(tmp_path)
        assert len(df)==3
        assert set(df["group"])=={"Classifier: fine-tune mode"}
        assert set(df["variant"])=={"frozen", "partial", "full"}

    def test_multiple_groups_are_kept_separate(self, tmp_path):
        _log_fake_ablation(tmp_path, "ablation_classifier_finetune_frozen", f1=0.8, roc_auc=0.85)
        _log_fake_ablation(tmp_path, "ablation_autoencoder_latent64", f1=0.7, roc_auc=0.75)
        _log_fake_ablation(tmp_path, "ablation_autoencoder_latent256", f1=0.85, roc_auc=0.88)

        df=build_ablation_table(tmp_path)
        assert set(df["group"])=={"Classifier: fine-tune mode", "Autoencoder: latent dimension"}

    def test_unrecognized_ablation_name_is_skipped_not_guessed(self, tmp_path):
        _log_fake_ablation(tmp_path, "ablation_something_unexpected", f1=0.5, roc_auc=0.5)
        df=build_ablation_table(tmp_path)
        assert df.empty

    def test_only_latest_run_kept_when_rerun(self, tmp_path):
        _log_fake_ablation(tmp_path, "ablation_classifier_finetune_frozen", f1=0.70, roc_auc=0.70)
        _log_fake_ablation(tmp_path, "ablation_classifier_finetune_frozen", f1=0.95, roc_auc=0.96)
        df=build_ablation_table(tmp_path)
        assert len(df)==1
        assert df.iloc[0]["f1"]==0.95


class TestPlotAblationGroup:
    def test_returns_none_for_missing_group(self, tmp_path):
        _log_fake_ablation(tmp_path, "ablation_classifier_finetune_frozen", f1=0.8, roc_auc=0.85)
        df=build_ablation_table(tmp_path)
        result=plot_ablation_group(df, "Nonexistent Group", tmp_path / "plots")
        assert result is None

    def test_saves_plot_file_for_valid_group(self, tmp_path):
        _log_fake_ablation(tmp_path, "ablation_classifier_finetune_frozen", f1=0.8, roc_auc=0.85)
        _log_fake_ablation(tmp_path, "ablation_classifier_finetune_full", f1=0.75, roc_auc=0.80)
        df=build_ablation_table(tmp_path)

        out_dir=tmp_path/"plots"
        result=plot_ablation_group(df, "Classifier: fine-tune mode", out_dir)
        assert result is not None
        assert result.exists()
        assert result.suffix==".png"
