from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data.dataset import MVTecDataset
from src.evaluation.experiment_log import log_experiment
from src.evaluation.metrics import anomaly_detection_report_dict, select_threshold_by_f1
from src.models.autoencoder import ConvAutoencoder, reconstruction_error_per_image
from src.utils import get_device, load_config, set_seed


def build_dataloaders(cfg: dict)->tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    data_cfg=cfg["data"]
    aug_cfg=cfg["augmentation"]
    seed=cfg["experiment"]["seed"]

    ae_train=MVTecDataset(
        root=data_cfg["root"], category=data_cfg["category"], split="ae_train",
        image_size=data_cfg["image_size"], seed=seed, augmentation_cfg=aug_cfg, normalize=False,
    )
    ae_val=MVTecDataset(
        root=data_cfg["root"], category=data_cfg["category"], split="ae_val",
        image_size=data_cfg["image_size"], seed=seed, normalize=False,
    )
    labeled_val=MVTecDataset(
        root=data_cfg["root"], category=data_cfg["category"], split="labeled_val",
        image_size=data_cfg["image_size"], seed=seed, val_fraction=0.3, normalize=False,
    )
    labeled_test=MVTecDataset(
        root=data_cfg["root"], category=data_cfg["category"], split="labeled_test",
        image_size=data_cfg["image_size"], seed=seed, val_fraction=0.3, normalize=False,
    )

    bs=cfg["training"]["batch_size"]
    nw=data_cfg["num_workers"]
    return (
        DataLoader(ae_train, batch_size=bs, shuffle=True, num_workers=nw),
        DataLoader(ae_val, batch_size=bs, shuffle=False, num_workers=nw),
        DataLoader(labeled_val, batch_size=bs, shuffle=False, num_workers=nw),
        DataLoader(labeled_test, batch_size=bs, shuffle=False, num_workers=nw),
    )


def run_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device, train: bool)->float:
    model.train() if train else model.eval()
    total_loss, n_batches=0.0, 0
    context=torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, _, _ in loader:
            images=images.to(device)
            if train:
                optimizer.zero_grad()
            reconstructed=model(images)
            loss=criterion(reconstructed, images)
            if train:
                loss.backward()
                optimizer.step()
            total_loss+=loss.item()
            n_batches+=1
    return total_loss/max(n_batches, 1)


@torch.no_grad()
def compute_scores(model: nn.Module, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_labels, all_scores = [], []
    for images, labels, _ in loader:
        images=images.to(device)
        reconstructed=model(images)
        scores=reconstruction_error_per_image(images, reconstructed)
        all_labels.extend(labels.tolist())
        all_scores.extend(scores.cpu().tolist())
    return np.array(all_labels), np.array(all_scores)


def train(config_path: str)->None:
    cfg=load_config(config_path)
    set_seed(cfg["experiment"]["seed"])
    device=get_device(cfg["device"]["prefer_cuda"])
    print(f"Using device: {device}")
    ae_train_loader, ae_val_loader, labeled_val_loader, labeled_test_loader=build_dataloaders(cfg)
    print(f"AE train (normal only): {len(ae_train_loader.dataset)} | AE val (normal only): {len(ae_val_loader.dataset)}")
    print(f"Labeled val (threshold selection): {len(labeled_val_loader.dataset)} | Labeled test: {len(labeled_test_loader.dataset)}")
    model_cfg=cfg["model"]
    model=ConvAutoencoder(
        image_size=cfg["data"]["image_size"],
        latent_dim=model_cfg["latent_dim"],
        base_channels=model_cfg["base_channels"],
        num_downsample_blocks=model_cfg["num_downsample_blocks"],
    ).to(device)
    n_params=sum(p.numel() for p in model.parameters())
    print(f"Model: conv_autoencoder | latent_dim={model_cfg['latent_dim']} | params={n_params:,}")

    train_cfg=cfg["training"]
    criterion=nn.MSELoss() if train_cfg["loss"] == "mse" else nn.L1Loss()
    optimizer=torch.optim.Adam(
        model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"]
    )
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=train_cfg["scheduler_patience"], factor=0.5
    )

    ckpt_dir=Path(cfg["checkpointing"]["save_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss, best_epoch, patience_counter=np.inf, -1, 0

    start_time=time.time()
    for epoch in tqdm(range(train_cfg["epochs"]), desc="Training"):
        train_loss=run_epoch(model, ae_train_loader, criterion, optimizer, device, train=True)
        val_loss=run_epoch(model, ae_val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{train_cfg['epochs']} | train_loss={train_loss:.5f} val_loss={val_loss:.5f}")

        if val_loss<best_val_loss:
            best_val_loss, best_epoch, patience_counter = val_loss, epoch, 0
            torch.save(
                {"model_state_dict": model.state_dict(), "config": cfg, "epoch": epoch},
                ckpt_dir / "best_model.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= train_cfg["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch+1} (best epoch: {best_epoch+1})")
                break

    training_time_sec=time.time()-start_time

    checkpoint=torch.load(ckpt_dir/"best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    #threshold selection: VALIDATION labeled images only.
    val_labels, val_scores=compute_scores(model, labeled_val_loader, device)
    threshold=select_threshold_by_f1(val_labels, val_scores)
    print(f"\nSelected threshold from validation data: {threshold:.6f}")

    val_report=anomaly_detection_report_dict(val_labels, val_scores, threshold)

    #final test report: touched exactly once, using the validation-derived threshold.
    test_labels, test_scores=compute_scores(model, labeled_test_loader, device)
    test_report=anomaly_detection_report_dict(test_labels, test_scores, threshold)

    metrics={
        "best_epoch": best_epoch + 1,
        "training_time_sec": training_time_sec,
        "selected_threshold": threshold,
        "validation": val_report,
        "test": test_report,
        "model_params": n_params,
    }
    print("\nFinal TEST set report (touched once):")
    print(test_report)

    log_path=log_experiment(cfg["experiment"]["name"], cfg, metrics)
    print(f"\nExperiment logged to: {log_path}")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/autoencoder.yaml")
    args=parser.parse_args()
    train(args.config)
