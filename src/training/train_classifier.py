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
from src.evaluation.metrics import classification_report_dict
from src.models.classifier import DefectClassifier
from src.utils import get_device, load_config, set_seed


def build_dataloaders(cfg: dict)->tuple[DataLoader, DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    aug_cfg = cfg["augmentation"]

    #`labeled_val` (from dataset.py's split) is what the classifier actually
    #trains + validates on; `labeled_test` is the untouched final test set.
    full_labeled = MVTecDataset(
        root=data_cfg["root"],
        category=data_cfg["category"],
        split="labeled_val",
        image_size=data_cfg["image_size"],
        seed=cfg["experiment"]["seed"],
        val_fraction=data_cfg["val_fraction"],
        augmentation_cfg=aug_cfg,
        normalize=True,
    )
    #internal split of labeled_val into classifier-train / classifier-val,
    #deterministic via a derived seed so it doesn't collide with the
    #dataset-level split logic, and touches only images already inside
    #labeled_val (never labeled_test).
    n=len(full_labeled)
    rng=np.random.default_rng(cfg["experiment"]["seed"]+1)
    indices=rng.permutation(n)
    split_point=int(n*0.7)
    train_idx, val_idx=indices[:split_point], indices[split_point:]

    train_subset=torch.utils.data.Subset(full_labeled, train_idx)
    val_subset=torch.utils.data.Subset(full_labeled, val_idx)

    test_set=MVTecDataset(
        root=data_cfg["root"],
        category=data_cfg["category"],
        split="labeled_test",
        image_size=data_cfg["image_size"],
        seed=cfg["experiment"]["seed"],
        val_fraction=data_cfg["val_fraction"],
        normalize=True,
    )

    train_loader=DataLoader(
        train_subset, batch_size=cfg["training"]["batch_size"], shuffle=True,
        num_workers=data_cfg["num_workers"],
    )
    val_loader=DataLoader(
        val_subset, batch_size=cfg["training"]["batch_size"], shuffle=False,
        num_workers=data_cfg["num_workers"],
    )
    test_loader=DataLoader(
        test_set, batch_size=cfg["training"]["batch_size"], shuffle=False,
        num_workers=data_cfg["num_workers"],
    )
    return train_loader, val_loader, test_loader


def compute_class_weights(loader: DataLoader, device: torch.device)->torch.Tensor:
    labels=[]
    for _, y, _ in loader:
        labels.extend(y.tolist())
    labels=np.array(labels)
    counts=np.bincount(labels, minlength=2)
    weights=counts.sum()/(len(counts)*np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(
    model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device, train: bool
)->float:
    model.train() if train else model.eval()
    total_loss, n_batches = 0.0, 0

    context=torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels, _ in loader:
            images, labels=images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            logits=model(images)
            loss=criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss+=loss.item()
            n_batches+=1
    return total_loss/max(n_batches, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device)->dict:
    model.eval()
    all_labels, all_preds, all_scores = [], [], []
    for images, labels, _ in loader:
        images=images.to(device)
        logits=model(images)
        probs=torch.softmax(logits, dim=1)[:, 1]
        preds=(probs >= 0.5).long()
        all_labels.extend(labels.tolist())
        all_preds.extend(preds.cpu().tolist())
        all_scores.extend(probs.cpu().tolist())
    return classification_report_dict(
        np.array(all_labels), np.array(all_preds), np.array(all_scores)
    )


def train(config_path: str)->None:
    cfg=load_config(config_path)
    set_seed(cfg["experiment"]["seed"])
    device=get_device(cfg["device"]["prefer_cuda"])
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    model_cfg=cfg["model"]
    model=DefectClassifier(
        architecture=model_cfg["architecture"],
        pretrained=model_cfg["pretrained"],
        num_classes=model_cfg["num_classes"],
        dropout=model_cfg["dropout"],
        finetune_mode=model_cfg["finetune_mode"],
        unfrozen_layers=model_cfg.get("unfrozen_layers", []),
    ).to(device)
    print(
        f"Model: {model_cfg['architecture']} ({model_cfg['finetune_mode']}) | "
        f"Trainable params: {model.trainable_parameter_count():,} / {model.total_parameter_count():,}"
    )

    train_cfg=cfg["training"]
    class_weights=(
        compute_class_weights(train_loader, device) if train_cfg["class_weighting"] else None
    )
    criterion=nn.CrossEntropyLoss(weight=class_weights)
    optimizer=torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"], eta_min=train_cfg["scheduler_min_lr"]
    )

    ckpt_dir=Path(cfg["checkpointing"]["save_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_metric, best_epoch, patience_counter= -np.inf, -1, 0

    start_time=time.time()
    for epoch in tqdm(range(train_cfg["epochs"]), desc="Training"):
        train_loss=run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss=run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        val_report=evaluate(model, val_loader, device)
        scheduler.step()

        monitor=val_report[cfg["checkpointing"]["monitor_metric"].replace("val_", "")]
        print(
            f"Epoch {epoch+1}/{train_cfg['epochs']} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_f1={val_report['f1']:.4f} val_auc={val_report['roc_auc']:.4f}"
        )

        if monitor>best_metric:
            best_metric, best_epoch, patience_counter = monitor, epoch, 0
            torch.save(
                {"model_state_dict": model.state_dict(), "config": cfg, "epoch": epoch},
                ckpt_dir/"best_model.pt",
            )
        else:
            patience_counter+=1
            if patience_counter>=train_cfg["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch+1} (best epoch: {best_epoch+1})")
                break

    training_time_sec=time.time()-start_time

    #reload best checkpoint (by VALIDATION metric) before final test-set report
    checkpoint=torch.load(ckpt_dir/"best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_report=evaluate(model, val_loader, device)
    test_report=evaluate(model, test_loader, device)  # touched exactly once, here

    metrics={
        "best_epoch": best_epoch + 1,
        "training_time_sec": training_time_sec,
        "validation": val_report,
        "test": test_report,
        "trainable_params": model.trainable_parameter_count(),
        "total_params": model.total_parameter_count(),
    }
    print("\nFinal TEST set report (touched once):")
    print(test_report)

    log_path = log_experiment(cfg["experiment"]["name"], cfg, metrics)
    print(f"\nExperiment logged to: {log_path}")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/classifier.yaml")
    args=parser.parse_args()
    train(args.config)
