"""Training loop for compact temporal jersey recognition."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .dataset import (
    JerseyTrackletDataset,
    build_records,
    deterministic_tracklet_split,
    load_ground_truth,
    seed_worker,
)
from .model import build_model
from .schemas import NUM_CLASSES, label_to_class


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    return config


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _make_loader(
    dataset: JerseyTrackletDataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total = 0
    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True)
        mask = batch["frame_mask"].to(device, non_blocking=True)
        quality = batch["frame_quality"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(frames, mask, quality)["logits"]
                loss = criterion(logits, labels)
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
        total_loss += float(loss.detach()) * labels.numel()
        total_correct += int((logits.argmax(1) == labels).sum())
        total += labels.numel()
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": total_correct / max(total, 1),
    }


def _save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def train_jersey_recognizer(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 42))
    set_determinism(seed)
    data_cfg = config["dataset"]
    train_cfg = config["training"]
    runtime_cfg = config.get("runtime", {})
    root = Path(data_cfg["root"])
    labels = load_ground_truth(root, "train")
    train_ids, val_ids = deterministic_tracklet_split(
        labels, val_fraction=float(data_cfg.get("val_fraction", 0.15)), seed=seed
    )
    train_records = build_records(
        root,
        "train",
        train_ids,
        subset_size=data_cfg.get("train_subset"),
        seed=seed,
        include_unknown=bool(data_cfg.get("include_unknown", True)),
    )
    val_records = build_records(
        root,
        "train",
        val_ids,
        subset_size=data_cfg.get("val_subset"),
        seed=seed + 1,
        include_unknown=bool(data_cfg.get("include_unknown", True)),
    )
    if not train_records or not val_records:
        raise ValueError("training and validation splits must both contain tracklets")
    assert {record.tracklet_id for record in train_records}.isdisjoint(
        record.tracklet_id for record in val_records
    )

    image_size = tuple(map(int, data_cfg.get("image_size", [128, 64])))
    num_frames = int(data_cfg.get("num_frames", 8))
    train_data = JerseyTrackletDataset(
        train_records, num_frames=num_frames, image_size=image_size, training=True
    )
    val_data = JerseyTrackletDataset(
        val_records, num_frames=num_frames, image_size=image_size, training=False
    )
    device = resolve_device(str(runtime_cfg.get("device", "auto")))
    workers = int(runtime_cfg.get("workers", 2))
    batch_size = int(train_cfg.get("batch_size", 16))
    train_loader = _make_loader(
        train_data, batch_size=batch_size, workers=workers, shuffle=True, seed=seed, device=device
    )
    val_loader = _make_loader(
        val_data, batch_size=batch_size, workers=workers, shuffle=False, seed=seed, device=device
    )

    model = build_model(config.get("model")).to(device)
    class_weights = None
    if bool(train_cfg.get("balanced_class_weights", True)):
        counts = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        for record in train_records:
            counts[label_to_class(record.label)] += 1.0
        present = counts > 0
        weights = torch.ones(NUM_CLASSES, dtype=torch.float32)
        weights[present] = counts[present].sum() / (present.sum() * counts[present])
        class_weights = weights.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(train_cfg.get("label_smoothing", 0.05)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(train_cfg.get("epochs", 20))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    output_dir = Path(train_cfg.get("output_dir", "artifacts/jersey"))
    latest_path = output_dir / "latest.pt"
    best_path = output_dir / "best.pt"
    start_epoch, best_loss, stale_epochs = 0, float("inf"), 0
    history: list[dict[str, Any]] = []

    resume = train_cfg.get("resume")
    resume_path = latest_path if resume is True else Path(resume) if resume else None
    if resume_path is not None and resume_path.exists():
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint.get("best_val_loss", best_loss))
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
        history = list(checkpoint.get("history", []))

    patience = int(train_cfg.get("early_stopping_patience", 5))
    for epoch in range(start_epoch, epochs):
        train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        )
        val_metrics = _run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        row = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        history.append(row)
        improved = val_metrics["loss"] < best_loss
        if improved:
            best_loss, stale_epochs = val_metrics["loss"], 0
        else:
            stale_epochs += 1
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_loss": best_loss,
            "stale_epochs": stale_epochs,
            "history": history,
            "config": config,
        }
        _save_checkpoint(latest_path, state)
        if improved:
            _save_checkpoint(best_path, state)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if stale_epochs >= patience:
            break
    return {
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
        "best_val_loss": best_loss,
        "epochs_completed": len(history),
        "train_tracklets": len(train_records),
        "val_tracklets": len(val_records),
    }
