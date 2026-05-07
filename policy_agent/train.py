from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from backend.app.config import load_config
from datasets.loader import ACTIONS, build_routing_loader_from_manifest
from policy_agent.model import TransformerPolicyAgent


def _log(msg: str) -> None:
    print(msg, flush=True)
    sys.stdout.flush()


def _compute_class_weights(manifest_path: Path, num_classes: int, device: str) -> torch.Tensor:
    """
    Inverse-frequency weights w_c = N / (num_classes * count_c), clamped to a reasonable range.
    Falls back to uniform weights if the manifest lacks oracle_action.
    """
    try:
        df = pd.read_csv(manifest_path, usecols=lambda c: c == "oracle_action")
    except Exception:
        df = pd.DataFrame()
    weights = np.ones(num_classes, dtype=np.float32)
    if not df.empty and "oracle_action" in df.columns:
        counts = df["oracle_action"].value_counts().to_dict()
        total = sum(counts.values())
        for c in range(num_classes):
            n = float(counts.get(c, 0))
            weights[c] = total / (num_classes * n) if n > 0 else 0.0
        # Avoid zero weights for absent classes (e.g. MossFormer2 with 0 oracle wins): set
        # to the median nonzero weight so the head can still learn that class is rarely best.
        nonzero = weights[weights > 0]
        if nonzero.size:
            fallback = float(np.median(nonzero))
            for c in range(num_classes):
                if weights[c] == 0.0:
                    weights[c] = fallback
        # Clamp ratio so a tiny class doesn't blow up the loss.
        max_ratio = 8.0
        wmin = float(weights.min())
        for c in range(num_classes):
            weights[c] = float(min(weights[c], wmin * max_ratio))
        weights = weights / max(float(weights.mean()), 1e-6)  # normalize to mean 1.0
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _evaluate(
    model: nn.Module,
    loader,
    device: str,
    epoch: int | None = None,
    ce_loss: nn.Module | None = None,
) -> dict[str, float]:
    model.eval()
    ce = ce_loss if ce_loss is not None else nn.CrossEntropyLoss()
    bce = nn.BCELoss()
    total_loss = 0.0
    n = 0
    correct = 0
    preds_all: list[int] = []
    labels_all: list[int] = []
    desc = f"val   ep{epoch}" if epoch is not None else "val"
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, file=sys.stdout, mininterval=0.5)
    with torch.no_grad():
        for batch in pbar:
            wave = batch["wave"].to(device, non_blocking=True)
            distortion = batch["distortion"].to(device, non_blocking=True)
            labels = batch["action"].to(device, non_blocking=True)
            target_strength = batch["strength"].to(device, non_blocking=True)
            target_refine = batch["refine"].to(device, non_blocking=True)

            action_logits, strength_pred, refine_pred = model.forward_logits(wave, distortion)
            with torch.amp.autocast(device_type="cuda", enabled=False):
                refine_loss = bce(refine_pred.float(), target_refine.float())
            loss = ce(action_logits, labels) + nn.functional.mse_loss(strength_pred, target_strength) + refine_loss
            total_loss += float(loss.item())
            pred = torch.argmax(action_logits, dim=-1)
            correct += int((pred == labels).sum().item())
            n += int(labels.numel())
            preds_all.extend(pred.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(correct/max(n,1)):.3f}")

    accuracy = (correct / n) if n else 0.0
    cm = np.zeros((len(ACTIONS), len(ACTIONS)), dtype=np.int64)
    for y, p in zip(labels_all, preds_all):
        cm[y, p] += 1
    return {"val_loss": total_loss / max(len(loader), 1), "val_accuracy": accuracy, "confusion_matrix": cm.tolist()}


def _save_confusion_matrix(cm: list[list[int]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues", xticklabels=ACTIONS, yticklabels=ACTIONS, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Policy Action Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def train_supervised(config_path: str = "configs/base.yaml") -> None:
    cfg = load_config(config_path)
    device = "cuda" if cfg.system.device == "cuda" and torch.cuda.is_available() else "cpu"
    _log(f"[train] config={config_path} device={device} mixed_precision={cfg.system.mixed_precision}")
    if device == "cuda":
        _log(f"[train] gpu={torch.cuda.get_device_name(0)} torch={torch.__version__}")

    if cfg.training.distributed and torch.cuda.device_count() > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    _log(f"[train] building model wavlm={cfg.policy.wavlm_name} hidden={cfg.policy.hidden_dim}")
    t0 = time.time()
    model = TransformerPolicyAgent(
        wavlm_name=cfg.policy.wavlm_name,
        distortion_dim=6,
        hidden_dim=cfg.policy.hidden_dim,
        num_heads=cfg.policy.num_heads,
        num_layers=cfg.policy.num_layers,
        dropout=cfg.policy.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log(f"[train] model ready in {time.time()-t0:.1f}s params={n_params/1e6:.2f}M trainable={n_trainable/1e6:.2f}M")

    if cfg.training.distributed and dist.is_initialized():
        model = DDP(model, device_ids=[dist.get_rank()])

    # Optimize only trainable params (frozen WavLM is excluded automatically).
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_real_trainable = sum(p.numel() for p in trainable_params)
    _log(f"[train] real_trainable_params={n_real_trainable/1e6:.2f}M (frozen-wavlm head only)")

    optimizer = optim.AdamW(trainable_params, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    bce = nn.BCELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    writer = SummaryWriter("logs/tensorboard/policy")
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = Path(cfg.training.train_manifest)
    val_manifest = Path(cfg.training.val_manifest)
    if not train_manifest.exists() or not val_manifest.exists():
        raise FileNotFoundError(
            f"Missing manifest files: {train_manifest} / {val_manifest}. "
            "Run scripts/prepare_dataset.py or scripts/build_paired_manifest.py."
        )

    # Inverse-frequency class weighting for the action head — handles label imbalance
    # (e.g. BYPASS-heavy oracle distribution) without changing the loss formulation.
    class_weights_tensor = _compute_class_weights(train_manifest, len(ACTIONS), device)
    _log(f"[train] class_weights={class_weights_tensor.detach().cpu().tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
    val_criterion = nn.CrossEntropyLoss()

    _log(f"[train] manifests train={train_manifest} val={val_manifest}")
    train_loader = build_routing_loader_from_manifest(
        manifest_csv=str(train_manifest),
        batch_size=cfg.training.batch_size,
        sample_rate=cfg.system.sample_rate,
        shuffle=True,
    )
    val_loader = build_routing_loader_from_manifest(
        manifest_csv=str(val_manifest),
        batch_size=cfg.training.batch_size,
        sample_rate=cfg.system.sample_rate,
        shuffle=False,
    )
    _log(
        f"[train] dataset_sizes train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
        f"batch_size={cfg.training.batch_size} epochs={cfg.training.epochs} lr={cfg.training.lr}"
    )

    # Cosine LR with a brief linear warmup (5% of steps) — stable for short schedules.
    total_steps = max(1, cfg.training.epochs * max(1, len(train_loader)))
    warmup_steps = max(50, int(0.05 * total_steps))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + float(np.cos(np.pi * min(1.0, progress))))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    best_loss = float("inf")
    patience = 0
    history: list[dict[str, float]] = []
    for epoch in range(cfg.training.epochs):
        model.train()
        epoch_loss_total = 0.0
        batch_count = 0
        ep_start = time.time()
        pbar = tqdm(
            train_loader,
            desc=f"train ep{epoch}",
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            mininterval=0.5,
        )
        for batch in pbar:
            wave = batch["wave"].to(device, non_blocking=True)
            distortion = batch["distortion"].to(device, non_blocking=True)
            labels = batch["action"].to(device, non_blocking=True)
            target_strength = batch["strength"].to(device, non_blocking=True)
            target_refine = batch["refine"].to(device, non_blocking=True)

            soft_targets = batch.get("soft_action")
            if soft_targets is not None:
                soft_targets = soft_targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda" and cfg.system.mixed_precision)):
                action_logits, strength_pred, refine_pred = model.forward_logits(wave, distortion)
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    refine_loss = bce(refine_pred.float(), target_refine.float())
                # Hybrid action loss: hard CE (with class weights + label smoothing) + soft-label KL.
                # Soft targets reflect oracle composite-score margins between experts and reduce
                # overfitting on ambiguous clips; the hard term keeps decisions sharp.
                hard_ce = criterion(action_logits, labels)
                if soft_targets is not None:
                    log_probs = nn.functional.log_softmax(action_logits, dim=-1)
                    kl_term = -(soft_targets * log_probs).sum(dim=-1).mean()
                    action_loss = 0.5 * hard_ce + 0.5 * kl_term
                else:
                    action_loss = hard_ce
                loss = action_loss + nn.functional.mse_loss(strength_pred, target_strength) + refine_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.training.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss_total += float(loss.item())
            batch_count += 1
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{epoch_loss_total/batch_count:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        epoch_loss = epoch_loss_total / max(batch_count, 1)
        ep_train_dt = time.time() - ep_start
        eval_stats = _evaluate(model, val_loader, device, epoch=epoch, ce_loss=val_criterion)
        writer.add_scalar("train/loss", epoch_loss, epoch)
        writer.add_scalar("val/loss", eval_stats["val_loss"], epoch)
        writer.add_scalar("val/accuracy", eval_stats["val_accuracy"], epoch)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(epoch_loss),
                "val_loss": float(eval_stats["val_loss"]),
                "val_accuracy": float(eval_stats["val_accuracy"]),
            }
        )
        torch.save({"epoch": epoch, "state_dict": model.state_dict()}, ckpt_dir / "policy_last.pt")
        improved = eval_stats["val_loss"] < best_loss
        if improved:
            best_loss = float(eval_stats["val_loss"])
            patience = 0
            torch.save({"epoch": epoch, "state_dict": model.state_dict()}, ckpt_dir / "policy_best.pt")
            _save_confusion_matrix(eval_stats["confusion_matrix"], Path("outputs/reports/confusion_matrix_best.png"))
        else:
            patience += 1
        _log(
            f"[epoch {epoch:02d}/{cfg.training.epochs}] "
            f"train_loss={epoch_loss:.4f} val_loss={eval_stats['val_loss']:.4f} "
            f"val_acc={eval_stats['val_accuracy']:.4f} best={best_loss:.4f} "
            f"patience={patience}/{cfg.training.early_stopping_patience} "
            f"train_time={ep_train_dt:.1f}s {'[*best -> policy_best.pt]' if improved else ''}"
        )
        # Persist history every epoch so partial runs are inspectable.
        Path("outputs/reports").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(Path("outputs/reports/training_history.csv"), index=False)
        if not improved and patience >= cfg.training.early_stopping_patience:
            _log(f"[train] early stop at epoch {epoch} (patience {patience})")
            break
    reports = Path("outputs/reports")
    reports.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(reports / "training_history.csv", index=False)
    _log("[train] running final evaluation on val set")
    final_eval = _evaluate(model, val_loader, device, epoch=None, ce_loss=val_criterion)
    pd.DataFrame([{"best_val_loss": best_loss, "final_val_accuracy": final_eval["val_accuracy"]}]).to_csv(
        reports / "model_performance.csv", index=False
    )
    _save_confusion_matrix(final_eval["confusion_matrix"], reports / "confusion_matrix_final.png")
    writer.close()
    if cfg.training.distributed and dist.is_initialized():
        dist.destroy_process_group()
    _log(
        f"[train] DONE best_val_loss={best_loss:.4f} final_val_acc={final_eval['val_accuracy']:.4f} "
        f"checkpoints=checkpoints/policy_best.pt,policy_last.pt"
    )


def ppo_finetune_stub(reward_weights: Dict[str, float]) -> None:
    _ = reward_weights
    # Add PPO loop using policy rollout and reward from evaluation metrics.
