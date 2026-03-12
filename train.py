import argparse
import torch
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb
import random
import copy
import numpy as np
import os
from config import YOHOConfig
from model.yoho import YOHO
from data.audio_dataset import AudioSEDDataset, create_dataloader
from utils.metrics import event_based_f1, proper_psds

class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for param in self.ema.parameters():
            param.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            for ema_v, model_v in zip(self.ema.state_dict().values(), model.state_dict().values()):
                if ema_v.dtype.is_floating_point:
                    ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)

def get_sample_weights(dataset):
    weights = []
    for idx in range(len(dataset)):
        _, labels = dataset[idx]
        num_events = len(labels['labels']) if 'labels' in labels else 1
        overlap_estimate = 1.0 + (num_events - 1) * 0.3
        weights.append(overlap_estimate)
    return weights

def get_dataset(config):
    if config.data.dataset == 'esc50':
        return AudioSEDDataset(config.data.root_dir, augment_prob=config.data.augment_prob, max_dur_sec=config.data.max_dur_sec)
    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset}")

def train_epoch(model, ema, loader, optimizer, scaler, device, epoch, config):
    model.train()
    total_loss = 0.0
    for audio, labels in tqdm(loader, desc=f"Epoch {epoch}"):
        audio = audio.to(device)
        labels = {k: [v.to(device) for v in labels[k]] for k in labels}

        if random.random() < config.data.aug.mixup_prob:
            batch_size = audio.size(0)
            perm = torch.randperm(batch_size).to(device)
            lam = np.random.beta(config.data.aug.mixup_alpha, config.data.aug.mixup_alpha)
            audio = lam * audio + (1 - lam) * audio[perm]
            for i in range(batch_size):
                labels['boxes'][i] = torch.cat([labels['boxes'][i], labels['boxes'][perm[i]]])
                labels['labels'][i] = torch.cat([labels['labels'][i], labels['labels'][perm[i]]])

        with autocast(enabled=config.train.amp):
            loss_dict = model(audio, labels)
            loss = loss_dict['total_loss']

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if ema is not None:
            ema.update(model)
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device, config):
    model.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        for audio, batch_targets in loader:
            audio = audio.to(device)
            pred_dicts = model.infer(audio)  # List[Dict] per sampl
            for i, pred_d in enumerate(pred_dicts):
                gt_d = {
                    "boxes": batch_targets["boxes"][i],
                    "labels": batch_targets["labels"][i]
                }
                all_preds.append(pred_d)
                all_gts.append(gt_d)
    f1 = np.mean([event_based_f1(p, g) for p, g in zip(all_preds, all_gts)])
    psds = proper_psds(all_preds, all_gts)
    return {"f1": f1, "psds": psds['psds_scenario1']}

def main(args):
    config = YOHOConfig()
    config.train.batch_size = args.batch_size or config.train.batch_size
    config.train.lr = args.lr or config.train.lr
    config.train.epochs = args.epochs or config.train.epochs

    os.makedirs("checkpoints", exist_ok=True)
    wandb.init(project=config.train.wandb_project, config=config.__dict__)
    device = config.train.device

    dataset = get_dataset(config)
    if config.data.curriculum_enabled:
        weights = get_sample_weights(dataset)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        loader = create_dataloader(dataset, config.train.batch_size, sampler=sampler, num_workers=4)
    else:
        loader = create_dataloader(dataset, config.train.batch_size, shuffle=True, num_workers=4)
    val_loader = create_dataloader(dataset, config.train.batch_size, shuffle=False, num_workers=4)

    model = YOHO(config).to(device)
    model = torch.compile(model)
    ema = ModelEMA(model, decay=config.train.ema_decay)

    optimizer = optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.epochs)
    scaler = GradScaler(enabled=config.train.amp)

    best_f1 = 0.0
    patience_counter = 0
    for epoch in range(1, config.train.epochs + 1):
        train_loss = train_epoch(model, ema, loader, optimizer, scaler, device, epoch, config)
        metrics = evaluate(ema.ema, val_loader, device, config)

        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_f1": metrics["f1"], "val_psds": metrics["psds"], "lr": optimizer.param_groups[0]['lr']})
        scheduler.step()

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            patience_counter = 0
            torch.save(ema.ema.state_dict(), "checkpoints/best_yoho_ema.pth")
            print(f"New best F1 (EMA): {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config.train.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {config.train.patience} epochs)")
                break

    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    main(args)
