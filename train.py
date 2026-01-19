import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb
import random
import numpy as np
from config import YOHOConfig
from model.yoho import YOHO
from data.audio_dataset import AudioSEDDataset
from utils.metrics import event_based_f1, psds
from utils.visualize import plot_spectrogram_with_preds

def get_sample_weights(dataset):
    """Compute difficulty weights: higher for more complex samples"""
    weights = []
    for idx in range(len(dataset)):
        _, labels = dataset[idx]
        num_events = len(labels['cls']) if 'cls' in labels else 1
        overlap_estimate = 1.0 + (num_events - 1) * 0.3  # simple heuristic
        weights.append(overlap_estimate)
    return weights


def train_epoch(model, loader, optimizer, scaler, device, epoch, config):
    model.train()
    total_loss = 0.0
    for audio, labels in tqdm(loader, desc=f"Epoch {epoch}"):
        audio = audio.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}
        
        with autocast():
            loss = model(audio, labels)  # assumes model returns loss in train mode
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    f1_sum, psds_sum, count = 0.0, 0.0, 0
    with torch.no_grad():
        for audio, labels in loader:
            audio = audio.to(device)
            preds = model.infer(audio)
            f1_sum += event_based_f1(preds, labels)
            psds_sum += psds(preds, labels)
            count += 1
    return {"f1": f1_sum / count, "psds": psds_sum / count}


def main(args):
    config = YOHOConfig()
    config.train.batch_size = args.batch_size or config.train.batch_size
    config.train.lr = args.lr or config.train.lr
    config.train.epochs = args.epochs or config.train.epochs
    
    wandb.init(project="yoho-audio", config=config.__dict__)
    device = config.train.device
    
    dataset = AudioSEDDataset(config.data.root_dir, augment_prob=config.data.augment_prob)

    if config.data.curriculum_enabled:
        weights = get_sample_weights(dataset)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        loader = DataLoader(dataset, config.train.batch_size, sampler=sampler, num_workers=4)
    else:
        loader = DataLoader(dataset, config.train.batch_size, shuffle=True, num_workers=4)
        
    val_loader = DataLoader(dataset, config.train.batch_size, shuffle=False, num_workers=4)
    
    model = YOHO(config.model).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.epochs)
    scaler = GradScaler(enabled=config.train.amp)
    
    best_f1 = 0.0
    for epoch in range(1, config.train.epochs + 1):
        if config.data.curriculum_enabled and epoch >= config.data.curriculum_start_epoch:
            progress = min(1.0, (epoch - config.data.curriculum_start_epoch) / config.data.curriculum_ramp_epochs)
        
        train_loss = train_epoch(model, loader, optimizer, scaler, device, epoch, config)
        metrics = evaluate(model, val_loader, device)
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_f1": metrics["f1"],
            "val_psds": metrics["psds"],
            "lr": optimizer.param_groups[0]['lr']
        })
        
        scheduler.step()
        
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), "checkpoints/best_yoho.pth")
            print(f"New best F1: {best_f1:.4f}")
    
    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOHO Audio SED Training")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    main(args)
