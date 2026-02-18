import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb
import random
import copy
import numpy as np
from config import YOHOConfig
from model.yoho import YOHO
from data.audio_dataset import AudioSEDDataset
from data.desed_dataset import DESEDDataset
from utils.metrics import event_based_f1, proper_psds

class ModelEMA:
    """Exponential Moving Average of Model weights for stable training"""
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
        num_events = len(labels['cls']) if 'cls' in labels else 1
        overlap_estimate = 1.0 + (num_events - 1) * 0.3  
        weights.append(overlap_estimate)
    return weights

def get_dataset(config):
    if config.data.dataset == 'esc50':
        return AudioSEDDataset(config.data.root_dir, augment_prob=config.data.augment_prob)
    elif config.data.dataset == 'desed':
        return DESEDDataset(config.data.root_dir, mode='weak', augment_prob=config.data.augment_prob)
    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset}")

def train_epoch(model, ema, loader, optimizer, scaler, device, epoch, config):
    model.train()
    total_loss = 0.0
    for audio, labels in tqdm(loader, desc=f"Epoch {epoch}"):
        audio = audio.to(device)
        labels = {k: [v.to(device) for v in labels[k]] for k in labels[0]}
        
        # Audio MixUp (Crucial for Polyphony)
        if random.random() < config.data.aug.mixup_prob:
            batch_size = audio.size(0)
            perm = torch.randperm(batch_size).to(device)
            lam = np.random.beta(config.data.aug.mixup_alpha, config.data.aug.mixup_alpha)
            
            # Mix waveforms
            audio = lam * audio + (1 - lam) * audio[perm]
            
            # Concat Targets
            for i in range(batch_size):
                labels['boxes'][i] = torch.cat([labels['boxes'][i], labels['boxes'][perm[i]]])
                labels['cls'][i] = torch.cat([labels['cls'][i], labels['cls'][perm[i]]])
        
        with autocast(enabled=config.train.amp):
            loss_dict = model(audio, labels)
            loss = loss_dict['total_loss']
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # Update EMA
        if ema is not None:
            ema.update(model)
            
        total_loss += loss.item()
    
    return total_loss / len(loader)

def evaluate(model, loader, device, config):
    model.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        for audio, labels in loader:
            audio = audio.to(device)
            preds = model.infer(audio)
            all_preds.append(preds)
            all_gts.append(labels)
    
    f1 = np.mean([event_based_f1(p, g) for p, g in zip(all_preds, all_gts)])
    psds = proper_psds(all_preds, all_gts)  
    return {"f1": f1, "psds": psds['psds_scenario1']}

def main(args):
    config = YOHOConfig()
    config.train.batch_size = args.batch_size or config.train.batch_size
    config.train.lr = args.lr or config.train.lr
    config.train.epochs = args.epochs or config.train.epochs
    
    wandb.init(project=config.train.wandb_project, config=config.__dict__)
    device = config.train.device
    
    dataset = get_dataset(config)

    if config.data.curriculum_enabled:
        weights = get_sample_weights(dataset)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        loader = DataLoader(dataset, config.train.batch_size, sampler=sampler, num_workers=4)
    else:
        loader = DataLoader(dataset, config.train.batch_size, shuffle=True, num_workers=4)
        
    val_loader = DataLoader(dataset, config.train.batch_size, shuffle=False, num_workers=4)
    
    model = YOHO(config.model).to(device)
    ema = ModelEMA(model, decay=config.train.ema_decay)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.epochs)
    scaler = GradScaler(enabled=config.train.amp)
    
    best_f1 = 0.0
    for epoch in range(1, config.train.epochs + 1):
        train_loss = train_epoch(model, ema, loader, optimizer, scaler, device, epoch, config)
        
        # Evaluate on EMA weights for much better stability
        metrics = evaluate(ema.ema, val_loader, device, config)
        
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
            torch.save(ema.ema.state_dict(), "checkpoints/best_yoho_ema.pth")
            print(f"New best F1 (EMA): {best_f1:.4f}")
    
    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    main(args)
