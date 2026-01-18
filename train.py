import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb
from config import YOHOConfig
from model.yoho import YOHO
from data.audio_dataset import AudioSEDDataset
from utils.metrics import event_based_f1, psds
from utils.visualize import plot_spectrogram_with_preds

def train_epoch(model, loader, optimizer, scaler, device, clip_grad):
    model.train()
    total_loss = 0
    for audio, labels in tqdm(loader):
        audio = audio.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}
        with autocast():
            loss = model(audio, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    f1_total = 0
    psds_total = 0
    with torch.no_grad():
        for audio, labels in loader:
            audio = audio.to(device)
            preds = model.infer(audio)
            f1_total += event_based_f1(preds, labels)
            psds_total += psds(preds, labels)
    n = len(loader)
    return {"f1": f1_total / n, "psds": psds_total / n}

def main(args):
    config = YOHOConfig()
    # Override args
    config.train.batch_size = args.batch_size
    config.train.lr = args.lr
    config.train.epochs = args.epochs
    # etc.

    wandb.init(project="yoho-audio")
    device = config.train.device
    dataset = AudioSEDDataset(config.data.root_dir)
    loader = DataLoader(dataset, config.train.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset, config.train.batch_size, num_workers=4)  # Split in real

    model = YOHO(config.model).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.epochs)
    scaler = GradScaler()
    clip_grad = 10.0

    best_f1 = 0
    for epoch in range(config.train.epochs):
        train_loss = train_epoch(model, loader, optimizer, scaler, device, clip_grad)
        metrics = evaluate(model, val_loader, device)
        wandb.log({"epoch": epoch, "train_loss": train_loss, **metrics})
        scheduler.step()
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), "best_yoho.pth")
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="esc50")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--clip-grad", type=float, default=10.0)
    args = parser.parse_args()
    main(args)
