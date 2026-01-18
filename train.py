import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from config import cfg, YOHOConfig
from model.yoho import YOHO
from data.audio_dataset import AudioSEDDataset
from utils.metrics import compute_f1, compute_map

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for audio, labels in loader:
        audio, labels = audio.to(device), {k: v.to(device) for k,v in labels.items()}
        out = model(audio)
        # Loss: YOLO-style (bbox + class + conf) + mask BCE if seg
        loss = torch.tensor(1.0)  # Placeholder: Implement full loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    metrics = {"f1": 0, "map": 0}
    with torch.no_grad():
        for audio, labels in loader:
            audio = audio.to(device)
            out = model.infer(audio)
            metrics["f1"] += compute_f1(out, labels)
            metrics["map"] += compute_map(out, labels)
    return {k: v / len(loader) for k,v in metrics.items()}

def main(args):
    config = YOHOConfig()  # Override with args if needed
    wandb.init(project=config.train.wandb_project)
    device = config.train.device
    dataset = AudioSEDDataset(config.data.root_dir, config.data.augment_prob)
    loader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=config.train.batch_size)  # Placeholder

    model = YOHO(config.model).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    best_f1 = 0
    patience_cnt = 0
    for epoch in range(config.train.epochs):
        train_loss = train_epoch(model, loader, optimizer, device)
        metrics = evaluate(model, val_loader, device)
        wandb.log({"epoch": epoch, "train_loss": train_loss, **metrics})
        scheduler.step(metrics["f1"])
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), "best_yoho.pth")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= config.train.patience:
                break
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="esc50")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    main(args)
