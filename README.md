# YOHO: You Only Hear Once
**Real-time Audio Event Detection on Spectrograms**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

YOHO is a modern, single-stage architecture designed for **polyphonic Sound Event Detection (SED)**.  
It combines efficient CNN-based feature extraction, long-range context via Conformer blocks, BiFPN multi-scale fusion, and DETR-style bipartite matching to handle overlapping events and variable durations in audio spectrograms.

## 🚀 Key Features (as of March 2026)

- BiPath Conformer Backbone: dual-path (time + frequency) processing with Audio Conformer blocks
- BiFPN (Bi-directional Feature Pyramid Network) with learnable weighted fusion
- Anchor-free detection head using Distribution Focal Loss (DFL) + Varifocal Loss
- Asymmetric Audio IoU Loss - strong penalty on temporal errors, softer on frequency bounds
- Polyphonic MixUp on waveform level with correct bounding box mixing
- Model EMA (Exponential Moving Average) for better generalization
- SpecAugment (frequency & time masking) applied during training
- Hungarian bipartite matching during training (DETR-style)
- Lightweight NMS (IoU threshold 0.45) applied optionally during inference
- Event-based F1 and PSDS evaluation support
- Demo notebook with visualization of predictions on spectrograms

## Current Capabilities & Limitations

✅ Fully working end-to-end training & inference pipeline  
✅ Tested primarily on ESC-50 (converted to pseudo-SED with single-event bounding boxes)  
✅ Correct normalization of time (seconds) and frequency ([0,1]) coordinates  
✅ Proper PSDS evaluation via `psds_eval` library  

⚠️ Not yet fully implemented / integrated:
- True streaming / chunk-based real-time inference
- Instance segmentation (masks)
- Full strong-labeled DESED support (loader exists as prototype only)
- Temporal memory bank for continual learning

## 🛠 Installation

```bash
git clone https://github.com/Ant1pozitive/you-only-hear-once.git
cd you-only-hear-once

# Install dependencies
pip install -r requirements.txt

# Recommended: install recent torch + torchaudio with CUDA (adjust for your version)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 📦 Requirements (core)

```text
torch>=2.3.0
torchaudio>=2.3.0
librosa>=0.10.0
scipy>=1.10.0
matplotlib>=3.8.0
numpy>=1.26.0
wandb>=0.16.0
tqdm>=4.66.0
psds-eval>=0.2.0
sed-eval>=0.2.1
pandas>=2.0.0
```

## 🚀 Quick Start

### Training (ESC-50 example)

```bash
# Default training (ESC-50 pseudo-SED)
python train.py --batch-size 16 --lr 1e-3 --epochs 100

# Or override via command line (if you extend the argparser)
python train.py --batch-size 24 --lr 5e-4 --epochs 150
```

### Inference & Visualization (demo notebook)

1. Train or download a checkpoint (`checkpoints/best_yoho_ema.pth`)
2. Open `demo.ipynb`
3. Run all cells - includes synthetic audio generation + visualization

## 📂 Project Structure

```text
├── config.py               # All hyperparameters (model, training, data, aug)
├── train.py                # Main training loop with EMA, MixUp, AMP
├── model/
│   ├── yoho.py             # Core model + inference logic
│   ├── backbone.py         # BiPath + Conformer + BiFPN
│   ├── heads.py            # Detection head, losses, Hungarian matcher, NMS
│   └── memory_aug.py       # Memory bank (not yet integrated)
├── data/
│   ├── audio_dataset.py    # ESC-50 loader + pseudo-SED conversion
│   ├── desed_dataset.py    # DESED prototype loader (not yet wired)
│   └── augmentations.py    # Waveform-level augmentations
└── utils/
    ├── metrics.py          # Event-based F1 + PSDS
    └── visualize.py        # Spectrogram + bounding box plotting
```

## 🗺 Roadmap (March 2026)

### Completed / Working Well

- [x] Anchor-free head with DFL + Varifocal Loss
- [x] Asymmetric Audio IoU Loss (time-heavy)
- [x] BiPath + Conformer backbone + BiFPN fusion
- [x] Polyphonic MixUp (waveform + boxes)
- [x] Model EMA + AMP + gradient clipping
- [x] Event-based F1 + PSDS evaluation
- [x] Clean demo notebook with visualization

### Next 1–4 months (high priority)

- [ ] Full strong-labeled DESED integration + PSDS in validation loop
- [ ] Lightweight NMS tuning / ablation study
- [ ] Curriculum learning improvements
- [ ] Better handling of variable-length clips

### Medium-term

- [ ] True streaming / chunk-based inference with state carry-over
- [ ] Temporal memory bank for continual / lifelong learning
- [ ] Instance segmentation prototype (proto-masks)

### Long-term / Research Ideas

- [ ] Weak-to-strong consistency regularization
- [ ] Large-scale pre-training (AudioSet, masked modeling)
- [ ] Language-guided audio event detection (CLAP / CLAP-like)
- [ ] ONNX / TensorRT export for real-time edge deployment

## 📜 License

MIT License

Feel free to use, modify, contribute - any feedback is very welcome!
