# YOHO: You Only Hear Once  
**Real-time Audio Event Detection and Instance Segmentation on Spectrograms**

YOHO is a modern, one-stage architecture for polyphonic **Sound Event Detection (SED)** and **temporal-frequency instance segmentation**, heavily inspired by YOLOv8 but specifically re-designed for audio domain challenges.

## Key Features

- One forward pass → bounding boxes + class probabilities + instance masks
- Multi-scale feature pyramid optimized for very different event durations
- Auditory-inspired multi-head self-attention over time-frequency grid
- Optional active memory bank for continual learning / domain adaptation
- Strong real-time focus (streaming inference support planned)
- Comprehensive evaluation metrics: event-based F1, PSDS, mAP@IoU, segmentation IoU

## Current Capabilities (2026-01 status)

- Mel-spectrogram preprocessing with proper normalization
- Multi-scale CSP-like backbone with temporal pooling variants
- Detection head with anchor-free regression + classification
- Prototype instance segmentation head (proto + mask coefficients)
- Basic memory consolidation mechanism (Hebbian-style)
- Training pipeline with mixed precision & gradient clipping
- Visualizations: spectrogram + predictions overlay

## Installation

```bash
git clone https://github.com/Ant1pozitive/you-only-hear-once.git
cd you-only-hear-once
pip install -r requirements.txt
# Optional: for better audio processing & faster training
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Requirements (minimal working set)

```
torch>=2.3.0
torchaudio>=2.3.0
librosa>=0.10.0
matplotlib>=3.8.0
numpy>=1.26.0
wandb>=0.16.0
tqdm>=4.66.0
```

## Training example

```bash
# ESC-50 classification → we convert to pseudo-SED format
python train.py --dataset esc50 --epochs 80 --batch-size 24 --lr 3e-4

# Stronger training schedule (recommended)
python train.py \
  --dataset deSED \
  --batch-size 16 \
  --lr 2e-4 \
  --weight-decay 5e-5 \
  --epochs 120 \
  --amp \
  --clip-grad 10.0
```

## Project Structure

```
yoho-audio/
├── config.py               # All hyperparameters in dataclasses
├── train.py                # Main training & validation loop
├── demo.ipynb              # Interactive demonstration & visualization
├── model/
│   ├── __init__.py
│   ├── yoho.py             # Main model class
│   ├── backbone.py         # Multi-scale feature extractor
│   ├── heads.py            # Detection & segmentation heads
│   └── memory_aug.py       # Continual learning memory component
├── data/
│   ├── audio_dataset.py    # Base dataset + pseudo-SED conversion
│   └── augmentations.py    # Audio-specific augmentations
└── utils/
    ├── metrics.py          # PSDS, event-F1, segmentation IoU
    └── visualize.py        # Spectrogram + predictions plotting
```

## Current Limitations & Roadmap (2026 Q1–Q2)

### Short-term (next 1–3 months)

- Full anchor-free detection head (distribution focal loss + varifocal)
- Proper PSDS evaluation implementation
- Streaming inference mode (chunk-by-chunk processing)
- Better data loaders for DESED / AudioSet / MAESTRO

### Medium-term (3–9 months)

- Frequency-selective auditory attention (cochlear-like filtering)
- Temporal memory with decay & consolidation
- Weakly-supervised / self-supervised pretraining
- ONNX / TensorRT export for edge deployment

### Long-term ideas

- Multi-modal (audio + video) event detection
- Language-guided audio segmentation
- Neuro-symbolic integration for sound source reasoning

## License

MIT License

Feel free to use, modify, contribute — any feedback is very welcome!
