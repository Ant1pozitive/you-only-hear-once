# YOHO: You Only Hear Once
**Real-time Audio Event Detection and Instance Segmentation on Spectrograms**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

YOHO is a modern, single-stage architecture for polyphonic **Sound Event Detection (SED)**. Originally inspired by YOLOv8, YOHO has evolved into a hybrid model that combines the local precision of CNNs, the global context of Transformers, and the elegant bipartite matching of DETR to solve audio-specific challenges (like overlapping sounds and variable event durations).


## 🚀 Key Innovations & Features

- **BiPath Conformer Backbone:** A dual-path feature extractor that processes time and frequency patterns separately, enhanced with **Audio Conformer** blocks to capture long-range global dependencies across the spectrogram.
- **BiFPN (Bi-directional Feature Pyramid Network):** Replaces standard top-down FPN. Uses learnable weights for fast normalized fusion, ensuring sharp onset/offset boundaries for short sounds aren't lost in deep layers.
- **NMS-Free Detection (DETR-style Bipartite Matching):** Standard NMS aggressively deletes overlapping predictions, which is disastrous for polyphonic audio. YOHO uses the **Hungarian Matcher** during training, allowing the model to naturally predict unique, distinct events without post-processing.
- **Asymmetric Audio IoU Loss:** Unlike objects in images, time boundaries in audio are far more critical than frequency boundaries. Our custom loss heavily penalizes temporal errors while being lenient on frequency bounds.
- **Polyphonic MixUp & SpecAugment:** Native support for mixing audio waveforms/spectrograms and their bounding boxes in the same batch, forcing the model to learn complex sound polyphony.
- **Model EMA (Exponential Moving Average):** Stabilizes training and significantly boosts generalization on validation/test sets, a standard trick in DCASE winning solutions.

## 🏗 Current Capabilities

- End-to-end mel-spectrogram processing with embedded augmentations.
- Anchor-free detection head with **Distribution Focal Loss (DFL)** and **Varifocal Loss**.
- Streaming inference mode for chunk-by-chunk real-time processing.
- Prototype instance segmentation head (proto + mask coefficients).
- Robust training pipeline with mixed precision (AMP) and gradient clipping.

## 🛠 Installation

```bash
git clone [https://github.com/Ant1pozitive/you-only-hear-once.git](https://github.com/Ant1pozitive/you-only-hear-once.git)
cd you-only-hear-once
pip install -r requirements.txt
# Optional: for better audio processing & faster training
pip install torch torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

```

## 📦 Requirements

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

## 🚀 Training Example

```bash
# Standard training schedule with Model EMA and MixUp enabled by default
python train.py \
  --dataset deSED \
  --batch-size 16 \
  --lr 2e-4 \
  --epochs 120

# Overriding configs via CLI (if implemented in your argparser)
# Ensure your config.py has curriculum learning or augmentations tuned!

```

## 📂 Project Structure

```text
yoho-audio/
├── config.py               # Hyperparameters (Augmentations, BiFPN, EMA configs)
├── train.py                # Main training loop with MixUp & EMA support
├── model/
│   ├── yoho.py             # Main model class (NMS-free inference & SpecAugment)
│   ├── backbone.py         # BiPath ConvNext-style + Conformer + BiFPN
│   ├── heads.py            # Anchor-free head, AudioIoU, Hungarian Matcher
│   └── memory_aug.py       # Continual learning memory component
├── data/
│   ├── audio_dataset.py    # Base dataset + pseudo-SED conversion
│   └── augmentations.py    # Audio-specific augmentations
└── utils/
    ├── metrics.py          # PSDS, event-F1, segmentation IoU
    └── visualize.py        # Spectrogram + predictions plotting

```

## 🗺 Roadmap

### Completed (v2.0 Update)

* [x] Anchor-free detection head (DFL + Varifocal)
* [x] Streaming inference mode
* [x] Bipartite Matching (NMS-Free inference)
* [x] BiFPN multi-scale fusion
* [x] Audio-specific Asymmetric IoU

### Medium-term

* [ ] Proper PSDS evaluation integration within the validation loop
* [ ] Temporal memory with active decay & consolidation
* [ ] Better pre-built data loaders for DESED / AudioSet / MAESTRO

### Long-term ideas

* [ ] Weakly-supervised / self-supervised pretraining (Masked Autoencoders)
* [ ] ONNX / TensorRT export for edge deployment
* [ ] Language-guided audio segmentation (CLAP integration)

## 📜 License

MIT License

Feel free to use, modify, contribute - any feedback is very welcome!
