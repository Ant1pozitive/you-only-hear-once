import matplotlib.pyplot as plt
import numpy as np
import librosa.display

def plot_spectrogram_with_preds(spec: np.ndarray, preds: dict, n_mels: int = 128, save_path: str = None):
    """freq [0,1] -> mel bins"""
    fig, ax = plt.subplots(figsize=(10, 6))
    librosa.display.specshow(spec, ax=ax, x_axis='time', y_axis='mel', sr=44100, hop_length=512)
    for box in preds.get('boxes', []):
        t0, f0, t1, f1 = box.tolist()
        f0_mel = f0 * n_mels
        f1_mel = f1 * n_mels
        ax.add_patch(plt.Rectangle((t0, f0_mel), t1 - t0, f1_mel - f0_mel,
                                   fill=False, color='red', linewidth=2))
    ax.set_title("YOHO Predictions")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
