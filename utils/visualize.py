import matplotlib.pyplot as plt
import numpy as np
import librosa.display

def plot_spectrogram_with_preds(spec: np.ndarray, preds: dict, save_path: str = None):
    fig, ax = plt.subplots()
    librosa.display.specshow(spec, ax=ax, x_axis='time', y_axis='mel')
    # Draw boxes/masks
    for box in preds.get('boxes', []):
        ax.add_patch(plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], fill=False, color='r'))
    if save_path:
        plt.savefig(save_path)
    plt.close()
