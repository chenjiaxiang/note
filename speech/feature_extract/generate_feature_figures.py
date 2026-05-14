import os

import librosa
import librosa.display
import matplotlib
import numpy as np
from scipy.fftpack import dct, idct

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(__file__)
AUDIO_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../../data/wav/0_jackson_18.wav"))
OUT_DIR = os.path.join(BASE_DIR, "images")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_waveform_and_frame(x, x_pre, sr, frame_plain, frame_win):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    t = np.arange(len(x)) / sr
    axes[0].plot(t, x, linewidth=0.9, label="raw waveform")
    axes[0].plot(t, x_pre, linewidth=0.9, label="pre-emphasis")
    axes[0].set_title("Waveform Before and After Pre-emphasis")
    axes[0].set_xlabel("Time (s)")
    axes[0].legend()

    axes[1].plot(frame_plain, linewidth=0.9, label="frame")
    axes[1].plot(frame_win, linewidth=0.9, label="frame * hamming")
    axes[1].set_title("One Short-time Frame")
    axes[1].set_xlabel("Sample index")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "waveform_and_frame.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_spectrum_pipeline(freqs, power_db, mel_energy, log_mel, mfcc_full):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    axes[0].plot(freqs, power_db, linewidth=1.0)
    axes[0].set_title("Power Spectrum of One Frame")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("dB")
    axes[0].set_xlim(0, freqs[-1])

    axes[1].plot(np.arange(len(mel_energy)), 10 * np.log10(mel_energy + 1e-10), marker="o", linewidth=0.9, label="mel energy (dB)")
    axes[1].plot(np.arange(len(log_mel)), log_mel, marker=".", linewidth=0.9, label="log-mel")
    axes[1].set_title("Mel Filter Bank Output")
    axes[1].set_xlabel("Mel bin")
    axes[1].legend()

    markerline, stemlines, baseline = axes[2].stem(np.arange(len(mfcc_full)), mfcc_full, basefmt=" ")
    plt.setp(markerline, markersize=4)
    plt.setp(stemlines, linewidth=1.0)
    axes[2].set_title("MFCC Coefficients from DCT")
    axes[2].set_xlabel("Cepstral index")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "spectrum_mel_mfcc.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_mfcc_reconstruction(log_mel, recon_5, recon_13, recon_20):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(log_mel, linewidth=2.0, label="original log-mel")
    ax.plot(recon_5, linewidth=1.1, label="recon with 5 coeffs")
    ax.plot(recon_13, linewidth=1.1, label="recon with 13 coeffs")
    ax.plot(recon_20, linewidth=1.1, label="recon with 20 coeffs")
    ax.set_title("Low-order MFCCs Preserve the Spectral Envelope")
    ax.set_xlabel("Mel bin")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "mfcc_reconstruction.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_global_views(S_db, S_mel_db, mfcc_all, sr, hop_length):
    fig = plt.figure(figsize=(12, 10))

    ax1 = plt.subplot(3, 1, 1)
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz", ax=ax1)
    fig.colorbar(ax1.collections[0], ax=ax1, format="%+2.0f dB")
    ax1.set_title("STFT Spectrogram")

    ax2 = plt.subplot(3, 1, 2)
    librosa.display.specshow(S_mel_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel", ax=ax2)
    fig.colorbar(ax2.collections[0], ax=ax2, format="%+2.0f dB")
    ax2.set_title("Mel Spectrogram")

    ax3 = plt.subplot(3, 1, 3)
    librosa.display.specshow(mfcc_all, sr=sr, hop_length=hop_length, x_axis="time", ax=ax3)
    fig.colorbar(ax3.collections[0], ax=ax3)
    ax3.set_title("MFCC Over Time")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "stft_mel_mfcc_overview.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    ensure_dir(OUT_DIR)

    x, sr = librosa.load(AUDIO_PATH, sr=16000, mono=True)

    alpha = 0.97
    x_pre = np.append(x[0], x[1:] - alpha * x[:-1])

    frame_length = int(sr * 25 / 1000)
    hop_length = int(sr * 10 / 1000)
    n_fft = 1024
    n_mels = 40
    n_mfcc = 13

    frames = librosa.util.frame(x_pre, frame_length=frame_length, hop_length=hop_length).copy()
    window = np.hamming(frame_length).astype(np.float32)
    frames_win = frames * window[:, None]
    mid_idx = frames_win.shape[1] // 2
    frame_plain = frames[:, mid_idx]
    frame_win = frames_win[:, mid_idx]

    spec = np.fft.rfft(frame_win, n=n_fft)
    mag = np.abs(spec)
    power = (mag ** 2) / n_fft
    freqs = np.fft.rfftfreq(n_fft, d=1 / sr)
    power_db = 10 * np.log10(power + 1e-10)

    mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0, fmax=sr / 2)
    mel_energy = mel_fb @ power
    log_mel = np.log(mel_energy + 1e-10)

    mfcc_full = dct(log_mel, type=2, norm="ortho")
    recon_5 = idct(np.pad(mfcc_full[:5], (0, n_mels - 5)), type=2, norm="ortho")
    recon_13 = idct(np.pad(mfcc_full[:13], (0, n_mels - 13)), type=2, norm="ortho")
    recon_20 = idct(np.pad(mfcc_full[:20], (0, n_mels - 20)), type=2, norm="ortho")

    S = librosa.stft(x_pre, n_fft=n_fft, hop_length=hop_length, win_length=frame_length, window="hamming")
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    S_mel = librosa.feature.melspectrogram(
        y=x_pre,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=frame_length,
        window="hamming",
        n_mels=n_mels,
        power=2.0,
    )
    S_mel_db = librosa.power_to_db(S_mel, ref=np.max)
    mfcc_all = librosa.feature.mfcc(S=librosa.power_to_db(S_mel, ref=1.0), n_mfcc=n_mfcc)

    save_waveform_and_frame(x, x_pre, sr, frame_plain, frame_win)
    save_spectrum_pipeline(freqs, power_db, mel_energy, log_mel, mfcc_full)
    save_mfcc_reconstruction(log_mel, recon_5, recon_13, recon_20)
    save_global_views(S_db, S_mel_db, mfcc_all, sr, hop_length)


if __name__ == "__main__":
    main()
