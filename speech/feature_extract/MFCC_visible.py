import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.fftpack import dct

# -----------------------------
# 0) 读音频
# -----------------------------
audio_path = "/root/blog/note/data/wav/0_jackson_18.wav"  # TODO: 改成你的路径
x, sr = librosa.load(audio_path, sr=16000, mono=True)

# -----------------------------
# 1) 预加重
# -----------------------------
alpha = 0.97
x_pre = np.append(x[0], x[1:] - alpha * x[:-1])

# -----------------------------
# 2) 分帧 + 加窗
# -----------------------------
frame_length_ms = 25
hop_length_ms = 10
frame_length = int(sr * frame_length_ms / 1000)
hop_length = int(sr * hop_length_ms / 1000)

# librosa.util.frame 生成 (frame_length, num_frames) 的视图
frames = librosa.util.frame(x_pre, frame_length=frame_length, hop_length=hop_length).copy()

# Hamming window
window = np.hamming(frame_length).astype(np.float32)
frames_win = frames * window[:, None]

# 选一帧来做“逐步可视化”（取中间那帧，比较稳）
mid_idx = frames_win.shape[1] // 2
frame_t = frames[:, mid_idx]
frame_tw = frames_win[:, mid_idx]

# -----------------------------
# 3) FFT -> 幅度谱/功率谱
# -----------------------------
n_fft = 1024  # 常用 >= frame_length 的2次幂
spec = np.fft.rfft(frame_tw, n=n_fft)
mag = np.abs(spec)
power = (mag ** 2) / n_fft
freqs = np.fft.rfftfreq(n_fft, d=1/sr)

# dB 版本更好看（避免 log(0)）
eps = 1e-10
mag_db = 20 * np.log10(mag + eps)
power_db = 10 * np.log10(power + eps)

# -----------------------------
# 4) Mel filterbank + log
# -----------------------------
n_mels = 40
mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0, fmax=sr/2)
mel_energy = mel_fb @ power  # (n_mels,)
log_mel = np.log(mel_energy + eps)

# -----------------------------
# 5) DCT -> MFCC
#    经典 MFCC 是对 log-mel 做 DCT-II，然后取前 12/13 维
# -----------------------------
mfcc_full = dct(log_mel, type=2, norm="ortho")  # (n_mels,)
n_mfcc = 13
mfcc = mfcc_full[:n_mfcc]

# 为了验证“低阶=慢变化(包络)”：做一个只保留低阶系数的重建
def idct_reconstruct_from_mfcc(mfcc_coeffs, n_mels_total):
    """用部分 DCT 系数重建 log-mel（其余补0），看低阶/高阶的含义"""
    coeff = np.zeros(n_mels_total, dtype=np.float32)
    coeff[:len(mfcc_coeffs)] = mfcc_coeffs
    # scipy 的 idct：type=2 对应逆一般用 type=3；norm="ortho" 配对是可逆的
    from scipy.fftpack import idct
    return idct(coeff, type=2, norm="ortho")

log_mel_recon_13 = idct_reconstruct_from_mfcc(mfcc, n_mels)
log_mel_recon_5  = idct_reconstruct_from_mfcc(mfcc_full[:5], n_mels)
log_mel_recon_20 = idct_reconstruct_from_mfcc(mfcc_full[:20], n_mels)

# -----------------------------
# 6) 再做一个“整段的谱图”用于观察谐波 vs 包络
# -----------------------------
S = librosa.stft(x_pre, n_fft=n_fft, hop_length=hop_length, win_length=frame_length, window="hamming")
S_mag = np.abs(S)
S_db = librosa.amplitude_to_db(S_mag, ref=np.max)

S_mel = librosa.feature.melspectrogram(
    y=x_pre, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=frame_length,
    window="hamming", n_mels=n_mels, power=2.0
)
S_mel_db = librosa.power_to_db(S_mel, ref=np.max)

MFCC_all = librosa.feature.mfcc(S=librosa.power_to_db(S_mel, ref=1.0), n_mfcc=n_mfcc)

# -----------------------------
# 7) 画图
# -----------------------------
plt.figure(figsize=(14, 10))

# (A) 波形：原始 vs 预加重
ax1 = plt.subplot(3, 2, 1)
t = np.arange(len(x)) / sr
ax1.plot(t, x, linewidth=0.8, label="raw")
ax1.plot(t, x_pre, linewidth=0.8, label="pre-emphasis")
ax1.set_title("Waveform (raw vs pre-emphasis)")
ax1.set_xlabel("Time (s)")
ax1.legend()

# (B) 单帧：未加窗 vs 加窗
ax2 = plt.subplot(3, 2, 2)
ax2.plot(frame_t, linewidth=0.8, label="frame (no window)")
ax2.plot(frame_tw, linewidth=0.8, label="frame * Hamming")
ax2.set_title("One frame (framing + windowing)")
ax2.set_xlabel("Sample")
ax2.legend()

# (C) 单帧频谱：幅度/功率（dB）
ax3 = plt.subplot(3, 2, 3)
ax3.plot(freqs, mag_db, linewidth=0.9)
ax3.set_title("Magnitude spectrum of the frame (dB)")
ax3.set_xlabel("Frequency (Hz)")
ax3.set_ylabel("dB")
ax3.set_xlim(0, sr/2)

ax4 = plt.subplot(3, 2, 4)
ax4.plot(freqs, power_db, linewidth=0.9)
ax4.set_title("Power spectrum of the frame (dB)")
ax4.set_xlabel("Frequency (Hz)")
ax4.set_ylabel("dB")
ax4.set_xlim(0, sr/2)

# (D) Mel + log + DCT（单帧）
ax5 = plt.subplot(3, 2, 5)
ax5.plot(np.arange(n_mels), 10*np.log10(mel_energy + eps), marker="o", linewidth=0.9, label="mel energy (dB)")
ax5.plot(np.arange(n_mels), log_mel, marker=".", linewidth=0.9, label="log-mel (natural log)")
ax5.set_title("Mel filterbank energies (frame) & log-mel")
ax5.set_xlabel("Mel bin")
ax5.legend()

ax6 = plt.subplot(3, 2, 6)
ax6.stem(np.arange(n_mels), mfcc_full, basefmt=" ", linefmt="C0-", markerfmt="C0o")
ax6.set_title("DCT of log-mel = MFCC coefficients (frame)")
ax6.set_xlabel("Cepstral index")
plt.tight_layout()
plt.show()

# -----------------------------
# 8) “低阶=包络”验证：用不同数量的 MFCC 重建 log-mel
# -----------------------------
plt.figure(figsize=(14, 4))
plt.plot(log_mel, linewidth=2, label="original log-mel")
plt.plot(log_mel_recon_5,  linewidth=1.2, label="recon from 5 coeffs (very smooth)")
plt.plot(log_mel_recon_13, linewidth=1.2, label="recon from 13 coeffs")
plt.plot(log_mel_recon_20, linewidth=1.2, label="recon from 20 coeffs (more detail)")
plt.title("Reconstruct log-mel from low-order MFCCs (shows 'slow vs fast variation')")
plt.xlabel("Mel bin")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 9) 整段可视化：STFT vs Mel-spectrogram vs MFCC
# -----------------------------
plt.figure(figsize=(14, 10))

ax1 = plt.subplot(3, 1, 1)
librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz")
plt.colorbar(format="%+2.0f dB")
plt.title("STFT magnitude spectrogram (dB) - look for harmonic stacks")

ax2 = plt.subplot(3, 1, 2)
librosa.display.specshow(S_mel_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel")
plt.colorbar(format="%+2.0f dB")
plt.title("Mel spectrogram (dB) - harmonics get smoothed, envelope clearer")

ax3 = plt.subplot(3, 1, 3)
librosa.display.specshow(MFCC_all, sr=sr, hop_length=hop_length, x_axis="time")
plt.colorbar()
plt.title("MFCC (time x cepstral) - low indices track spectral envelope dynamics")

plt.tight_layout()
plt.show()

print("Done.")
print(f"sr={sr}, frame_length={frame_length}, hop_length={hop_length}, n_fft={n_fft}, n_mels={n_mels}, n_mfcc={n_mfcc}")
print("Example frame MFCC(0..12):", mfcc)