import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy


def load_file(filename):
    y, sr, = librosa.load(filename, sr=None)

    cut = int(len(y) * 0.75)
    y = y[:cut]

    t = len(y) / sr

    return sr, y, t


def autocorrelation(windowed_data):
    # 自己相関関数の計算
    N = len(windowed_data)
    r_ac = np.zeros(N)
    for m in range(N):
        for n in range(N - m):
            r_ac[m] += windowed_data[n] * windowed_data[n + m]

    return r_ac


def estimate_f0_by_ac(data, fs, window_size, overlap):
    shift_times = int((data.shape[0] - overlap) // (window_size - overlap))
    f0 = np.zeros(shift_times)
    window = np.hamming(window_size)

    for t in range(shift_times):
        windowed_data = data[t * overlap: t * overlap + window_size] * window
        # 自己相関関数
        r = autocorrelation(windowed_data)
        # ピーク時の m0 取得
        m0 = get_peak(r)
        if m0 == 0:
            f0[t] = 0
        else:
            f0[t] = fs / m0

    return f0


def cepstrum(windowed_data):
    data_fft = np.fft.fft(windowed_data)
    log_power = 20 * np.log10(np.abs(data_fft))
    cepstrum = np.real(np.fft.ifft(log_power))

    return cepstrum


def estimate_f0_by_cep(data, fs, window_size, overlap, lifter_cutoff=20):
    shift_times = int((data.shape[0] - overlap) // (window_size - overlap))
    f0 = np.zeros(shift_times)
    window = np.hamming(window_size)
    lifter = np.ones(window_size)
    lifter[0: lifter_cutoff] = 0

    for t in range(shift_times):
        windowed_data = data[t * overlap: t * overlap + window_size] * window
        # ケプストラム
        r = cepstrum(windowed_data)
        # リフタリング
        liftered_r = r * lifter
        # ピーク時の m0 取得
        m0 = get_peak(liftered_r)
        if m0 == 0:
            f0[t] = 0
        else:
            f0[t] = fs / m0

    return f0


def get_peak(r):
    peak = np.zeros(r.shape[0] - 2)
    for i in range(peak.shape[0]):
        if r[i] < r[i + 1] and r[i + 1] > r[i + 2]:
            peak[i] = r[i + 1]
    m0 = np.argmax(peak)

    return m0


def envelope_by_cepstrum(data, lifter_cutoff=30):
    r = cepstrum(data)
    lifter = np.ones(len(r))
    lifter[lifter_cutoff: len(r) - lifter_cutoff + 1] = 0
    liftered_r = r * lifter
    envelope_cep = np.real(np.fft.fft(liftered_r))

    return envelope_cep


def envelope_by_lpc(data, dim, n_freq):
    data = data * np.hamming(len(data))
    r = autocorrelation(data)
    a, e = levinson_durbin(r, dim)
    w, h = scipy.signal.freqz(np.sqrt(e), a, n_freq, whole=True)
    envelope_lcp = 20 * np.log10(np.abs(h))

    return envelope_lcp


def levinson_durbin(r, dim):
    a = np.zeros(dim + 1)
    a[0] = 1
    a[1] = - r[1] / r[0]
    e = r[0] + r[1] * a[1]

    for i in range(1, dim):
        w = np.sum(a[: i + 1] * r[i + 1: 0: -1])
        k = w / e
        tmp1 = a[0: i + 2]
        tmp2 = tmp1[:: -1]
        a[0: i + 2] = tmp1 - k * tmp2
        e -= k * w

    return a, e


def main():
    # 音声の読込
    fs, data, t = load_file("sample.mp3")

    # パラメータ
    window_size = 512
    overlap = 256
    dimension = 32

    # グラフ準備
    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(hspace=0.7)
    fig.subplots_adjust(wspace=0.4)

    # スペクトログラム作成用データ
    data_stft = librosa.stft(data)
    spectrogram, phase = librosa.magphase(data_stft)
    spectrogram_db = librosa.amplitude_to_db(spectrogram)

    # 基本周波数推定
    # 自己相関法による f0 表示
    f0_ac = estimate_f0_by_ac(data, fs, window_size, overlap)
    ax1 = fig.add_subplot(221)
    img = librosa.display.specshow(
        spectrogram_db,
        sr=fs,
        ax=ax1,
        x_axis="time",
        y_axis="log",
        cmap="plasma",
    )
    ax1.set(title="Spectrogram",
            xlabel="Time [sec]",
            ylabel="Freq [Hz]")
    fig.colorbar(img, ax=ax1, format="%+2.f dB")
    ax1.plot(np.arange(0, t, t / len(f0_ac)), f0_ac,
             color="black", label="f0 by autocorrelation")
    plt.legend()

    # ケプストラム法による f0 表示
    f0_cep = estimate_f0_by_cep(data, fs, window_size, overlap)
    ax2 = fig.add_subplot(222)
    img = librosa.display.specshow(
        spectrogram_db,
        sr=fs,
        ax=ax2,
        x_axis="time",
        y_axis="log",
        cmap="plasma",
    )
    ax2.set(title="Spectrogram",
            xlabel="Time [sec]",
            ylabel="Freq [Hz]",
            label="f0 by cepstrum")
    fig.colorbar(img, ax=ax2, format="%+2.f dB")
    ax2.plot(np.arange(0, t, t / len(f0_cep)), f0_cep,
             color="black", label="f0 by cepstrum")
    plt.legend()

    # スペクトル包絡の比較
    frame_length = 2048
    start = int(len(data) * 0.2)
    target_data = data[start: start + frame_length]

    # スペクトル表示用データ
    fft_data = np.fft.fft(target_data)
    spectrum = 20 * np.log10(np.abs(fft_data))
    f = np.fft.fftfreq(frame_length, d=1.0/fs)

    # スペクトラム包絡の表示
    wav_cepstrum = envelope_by_cepstrum(target_data)
    wav_lpc = envelope_by_lpc(target_data, dimension, frame_length)
    ax2 = fig.add_subplot(223)
    ax2.set(title="Spectrum envelope",
            xlabel="Freq [Hz]",
            ylabel="log amplitude spectrum [dB]")
    ax2.plot(f[: len(f) // 2], spectrum[: frame_length // 2],
             color="blue", label="Spectrum")
    ax2.plot(f[: len(f) // 2], wav_cepstrum[: frame_length // 2],
             color="lime", label="Cepstrum")
    ax2.plot(f[: len(f) // 2], wav_lpc[: frame_length // 2],
             color="red", label="LPC")
    plt.legend()

    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
