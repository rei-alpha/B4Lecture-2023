import numpy as np
import matplotlib.pyplot as plt
import librosa


def sinc(x):
    return np.where(x == 0, 1, np.sin(x)/x)


def load_file(filename):
    data, sr = librosa.load(filename, sr=None)
    return data, sr


def bef(fs, f_cut_low, f_cut_high, fil_size):
    # 角周波数へ変換
    w_cut_low = 2 * np.pi * f_cut_low / fs
    w_cut_high = 2 * np.pi * f_cut_high / fs

    # 時間領域でのbefフィルタ
    n_arange = np.arange(-fil_size//2, fil_size//2 + 1)
    filter = (
            sinc(np.pi*n_arange)
            + (w_cut_low*sinc(w_cut_low*n_arange)
                - w_cut_high*sinc(w_cut_high*n_arange))/np.pi
            )

    # 窓関数作成
    window = np.hamming(fil_size + 1)

    return filter * window


def conv(a, v):
    # 結果保持用
    result = np.zeros(len(a)+len(v))

    # 畳み込み演算
    for i in range(len(v)):
        result[i: i+len(a)] += a * v[i]

    return result


def main():
    # 音声の読み込み
    data, fs = load_file("sample.wav")

    # パラメータ設定
    f_cut_low = 1000
    f_cut_high = 10000
    fil_size = 255

    # フィルタ作成
    filter = bef(fs, f_cut_low, f_cut_high, fil_size)

    # 周波数特性の取得
    filter_fft = np.fft.rfft(filter)    # フーリエ変換
    amplitude = np.abs(filter_fft)      # 振幅取得
    amplitude_db = librosa.amplitude_to_db(amplitude)   # デシベル変換
    phase = np.unwrap(np.angle(filter_fft))             # 位相取得
    frequency = np.arange(0, fs/2, (fs//2)/(fil_size//2+1))  # 周波数取得

    # フィルタの周波数特性画像の保存
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.9)
    ax[0].plot(frequency[1:fil_size//2+1], amplitude_db[1:fil_size//2+1])
    ax[1].plot(frequency[1:fil_size//2+1], phase[1:fil_size//2+1])
    fig.savefig("compare_frequency.png")

    # フィルタの畳み込み
    data_filtered = conv(data, filter)

    # スペクトログラムの比較画像を保存
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.9)
    # オリジナル波形のスペクトログラム
    data_stft = librosa.stft(data)
    spectrogram, phase = librosa.magphase(data_stft)
    spectrogram_db = librosa.amplitude_to_db(spectrogram)
    img = librosa.display.specshow(
        spectrogram_db,
        sr=fs,
        ax=ax[0],
        x_axis="time",
        y_axis="log",
        cmap="plasma",
    )
    ax[0].set(
            title="Spectrogram",
            xlabel="Time [sec]",
            ylabel="Freq [Hz]"
            )
    fig.colorbar(img, ax=ax[0], format="%+2.f dB")
    # フィルタ適用後の波形のスペクトログラム
    data_filtered_stft = librosa.stft(data_filtered)
    spectrogram_filtered, phase = librosa.magphase(data_filtered_stft)
    spectrogram_filtered_db = librosa.amplitude_to_db(spectrogram_filtered)
    img = librosa.display.specshow(
        spectrogram_filtered_db,
        sr=fs,
        ax=ax[1],
        x_axis="time",
        y_axis="log",
        cmap="plasma",
    )
    ax[1].set(
            title="Spectrogram",
            xlabel="Time [sec]",
            ylabel="Freq [Hz]"
            )
    fig.colorbar(img, ax=ax[1], format="%+2.f dB")
    fig.savefig("compare_spectrogram.png")

    # 波形の比較画像の保存
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.9)
    # オリジナル波形
    librosa.display.waveshow(data, sr=fs, ax=ax[0])
    ax[0].set(title="Original Signal", ylabel="Amplitude")
    # フィルタ適用後の波形
    librosa.display.waveshow(data_filtered, sr=fs, ax=ax[1])
    ax[1].set(title="Filtered Signal", ylabel="Amplitude")
    fig.savefig("compare_wave.png")


if __name__ == '__main__':
    main()
