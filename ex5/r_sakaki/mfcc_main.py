import numpy as np
import librosa
import librosa.display
import scipy
import matplotlib.pyplot as plt


def f_to_mel(f, f0):
    return 1000 * np.log(f / f0 + 1) / np.log(1000 / f0 + 1)


def mel_to_f(m, f0):
    m0 = 1000 / np.log(1000 / f0 + 1)
    return f0 * (np.exp(m / m0) - 1)


def melFilterBank(sr, f0, N, channels_n):
    # ナイキスト周波数(mel)
    fmax = sr / 2
    melmax = f_to_mel(fmax, f0)

    # 周波数インデックスの最大数
    nmax = N // 2
    # 周波数解像度
    df = sr / N

    # 中心周波数
    dmel = melmax / (channels_n + 1)
    melcenters = np.arange(1, channels_n + 1) * dmel
    # hzに変換
    fcenters = mel_to_f(melcenters, f0)
    # 周波数インデックスに変換
    indexcenter = np.round(fcenters / df)

    # 各フィルタの開始位置インデックス
    indexstart = np.hstack(([0], indexcenter[0:channels_n - 1]))
    # 終了位置インデックス
    indexstop = np.hstack((indexcenter[1:channels_n], [nmax]))
    filterbank = np.zeros((channels_n, nmax))
    for c in range(0, channels_n):
        # 三角フィルタの左の直線の傾きから点を求める
        increment = 1.0 / (indexcenter[c] - indexstart[c])
        for i in range(int(indexstart[c]), int(indexcenter[c])):
            filterbank[c, i] = (i - indexstart[c]) * increment
        # 三角フィルタの右の直線の傾きから点を求める
        decrement = 1.0 / (indexstop[c] - indexcenter[c])
        for i in range(int(indexcenter[c]), int(indexstop[c])):
            filterbank[c, i] = 1.0 - ((i - indexcenter[c]) * decrement)

    return filterbank


def calc_mfcc(x, sr, f0, win_length=1024, hop_length=512, mfcc_dim=12):
    xnum = x.shape[0]
    window = np.hamming(win_length)

    mfcc = []
    for i in range(int((xnum - hop_length) / hop_length)):
        # データの切り取り
        tmp = x[i * hop_length: i * hop_length + win_length]
        # 窓関数を適用
        tmp = tmp * window
        # FFTの適用
        tmp = np.fft.rfft(tmp)
        # パワースペクトルの取得
        tmp = np.abs(tmp)
        tmp = tmp[:win_length//2]

        # フィルタバンク
        channels_n = 20
        filterbank = melFilterBank(sr, f0, win_length, channels_n)
        # フィルタバンクの適用
        tmp = np.dot(filterbank, tmp)
        # log
        tmp = 20 * np.log10(tmp)
        # 離散コサイン変換
        tmp = scipy.fftpack.dct(tmp, norm='ortho')
        # リフタの適用
        tmp = tmp[1:mfcc_dim+1]

        mfcc.append(tmp)

    mfcc = np.transpose(mfcc)
    return mfcc


def delta(input, mfcc_dim, k):
    output = np.zeros((mfcc_dim, input.shape[1] - 2 * k))
    for i in range(input.shape[1] - 2 * k):
        tmp = input[:, i: i + 2 * k + 1]
        delta = [0] * mfcc_dim
        for j in range(mfcc_dim):
            a, _ = np.polyfit(np.arange(-k, k + 1, 1), tmp[j:j + 1, :][0], 1)
            delta[j] = a
        delta = np.array(delta).reshape(mfcc_dim, -1)
        output[:, i:i + 1] = delta

    return output


def main():
    data, fs = librosa.load("sample.mp3", mono=True)

    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(hspace=0.3)

    # パラメータ設定
    f0 = 1000
    window_length = 512
    hop_length = 256

    # スペクトログラム取得
    spectrogram = \
        librosa.stft(data, win_length=window_length, hop_length=hop_length)
    spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram))

    # スペクトログラム表示
    ax0 = fig.add_subplot(411)
    img = librosa.display.specshow(
        spectrogram_db,
        y_axis="log",
        sr=fs,
        cmap="rainbow",
        ax=ax0
        )
    ax0.set(
        title="Spectrogram",
        ylabel="frequency [Hz]"
        )
    fig.colorbar(
        img,
        aspect=10,
        pad=0.01,
        extend="both",
        ax=ax0,
        format="%+2.f dB"
        )

    # mfcc 表示
    mfcc_dim = 12
    ax1 = fig.add_subplot(412)
    mfcc = calc_mfcc(
        data, fs, f0,
        win_length=window_length,
        hop_length=hop_length,
        mfcc_dim=mfcc_dim
        )
    wav_time = data.shape[0] // fs
    extent = [0, wav_time, 0, mfcc_dim]
    img1 = ax1.imshow(
        np.flipud(mfcc),
        aspect="auto",
        extent=extent,
        cmap="rainbow"
        )
    ax1.set(
        title="MFCC sequence",
        ylabel="MFCC",
        yticks=range(0, 13, 2)
        )
    fig.colorbar(
        img1,
        aspect=10,
        pad=0.01,
        extend="both",
        ax=ax1,
        format="%+2.f dB"
        )

    # Δmfcc 表示
    ax2 = fig.add_subplot(413)
    dmfcc = delta(mfcc, mfcc_dim, k=2)
    img2 = ax2.imshow(
        np.flipud(dmfcc),
        aspect="auto",
        extent=extent,
        cmap="rainbow"
        )
    ax2.set(
        title="ΔMFCC sequence",
        ylabel="ΔMFCC",
        yticks=range(0, 13, 2)
        )
    fig.colorbar(
        img2,
        aspect=10,
        pad=0.01,
        extend="both",
        ax=ax2,
        format="%+2.f dB"
        )

    # ΔΔmfcc 表示
    ax3 = fig.add_subplot(414)
    ddmfcc = delta(dmfcc, mfcc_dim, k=2)
    img3 = ax3.imshow(
        np.flipud(ddmfcc),
        aspect="auto",
        extent=extent,
        cmap="rainbow"
        )
    ax3.set(
        title="ΔΔMFCC sequence",
        xlabel="time[s]",
        ylabel="ΔΔMFCC",
        yticks=range(0, 13, 2)
        )
    fig.colorbar(img3,
                 aspect=10,
                 pad=0.01,
                 extend="both",
                 ax=ax3,
                 format="%+2.f dB"
                 )

    fig.tight_layout()
    fig.savefig("mfcc.png")


if __name__ == "__main__":
    main()
