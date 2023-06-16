import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
import os
from matplotlib.animation import FuncAnimation as animation


class PCA:
    def __init__(self, num_eigen=None):
        self.num_eigen = num_eigen
        self.eigen_value = None
        self.eigen_vector = None
        self.W = None

    def standardize(self, x):
        standardized_x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
        # 標準化したデータを返す
        return standardized_x

    def fit(self, data):
        # データの標準化
        standardized_data = self.standardize(data)
        # データの分散共分散行列の取得
        v_cov = np.cov(standardized_data.T)
        # 分散共分散行列の固有値と固有ベクトルの取得
        self.eigen_value, self.eigen_vector = np.linalg.eig(v_cov)
        # 固有値・固有ベクトルを降順ソート
        sorted_index = np.argsort(self.eigen_value)[::-1]
        self.eigen_value = self.eigen_value[sorted_index]
        self.eigen_vector = self.eigen_vector[sorted_index]
        # 上位k(num_eigen)個の固有ベクトルから射影行列の作成
        self.W = self.eigen_vector[:, :self.num_eigen]
        # 射影行列を返す
        return self.W

    def compress(self, x):
        # 圧縮したデータを返す
        return x @ self.W

    def get_contribution_rate(self):
        contribution_rate = self.eigen_value / np.sum(self.eigen_value)
        cumulative_contribution_rate = np.cumsum(contribution_rate)
        # 寄与率、累積寄与率を返す
        return contribution_rate, cumulative_contribution_rate


def load_file(path):
    with open(path) as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
    data = np.float_(np.array(rows))
    # データを返す
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="the path to input data")
    args = parser.parse_args()

    path = args.path
    filename = os.path.splitext(os.path.basename(path))[0]

    # データの取得
    data = load_file(path)
    dimension = data.shape[1]

    # pca実行
    model = PCA()
    W = model.fit(data)
    compressed_data = model.compress(data)
    con_rate, cum_con_rate = model.get_contribution_rate()

    # 寄与率の表示
    for i in range(dimension):
        print(f"element {i+1}: {con_rate[i]:.6f}")

    if dimension == 2:
        fig1 = plt.figure(figsize=(10, 10))
        ax1 = fig1.add_subplot(111)
        ax1.scatter(
            data[:, 0], data[:, 1], color="magenta",
            label="observed data", marker="$○$"
        )
        ax1.axline(
            (0, 0), W[0], color="blue",
            label=f"Contribution rate: {con_rate[0]:.3f}"
        )
        ax1.axline(
            (0, 0), W[1], color="green",
            label=f"Contribution rate: {con_rate[1]:.3f}"
        )
        ax1.set(
            xlabel="x1",
            ylabel="x2",
            title=f"principal components analysis to {filename}",
        )
        ax1.legend(loc="upper left")
        ax1.grid()
        fig1.savefig(f"result_{filename}.png")

    elif dimension == 3:
        # 2次元圧縮したデータの表示
        fig1 = plt.figure(figsize=(10, 10))
        ax1 = fig1.add_subplot(111)
        ax1.scatter(
            compressed_data[:, 0], compressed_data[:, 1], color="magenta",
            label="compressed data", marker="$○$"
        )
        ax1.set(
            xlabel="x",
            ylabel="y",
            title=f"compressed data of {filename}"
        )
        ax1.legend(loc="upper left")
        ax1.grid()
        fig1.savefig(f"result_compressed_{filename}.png")

        # 3次元の散布図表示
        fig2 = plt.figure(figsize=(10, 10))
        ax2 = fig2.add_subplot(111, projection="3d")
        slope_xy = W[1] / W[0]
        slope_yz = W[2] / W[0]
        x = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 100)

        def animation_init():
            ax2.scatter(
                data[:, 0], data[:, 1], data[:, 2],
                c="magenta", label="observed data", marker="$○$"
            )
            ax2.plot(
                x,
                slope_xy[0] * x,
                slope_yz[0] * x,
                color="blue",
                label=f"Contribution rate:{con_rate[0]:.3f}",
            )
            ax2.plot(
                x,
                slope_xy[1] * x,
                slope_yz[1] * x,
                color="green",
                label=f"Contribution rate:{con_rate[1]:.3f}",
            )
            ax2.plot(
                x,
                slope_xy[2] * x,
                slope_yz[2] * x,
                color="red",
                label=f"Contribution rate:{con_rate[2]:.3f}",
            )
            ax2.view_init(elev=0, azim=60, roll=5)
            return fig2,

        rotate_elev = np.linspace(0, 180, 50, endpoint=False)
        rotate_azim = np.linspace(60, 420, 120)

        def animate(i):
            ax2.view_init(elev=rotate_elev[i], azim=rotate_azim[i], roll=5)
            return fig2,

        ani = animation(
            fig2,
            func=animate,
            init_func=animation_init,
            frames=101,
            interval=100,
            blit=True,
            repeat=False,
        )
        ax2.set_title(f"principal component analysis to {filename}")
        ax2.legend()
        ani.save(f"result_pca_{filename}.gif", writer="pillow")

    else:
        x = np.linspace(0, dimension, dimension)
        order = np.min(np.where(cum_con_rate >= 0.9))
        fig1 = plt.figure(figsize=(10, 10))
        ax1 = fig1.add_subplot(111)
        ax1.plot(
            x, cum_con_rate, color="blue",
            label="cumulative contribution rate"
        )
        ax1.axhline(y=0.9, color="red", label="threshold 90%")
        ax1.axvline(x=order, color="orange", label=f"order={order}")
        ax1.set(
            xlabel="number of principal components",
            ylabel="contribution rate",
            xlim=[0, dimension],
            title=f"cumulative contribution rate of {filename}"
        )
        ax1.legend(loc="upper left")
        ax1.grid()
        fig1.savefig(f"result_contribution_rate_{filename}.png")


if __name__ == "__main__":
    main()
