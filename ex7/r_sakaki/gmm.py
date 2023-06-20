import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
import os
from matplotlib import cm
from matplotlib.animation import FuncAnimation as animation


def load_file(path):
    with open(path) as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
    data = np.float_(np.array(rows))
    # データを返す
    return data


def get_gaussian(x, mu, sigma):
    dimension = x.shape[0]
    gauss_numerator = np.exp(
        - 1 / 2 * (x - mu).T @ np.linalg.inv(sigma) @ (x - mu)
        )
    gauss_denominator = np.power(2 * np.pi, dimension / 2) * \
        np.power(np.linalg.det(sigma), 1 / 2)
    gaussian_value = gauss_numerator / gauss_denominator
    # ガウス分布の確率密度関数を返す
    return gaussian_value


def get_mixture_gaussian(X, Mu, Sigma, Pi):
    num_gauss = Pi.shape[0]
    num_data = X.shape[0]
    mixture_gaussian = np.zeros((num_gauss, num_data))
    for k in range(num_gauss):
        mixture_gaussian[k] = [Pi[k] * get_gaussian(x, Mu[k], Sigma[k]) for x in X]
    # 混合ガウス分布の確率密度関数を返す
    return mixture_gaussian


def log_likelihood(data, Mu, Sigma, Pi):
    loglikelihood = np.sum(np.log(np.sum(get_mixture_gaussian(data, Mu, Sigma, Pi), axis=0)))
    # 対数尤度関数の値を返す
    return loglikelihood


def em_algorithm(data, Mu, Sigma, Pi):
    # Eステップ
    mixture_gaussian = get_mixture_gaussian(data, Mu, Sigma, Pi)
    gamma = mixture_gaussian / np.sum(mixture_gaussian, axis=0)[np.newaxis, :]
    # Mステップ
    N_k = np.sum(gamma, axis=1)
    # mu更新
    Mu = (gamma @ data) / N_k[:, np.newaxis]
    # sigma更新
    deviation = data - Mu[:, np.newaxis, :]
    for k in range(len(N_k)):
        Sigma[k] = gamma[k] * deviation[k].T @ deviation[k]
    Sigma /= N_k[:, np.newaxis, np.newaxis]
    # pi更新
    N = data.shape[0]
    Pi = N_k / N
    # 各パラメータを返す
    return Mu, Sigma, Pi


def gmm(data, Mu, Sigma, Pi, epsilon):
    # 初期パラメータでの対数尤度関数計算
    loglikelihood = [log_likelihood(data, Mu, Sigma, Pi)]

    while True:
        # EMステップ
        Mu, Sigma, Pi = em_algorithm(data, Mu, Sigma, Pi)
        # 対数尤度関数再計算
        loglikelihood.append(log_likelihood(data, Mu, Sigma, Pi))
        # 収束判定
        if np.abs(loglikelihood[-1] - loglikelihood[-2]) < epsilon:
            break

    return loglikelihood, Mu, Sigma, Pi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="the path to input data")
    args = parser.parse_args()

    path = args.path
    filename = os.path.splitext(os.path.basename(path))[0]

    # データの取得
    data = load_file(path)
    dimension = data.shape[1]

    # 初期パラメータ設定
    num_gauss = 3
    Mu_init = np.random.randn(num_gauss, dimension)
    Sigma_init = np.array([np.eye(dimension) for i in range(num_gauss)])
    Pi_init = np.ones(num_gauss) / num_gauss
    epsilon = 0.00001

    # gmm実行
    loglikelihood, Mu, Sigma, Pi = gmm(data, Mu_init, Sigma_init, Pi_init, epsilon)

    # グラフ表示
    # 1次元データ
    if dimension == 1:
        fig1 = plt.figure(figsize=(18, 10))
        # 対数尤度関数表示
        ax1 = fig1.add_subplot(121)
        ax1.plot(np.arange(0, len(loglikelihood), 1),
                 loglikelihood,
                 color="green"
                 )
        ax1.set(
            xlabel="Iteration",
            ylabel="Log likelihood",
            title=f"log_likelihood {filename}"
            )
        # gmmの表示
        ax2 = fig1.add_subplot(122)
        # データ
        ax2.scatter(
            data[:, 0],
            np.zeros(data.shape[0]),
            color="blue",
            label="observed data",
            marker="$○$"
        )
        # セントロイド
        ax2.scatter(
            Mu[:, 0],
            np.zeros(Mu.shape[0]),
            color="red",
            label="Centroid",
            marker="$×$"
        )
        # 混合ガウス分布
        X = np.linspace(
            np.min(data[:, 0]),
            np.max(data[:, 0]),
            100
            )[:, np.newaxis]
        pdf = np.sum(get_mixture_gaussian(X, Mu, Sigma, Pi), axis=0)
        ax2.plot(X, pdf, label="GMM", color="green")
        ax2.set(
            xlabel="x",
            ylabel="Probability density",
            xlim=(np.min(data[:, 0]) - 1, np.max(data[:, 0]) + 1),
            title=f"Probability density of {filename} (K = {num_gauss})",
        )
        ax2.grid()
        fig1.savefig(f"result_{filename}_K{num_gauss}.png")

    # 2次元データ
    if dimension == 2:
        fig1 = plt.figure(figsize=(18, 10))
        # 対数尤度関数表示
        ax1 = fig1.add_subplot(121)
        ax1.plot(
            np.arange(0, len(loglikelihood), 1),
            loglikelihood,
            color="green"
            )
        ax1.set(
            xlabel="Iteration",
            ylabel="Log likelihood",
            title=f"log_likelihood {filename}"
            )
        # gmmの等高線表示
        ax2 = fig1.add_subplot(122)
        # データ
        ax2.scatter(
            data[:, 0],
            data[:, 1],
            label="observed data",
            marker="$○$",
        )
        # セントロイド
        ax2.scatter(
            Mu[:, 0],
            Mu[:, 1],
            color="red",
            label="Centroid",
            marker="$×$",
        )
        ax2.set(
            xlim=(np.min(data[:, 0]) - 1, np.max(data[:, 1]) + 1),
            ylim=(np.min(data[:, 1]) - 1, np.max(data[:, 1]) + 1),
            title=f"contour map of {filename} (K = {num_gauss})"
        )
        # 混合ガウス分布
        x1 = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 80)
        x2 = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 80)
        X1, X2 = np.meshgrid(x1, x2)
        X = np.c_[X1.ravel(), X2.ravel()]
        pdf = np.sum(get_mixture_gaussian(X, Mu, Sigma, Pi), axis=0)
        pdf = pdf.reshape(X1.shape)
        cset = ax2.contour(X1, X2, pdf, cmap=cm.jet)
        ax2.clabel(cset, fmt='%1.1f', fontsize=9)
        fig1.savefig(f"result_{filename}_K{num_gauss}.png")

        # gmmのアニメーション表示
        fig2 = plt.figure(figsize=(10, 10))
        ax2 = fig2.add_subplot(111, projection="3d")

        def plot_graph():
            ax2.scatter(
                data[:, 0],
                data[:, 1],
                np.zeros(data.shape[0]),
                color="red",
                label="observed data",
                marker="$○$",
            )
            ax2.scatter(
                Mu[:, 0],
                Mu[:, 1],
                np.zeros(Mu.shape[0]),
                color="orange",
                label="Centroid",
                marker="$×$",
            )
            ax2.plot_surface(
                X1,
                X2,
                pdf,
                label="GMM",
                rstride=1,
                cstride=1,
                cmap=cm.coolwarm
            )

        def plt_graph3d(angle):
            ax2.view_init(azim=angle*5)

        ani = animation(
            fig2,
            func=plt_graph3d,
            frames=72,
            init_func=plot_graph,
            interval=300
        )
        ax2.set(
            xlabel="x1",
            ylabel="x2",
            zlabel="Probability density",
            title=f"Probability density of {filename} (K = {num_gauss})"
        )
        ax2.grid()
        ani.save(f"result_{filename}_K{num_gauss}_ani.gif", writer="pillow")


if __name__ == "__main__":
    main()
