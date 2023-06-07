import matplotlib.pyplot as plt
import numpy as np
import csv


def load_file(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
    header = rows.pop(0)
    data = np.float_(np.array(rows))

    return header, data


def k_means(header, data, cluster_n):
    data_length = data.shape[0]
    dimension = data.shape[1]

    # データからランダムな点をセントロイドとする
    centroids_index = np.random.randint(0, data_length, cluster_n)
    centroids = data[centroids_index]

    distance = np.zeros((data_length, cluster_n))
    while True:
        # 距離の計算
        for i in range(data_length):
            distance[i] = np.sum(np.power(data[i] - centroids, 2), axis=1)
        # 属するクラスタの取得
        cluster_list = np.argmin(distance, axis=1)

        # 新たなセントロイドの取得
        new_centroids = np.zeros((cluster_n, dimension))
        for k in range(cluster_n):
            new_centroids[k] = \
                np.average(data[np.where(cluster_list == k)], axis=0)

        # 新旧セントロイドで変化がなければ終了
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return cluster_list


def main():
    clusters = [2, 3, 4, 5]
    header, data = load_file("data3.csv")

    # 2次元散布図の表示
    if len(header) == 2:
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        plt.subplots_adjust(hspace=0.3)
        ax = ax.flatten()

        for i in range(len(clusters)):
            # クラスタリング
            cluster_list = k_means(header, data, clusters[i])

            # 表示
            for k in range(clusters[i]):
                cluster_index = np.where(cluster_list == k)
                ax[i].scatter(
                            data[cluster_index][:, 0],
                            data[cluster_index][:, 1],
                            marker="$○$"
                            )
            ax[i].set_xlabel(header[0])
            ax[i].set_ylabel(header[1])
            ax[i].set_title(f"cluster_num = {clusters[i]}")
            ax[i].grid()

    # 3次元散布図の表示
    elif len(header) == 3:
        fig, ax = plt.subplots(
                            2, 2, figsize=(10, 10),
                            subplot_kw=dict(projection='3d')
                            )
        ax = ax.flatten()

        for i in range(len(clusters)):
            # クラスタリング
            cluster_list = k_means(header, data, clusters[i])

            # 表示
            for k in range(clusters[i]):
                cluster_index = np.where(cluster_list == k)
                ax[i].scatter(
                            data[cluster_index][:, 0],
                            data[cluster_index][:, 1],
                            data[cluster_index][:, 2],
                            marker="$○$"
                            )
            ax[i].set_xlabel(header[0])
            ax[i].set_ylabel(header[1])
            ax[i].set_zlabel(header[2])
            ax[i].set_title(f"cluster_num = {clusters[i]}")
            ax[i].grid()

    plt.show()


if __name__ == "__main__":
    main()
