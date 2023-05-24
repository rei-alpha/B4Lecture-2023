import matplotlib.pyplot as plt
import numpy as np
import csv



def load_file(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
    header = rows.pop(0)
    data = np.float_(np.array(rows).T)

    return header, data


def reg_2D(lam, dim, header, data_x, data_y):
    # 回帰
    X = np.array([data_x ** i for i in range(dim+1)]).T
    I = np.eye(dim + 1)
    B = np.linalg.inv((X.T @ X) + lam * I) @ X.T @ data_y
    
    # グラフデータの用意
    plot_x = np.linspace(data_x.min(), data_x.max(), 100)
    plot_y = B @ (np.array([plot_x ** i for i in range(dim + 1)]))

    # 数式の取得
    equation = f"{B[0]:.2f}"
    for i in range(1, dim + 1):
        if B[i] > 0:
            equation += "+" + f"${B[i]:.2f}x^{i}$"
        else:
            equation += f"${B[i]:.2f}x^{i}$"

    # 散布図、グラフの描画
    plt.scatter(data_x, data_y, marker="$○$", s=40, label="Observed data")
    plt.plot(plot_x, plot_y, color='r', label="y="+equation)
    plt.title("correlation")
    plt.xlabel(header[0])
    plt.ylabel(header[1])
    plt.grid()
    plt.legend()
    plt.show()


def reg_3D(lam, dim, header, data_x1, data_x2, data_y):
    # 回帰
    X = []
    for x1, x2 in zip(data_x1, data_x2):
        list_tmp = []
        for d1 in range(dim + 1):
            for d2 in range(dim + 1 - d1):
                list_tmp.append((x1 ** d1) * (x2 ** d2))
        X.append(list_tmp)
    X = np.array(X, dtype=float)
    I = np.eye(X.shape[1])
    B = np.linalg.inv((X.T @ X) + lam * I) @ X.T @ data_y

    # グラフデータの用意
    plot_x1, plot_x2 = np.meshgrid(np.linspace(data_x1.min(), data_x1.max(), 100), 
                                   np.linspace(data_x2.min(), data_x2.max(), 100))
    plot_y = 0
    i = 0
    equation = f"{B[0]:.2f}"
    for d1 in range(dim + 1):
        for d2 in range(dim + 1 - d1):
            # yの取得
            plot_y += B[i] * (plot_x1 ** d1) * (plot_x2 ** d2)
            # 数式の取得
            if B[i] > 0:
                equation += "+" + f"${B[i]:.2f}x_1^{d1}x_2^{d2}$"
            else:
                equation += f"${B[i]:.2f}x_1^{d1}x_2^{d2}$"
            i += 1

    # 散布図、グラフの描画
    fig = plt.figure(figsize=(1,2))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(data_x1, data_x2, data_y, c="b", label="Observed data")
    ax.plot_wireframe(plot_x1, plot_x2, plot_y, color="r", label="y="+equation)
    ax.set_title("correlation")
    ax.set_xlabel(header[0])
    ax.set_ylabel(header[1])
    ax.set_zlabel(header[2]) 
    plt.grid()
    plt.legend()
    plt.show()


def main():
    lamda = 0.5
    dimension = 3
    header, data = load_file("data3.csv")
    if len(header) == 2:
        reg_2D(lamda, dimension, header, data[0], data[1])
    else:
        reg_3D(lamda, dimension, header, data[0], data[1], data[2])


if __name__ == "__main__":
    main()
