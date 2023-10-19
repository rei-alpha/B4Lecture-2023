#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
B4輪講最終課題 パターン認識に挑戦してみよう
ベースラインスクリプト
特徴量；MFCCの平均（0次項含まず）
識別器；MLP
バギング；作成した3つのモデルから得た結果を多数決で最終結果とする
"""


from __future__ import division
from __future__ import print_function

import argparse

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import Nadam
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import time


def my_MLP(input_shape, output_dim):
    """
    MLPモデルの構築
    Args:
        input_shape: 入力の形
        output_dim: 出力次元
    Returns:
        model: 定義済みモデル
    """

    model = Sequential()

    model.add(Dense(256, input_dim=input_shape))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    model.add(Dense(output_dim))
    model.add(Activation("softmax"))

    # モデル構成の表示
    model.summary()

    return model


def feature_extraction(path_list):
    """
    wavファイルのリストから特徴抽出を行い，リストで返す
    扱う特徴量はMFCC100次元の平均（0次は含めない）
    Args:
        path_list: 特徴抽出するファイルのパスリスト
    Returns:
        features: 特徴量
    """

    load_data = lambda path: librosa.load(path)[0]

    data = list(map(load_data, path_list))
    features = np.array(
        [np.mean(librosa.feature.mfcc(y=y, n_mfcc=100), axis=1) for y in data]
    )

    return features


def plot_confusion_matrix(predict, ground_truth, title=None, cmap=plt.cm.Blues):
    """
    予測結果の混合行列をプロット
    Args:
        predict: 予測結果
        ground_truth: 正解ラベル
        title: グラフタイトル
        cmap: 混合行列の色
    Returns:
        Nothing
    """

    cm = confusion_matrix(predict, ground_truth)
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Ground truth")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig("result/cm_3nadam100.png", transparent=True)
    plt.show()


def write_result(paths, outputs):
    """
    結果をcsvファイルで保存する
    Args:
        paths: テストする音声ファイルリスト
        outputs:
    Returns:
        Nothing
    """

    with open("result_3nadam100.csv", "w") as f:
        f.write("path,output\n")
        assert len(paths) == len(outputs)
        for path, output in zip(paths, outputs):
            f.write("{path},{output}\n".format(path=path, output=output))


def plot_history(history):
    # 学習過程をグラフで出力
    # print(history.history.keys())
    acc = history.history["accuracy"]
    loss = history.history["loss"]
    epochs = range(1, len(acc) + 1)
    plt.figure()
    plt.plot(epochs, acc)
    plt.grid()
    plt.title("Model accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig("result/history_1of3nadam100_acc.png", transparent=True)
    plt.show()

    plt.figure()
    plt.plot(epochs, loss)
    plt.grid()
    plt.title("Model loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("result/history_1of3nadam100_loss.png", transparent=True)
    plt.show()


def vote(pre_val1, pre_val2, pre_val3, idx_best_model):
    """
     各モデルから得られた予測結果に多数決を行い、最終的な結果を出力する
    Args:
        pre_val1: モデル1が予測した結果
        pre_val2: モデル2が予測した結果
        pre_val3: モデル3が予測した結果
        idx_best_model: バリューデータに対して最も正解率が高かったモデルの番号
    Returns:
        predicted_values: 多数決後の最終的な予測結果
    """
    # 票の集計
    three_pre_val = np.zeros((len(pre_val1), 10))
    for i in range(len(pre_val1)):
        three_pre_val[i][pre_val1[i]] +=1
        three_pre_val[i][pre_val2[i]] +=1
        three_pre_val[i][pre_val3[i]] +=1
    
    # 多数決で一つに絞る
    predicted_values = np.argmax(three_pre_val, axis=1)
    
    # 票が均等になった場合は、最もaccuracyが高いモデルの予測を使用する
    if idx_best_model == 0:
        best_pred = pre_val1
    elif idx_best_model == 1:
        best_pred = pre_val2
    else:
        best_pred = pre_val3
    for i in range(len(predicted_values)):
        if three_pre_val[i][predicted_values[i]] == 1:
            predicted_values[i] = best_pred[i]
    
    return predicted_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_truth", type=str, help="テストデータの正解ファイルCSVのパス")
    args = parser.parse_args()

    # データの読み込み
    training = pd.read_csv("training.csv")
    test = pd.read_csv("test.csv")

    # 学習データの特徴抽出
    X_train = feature_extraction(training["path"].values)
    X_test = feature_extraction(test["path"].values)

    # 正解ラベルをone-hotベクトルに変換 ex. 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    Y_train = np_utils.to_categorical(y=training["label"], num_classes=10)

    print("時間計測開始\n")
    start = time.time()
    # 学習データを学習データとバリデーションデータに分割 (バリデーションセットを20%とした例)
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X_train,
        Y_train,
        test_size=0.2,
    )
    # さらに生成した80%の学習データを、各モデルの学習データとバリデーションデータに分割（バリデーションデータを20%とした）
    # モデル1用
    X_train1, X_validation1, Y_train1, Y_validation1 = train_test_split(
        X_train,
        Y_train,
        test_size=0.2,
        random_state=20200616,
    )
    # モデル2用
    X_train2, X_validation2, Y_train2, Y_validation2 = train_test_split(
        X_train,
        Y_train,
        test_size=0.2,
        random_state=20231016,
    )
    # モデル3用
    X_train3, X_validation3, Y_train3, Y_validation3 = train_test_split(
        X_train,
        Y_train,
        test_size=0.2,
        random_state=20231025,
    )

    # モデルの構築
    model1 = my_MLP(input_shape=X_train1.shape[1], output_dim=10)
    model2 = my_MLP(input_shape=X_train2.shape[1], output_dim=10)
    model3 = my_MLP(input_shape=X_train3.shape[1], output_dim=10)

    # モデルの学習基準の設定
    model1.compile(
        loss="categorical_crossentropy", optimizer=Nadam(lr=1e-4), metrics=["accuracy"]
    )
    model2.compile(
        loss="categorical_crossentropy", optimizer=Nadam(lr=1e-4), metrics=["accuracy"]
    )
    model3.compile(
        loss="categorical_crossentropy", optimizer=Nadam(lr=1e-4), metrics=["accuracy"]
    )

    # モデルの学習
    history1 = model1.fit(X_train1, Y_train1, batch_size=32, epochs=250, verbose=1)
    history2 = model2.fit(X_train2, Y_train2, batch_size=32, epochs=250, verbose=1)
    history3 = model3.fit(X_train3, Y_train3, batch_size=32, epochs=250, verbose=1)

    # モデル構成，学習した重みの保存
    model1.save("keras_model/my_model1_nadam100.h5")
    model2.save("keras_model/my_model2_nadam100.h5")
    model3.save("keras_model/my_model3_nadam100.h5")

    plot_history(history1)

    # バリデーションセットによるモデルの評価
    # モデルをいろいろ試すときはテストデータを使ってしまうとリークになる可能性があるため、このバリデーションセットによる指標を用いてください
    score1 = model1.evaluate(X_validation1, Y_validation1, verbose=0)
    score2 = model2.evaluate(X_validation2, Y_validation2, verbose=0)
    score3 = model3.evaluate(X_validation3, Y_validation3, verbose=0)
    print("Validation accuracy with model1: ", score1[1])
    print("Validation accuracy with model2: ", score2[1])
    print("Validation accuracy with model3: ", score3[1])

    # 同数票の際に利用するため、最もaccuracyの高いモデルを選定
    idx_best_model = np.argmax([score1[1], score2[1], score3[1]])

    ############################################################################
    # バリデーションセットに対する予測
    # 学習データを学習データとバリデーションデータに分割 (バリデーションセットを20%とした例)
   
    # モデル1
    predict1 = model1.predict(X_validation)
    predicted_values1 = np.argmax(predict1, axis=1)
    # モデル2
    predict2 = model2.predict(X_validation)
    predicted_values2 = np.argmax(predict2, axis=1)
    # モデル3
    predict3 = model3.predict(X_validation)
    predicted_values3 = np.argmax(predict3, axis=1)

    # 多数決後の最終結果
    predicted_values = vote(predicted_values1, predicted_values2, predicted_values3, idx_best_model)
    
    # バリデーションでの正解率
    Y_val_label = np.argmax(Y_validation, axis=1)
    val_accuracy = accuracy_score(Y_val_label, predicted_values)
    print("Validatoion accuracy with new model: ", val_accuracy)
    print("評価終了\n")

    end = time.time() - start
    print(f"{end}秒かかりました\n")


    ############################################################################
    # テストデータに対する予測
    # モデル1の予測
    predict1 = model1.predict(X_test)
    predicted_values1 = np.argmax(predict1, axis=1)
    # モデル2の予測
    predict2 = model2.predict(X_test)
    predicted_values2 = np.argmax(predict2, axis=1)
    # モデル3の予測
    predict3 = model3.predict(X_test)
    predicted_values3 = np.argmax(predict3, axis=1)

    # 多数決後の最終結果
    predicted_values = vote(predicted_values1, predicted_values2, predicted_values3, idx_best_model)

    # テストデータに対して推論した結果の保存
    write_result(test["path"].values, predicted_values)

    # テストデータに対する正解ファイルが指定されていれば評価を行う（accuracyと混同行列）
    if args.path_to_truth:
        test_truth = pd.read_csv(args.path_to_truth)
        truth_values = test_truth["label"].values
        test_accuracy = accuracy_score(truth_values, predicted_values)
        plot_confusion_matrix(
            predicted_values,
            truth_values,
            title=f"Acc. {round(test_accuracy*100,2)}%",
        )
        print("Test accuracy: ", test_accuracy)


if __name__ == "__main__":
    main()
