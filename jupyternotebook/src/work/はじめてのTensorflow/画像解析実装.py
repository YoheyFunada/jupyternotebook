# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# Pillowの基本的な使い方に慣れてきたところで、一般的な画像に対する分類問題を解いてみましょう。 画像データを学習に使用できるよう、入力層へ渡せる形式へ変換した後、4.2節と同じくCNNを実装し画像を分類してみます。


# +
import tensorflow as tf
import tflearn

#層の作成、学習に必要なライブラリの読み込み
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import os
import numpy as np
from PIL import Image

## 画像データの処理

# 学習用の画像ファイルを格納しているディレクトリ
train_dirs = ['pos','neg']

# 学習データを格納する配列の準備
trainX = [] #画像ピクセル値
trainY = [] #正解データ

for i,d in enumerate(train_dirs):
    #ファイル名の取得
    files = os.listdir('./data/pict/' + d)
    for f in files:
        #画像読み込み
        image = Image.open('./data/pict/' + d + '/' + f, 'r')
        #グレースケールへ変換
        gray_image = image.convert('L')
        # 画像ファイルをピクセル値へ変換
        gray_image_px = np.array(gray_image)
        gray_image_flatten = gray_image_px.flatten().astype(np.float32)/255.0
        trainX.append(gray_image_flatten)
        
        # 正解データをone_hot形式へ変換
        tmp = np.zeros(2)
        tmp[i] = 1
        trainY.append(tmp)

# numpy配列に変換
trainX = np.asarray(trainX)
trainY = np.asarray(trainY)
# -

#1枚目の画像ピクセルデータを表示
print(trainX[0])
print(trainY[0])

# 1枚目の画像ピクセル値と正解データの長さを表示
print(len(trainX[0]))
print(len(trainY[0]))

# 1枚目の画像ピクセルデータは、サイズが1024（32×32×1）の1次元配列に格納されており、正解データはサイズが2（posとnegの2値）の1次元配列に格納されていることが分かります。

#画像ピクセルデータを1次元から2次元へ変換
#CNNを使って学習するために、学習（入力）データは2次元でなければなりません。
trainX = trainX.reshape([-1,32,32,1])
#1枚目の画像ピクセル値を表示
trainX[0]

#1枚目のサイズ
print(len(trainX[0]))

# ここでは、入力層が32×32ノード、中間層は畳み込み層、プーリング層が2層ずつ、全結合層が1層、出力層が2ノード（posとnegの2種類）から成るCNNを構築し、
# モデルの分類精度を確かめます。畳み込み層で使用するフィルタのサイズは5とし、プーリング層の領域のサイズは2とします。また、全結合層のノード数を128とします。

# +
# ニューラルネットワークの作成

## 初期化
tf.reset_default_graph()

## 入力層の作成
# input_data関数を使って入力層を作成します。
# 1番目の引数shapeには、入力する学習データの形状として、バッチサイズとノード数を設定します。
# ここでは、None（ここでは指定しない）と、32×32（画像1枚あたりのピクセル数）、1（グレースケール画像）とします。
net = input_data(shape=[None, 32, 32, 1])

## 中間層の作成
# 畳み込み層の作成
#●1番目の引数：作成する層の1つ前の層（結合の対象となる層）を設定します。ここでは、netにあたります。
#●2番目の引数：畳み込みフィルタ数（出力次元数）を設定します。ここでは、1番目の畳み込み層では32とします。
#●3番目の引数：フィルタのサイズを設定します。ここでは、5×5とします。
#●4番目の引数：使用する活性化関数を設定します。ここでは、relu（ReLU関数）を使用します。
#●また、引数として明示的に設定していませんが、ゼロパディングを行ってサイズを保持しています。同様に、フィルタは1ずつスライドします。
net = conv_2d(net, 32, 5, activation='relu')

# プーリング層の作成
# ●1番目の引数：作成する層の1つ前の層を設定します。ここでは、netにあたります。
# ●2番目の引数：最大プーリングを行う領域を設定します。ここでは、2×2とします。
net = max_pool_2d(net, 2)

# 畳み込み層の作成
# ●1番目の引数：作成する層の1つ前の層（結合の対象となる層）を設定します。ここでは、netにあたります。
# ●2番目の引数：畳み込みフィルタ数（出力次元数）を設定します。ここでは、1つ目の畳み込み層では64とします。
#●3番目の引数：フィルタのサイズを設定します。ここでは、5×5とします。
#●4番目の引数：使用する活性化関数を設定します。ここでは、relu（ReLU関数）を使用します。
#●また、引数として明示的に設定していませんが、ゼロパディングを行ってサイズを保持しています。同様に、フィルタは1ずつスライドします。
net = conv_2d(net, 64, 5, activation='relu')

# プーリング層の作成
# ●1番目の引数：作成する層の1つ前の層を設定します。ここでは、netにあたります。
# ●2番目の引数：最大プーリングを行う領域を設定します。ここでは、2×2とします。
# 入力層における32×32×1サイズのデータは、1層目の畳み込み層で32×32×32サイズとなり、1層目のプーリング層で16×16×32サイズとなり、2層目の畳み込み層で16×16×64サイズとなり、2層目のプーリング層で8×8×64サイズとなります。
nat = max_pool_2d(net, 2)

# 全結合層の作成
# ●1番目の引数：作成する層の1つ前の層（結合の対象となる層）を設定します。ここでは、netにあたります。
# ●2番目の引数：作成する層のノード数を設定します。ここでは、128とします。
# ●3番目の引数：使用する活性化関数を設定します。ここでは、relu（ReLU関数）とします。
net = fully_connected(net, 128, activation='relu')

#dropout関数を使って、作成した層に対しドロップアウトを行います。
#●1番目の引数：ドロップアウトの対象とする層を設定します。ここでは、netにあたります。
#●2番目の引数：対象となる層の全ノードのうち何割を残しておくか、その比率を設定します。ここでは、0.5とします。
net = dropout(net, 0.5)

## 出力層の作成
# ●1番目の引数：作成する層の1つ前の層（結合の対象となる層）を設定します。ここでは、netにあたります。
# ●2番目の引数：作成する層のノード数を設定します。ここでは、2とします（posとnegの2種類あるため）。
# ●3番目の引数：作成する層で使用する活性化関数を設定します。ここでは、softmax（ソフトマックス関数）を使用します。
net = tflearn.fully_connected(net, 2, activation='softmax')

# regression関数を使って、学習の条件を設定します。
# ●1番目の引数：学習の対象となる層を設定します。ここでは、これまで作成してきた層であるnetにあたります。
# ●2番目の引数：最適化の手法を設定します。ここでは、sgd（確率的勾配降下法）を使用します。
# ●3番目の引数：学習係数の減衰係数を設定します。ここでは、0.5とします。
# ●4番目の引数：誤差関数を設定します。ここでは、categorical_crossentropy（交差エントロピー）を使用します。
net = tflearn.regression(net, optimizer='sgd', learning_rate=0.5, loss='categorical_crossentropy')

## モデルの作成（学習） ##
#学習の実行
# DNN関数を使って、作成したニューラルネットワークと学習条件をセットします。
#●1番目の引数：セットする対象のニューラルネットワークを設定します。ここでは、netにあたります。
model = tflearn.DNN(net)

# fit関数を使って学習を実行し、モデルを作成します。
# ●1番目の引数：学習データを設定します。ここでは、学習用の画像ピクセルデータを格納した配列trainXです。
# ●2番目の引数：正解データを設定します。ここでは、学習用の画像正解データを格納した配列trainYです。
# ●3番目の引数：エポック数（≒学習回数）を設定します。ここでは、20とします。
# ●4番目の引数：バッチサイズを設定します。ここでは32とします。
# ●5番目の引数：モデルの精度を検証するためのテストデータセットを設定します。ここでは、学習用データセットのうちの2割（0.2）とします。
# ●6番目の引数：学習のステップごとに精度を表示するかどうかを設定します。ここでは、True（表示する）とします。
model.fit(trainX, trainY, n_epoch=20, batch_size=32, validation_set=0.2, show_metric=True)
# -