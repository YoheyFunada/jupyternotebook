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
        image = Image.open('./data/pict/' + '/' + f, 'r')
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


