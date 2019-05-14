# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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

# # 一般的な画像の分類
#
# JPEG形式やPNG形式などの画像ファイルは、機械学習に使用できる形へ変換しなければなりません。画像処理ライブラリPillowを使って、その変換を行う方法を説明します。
# Pillow（PIL(注1)）はPythonの画像処理ライブラリです。画像のグレースケール化や二値化などの加工、拡大や縮小などの整形を行うことができます。PillowはAnacondaと一緒にインストールされています。

# +
import numpy as np
from PIL import Image

#画像ファイルの読み込み
image = Image.open('./data/pict/sample.jpg', 'r')
#画像ファイルの表示
image

# 画像ファイルをピクセル値で表示
image_px = np.array(image)
#画像ファイルの表示
print(image_px)

# 画像ピクセル値を1次元配列に変換
image_flatten = image_px.flatten().astype(np.float32)/255.0
print("# 画像ピクセル値を1次元配列に変換")
print(image_flatten)

#画像ピクセル値（配列）のサイズを表示
print("#画像ピクセル値（配列）のサイズを表示")
print(len(image_flatten))

#画像をグレースケールに変換
gray_image = image.convert('L')

#画像ファイルを表示
gray_image


# -


#画像ファイルをピクセル値で変換
gray_image_px = np.array(gray_image)
print("#gray画像ファイルをピクセル値で変換")
print(gray_image_px)


# +
#画像ピクセル値を1次元に変換
gray_image_flatten = gray_image_px.flatten().astype(np.float32)/255.0
print(gray_image_flatten)

# 画像ピクセル値（配列）のサイズを表示
print(len(gray_image_flatten))
# -

# 画像から分類モデルを作成するためには、十分な量の画像データを用意する必要があります。しかし、十分な量の画像データを用意できない場合は、もとある画像データを加工して枚数を増やすこともできます。ここでは、Pillowを使った画像の加工方法を紹介します。まずはライブラリを読み込みましょう。

# +
# ライブラリの読み込み
from PIL import ImageEnhance

#画像の彩度を調整
#画像imageの彩度を係数の値（ここでは0.5）によって調整し、その結果を変数conv1_imageへ格納します。係数が0.0なら白黒画像、係数が1.0なら元の画像になります。

conv1 = ImageEnhance.Color(image)
conv1_image = conv1.enhance(0.5)
conv1_image
# -
# +
#画像の明度を調整
conv2 = ImageEnhance.Brightness(image)
conv2_image = conv2.enhance(0.5)
conv2_image
# -
# +
# 画像のコントラストを調整
conv3= ImageEnhance.Contrast(image)
conv3_image = conv3.enhance(0.5)
conv3_image
# -
# +
# 画像のシャープネスを調整
# 画像imageのシャープネスを係数の値（ここでは2.0）によって調整し、その結果を変数conv4_imageへ格納します。係数が0.0なら輪郭がぼやけた画像、係数が1.0なら元の画像、係数が2.0なら輪郭が強調された鮮明な画像になります。
conv4= ImageEnhance.Sharpness(image)
conv4_image = conv4.enhance(2.0)
conv4_image
# -