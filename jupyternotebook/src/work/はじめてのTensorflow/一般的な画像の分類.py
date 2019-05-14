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
# -
print("test2")
