# -*- coding: utf-8 -*-
import cv2          # OpenCVを使うため
import numpy as np

#画像読み込み
img = cv2.imread('./train_data/sample1.jpg')

hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # BGR画像 -> HSV画像


# トラックバーを作るため，まず最初にウィンドウを生成
cv2.namedWindow("OpenCV Window", cv2.WINDOW_NORMAL)

# トラックバーのコールバック関数は何もしない空の関数
def nothing(x):
    pass


# トラックバーの生成
cv2.createTrackbar("H_min", "OpenCV Window", 0, 179, nothing)       # Hueの最大値は179
cv2.createTrackbar("H_max", "OpenCV Window", 179, 179, nothing)
cv2.createTrackbar("S_min", "OpenCV Window", 0, 255, nothing)
cv2.createTrackbar("S_max", "OpenCV Window", 255, 255, nothing)
cv2.createTrackbar("V_min", "OpenCV Window", 0, 255, nothing)
cv2.createTrackbar("V_max", "OpenCV Window", 255, 255, nothing)

while(1):
    # (B)ここから画像処理
    # トラックバーの値を取る
    h_min = cv2.getTrackbarPos("H_min", "OpenCV Window")
    h_max = cv2.getTrackbarPos("H_max", "OpenCV Window")
    s_min = cv2.getTrackbarPos("S_min", "OpenCV Window")
    s_max = cv2.getTrackbarPos("S_max", "OpenCV Window")
    v_min = cv2.getTrackbarPos("V_min", "OpenCV Window")
    v_max = cv2.getTrackbarPos("V_max", "OpenCV Window")

    # inRange関数で範囲指定２値化 -> マスク画像として使う
    mask_image = cv2.inRange(hsv_image, (h_min, s_min, v_min), (h_max, s_max, v_max)) # HSV画像なのでタプルもHSV並び

    # bitwise_andで元画像にマスクをかける -> マスクされた部分の色だけ残る
    result_image = cv2.bitwise_and(img, img, mask=mask_image)
    result_image = cv2.resize(result_image, dsize=None, fx=1.0, fy=0.8)

    # (X)ウィンドウに表示
    cv2.imshow('OpenCV Window', result_image)   # ウィンドウに表示するイメージを変えれば色々表示できる

    if cv2.waitKey(100) == 27: # ESCキー
            break
