#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import glob
import filter

def remove_objects(img, lower_size=None, upper_size=None):
    # find all objects
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)

    sizes = stats[1:, -1]
    _img = np.zeros((labels.shape))

    # process all objects, label=0 is background, objects are started from 1
    for i in range(1, nlabels):

        # remove small objects
        if (lower_size is not None) and (upper_size is not None):
            if lower_size < sizes[i - 1] and sizes[i - 1] < upper_size:
                _img[labels == i] = 255

        elif (lower_size is not None) and (upper_size is None):
            if lower_size < sizes[i - 1]:
                _img[labels == i] = 255

        elif (lower_size is None) and (upper_size is not None):
            if sizes[i - 1] < upper_size:
                _img[labels == i] = 255

    return _img

def filter_loss(input,test):

    input = input.astype(int)
    test = test.astype(int)

    miss_mask = cv2.bitwise_xor(input,test)

    loss = cv2.countNonZero(miss_mask)

    return loss

#学習データ
train_data = sorted(glob.glob('./output/*.jpg'))
test_data = sorted(glob.glob('./test_data/*.jpg'))

print(train_data)

#画像サイズ
img_size = 768*960
#学習データの量
file_num = len(train_data)

fin_loss = 1

for remove_area in range(5000,20000+1,1000):#面積による除去の許容面積の全探索
    print(remove_area)
    loss = 0
    for train_path,test_path in zip(train_data,test_data):

        train_img = cv2.imread(train_path, cv2.IMREAD_GRAYSCALE)
        test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

        fin_mask = remove_objects(train_img,lower_size=remove_area)

        loss += filter_loss(fin_mask,test_img)/img_size

    loss = loss/file_num
    if loss < fin_loss:
        output = remove_area
        fin_loss = loss

print('final output:')
print(output)

for i,path in enumerate(train_data):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    output_img = remove_objects(img,lower_size=output)

    path = './final_output/sample'+str(i)+'.jpg'
    cv2.imwrite(path,output_img)
