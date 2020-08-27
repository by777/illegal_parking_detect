# encoding: utf-8
# @Time : 20/08/27 13:49
# @Author : Xu Bai
# @File : cv_utils.py
# @Desc :

import cv2

img_path = 'inference/images/frame.jpg'

img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)

cv2.imshow('frame',img)