# encoding: utf-8
# @Time : 20/08/31 22:04
# @Author : Xu Bai
# @File : detect_move.py
# @Desc : 利用混合高斯模型用于背景建模检测物体是否运动 × 效果极差 光流估计 × 效果不咋样


import cv2
import os

import numpy as np

# 拿到车辆的图片在前frame里进行模板匹配，大于阈值则视为之前已存在
img_path = r'.\vehicle_imgs\motors\1598708918.jpg'
frame_path = r'.\inference\images\frame.jpg'


def first_appear(img_path, frame_path):
    template = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    h, w = template.shape[:2]
    # 取匹配程度大于80%的坐标
    loc = np.where(res >= threshold)
    print(loc)
    for pt in zip(*loc[::-1]):
        print(pt)
        bottom_right = (pt[0] + w, pt[1] + h)
        cv2.rectangle(frame, pt, bottom_right, (0, 0, 255), 2)
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


first_appear(img_path, frame_path)
