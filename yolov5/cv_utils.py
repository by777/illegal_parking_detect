# encoding: utf-8
# @Time : 20/08/27 13:49
# @Author : Xu Bai
# @File : cv_utils.py
# @Desc :

import cv2
import time
import os
import sys

# wd = os.getcwd()
img_path = r'inference\images\frame.jpg'

cars_img_path = r'vehicle_imgs\cars'

motors_img_path = r'vehicle_imgs\motors'

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

c1 = (1282, 371)
c2 = (1441, 476)
color = (0, 255, 255)
# 裁剪到的自行车
crop = img[371: 476, 1282:1441]
crop_path = os.path.join(motors_img_path, str(time.time()).split('.')[0] + '.jpg')
print(crop_path)

cv2.imwrite(crop_path, crop)
# r = cv2.imencode('.jpg', crop)[1].tofile(crop_path) #当有中文路径时使用此方法


img = cv2.rectangle(img, c1, c2, color, 4)


cv2.imshow('frame', img)
cv2.imshow('crop', crop)
cv2.waitKey(0)
cv2.destroyAllWindows()

