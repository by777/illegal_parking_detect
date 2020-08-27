# encoding: utf-8
# @Time : 20/08/27 13:49
# @Author : Xu Bai
# @File : cv_utils.py
# @Desc :

import cv2
import time

img_path = 'inference/images/frame.jpg'
cars_img_path = 'vehicle_imgs/cars/'
motors_img_path = 'vehicle_imgs/motors/'

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
c1 = (1282, 371)
c2 = (1441, 476)
color = (0, 255, 255)
# 裁剪到的自行车
crop = img[371: 476,1282:1441]
cv2.imwrite(motors_img_path + str(time.time()) + '.jpg',crop )
img = cv2.rectangle(img, c1, c2, color, 4)

# img = cv2.putText(img,text="自行车！！",(c1[0], c1[1]- 2))
cv2.imshow('frame', img)
cv2.imshow('crop', crop)
cv2.waitKey(0)
cv2.destroyAllWindows()
