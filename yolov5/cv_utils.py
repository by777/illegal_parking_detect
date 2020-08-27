# encoding: utf-8
# @Time : 20/08/27 13:49
# @Author : Xu Bai
# @File : cv_utils.py
# @Desc :

import cv2

img_path = 'inference/images/frame.jpg'

img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
c1 = (1282, 371)
c2 = (1441, 476)
color = (0,255,255)
img = cv2.rectangle(img, c1, c2, color, 4)
# img = cv2.putText(img,text="自行车！！",(c1[0], c1[1]- 2))
cv2.imshow('frame',img)
cv2.waitKey(0)
cv2.destroyAllWindows()