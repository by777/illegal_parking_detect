# encoding: utf-8
# @Time : 20/08/25 20:09
# @Author : Xu Bai
# @File : parking_detection.py
# @Desc : 使用yolov5检测车辆违停信息

from detect_obj import detect
import torch
import cv2
import sys
import os

print('Setup complete. Using torch %s %s' %
      (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

video_path = "./inference/car.mp4"


def get_images():
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("获取视频失败！")
        sys.exit()
    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('frame.jpg',frame)
        # cv2.imshow('frame',cv2.imread('frame.jpg'))
        # cv2.waitKey(0)
        detect('frame.jpg')
        key_pressed = cv2.waitKey(25)
        # ESC
        if key_pressed == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def run():
    pass


if __name__ == '__main__':
    get_images()
