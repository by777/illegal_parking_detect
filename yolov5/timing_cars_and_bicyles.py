# encoding: utf-8
# @Time : 20/08/27 13:59
# @Author : Xu Bai
# @File : timing_cars_and_bicyles.py
# @Desc :

import time
from car import Car
import cv2


def detected_cars(time_, position, img_path):
    car = Car()
    car.last_position = position
    car.stop_time = time_
    car.img_path = img_path
    # 开始检测是否第一次出现

def car_timing(car):
    print(car)
    if car.first_appear:
        '''
            第一次出现，当前目标保存第一张图片、在图片中的位置坐标和当前时间
        '''
        print('最后坐标为：' + str(car.last_position))


def bicycle_timing():
    pass
