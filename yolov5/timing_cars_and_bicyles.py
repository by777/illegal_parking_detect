# encoding: utf-8
# @Time : 20/08/27 13:59
# @Author : Xu Bai
# @File : timing_cars_and_bicyles.py
# @Desc :

import time
from car import Car
import cv2
# 暂定最大停车时长
MAX_PARKING_TIME = 3000


def detected_cars(time_, position, img_path):
    car = Car()
    car.last_position = position
    car.stop_time = time_
    car.img_path = img_path
    # 开始检测是否第一次出现
    if first_appear(car):
        print("第一次出现的汽车，可以暂停在停车区")
        t2 = time.time()
        if(t2 - car.stop_time) >= MAX_PARKING_TIME:
            print("违停！")
        else:



def car_timing(car):
    print(car)
    if car.first_appear:
        '''
            第一次出现，当前目标保存第一张图片、在图片中的位置坐标和当前时间
        '''
        print('最后坐标为：' + str(car.last_position))


def bicycle_timing():
    pass


def first_appear(car):
    return True
