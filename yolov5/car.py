# encoding: utf-8
# @Time : 20/08/27 14:15
# @Author : Xu Bai
# @File : car.py
# @Desc :


class Car:
    def __init__(self):
        self.name = 'car'
        self.first_appear = True
        self.last_position = ()
        self.stop_time = 0
        self.id = 0
        self.img_path = 'vehicle_imgs'

    def __str__(self):
        return " name: " + str(self.name) \
               + " first_appear: " + str(self.first_appear) \
               + " last_position: " + str(self.last_position) \
               + " stop_time: " + str(self.stop_time) \
               + " id: " + str(self.id) \
               + " img_path: " + str(self.img_path)
