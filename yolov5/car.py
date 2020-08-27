# encoding: utf-8
# @Time : 20/08/27 14:15
# @Author : Xu Bai
# @File : car.py
# @Desc :
class Car:
    def __init__(self):
        self.name = ''
        self.first_appear = True
        self.last_position = ()
        self.stop_time = 0
        self.id = 0
        self.img_path = ''
    def __str__(self):
        return "name:" + str(self.name)
