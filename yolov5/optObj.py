# encoding: utf-8
# @Time : 20/08/26 22:46
# @Author : Xu Bai
# @File : optObj.py
# @Desc :

class Opt:

    def __init__(self, source='', view_img=False):
        self.weights = 'weights/yolov5s.pt'
        self.source = "inference/images"
        self.output = 'inference/output'
        self.img_size = 640
        self.conf_thres = 0.4
        self.iou_thres = 0.5
        self.device = 'cpu'
        self.view_img = True
        self.save_txt = False
        self.classes = None
        self.agnostic_nms = False
        self.augment = False
        self.update = False

    def setOpts(**kwargs):
        print(kwargs)
