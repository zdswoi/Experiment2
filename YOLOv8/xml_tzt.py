#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from lxml.etree import Element, SubElement, tostring, ElementTree

# tensorboard --logdir=runs/train/exp14 --port=6006   通过这种方式打开exp14处的其他训练结果图示  上次训练结果50轮，其结果在exp11

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

classes = ["ctc1","ctc2","ctc3","ctc4","ctc5","ctc6","ctc7","ctc8","ctc9","dox1","dox2","dox3","dox4","dox5","dox6","dox7","dox8","dox9","otc1","otc2","otc3","otc4","otc5","otc6","otc7","otc8","otc9","tc1","tc2","tc3","tc4","tc5","tc6","tc7","tc8","tc9"]  # 类别
# classes = ["CTC0","CTC5","CTC10","CTC20","CTC30","DOX5","DOX10","DOX20","DOX30","OTC5","OTC10","OTC20","OTC30","TC5","TC10","TC20","TC30"]  # 类别
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0    # (x_min + x_max) / 2.0
    y = (box[2] + box[3]) / 2.0    # (y_min + y_max) / 2.0
    w = box[1] - box[0]   # x_max - x_min
    h = box[3] - box[2]   # y_max - y_min
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):
    in_file = open('VOCdevkit/VOC2007/Annotations/%s.xml' % (image_id), encoding='UTF-8')

    out_file = open('VOCdevkit/VOC2007/YOLOLabels/%s.txt' % (image_id), 'w')  # 生成txt格式文件
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        # print(cls)
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

xml_path = os.path.join(CURRENT_DIR, 'VOCdevkit\VOC2007\Annotations')

# xml list
img_xmls = os.listdir(xml_path)
for img_xml in img_xmls:
    label_name = img_xml.split('.')[0]
    print(label_name)
    convert_annotation(label_name)
