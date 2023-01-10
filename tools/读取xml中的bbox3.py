"""
本代码目的：统计label中框的尺寸
"""
import numpy as np

import xml.dom.minidom
import os
from PIL import ImageDraw, ImageFont, Image

from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config)
import cv2


def main():
    # JPG文件的地址
    img_path = '/Users/zhangshuowei/Documents/zsw/Data/rs/test/'
    # XML文件的地址
    anno_path = '/Users/zhangshuowei/Documents/zsw/Data/rs/voc/Annotations/'
    # 存结果的文件夹
    res_path = '/Users/zhangshuowei/Documents/zsw/Data/rs/results/gtbox_on_img_transparency/'
    # 获取文件夹中的文件
    imagelist = os.listdir(img_path)
    num = 0
    sizes = [0] * 60
    for image in imagelist:
        image_pre, ext = os.path.splitext(image)
        xml_file = anno_path + image_pre + '.xml'  # 获取图片对应的标签文件
        DOMTree = xml.dom.minidom.parse(xml_file)
        collection = DOMTree.documentElement
        objects = collection.getElementsByTagName("object")
        for object in objects:
            # print("start")
            bndbox = object.getElementsByTagName('bndbox')[0]
            xmin = bndbox.getElementsByTagName('xmin')[0]
            xmin_data = xmin.childNodes[0].data
            ymin = bndbox.getElementsByTagName('ymin')[0]
            ymin_data = ymin.childNodes[0].data
            xmax = bndbox.getElementsByTagName('xmax')[0]
            xmax_data = xmax.childNodes[0].data
            ymax = bndbox.getElementsByTagName('ymax')[0]
            ymax_data = ymax.childNodes[0].data
            xmin = int(xmin_data)
            xmax = int(xmax_data)
            ymin = int(ymin_data)
            ymax = int(ymax_data)
            h = ymax - ymin
            # print('h:', h)
            if h // 10 == 1:
                if h % 10 <= 1:
                    sizes[54] += 1
                elif h % 10 <= 2:
                    sizes[55] += 1
                elif h % 10 <= 4:
                    sizes[56] += 1
                elif h % 10 <= 6:
                    sizes[57] += 1
                elif h % 10 <= 8:
                    sizes[58] += 1
                else:
                    sizes[59] += 1
            else:
                sizes[h // 10] += 1
        print('sizes:', sizes)
        print(image_pre + ' is OK!')
        # break
    print('last sizes:', sizes)


if __name__ == '__main__':
    main()
