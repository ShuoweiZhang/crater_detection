"""
本代码目的：将xml标签中的bbox取出来，标注在原图上。 本代码是透明框
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

    for image in imagelist:
        image_pre, ext = os.path.splitext(image)
        # print(image_pre, ext)
        # image_pre = 'mars_1293'
        # image = 'mars_1293.jpg'
        img_file = img_path + image  # 获取图片路径
        image = cv2.imread(img_file)  # cv2读取图片
        # image = Image.open(img_file)
        blk = np.zeros(image.shape, np.uint8)  # bbox框的图片
        image_shape = np.array(np.shape(image)[0:2])
        # image = cvtColor(image)
        xml_file = anno_path + image_pre + '.xml'  # 获取图片对应的标签文件
        DOMTree = xml.dom.minidom.parse(xml_file)
        collection = DOMTree.documentElement
        objects = collection.getElementsByTagName("object")
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // 512, 1)
        boxes = []
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
            boxes.append([ymin, xmin, ymax, xmax])  # top, left, bottom, right

        for box in boxes:
            top, left, bottom, right = box
            # draw = ImageDraw.Draw(image)
            cv2.rectangle(blk, (left, top), (right, bottom), (0, 0, 255), 1)

            # for i in range(thickness):
            #     draw.rectangle([left + i, top + i, right - i, bottom - i], outline='green')
        image = cv2.addWeighted(image, 1.0, blk, 0.5, 1)
        # del draw
        # image.show() # PIL 显示图片
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        # image.save(res_path + image_pre + '.png')
        cv2.imwrite(res_path + image_pre + '.png', image)
        print(image_pre + ' is OK!')
        # break


if __name__ == '__main__':
    main()
