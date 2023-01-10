"""
为使用mac的gpu，119行 device设置成了mps
"""

import colorsys
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import ImageDraw, ImageFont, Image

from nets.centernet import CenterNet_HourglassNet, CenterNet_Resnet50
from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config)
from utils.utils_bbox import decode_bbox, postprocess
import torch.nn.functional


# --------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path、classes_path和backbone
#   训练时的model_path和classes_path参数的修改
# --------------------------------------------#
class CenterNet(object):
    _defaults = {

        # 模型路径
        "model_path": 'logs/loss_2022_08_14_17_32_17/ep075-loss1.363-val_loss1.387.pth',
        
        # 类别文件路径
        "classes_path": 'model_data/lroc_classes.txt',
        
        # backbone设置
        "backbone": 'hourglass',

        "input_shape": [512, 512],
        
        # 只有得分大于置信度的预测框会被保留下来
        "confidence": 0.3,

        # 非极大抑制所用到的nms_iou大小
        "nms_iou": 0.3,
        
        "nms": True,

        "letterbox_image": False,

        "cuda": False
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"


    # centernet初始化
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value

        # 计算总的类的数量
        self.class_names, self.num_classes = get_classes(self.classes_path)

        # 画框设置不同的颜色
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()

        show_config(**self._defaults)



    def generate(self, onnx=False):
        #  载入模型与权值
        assert self.backbone in ['resnet50', 'hourglass']
        if self.backbone == "resnet50":
            self.net = CenterNet_Resnet50(num_classes=self.num_classes, pretrained=False)
        else:
            self.net = CenterNet_HourglassNet({'hm': self.num_classes, 'wh': 2, 'reg': 2})

        # device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('mps')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = torch.nn.DataParallel(self.net)
                self.net = self.net.cuda()
    #   检测图片
    def detect_image(self, image, crop=False, count=False):
        # 计算输入图片的高和宽
        image_shape = np.array(np.shape(image)[0:2])
        # 在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        # 代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # 图片预处理，归一化。获得的photo的shape为[1, 512, 512, 3]
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            # 将图像输入网络当中进行预测
            outputs = self.net(images)
            if self.backbone == 'hourglass':
                outputs = [outputs[-1]["hm"].sigmoid(), outputs[-1]["wh"], outputs[-1]["reg"]]
            # 利用预测结果进行解码
            outputs = decode_bbox(outputs[0], outputs[1], outputs[2], self.confidence, self.cuda)
            results = postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, self.nms_iou)

            # 如果没有检测到撞击坑，则返回原图
            if results[0] is None:
                return image

            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

        #   设置字体与边框厚度
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0], 1)
        # 计数
        if count:
            print("top_label:", top_label)
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        # 是否进行目标的裁剪
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))

                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        # 图像绘制
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))
            label = '{} '.format(predicted_class)  
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                # draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline='green')
            # draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            # draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font) # 框左上角的类别标注
            del draw

        return image

    def detect_image_calc(self, image, sizes):
        # 计算输入图片的高和宽
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)

        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # 归一化
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            if self.backbone == 'hourglass':
                outputs = [outputs[-1]["hm"].sigmoid(), outputs[-1]["wh"], outputs[-1]["reg"]]
            outputs = decode_bbox(outputs[0], outputs[1], outputs[2], self.confidence, self.cuda)
            results = postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, self.nms_iou)

            if results[0] is None:
                return sizes

            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            h = bottom - top  # 计算预测框的尺寸
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
            label = '{} '.format(predicted_class) 
            # print(label, top, left, bottom, right, h)
        print('sizes:', sizes)
        return sizes

    # 计算最小尺寸的撞击坑
    def detect_image_calc_minsize(self, image, minsize):
        image_shape = np.array(np.shape(image)[0:2])

        image = cvtColor(image)

        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()

            outputs = self.net(images)
            if self.backbone == 'hourglass':
                outputs = [outputs[-1]["hm"].sigmoid(), outputs[-1]["wh"], outputs[-1]["reg"]]

            outputs = decode_bbox(outputs[0], outputs[1], outputs[2], self.confidence, self.cuda)

            results = postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, self.nms_iou)


            if results[0] is None:
                return minsize

            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

        # ---------------------------------------------------------#
        #   图像绘制
        # ---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            h = bottom - top  # 计算预测框的尺寸
            w = right - left
            if np.fabs(h - w) < 5 and h * w < minsize:
                minsize = h * w
                print('h:', h, '  w', w, '  minsize:', minsize)
            label = '{} '.format(predicted_class)  # 改后
            # print(label, top, left, bottom, right, h)

        return minsize

    def get_FPS(self, image, test_interval):
        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # -----------------------------------------------------------#
        #   图片预处理，归一化。获得的photo的shape为[1, 512, 512, 3]
        # -----------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            if self.backbone == 'hourglass':
                outputs = [outputs[-1]["hm"].sigmoid(), outputs[-1]["wh"], outputs[-1]["reg"]]
            # -----------------------------------------------------------#
            #   利用预测结果进行解码
            # -----------------------------------------------------------#
            outputs = decode_bbox(outputs[0], outputs[1], outputs[2], self.confidence, self.cuda)

            # -------------------------------------------------------#
            #   对于centernet网络来讲，确立中心非常重要。
            #   对于大目标而言，会存在许多的局部信息。
            #   此时对于同一个大目标，中心点比较难以确定。
            #   使用最大池化的非极大抑制方法无法去除局部框
            #   所以我还是写了另外一段对框进行非极大抑制的代码
            #   实际测试中，hourglass为主干网络时有无额外的nms相差不大，resnet相差较大。
            # -------------------------------------------------------#
            results = postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, self.nms_iou)

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                # ---------------------------------------------------------#
                #   将图像输入网络当中进行预测！
                # ---------------------------------------------------------#
                outputs = self.net(images)
                if self.backbone == 'hourglass':
                    outputs = [outputs[-1]["hm"].sigmoid(), outputs[-1]["wh"], outputs[-1]["reg"]]
                # -----------------------------------------------------------#
                #   利用预测结果进行解码
                # -----------------------------------------------------------#
                outputs = decode_bbox(outputs[0], outputs[1], outputs[2], self.confidence, self.cuda)

                # -------------------------------------------------------#
                #   对于centernet网络来讲，确立中心非常重要。
                #   对于大目标而言，会存在许多的局部信息。
                #   此时对于同一个大目标，中心点比较难以确定。
                #   使用最大池化的非极大抑制方法无法去除局部框
                #   所以我还是写了另外一段对框进行非极大抑制的代码
                #   实际测试中，hourglass为主干网络时有无额外的nms相差不大，resnet相差较大。
                # -------------------------------------------------------#
                results = postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image,
                                      self.nms_iou)
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def detect_heatmap(self, image, heatmap_save_path):
        import cv2
        import matplotlib.pyplot as plt
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # -----------------------------------------------------------#
        #   图片预处理，归一化。获得的photo的shape为[1, 512, 512, 3]
        # -----------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            if self.backbone == 'hourglass':
                outputs = [outputs[-1]["hm"].sigmoid(), outputs[-1]["wh"], outputs[-1]["reg"]]
        # print('image.shape:',image.size)
        img1 = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)  # 将PIL转化为cv2
        # cv2.imshow("img", img1)
        # cv2.waitKey()
        # plt.imshow(image, alpha=1)
        # plt.axis('off')
        mask = np.zeros((image.size[1], image.size[0]))
        score = np.max(outputs[0][0].permute(1, 2, 0).cpu().numpy(), -1)
        score = cv2.resize(score, (image.size[0], image.size[1]))
        normed_score = (score * 255).astype('uint8')
        mask = np.maximum(mask, normed_score)
        # print('mask.shape:',mask.shape)
        # print('mask.type:', type(mask))
        mask_img = Image.fromarray(mask)
        mask_img = cvtColor(mask_img)
        img2 = cv2.cvtColor(np.asarray(mask_img), cv2.COLOR_RGB2GRAY)

        img2 = cv2.applyColorMap(img2, cv2.COLORMAP_JET)
        img_mix = cv2.addWeighted(img1, 1, img2, 1, 0)
        # cv2.imshow("res", img_mix)
        # cv2.waitKey()
        cv2.imwrite(heatmap_save_path[:-3] + 'png', img_mix)
        # print('mask_img.shape',mask_img.size)

        # res = Image.blend(image, mask_img, 0.3)
        # res.show('res')
        # plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")
        # plt.imshow(mask, alpha=0.5) # 改
        # plt.show()
        # plt.axis('off')
        # plt.subplots_adjust(top=1, bottom=0, right=1,  left=0, hspace=0, wspace=0)
        # plt.margins(0, 0)
        # plt.savefig(heatmap_save_path, dpi=200, bbox_inches='tight', pad_inches = -0.1)
        # plt.savefig(heatmap_save_path, bbox_inches='tight', pad_inches = -0.1) # 改
        # print("Save to the " + heatmap_save_path)
        # plt.show()

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names = ["images"]
        output_layer_names = ["output"]

        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                          im,
                          f=model_path,
                          verbose=False,
                          opset_version=12,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=input_layer_names,
                          output_names=output_layer_names,
                          dynamic_axes=None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w")
        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # -----------------------------------------------------------#
        #   图片预处理，归一化。获得的photo的shape为[1, 512, 512, 3]
        # -----------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            if self.backbone == 'hourglass':
                outputs = [outputs[-1]["hm"].sigmoid(), outputs[-1]["wh"], outputs[-1]["reg"]]
            # -----------------------------------------------------------#
            #   利用预测结果进行解码
            # -----------------------------------------------------------#
            outputs = decode_bbox(outputs[0], outputs[1], outputs[2], self.confidence, self.cuda)

            # -------------------------------------------------------#
            #   对于centernet网络来讲，确立中心非常重要。
            #   对于大目标而言，会存在许多的局部信息。
            #   此时对于同一个大目标，中心点比较难以确定。
            #   使用最大池化的非极大抑制方法无法去除局部框
            #   所以我还是写了另外一段对框进行非极大抑制的代码
            #   实际测试中，hourglass为主干网络时有无额外的nms相差不大，resnet相差较大。
            # -------------------------------------------------------#
            results = postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, self.nms_iou)

            # --------------------------------------#
            #   如果没有检测到物体，则返回原图
            # --------------------------------------#
            if results[0] is None:
                return

            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])

            top, left, bottom, right = box

            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (
                predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return
