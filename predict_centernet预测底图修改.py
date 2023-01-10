"""
此代码主要目的是是用 mode = 'predict_dir'
输入图像进行bbox预测，然后将bbox画在heatmap上
使用的CenterNet是 centernet_my.py里面的
"""
import time

import cv2
import numpy as np
from PIL import Image

# from centernet import CenterNet
from centernet_my import CenterNet

if __name__ == "__main__":
    centernet = CenterNet()
    mode = "predict_dir2"
    crop = False
    count = False
    video_path = 0
    video_save_path = ""
    video_fps = 25.0
    test_interval = 100
    fps_image_path = "img/street.jpg"
    dir_origin_path = "img/"
    dir_save_path = "img_out/"
    heatmap_save_path = "model_data/heatmap_vision.png"
    simplify = True
    onnx_save_path = "model_data/models.onnx"

    if mode == "predict":
        while True:
            img = input('Input image filename:')
            bakimg = input('Input bakimage filename:')
            try:
                image = Image.open(img)
                bakimg = Image.open(bakimg)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = centernet.detect_image(image, bakimg, crop=crop, count=count)
                r_image.show()

    elif mode == "predict_dir":  # bbox绘制在heatmap上面
        import os

        img_dir = '/Users/zhangshuowei/Documents/zsw/Data/rs/test'
        # heatmap_dir = '/Users/zhangshuowei/Documents/zsw/Data/rs/heatmaps/' # 热力图目录
        heatmap_dir = '/Users/zhangshuowei/Documents/zsw/Data/rs/results/gtbox_on_img/' # 真值框画在img的路径
        # save_dir = '/Users/zhangshuowei/Documents/zsw/Data/rs/bbox_on_heatmap_png/'
        save_dir = '/Users/zhangshuowei/Documents/zsw/Data/rs/results/gt_pre_on_img/'
        g = os.walk(img_dir)
        for path, dir_list, file_list in g:
            for file_name in file_list:
                img_path = os.path.join(img_dir, file_name)
                heatmap_path = os.path.join(heatmap_dir, file_name[:-3] + 'png')
                image = Image.open(img_path)
                bakimg = Image.open(heatmap_path)
                r_image = centernet.detect_image(image, bakimg, crop=crop, count=count)
                r_image.save(save_dir + file_name[:-3] + 'png')
                print(file_name + ' is OK!')

    elif mode == "predict_dir2":  # 半透明bbox绘制在真值bbox已经在的原图上面
        import os

        img_dir = '/Users/zhangshuowei/Documents/zsw/Data/rs/test'
        # heatmap_dir = '/Users/zhangshuowei/Documents/zsw/Data/rs/heatmaps/' # 热力图目录
        heatmap_dir = '/Users/zhangshuowei/Documents/zsw/Data/rs/results/gtbox_on_img_transparency/' # 真值框画在img的路径
        # save_dir = '/Users/zhangshuowei/Documents/zsw/Data/rs/bbox_on_heatmap_png/'
        save_dir = '/Users/zhangshuowei/Documents/zsw/Data/rs/results/gt_pre_on_img_transparency/'
        g = os.walk(img_dir)
        for path, dir_list, file_list in g:
            for file_name in file_list:
                img_path = os.path.join(img_dir, file_name)
                heatmap_path = os.path.join(heatmap_dir, file_name[:-3] + 'png')
                image = Image.open(img_path)
                bakimg = Image.open(heatmap_path)
                centernet.detect_image2(image, bakimg, crop=crop, count=count)
                # r_image.save(save_dir + file_name[:-3] + 'png')
                print(file_name + ' is OK!')
                break



    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while (True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(centernet.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = centernet.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image = centernet.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    elif mode == "heatmap":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                centernet.detect_heatmap(image, heatmap_save_path)

    elif mode == "export_onnx":
        centernet.convert_to_onnx(simplify, onnx_save_path)

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
