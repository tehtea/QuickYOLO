#================================================================
#
#   File name   : detection_demo.py
#   Author      : PyLessons
#   Created date: 2020-09-27
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : object detection image and video example
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp
from yolov3.configs import *

if __name__ == "__main__":
    image_path   = "./IMAGES/kite.jpg"
    image_output_path = "./IMAGES/kite_pred.jpg"

    yolo = Load_Yolo_model()
    detect_image(yolo, image_path, image_output_path, CLASSES=TRAIN_CLASSES, input_size=YOLO_INPUT_SIZE, show=False, rectangle_colors=(255,0,0))
    print(f"sample predictions saved to {image_output_path}")