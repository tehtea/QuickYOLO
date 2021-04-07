import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import larq_compute_engine as lce
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from yolov3.dataset import Dataset
from yolov3.yolov4 import Create_Yolo
from yolov3.utils import load_yolo_weights, postprocess_boxes, nms_no_gather
from yolov3.configs import *
import shutil
import json
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError: print("RuntimeError in tf.config.experimental.list_physical_devices('GPU')")

# Custom Keras layer, for easy exporting
class PostProcess(tf.keras.layers.Layer):

    def __init__(self, iou_threshold, **kwargs):
        self.iou_threshold = iou_threshold
        super(PostProcess, self).__init__(**kwargs)

    def post_prediction_process(self, 
                                pred_boxes):
        # pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        # pred_bbox = tf.concat(pred_bbox, axis=0)
        pred_boxes = tf.reshape(pred_boxes, (-1, tf.shape(pred_boxes)[-1]))
        print(pred_boxes)
        boxes = postprocess_boxes(pred_boxes)
        boxes, selected_indices = nms_no_gather(boxes, iou_threshold=self.iou_threshold)

        boxes, box_scores, box_classes = tf.split(boxes, (4, 1, 1), axis=-1)
        box_scores = tf.squeeze(box_scores, axis=-1)
        box_classes = tf.cast(box_classes, dtype=tf.int32)
        box_classes = tf.squeeze(box_classes, axis=-1)
        
        return boxes, box_scores, box_classes, selected_indices

    def call(self, y_pred):
        return self.post_prediction_process(y_pred)

if __name__ == '__main__':       
    if YOLO_FRAMEWORK == "tf": # TensorFlow detection
        if YOLO_TYPE == "yolov4":
            Darknet_weights = YOLO_V4_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V4_WEIGHTS
        if YOLO_TYPE == "yolov3":
            Darknet_weights = YOLO_V3_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V3_WEIGHTS
        if YOLO_TYPE == "yolov2":
            Darknet_weights = YOLO_V2_WEIGHTS

        if YOLO_CUSTOM_WEIGHTS == False:
            yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=YOLO_COCO_CLASSES)
            load_yolo_weights(yolo, Darknet_weights) # use Darknet weights
        else:
            yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
            yolo.load_weights(f"./checkpoints/{TRAIN_MODEL_NAME}") # use custom weights
        
    elif YOLO_FRAMEWORK == "trt": # TensorRT detection
        saved_model_loaded = tf.saved_model.load(f"./checkpoints/{TRAIN_MODEL_NAME}", tags=[tag_constants.SERVING])
        signature_keys = list(saved_model_loaded.signatures.keys())
        yolo = saved_model_loaded.signatures['serving_default']

    post_processed_output = PostProcess(TEST_IOU_THRESHOLD)(yolo.output)
    yolo = tf.keras.models.Model(yolo.input, post_processed_output)

    flatbuffer_bytes = lce.convert_keras_model(yolo)

    # export
    exported_model_path = f'checkpoints/{TRAIN_MODEL_NAME}.tflite'
    with open(exported_model_path, "wb") as flatbuffer_file:
        flatbuffer_file.write(flatbuffer_bytes)
    print(f'exported to: {exported_model_path}')