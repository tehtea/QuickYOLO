import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import larq_compute_engine as lce
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from yolov3.dataset import Dataset
from yolov3.yolov4 import Create_Yolo
from yolov3.utils import load_yolo_weights, detect_image, image_preprocess, postprocess_boxes, nms, read_class_names
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

    def __init__(self, score_threshold, iou_threshold, area_threshold, **kwargs):
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.area_threshold = area_threshold
        super(PostProcess, self).__init__(**kwargs)

    def _convert_boxes_from_xywh(self, boxes):
        # Split boxes and convert them to xmin, ymin, xmax, ymax
        boxes_x, boxes_y, boxes_w, boxes_h = tf.split(boxes, (1, 1, 1, 1), axis=-1)
        boxes_x1 = tf.clip_by_value(boxes_x - boxes_w * 0.5, clip_value_min=0, clip_value_max=1)
        boxes_y1 = tf.clip_by_value( boxes_y - boxes_h * 0.5, clip_value_min=0, clip_value_max=1)
        boxes_x2 = tf.clip_by_value(boxes_x + boxes_w * 0.5, clip_value_min=0, clip_value_max=1)
        boxes_y2 = tf.clip_by_value(boxes_y + boxes_h * 0.5, clip_value_min=0, clip_value_max=1)

        # Concatenate the transformed boxes
        boxes = tf.concat([boxes_x1, boxes_y1, boxes_x2, boxes_y2], axis=-1)

        return boxes

    def _normalize_boxes(self, boxes):
        # Normalize xywh
        boxes /= YOLO_INPUT_SIZE
        boxes = tf.clip_by_value(boxes, clip_value_min=0, clip_value_max=1)

        return boxes

    def _rearrange_pred_boxes(self, pred_boxes):
        # Flatten the boxes
        flattened_boxes = tf.reshape(pred_boxes, (-1, tf.shape(pred_boxes)[-1]))

        # Get xywh, conf, class_probs of each box
        boxes, box_conf, box_class_prob = \
            tf.split(flattened_boxes, (4, 1, -1), axis=-1)
        
        return boxes, box_conf, box_class_prob

    def _get_box_scores(self, box_conf, box_class_prob):
        # Get box scores        
        box_scores = box_conf * tf.expand_dims(tf.reduce_max(box_class_prob, axis=-1), axis=-1)

        return box_scores

    def _get_box_classes(self, box_class_prob):
        # Get box classes
        box_classes = tf.argmax(box_class_prob, axis=-1, output_type=tf.int32)
        box_classes = tf.expand_dims(box_classes, axis=-1)

        return box_classes
    
    def _perform_nms(self, boxes, box_scores, box_classes):
        # Apply non-max-suppression on the filtered boxes
        selected_idx, selected_box_scores = tf.image.non_max_suppression_with_scores(boxes, 
                                                    tf.squeeze(box_scores, axis=-1),
                                                    10, 
                                                    iou_threshold=self.iou_threshold, 
                                                    score_threshold=self.score_threshold)

        return selected_idx, selected_box_scores

    def post_prediction_process(self, 
                                pred_boxes):
        boxes, box_conf, box_class_prob = self._rearrange_pred_boxes(pred_boxes)

        boxes = self._normalize_boxes(boxes)
        boxes = self._convert_boxes_from_xywh(boxes)

        box_scores = self._get_box_scores(box_conf, box_class_prob)
        box_classes = self._get_box_classes(box_class_prob)

        selected_idx, selected_box_scores = self._perform_nms(boxes, box_scores, box_classes)
        
        return boxes, box_scores, box_classes, selected_idx, selected_box_scores

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

    post_processed_output = PostProcess(TEST_SCORE_THRESHOLD, TEST_IOU_THRESHOLD, TEST_AREA_THRESHOLD)(yolo.output)
    yolo = tf.keras.models.Model(yolo.input, post_processed_output)

    flatbuffer_bytes = lce.convert_keras_model(yolo)

    # export
    exported_model_path = f'checkpoints/{TRAIN_MODEL_NAME}.tflite'
    with open(exported_model_path, "wb") as flatbuffer_file:
        flatbuffer_file.write(flatbuffer_bytes)
    print(f'exported to: {exported_model_path}')