"""
Sample script for getting camera feed and showing predictions from a Larq-ed model

References:
- https://github.com/aiden-dai/ai-tflite-opencv/blob/master/object_detection/test_camera.py
- https://stackoverflow.com/questions/5998245/get-current-time-in-milliseconds-in-python
"""
from datetime import datetime
from glob import glob

import numpy as np
import cv2
import larq_compute_engine as lce
from PIL import Image

# change these variables accordingly
INPUT_SIZE = 224
MODEL_FILE = "model.tflite"
LABEL_MAP_FILE = "labelmap.txt"
NUM_THREADS = 4

ESC = 27

class FpsManager:
    def __init__(self):
        self.last_time = datetime.now().microsecond

    def calculate_fps(self):
        current_time = datetime.now().microsecond
        delta = current_time - self.last_time + 1e-6
        self.last_time = current_time
        return max(10 ** 6 / delta, 0)

class Demo:
    def __init__(self):
        self.initialize_labels()
        self.load_model()

    def initialize_labels(self):
        with open(LABEL_MAP_FILE) as f:
            labels = f.readlines()
        labels = list(map(lambda label: label.strip(), labels))
        self.labels = labels
        print(labels)

    def load_model(self):
        with open(MODEL_FILE, 'rb') as f:
            flatbuffer_bytes = f.read()
        test_interpreter = lce.testing.Interpreter(
            flatbuffer_bytes, num_threads=4, use_reference_bconv=False
        )
        self.interpreter = test_interpreter

    def initialize_camera(self):
        self.cam = cv2.VideoCapture(0)

        # get camera resolution
        self.cam_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f'Got the camera, see cam_width and cam_height: {(self.cam_width, self.cam_height)}')

    def initialize_fps_manager(self):
        self.fps_manager = FpsManager()

    def postprocess(self, boxes, scores, classes, selected_idx):
        selected_boxes, selected_scores, selected_classes = [], [], []
        selected_idx = selected_idx[selected_idx > 0] # we don't use id 0 because selected_idx from tflite is zero-padded
        for selected_id in selected_idx:
            selected_boxes.append(boxes[selected_id])
            selected_scores.append(scores[selected_id])
            selected_classes.append(classes[selected_id])

        return selected_boxes, selected_scores, selected_classes

    def infer(self, frame):
        input_image = Image.fromarray(frame)
        input_image = input_image.resize((INPUT_SIZE, INPUT_SIZE))
        input_image = np.asarray(input_image, dtype=np.float32) / 255.

        input_image = np.expand_dims(input_image, 0)

        boxes, scores, classes, selected_idx = self.interpreter.predict(input_image)
        boxes, scores, classes = self.postprocess(boxes, scores, classes, selected_idx)

        return boxes, scores, classes

    def draw_on_frame(self, boxes, scores, classes, original_frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = 0.6
        color = (30, 255, 30)  # Green color
        thickness = 2

        frame_height, frame_width, _ = original_frame.shape

        for box, score, cls_id in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            x1 = int(x1 * frame_width)
            y1 = int(y1 * frame_height)
            x2 = int(x2 * frame_width)
            y2 = int(y2 * frame_height)

            prediction_text = f'{self.labels[cls_id]} ({str(round(score, 3))})'

            cv2.putText(original_frame, prediction_text, (int(x1 - size * 2), int(y1 - size * 2)), font, size, color, thickness)
            cv2.rectangle(original_frame, (x1, y1), (x2, y2), color, thickness)

        return original_frame

    def display_predictions(self, boxes, scores, classes, original_frame):
        original_frame = self.draw_on_frame(boxes, scores, classes, original_frame)
        cv2.imshow('Object Detection', original_frame)

    def draw_to_file(self, boxes, scores, classes, original_frame, output_path):
        original_frame = self.draw_on_frame(boxes, scores, classes, original_frame)
        cv2.imwrite(output_path, original_frame)
        print(f'Saved prediction in {output_path}')

    def display_fps(self, original_frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        size = 0.6
        color = (0, 0, 255)  # Red color
        thickness = 1

        fps = self.fps_manager.calculate_fps()
        fps = str(round(fps, 1))
        fps_string = f'FPS: {fps}'
        cv2.putText(original_frame, fps_string, (30, 30), font, size, color, thickness)

    def run_on_images(self):
        test_images = glob('../voc_data/test/*.jpg')
        test_images = np.random.choice(test_images, size=10)
        for i, test_image in enumerate(test_images):
            output_path = f'../IMAGES/pred_{i}.jpg'
            frame = cv2.imread(test_image)

            boxes, scores, classes = self.infer(frame)
            self.draw_to_file(boxes, scores, classes, frame, output_path)

    def run_on_webcam(self):
        self.initialize_camera()
        self.initialize_fps_manager()

        # start camera loop
        while True:
            ret, frame = self.cam.read()
            if ret != True:
                raise Exception('failed to read from camera')

            boxes, scores, classes = self.infer(frame)
            self.display_fps(frame)
            self.display_predictions(boxes, scores, classes, frame)

            key = cv2.waitKey(30)
            if key == ESC:
                break

        self.cam.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    show_images = True # change accordingly
    show_webcam = not show_images
    demo = Demo()
    if show_images:
        demo.run_on_images()
    elif show_webcam:
        demo.run_on_webcam()