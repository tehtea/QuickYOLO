#================================================================
#
#   File name   : utils.py
#   Author      : PyLessons
#   Created date: 2020-09-27
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : additional yolov3 and yolov4 functions
#
#================================================================
from multiprocessing import Process, Queue, Pipe
import cv2
import time
import random
import colorsys
import numpy as np
import tensorflow as tf
from yolov3.configs import *
from yolov3.yolov4 import *
from tensorflow.python.saved_model import tag_constants

def _load_yolov2_weights(model, weights_file):
    class WeightReader:
        def __init__(self, weight_file):
            self.offset = 4
            self.all_weights = np.fromfile(weight_file, dtype='float32')
            
        def read_bytes(self, size):
            self.offset = self.offset + size
            return self.all_weights[self.offset-size:self.offset]
        
        def reset(self):
            self.offset = 4

    weight_reader = WeightReader(weights_file)
    weight_reader.reset()
    nb_conv = 18

    for i in range(1, nb_conv+1):
        conv_layer_name = f'conv_{i}'
        conv_layer = model.get_layer(conv_layer_name)
        conv_layer.trainable = True
        
        norm_layer_name = f'norm_{i}'
            
        norm_layer = model.get_layer(norm_layer_name)
        norm_layer.trainable = True
        
        size = np.prod(norm_layer.get_weights()[0].shape)

        beta  = weight_reader.read_bytes(size)
        gamma = weight_reader.read_bytes(size)
        mean  = weight_reader.read_bytes(size)
        var   = weight_reader.read_bytes(size)

        weights = norm_layer.set_weights([gamma, beta, mean, var])       
            
        if len(conv_layer.get_weights()) > 1:
            bias   = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2,3,1,0])
            conv_layer.set_weights([kernel, bias])
        else:
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2,3,1,0])
            conv_layer.set_weights([kernel])
    assert weight_reader.offset == weight_reader.all_weights.size, 'failed to read all data'

def load_yolo_weights(model, weights_file):
    tf.keras.backend.clear_session() # used to reset layer names
    # load Darknet original weights to TensorFlow model
    if YOLO_TYPE == "yolov2":
        return _load_yolov2_weights(model, weights_file)
    if YOLO_TYPE == "yolov3":
        range1 = 75 if not TRAIN_YOLO_TINY else 13
        range2 = [58, 66, 74] if not TRAIN_YOLO_TINY else [9, 12]
    if YOLO_TYPE == "yolov4":
        range1 = 110 if not TRAIN_YOLO_TINY else 21
        range2 = [93, 101, 109] if not TRAIN_YOLO_TINY else [17, 20]
    
    with open(weights_file, 'rb') as wf:
        major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

        j = 0
        for i in range(range1):
            if i > 0:
                conv_layer_name = 'conv2d_%d' %i
            else:
                conv_layer_name = 'conv2d'
                
            if j > 0:
                bn_layer_name = 'batch_normalization_%d' %j
            else:
                bn_layer_name = 'batch_normalization'
            
            conv_layer = model.get_layer(conv_layer_name)
            filters = conv_layer.filters
            k_size = conv_layer.kernel_size[0]
            in_dim = conv_layer.input_shape[-1]

            if i not in range2:
                # darknet weights: [beta, gamma, mean, variance]
                bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                # tf weights: [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                bn_layer = model.get_layer(bn_layer_name)
                j += 1
            else:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, k_size, k_size)
            conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

            if i not in range2:
                conv_layer.set_weights([conv_weights])
                bn_layer.set_weights(bn_weights)
            else:
                conv_layer.set_weights([conv_weights, conv_bias])

        assert len(wf.read()) == 0, 'failed to read all data'

def Load_Yolo_model():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        print(f'GPUs {gpus}')
        try: tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError: pass
        
    if YOLO_FRAMEWORK == "tf": # TensorFlow detection
        if YOLO_TYPE == "yolov4":
            Darknet_weights = YOLO_V4_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V4_WEIGHTS
        if YOLO_TYPE == "yolov3":
            Darknet_weights = YOLO_V3_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V3_WEIGHTS
        if YOLO_TYPE == "yolov2":
            Darknet_weights = YOLO_V2_WEIGHTS
            
        if YOLO_CUSTOM_WEIGHTS == False:
            print("Loading Darknet_weights from:", Darknet_weights)
            yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=YOLO_COCO_CLASSES)
            load_yolo_weights(yolo, Darknet_weights) # use Darknet weights
        else:
            checkpoint = f"./checkpoints/{TRAIN_MODEL_NAME}"
            if TRAIN_YOLO_TINY:
                checkpoint += "_Tiny"
            print("Loading custom weights from:", checkpoint)
            yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
            yolo.load_weights(checkpoint)  # use custom weights
        
    elif YOLO_FRAMEWORK == "trt": # TensorRT detection
        saved_model_loaded = tf.saved_model.load(YOLO_CUSTOM_WEIGHTS, tags=[tag_constants.SERVING])
        signature_keys = list(saved_model_loaded.signatures.keys())
        yolo = saved_model_loaded.signatures['serving_default']

    return yolo

def image_preprocess(image, target_size, gt_boxes=None):
    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


def draw_bbox(image, bboxes, CLASSES=YOLO_COCO_CLASSES, show_label=True, show_confidence = True, Text_colors=(255,255,0), rectangle_colors='', tracking=False):   
    NUM_CLASS = read_class_names(CLASSES)
    num_classes = len(NUM_CLASS)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    #print("hsv_tuples", hsv_tuples)
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        if bbox_thick < 1: bbox_thick = 1
        fontScale = 0.75 * bbox_thick
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        # put object rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)

        if show_label:
            # get text label
            score_str = " {:.2f}".format(score) if show_confidence else ""

            if tracking: score_str = " "+str(score)

            try:
                label = "{}".format(NUM_CLASS[class_ind]) + score_str
            except KeyError:
                print("You received KeyError, this might be that you are trying to use yolo original weights")
                print("while using custom classes, if using custom model in configs.py set YOLO_CUSTOM_WEIGHTS = True")

            # get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                  fontScale, thickness=bbox_thick)
            # put filled text rectangle
            cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv2.FILLED)

            # put text above rectangle
            cv2.putText(image, label, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

    return image


def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious

def nms_no_gather(bboxes, iou_threshold, num_classes=20):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)
    
    Don't do gathering here because there is an issue in tflite when there are no boxes to gather
    
    TODO: add back soft-nms
    """
    bboxes = tf.cast(bboxes, dtype=tf.float32)
    boxes, scores, classes = tf.split(bboxes, (4, 1, 1), axis=-1)
    selected_indices = tf.zeros((0,), dtype=tf.int32)
    for cls_id in range(num_classes):
        cls_mask = tf.cast(classes, dtype=tf.int32)
        cls_mask = tf.equal(cls_mask, cls_id)
        cls_mask = tf.cast(cls_mask, dtype=tf.float32)
        cls_boxes = boxes * cls_mask
        cls_scores = scores * cls_mask
        cls_selected_indices = tf.image.non_max_suppression(
            cls_boxes, 
            tf.squeeze(cls_scores, axis=-1), 
            3, 
            iou_threshold, 
            score_threshold=0.0)
        selected_indices = tf.concat((selected_indices, cls_selected_indices), axis=-1)

    combined_boxes = tf.concat((boxes, scores, classes), axis=-1)
    
    return combined_boxes, selected_indices

def nms(bboxes, iou_threshold, sigma=0.3, method='nms', num_classes=20):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)
    """
    combined_boxes, selected_indices = nms_no_gather(bboxes, iou_threshold, num_classes)
    combined_boxes = tf.gather(combined_boxes, selected_indices)

    return combined_boxes

def postprocess_boxes(pred_bbox, original_image=None, input_size=YOLO_INPUT_SIZE, score_threshold=TEST_SCORE_THRESHOLD):
    valid_scale=[0, np.inf]

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = tf.concat([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)

    # 2. (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org) (if original_image provided)
    if original_image is not None:
        org_h, org_w = original_image.shape[:2]
        resize_ratio = min(input_size / org_w, input_size / org_h)

        dw = (input_size - resize_ratio * org_w) / 2
        dh = (input_size - resize_ratio * org_h) / 2

        pred_xmin_xmax = pred_coor[:, 0::2]
        pred_xmin_xmax = 1.0 * (pred_xmin_xmax - dw) / resize_ratio
        pred_ymin_ymax = pred_coor[:, 1::2]
        pred_ymin_ymax = 1.0 * (pred_ymin_ymax - dh) / resize_ratio
        pred_coor = tf.concat([pred_xmin_xmax[:, 0:1], pred_ymin_ymax[:, 0:1], pred_xmin_xmax[:, 1:2], pred_ymin_ymax[:, 1:2]], axis=-1)

    # 3. clip some boxes those are out of range
    if original_image is not None:
        pred_coor = tf.concat([tf.maximum(pred_coor[:, 0:1], 0), tf.maximum(pred_coor[:, 1:2], 0), tf.minimum(pred_coor[:, 2:3], org_w - 1), tf.minimum(pred_coor[:, 3:4], org_h - 1)], axis=-1)
    else:
        pred_coor = tf.concat([tf.maximum(pred_coor[:, 0:1], 0), tf.maximum(pred_coor[:, 1:2], 0), tf.minimum(pred_coor[:, 2:3], 1), tf.minimum(pred_coor[:, 3:4], 1)], axis=-1)
    invalid_mask = tf.logical_or(pred_coor[:, 0:1] > pred_coor[:, 2:3], pred_coor[:, 1:2] > pred_coor[:, 3:4])
    valid_mask = tf.logical_not(invalid_mask)
    valid_mask = tf.cast(valid_mask, dtype=tf.float32)
    pred_coor = pred_coor * valid_mask

    # 4. discard some invalid boxes
    bboxes_scale = tf.sqrt((pred_coor[:, 2:3] - pred_coor[:, 0:1]) * (pred_coor[:, 3:4] - pred_coor[:, 1:2]))
    scale_mask = tf.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))
    scale_mask = tf.squeeze(scale_mask, axis=-1)

    # 5. discard boxes with low scores
    classes = tf.argmax(pred_prob, axis=-1)
    classes = tf.expand_dims(classes, axis=-1)
    classes = tf.cast(classes, dtype=tf.float32)
    scores = pred_conf * tf.reduce_max(pred_prob, axis=-1)
    scores = tf.expand_dims(scores, axis=-1)

    score_mask = scores > score_threshold
    score_mask = tf.squeeze(score_mask, axis=-1)
    mask = tf.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return tf.concat([coors, scores, classes], axis=-1)


def detect_image(Yolo, image_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors=''):
    original_image      = cv2.imread(image_path)
    image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    if YOLO_FRAMEWORK == "tf":
        pred_bbox = Yolo.predict(image_data)
    elif YOLO_FRAMEWORK == "trt":
        batched_input = tf.constant(image_data)
        result = Yolo(batched_input)
        pred_bbox = []
        for key, value in result.items():
            value = value.numpy()
            pred_bbox.append(value)
        
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    
    bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
    bboxes = nms(bboxes, iou_threshold, method='nms')

    original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image = draw_bbox(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)
    # CreateXMLfile("XML_Detections", str(int(time.time())), original_image, bboxes, read_class_names(CLASSES))

    if output_path != '': cv2.imwrite(output_path, image)
    if show:
        # Show the image
        cv2.imshow("predicted image", image)
        # Load and hold the image
        cv2.waitKey(0)
        # To close the window after the required kill value was provided
        cv2.destroyAllWindows()
        
    return image

def Predict_bbox_mp(Frames_data, Predicted_data, Processing_times):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        try: tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError: print("RuntimeError in tf.config.experimental.list_physical_devices('GPU')")
    Yolo = Load_Yolo_model()
    times = []
    while True:
        if Frames_data.qsize()>0:
            image_data = Frames_data.get()
            t1 = time.time()
            Processing_times.put(time.time())
            
            if YOLO_FRAMEWORK == "tf":
                pred_bbox = Yolo.predict(image_data)
            elif YOLO_FRAMEWORK == "trt":
                batched_input = tf.constant(image_data)
                result = Yolo(batched_input)
                pred_bbox = []
                for key, value in result.items():
                    value = value.numpy()
                    pred_bbox.append(value)

            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
            pred_bbox = tf.concat(pred_bbox, axis=0)
            
            Predicted_data.put(pred_bbox)


def postprocess_mp(Predicted_data, original_frames, Processed_frames, Processing_times, input_size, CLASSES, score_threshold, iou_threshold, rectangle_colors, realtime):
    times = []
    while True:
        if Predicted_data.qsize()>0:
            pred_bbox = Predicted_data.get()
            if realtime:
                while original_frames.qsize() > 1:
                    original_image = original_frames.get()
            else:
                original_image = original_frames.get()
            
            bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
            bboxes = nms(bboxes, iou_threshold, method='nms')
            image = draw_bbox(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)
            times.append(time.time()-Processing_times.get())
            times = times[-20:]
            
            ms = sum(times)/len(times)*1000
            fps = 1000 / ms
            image = cv2.putText(image, "Time: {:.1f}FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            #print("Time: {:.2f}ms, Final FPS: {:.1f}".format(ms, fps))
            
            Processed_frames.put(image)

def Show_Image_mp(Processed_frames, show, Final_frames):
    while True:
        if Processed_frames.qsize()>0:
            image = Processed_frames.get()
            Final_frames.put(image)
            if show:
                cv2.imshow('output', image)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break

# detect from webcam
def detect_video_realtime_mp(video_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors='', realtime=False):
    if realtime:
        vid = cv2.VideoCapture(0)
    else:
        vid = cv2.VideoCapture(video_path)

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4
    no_of_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    original_frames = Queue()
    Frames_data = Queue()
    Predicted_data = Queue()
    Processed_frames = Queue()
    Processing_times = Queue()
    Final_frames = Queue()
    
    p1 = Process(target=Predict_bbox_mp, args=(Frames_data, Predicted_data, Processing_times))
    p2 = Process(target=postprocess_mp, args=(Predicted_data, original_frames, Processed_frames, Processing_times, input_size, CLASSES, score_threshold, iou_threshold, rectangle_colors, realtime))
    p3 = Process(target=Show_Image_mp, args=(Processed_frames, show, Final_frames))
    p1.start()
    p2.start()
    p3.start()
        
    while True:
        ret, img = vid.read()
        if not ret:
            break

        original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_frames.put(original_image)

        image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        Frames_data.put(image_data)
        
    while True:
        if original_frames.qsize() == 0 and Frames_data.qsize() == 0 and Predicted_data.qsize() == 0  and Processed_frames.qsize() == 0  and Processing_times.qsize() == 0 and Final_frames.qsize() == 0:
            p1.terminate()
            p2.terminate()
            p3.terminate()
            break
        elif Final_frames.qsize()>0:
            image = Final_frames.get()
            if output_path != '': out.write(image)

    cv2.destroyAllWindows()

def detect_video(Yolo, video_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors=''):
    times, times_2 = [], []
    vid = cv2.VideoCapture(video_path)

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4

    while True:
        _, img = vid.read()

        try:
            original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        except:
            break

        image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        t1 = time.time()
        if YOLO_FRAMEWORK == "tf":
            pred_bbox = Yolo.predict(image_data)
        elif YOLO_FRAMEWORK == "trt":
            batched_input = tf.constant(image_data)
            result = Yolo(batched_input)
            pred_bbox = []
            for key, value in result.items():
                value = value.numpy()
                pred_bbox.append(value)
        
        t2 = time.time()
        
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')
        
        image = draw_bbox(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)

        t3 = time.time()
        times.append(t2-t1)
        times_2.append(t3-t1)
        
        times = times[-20:]
        times_2 = times_2[-20:]

        ms = sum(times)/len(times)*1000
        fps = 1000 / ms
        fps2 = 1000 / (sum(times_2)/len(times_2)*1000)
        
        image = cv2.putText(image, "Time: {:.1f}FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        # CreateXMLfile("XML_Detections", str(int(time.time())), original_image, bboxes, read_class_names(CLASSES))
        
        print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))
        if output_path != '': out.write(image)
        if show:
            cv2.imshow('output', image)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

    cv2.destroyAllWindows()

# detect from webcam
def detect_realtime(Yolo, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors=''):
    times = []
    vid = cv2.VideoCapture(0)

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4

    while True:
        _, frame = vid.read()

        try:
            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        except:
            break
        image_data = image_preprocess(np.copy(original_frame), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        t1 = time.time()
        if YOLO_FRAMEWORK == "tf":
            pred_bbox = Yolo.predict(image_data)
        elif YOLO_FRAMEWORK == "trt":
            batched_input = tf.constant(image_data)
            result = Yolo(batched_input)
            pred_bbox = []
            for key, value in result.items():
                value = value.numpy()
                pred_bbox.append(value)
        
        t2 = time.time()
        
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_frame, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')
        
        times.append(t2-t1)
        times = times[-20:]
        
        ms = sum(times)/len(times)*1000
        fps = 1000 / ms
        
        print("Time: {:.2f}ms, {:.1f} FPS".format(ms, fps))

        frame = draw_bbox(original_frame, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)
        # CreateXMLfile("XML_Detections", str(int(time.time())), original_frame, bboxes, read_class_names(CLASSES))
        image = cv2.putText(frame, "Time: {:.1f}FPS".format(fps), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        if output_path != '': out.write(frame)
        if show:
            cv2.imshow('output', frame)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

    cv2.destroyAllWindows()
