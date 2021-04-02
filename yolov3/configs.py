#================================================================
#
#   File name   : configs.py
#   Author      : PyLessons
#   Created date: 2020-08-18
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : yolov3 configuration file
#
#================================================================

# YOLO options
YOLO_TYPE                   = "quickyolov2" # yolov4 or yolov3 or yolov2 or quickyolov2 (no tiny for v2)
YOLO_FRAMEWORK              = "tf" # "tf" or "trt"
YOLO_V2_WEIGHTS             = "model_data/darknet19_448.conv.23"
YOLO_V3_WEIGHTS             = "model_data/yolov3.weights"
YOLO_V4_WEIGHTS             = "model_data/yolov4.weights"
YOLO_V3_TINY_WEIGHTS        = "model_data/yolov3-tiny.weights"
YOLO_V4_TINY_WEIGHTS        = "model_data/yolov4-tiny.weights"
YOLO_TRT_QUANTIZE_MODE      = "INT8" # INT8, FP16, FP32
YOLO_CUSTOM_WEIGHTS         = "checkpoints/quickyolov2_custom" # "checkpoints/yolov3_custom" # used in evaluate_mAP.py and custom model detection, if not using leave False
                            # YOLO_CUSTOM_WEIGHTS also used with TensorRT and custom model detection
YOLO_COCO_CLASSES           = "model_data/coco/coco.names"
YOLO_STRIDES                = [8, 16, 32]
YOLO_IOU_LOSS_THRESH        = 0.5
YOLO_ANCHOR_PER_SCALE       = 3
YOLO_MAX_BBOX_PER_SCALE     = 100
YOLO_INPUT_SIZE             = 224
if YOLO_TYPE                == "yolov4":
    YOLO_ANCHORS            = [[[12,  16], [19,   36], [40,   28]],
                               [[36,  75], [76,   55], [72,  146]],
                               [[142,110], [192, 243], [459, 401]]]
if YOLO_TYPE                == "yolov3":
    YOLO_ANCHORS            = [[[10,  13], [16,   30], [33,   23]],
                               [[30,  61], [62,   45], [59,  119]],
                               [[116, 90], [156, 198], [373, 326]]]
if YOLO_TYPE                == "yolov2" or YOLO_TYPE == "quickyolov2":
    YOLO_STRIDES            = [32, 32, 32]
    YOLO_ANCHORS            = [[[62, 48], [84,  107], [200, 176]],
                               [[0,   0], [0,     0], [0,     0]],
                               [[0,   0], [0,     0], [0,     0]]]

# Train options
TRAIN_YOLO_TINY             = False
TRAIN_SAVE_BEST_ONLY        = True # saves only best model according validation loss (True recommended)
TRAIN_SAVE_CHECKPOINT       = False # saves all best validated checkpoints in training process (may require a lot disk space) (False recommended)
TRAIN_CLASSES               = "model_data/voc_names.txt"
TRAIN_ANNOT_PATH            = "model_data/voc_train.txt"
TRAIN_LOGDIR                = "log" + f"_{YOLO_TYPE}"
TRAIN_CHECKPOINTS_FOLDER    = "checkpoints"
TRAIN_MODEL_NAME            = f"{YOLO_TYPE}_custom"
TRAIN_LOAD_IMAGES_TO_RAM    = False # With True faster training, but need more RAM
TRAIN_BATCH_SIZE            = 32
TRAIN_INPUT_SIZE            = YOLO_INPUT_SIZE
TRAIN_DATA_AUG              = True
TRAIN_TRANSFER              = False # must be false for quickyolo
TRAIN_FROM_CHECKPOINT       = False # "checkpoints/yolov3_custom"
TRAIN_LR_INIT               = 1e-3
TRAIN_LR_END                = 1e-5
TRAIN_WARMUP_EPOCHS         = 2
TRAIN_EPOCHS                = 90

# TEST options
TEST_ANNOT_PATH             = "model_data/voc_test.txt"
TEST_BATCH_SIZE             = 32
TEST_INPUT_SIZE             = YOLO_INPUT_SIZE
TEST_DATA_AUG               = False
TEST_DECTECTED_IMAGE_PATH   = ""
TEST_SCORE_THRESHOLD        = 0.4
TEST_IOU_THRESHOLD          = 0.5

#YOLOv3-TINY and YOLOv4-TINY WORKAROUND
if TRAIN_YOLO_TINY:
    TRAIN_MODEL_NAME        += '_Tiny'
    YOLO_STRIDES            = [16, 32, 64]    
    YOLO_ANCHORS            = [[[10,  14], [23,   27], [37,   58]],
                               [[81,  82], [135, 169], [344, 319]],
                               [[0,    0], [0,     0], [0,     0]]]