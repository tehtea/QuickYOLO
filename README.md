# QuickYOLO

This repository is an end-to-end example for training a binarized object detector on the PASCAL VOC dataset, exporting it for inference, and using the exported model in a demo application. 

It is forked from https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3, which had the most comprehensive training pipeline based on TensorFlow 2 that I can find for training a YOLO detector.

Libraries used:

- Larq 0.5.0
- TensorFlow 2.4
- OpenCV 4

Platform tested on:

- NVIDIA Jetson Nano 2GB

The example model is based off YOLOv2, with QuickNet as the backbone, and a head that is based off the transition blocks in QuickNet. Here are the statistics for this model:

- mAP@0.5 for VOC2007 Test Set: 0.214 (A bit sad, I know :cry: )
- Average Inference Time (Tested using the LCE benchmarking tool on a Jetson Nano):  26.5 ms

## Usage

### 1. Prepare datset
1. Run `dataset_preparation/dataset_preparation.sh`
2. Run `python dataset_preparation/migrate_data.py`
3. Run `python dataset_preparation/flatten_voc.py`
4. Run `mv VOCdevkit/train voc_data; mv VOCdevkit/test voc_data`
5. Run `python tools/XML_to_YOLOv3.py`

### 2. Train
1. Run `python train.py`

### 3. Export
1. Change `CHECKPOINT_WEIGHTS` to `True` in `configs.py`
2. Run `python export_to_larq.py`

### 4. Deploy
1. Install OpenCV4 (run `demo_application/install_lce.sh` from current directory)
2. Install Larq Compute Engine (run `install_lce.sh` from current directory)
3. Move `demo_application/detection.cc` to `${larq-compute-engine}/examples`
4. Move `demo_application/Makefile` to `${larq-compute-engine}/larq_compute_engine/tflite/build_make/Makefile`
5. Build using `make`

## References
- A lot of the training code was forked from `https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3`
- Demo application took reference from `https://github.com/finnickniu/tensorflow_object_detection_tflite/blob/master/demo.cpp`

## License
MIT
