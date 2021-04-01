# QuickYOLO

This repository is an end-to-end example for training a binarized object detector on the PASCAL VOC dataset, exporting it for inference, and using the exported model in a demo application. 

It is forked from https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3, which had the most comprehensive training pipeline based on TensorFlow 2 that I can find for training a YOLO detector.

Libraries used:

- Larq 0.11.2
- TensorFlow 2.3.2
- OpenCV 4

Platform tested on:

- NVIDIA Jetson Nano 2GB

The example model is based off YOLOv2, with QuickNet as the backbone, and a head that is based off the transition blocks in QuickNet. Here are the statistics for this model:

- mAP@0.5 for VOC2007 Test Set: 0.50
- Average Inference Time (Tested using the LCE benchmarking tool on a Jetson Nano):  26.5 ms


## Demo through your computer (no speedups, just a rather meh detector)
1. `cd demo_application`
2. `python demo_lce.py`

## End-to-End Usage
For a quick demo, just skip to the deployment step with the provided 
`model.tflite` and `labelmap.tflite` in `demo_application/`

### 1. Prepare datset
1. Run `dataset_preparation/dataset_preparation.sh`
2. Run `python dataset_preparation/migrate_data.py`
3. Run `python dataset_preparation/flatten_voc.py`
4. Run `mv VOCdevkit/train voc_data && mv VOCdevkit/test voc_data`
5. Run `python tools/XML_to_YOLOv3.py`

### 2. Train
1. Run `python train.py`

### 3. (Optional) Evaluate accuracy
1. Change `YOLO_CUSTOM_WEIGHTS` in `configs.py` to the checkpoint for the trained model, e.g. `checkpoints/quickyolov2_custom`
2. Run `python evaluate_mAP.py`

### 4. Export
1. If haven't, change `YOLO_CUSTOM_WEIGHTS` in `configs.py` to the checkpoint for the trained model, e.g. `checkpoints/quickyolov2_custom`
2. Run `python export_larq_model.py`

### 5. Deploy
1. Install OpenCV4 (run `demo_application/install_lce.sh` from current directory)
2. Install Larq Compute Engine (run `install_lce.sh` from current directory)
3. Move `demo_application/detection.cc` to `${larq-compute-engine}/examples`
4. Move `demo_application/Makefile` to `${larq-compute-engine}/larq_compute_engine/tflite/build_make/Makefile`
5. Build using `larq_compute_engine/tflite/build_make/build_lce.sh --native`
6. Move the exported model, and `voc_names.txt` to `gen/linux_aarch64`, renaming them to `model.tflite` and `labelmap.txt` respectively.

## References
- A lot of the training code was forked from `https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3`
- Demo application took reference from `https://github.com/finnickniu/tensorflow_object_detection_tflite/blob/master/demo.cpp`

## TODOs
- [ ] Clean up the code base in this repo
- [ ] Incorporate spatial pyramids
- [ ] Make example application buildable using Bazel
- [ ] Add multiprocessing for camera I/O in demo application

## License
MIT
