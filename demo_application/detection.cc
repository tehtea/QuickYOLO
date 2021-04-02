/**
 * Adapted from
 *https://github.com/finnickniu/tensorflow_object_detection_tflite/blob/master/demo.cpp
 * Additional reference from
 *https://github.com/aiden-dai/ai-tflite-opencv/blob/master/object_detection/test_camera.py
 * Last reference for normalizing input from
 *https://stackoverflow.com/questions/42266742/how-to-normalize-image-in-opencv
 * A simple demo object detection application which loads model.tflite and
 *labelmap.txt from current directory and performs inference on camera output
 **/
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>

#include "larq_compute_engine/tflite/kernels/lce_ops_register.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include <opencv2/opencv.hpp>

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

// change these variables accordingly
#define INPUT_SIZE 224
#define MODEL_FILE "model.tflite"
#define LABEL_MAP_FILE "labelmap.txt"
#define NUM_THREADS 4

#define NO_PRESS 255

using namespace tflite;
using namespace std;
using namespace cv;
using namespace std::chrono;

struct Object {
  cv::Rect rec;
  int class_id;
  float prob;
};

// global variables
int frame_counter = 0;
std::time_t time_begin = std::time(0);
std::time_t last_delta_t = 0;

std::vector<std::string> initialize_labels() {
  // get the list of labels from labelmap
  std::vector<std::string> labels;
  std::ifstream input(LABEL_MAP_FILE);
  for (std::string line; getline(input, line);) {
    labels.push_back(line);
  }
  return labels;
}

void preprocess(
  cv::VideoCapture& cam,
  cv::Mat& original_image,
  cv::Mat& resized_image
) {
  // read frame from camera
  auto success = cam.read(original_image);
  if (!success) {
    std::cerr << "cam fail" << std::endl;
    throw new exception();
  }

  // Resize the original image
  resize(original_image, resized_image, Size(INPUT_SIZE, INPUT_SIZE));

  // Convert input image to Float and normalize
  resized_image.convertTo(resized_image, CV_32FC3, 1.0 / 255, 0);
}

std::vector<Object> postprocess(
  float* output_scores_tensor,
  float* output_boxes_tensor,
  int* output_classes_tensor,
  int* output_selected_idx_tensor,
  unsigned long num_selected_idx,
  double cam_width,
  double cam_height
) {
  // filter selected_idx to get non-zeros (don't use selected_idx == 0 because output is zero-padded by tflite)
  std::vector<int> selected_idx;
  for (int i = 0; i < num_selected_idx; i++) {
    auto selected_id = output_selected_idx_tensor[i];
    if (selected_id == 0) {
      continue;
    }
    selected_idx.push_back(selected_id);
  }

  // populate the vector of objects
  std::vector<Object> objects;
  for (auto selected_id : selected_idx) {
    // get box dimensions
    auto xmin = output_boxes_tensor[selected_id * 4 + 0] * cam_width;
    auto ymin = output_boxes_tensor[selected_id * 4 + 1] * cam_height;
    auto xmax = output_boxes_tensor[selected_id * 4 + 2] * cam_width;
    auto ymax = output_boxes_tensor[selected_id * 4 + 3] * cam_height;
    auto width = xmax - xmin;
    auto height = ymax - ymin;

    // populate an object proper
    Object object;
    object.class_id = output_classes_tensor[selected_id];
    object.rec.x = xmin;
    object.rec.y = ymin;
    object.rec.width = width;
    object.rec.height = height;
    object.prob = output_scores_tensor[selected_id];
    objects.push_back(object);
  }

  return objects;
}

void draw_boxes(std::vector<Object> objects, cv::Mat& original_image, std::vector<string>& labels) {
  // show the bounding boxes on GUI
  for (int l = 0; l < objects.size(); l++) {
    Object object = objects.at(l);
    auto score = object.prob;
    auto score_rounded = ((float)((int)(score * 100 + 0.5)) / 100);

    Scalar color = Scalar(255, 0, 0);
    auto class_id = object.class_id;
    auto class_label = labels[class_id];

    std::ostringstream label_txt_stream;
    label_txt_stream << class_label << " (" << score_rounded << ")";
    std::string label_txt = label_txt_stream.str();

    cv::rectangle(original_image, object.rec, color, 1);
    cv::putText(original_image, 
                label_txt,
                cv::Point(object.rec.x, object.rec.y - 5),
                cv::FONT_HERSHEY_COMPLEX, .8, cv::Scalar(10, 255, 30));
  }
}

void profile_execution() {
  frame_counter++;
  std::time_t delta_t = std::time(0) - time_begin;
  if (delta_t % 60 == 0 && delta_t != last_delta_t) { // delta_t % 60 == 0 can be true multiple times
    std::cout << "Frames Processed in the last minute: " << frame_counter << std::endl;
    frame_counter = 0;
    last_delta_t = delta_t;
  }
}

void test() {
  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(MODEL_FILE);

  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  compute_engine::tflite::RegisterLCECustomOps(&resolver);

  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> interpreter;
  builder(&interpreter, NUM_THREADS);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  std::vector<std::string> labels = initialize_labels();

  std::cout << "Initialized interpreter and labels" << std::endl;

  // declare the camera
  auto cam = cv::VideoCapture(0, cv::CAP_V4L);

  // get camera resolution
  auto cam_width = cam.get(cv::CAP_PROP_FRAME_WIDTH);
  auto cam_height = cam.get(cv::CAP_PROP_FRAME_HEIGHT);

  std::cout << "Got the camera, see cam_width and cam_height: " << cam_width
            << ',' << cam_height << std::endl;

  // initialize frame counter
  int frameCounter = 0;
  std::time_t timeBegin = std::time(0);

  // allocate tensor before inference loop
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  
  // start camera loop
  while (true) {
    // declare image buffers
    cv::Mat original_image;
    cv::Mat resized_image;

    preprocess(
      cam,
      original_image,
      resized_image
    );

    // Declare the input
    float* input = interpreter->typed_input_tensor<float>(0);

    // feed input
    memcpy(input, resized_image.data,
            resized_image.total() * resized_image.elemSize());

    // run inference
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

    // declare the output buffers
    float* output_boxes_tensor = interpreter->typed_output_tensor<float>(0);
    float* output_scores_tensor = interpreter->typed_output_tensor<float>(1);
    int* output_classes_tensor = interpreter->typed_output_tensor<int>(2);
    int* output_selected_idx_tensor = interpreter->typed_output_tensor<int>(3);

    auto num_selected_idx = *(interpreter->output_tensor(3)->dims[0].data);

    // get boxes from the output buffers
    std::vector<Object> objects = postprocess(
      output_scores_tensor, 
      output_boxes_tensor, 
      output_classes_tensor, 
      output_selected_idx_tensor, 
      num_selected_idx,
      cam_width,
      cam_height
    );

    // draw the boxes on the original image
    draw_boxes(objects, original_image, labels);

    // profile the code whenever you can
    profile_execution();

    // show image on screen
    cv::imshow("QuickYOLO", original_image);

    // go to next frame after 1ms if no key pressed
    auto k = cv::waitKey(1) & 0xFF;
    if (k != NO_PRESS) {
      std::cout << "See k: " << k << std::endl;
      break;
    }
  }
}

int main(int argc, char** argv) {
  test();
  return 0;
}
