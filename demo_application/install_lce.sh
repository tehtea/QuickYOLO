# 0. Clone LCE
git clone --depth 1 --branch v0.5.0 https://github.com/larq/compute-engine.git

# 1. Install bazel
sudo wget https://github.com/bazelbuild/bazel/releases/download/4.0.0/bazel-4.0.0-linux-arm64 -O /usr/local/bin/bazel
sudo chmod +x /usr/local/bin/bazel
bazel

# 2. Build TensorFlow dependency. 
# Steps for third_party/tensorflow referenced from here: https://www.tensorflow.org/lite/guide/build_arm64#compile_natively_on_arm64
cd third_party/tensorflow
sudo apt-get install build-essential
./tensorflow/lite/tools/make/download_dependencies.sh
./tensorflow/lite/tools/make/build_aarch64_lib.sh

# 3. Then, build at root dir of larq compute engine using:
larq_compute_engine/tflite/build_make/build_lce.sh --native

