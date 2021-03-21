# clean existing stuff
rm -rf opencv/
rm -rf opencv_contrib/

# download opencv
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.1.2.zip
unzip opencv.zip
mv opencv-4.1.2 opencv
rm opencv.zip

wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.1.2.zip
unzip opencv_contrib.zip
mv opencv_contrib-4.1.2 opencv_contrib
rm opencv_contrib.zip

# make build folder and compile
cd opencv
mkdir build
cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D WITH_CUDA=ON \
	-D CUDA_ARCH_PTX="" \
	-D CUDA_ARCH_BIN="5.3,6.2,7.2" \
	-D WITH_CUBLAS=ON \
	-D WITH_LIBV4L=ON \
	-D BUILD_opencv_python3=OFF \
	-D BUILD_opencv_python2=OFF \
	-D BUILD_opencv_java=OFF \
	-D WITH_GSTREAMER=ON \
	-D WITH_GTK=ON \
	-D BUILD_TESTS=OFF \
	-D BUILD_PERF_TESTS=OFF \
	-D BUILD_EXAMPLES=OFF \
    -D OPENCV_ENABLE_NONFREE=OFF \
    -D BUILD_opencv_world=OFF \
    -D OPENCV_EXTRA_MODULES_PATH=/home/`whoami`/opencv_contrib/modules \
    ..

make -j4

sudo make install

# # do cleanup in case of errors
# cd ..
# rm -rf build
# mkdir build
# cd build
