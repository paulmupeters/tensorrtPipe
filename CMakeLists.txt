cmake_minimum_required(VERSION 3.0.0)
project(pipetrt VERSION 0.1.0)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -D_GLIBCXX_USE_CXX11_ABI=1" )

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)



set(TENSORRT_ROOT /usr/src/tensorrt/)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda/)
set(CUDA_CUDART_LIBRARY /usr/local/cuda/lib64/libcudart.so)
set(ARMNN_ROOT /home/nvidia/armnn-caffe/armnn)
set(CAFFE_LIBRARY_PATH /home/nvidia/armnn-caffe/caffe/build/lib )
set(CAFFE_INCLUDE_DIR /home/nvidia/armnn-caffe/caffe/include)
set(CAFFE_PROTO_INCLUDE /home/nvidia/armnn-caffe/caffe/build/src)
set(BOOST_INCLUDE_DIR /home/nvidia/armnn-caffe/boost/include)
set(PROTO_DIR /home/nvidia/armnn-caffe/protobuf-host)

# add subdirectories
add_subdirectory(pipeTrt)
add_subdirectory(example)
