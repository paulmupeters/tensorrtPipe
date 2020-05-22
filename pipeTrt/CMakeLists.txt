cmake_minimum_required(VERSION 3.0.0)
project(pipetrt VERSION 0.1.0)

set(CMAKE_BUILD_TYPE Debug)



#
# CUDA Configuration
#
find_package(CUDA REQUIRED)
set(CUDA_VERBOSE_BUILD ON)

# Specify the cuda host compiler to use the same compiler as cmake.
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(PLUGIN_SOURCES 
    src/multiStream.cpp
    src/pipe.cpp
    src/parsePrototxt.cpp
)

# ARMNN
find_path(ARMNN_INCLUDE_DIR armnn
  HINTS ${ARMNN_ROOT}
  PATH_SUFFIXES include)
MESSAGE(STATUS "Found ARMNN headers at ${ARMNN_INCLUDE_DIR}")
find_library(ARMNN_CAFFE_PARSER armnnCaffeParser 
  HINTS ${ARMNN_ROOT}
  PATH_SUFFIXES build)
find_library(ARMNN_INFER armnn
  HINTS ${ARMNN_ROOT}
  PATH_SUFFIXES build)
set(ARMNN_LIBRARY ${ARMNN_INFER} ${ARMNN_CAFFE_PARSER})
MESSAGE(STATUS "Found ARMNN libs at ${ARMNN_LIBRARY}")
find_package_handle_standard_args(
  ARMNN DEFAULT_MSG ARMNN_LIBRARY ARMNN_INCLUDE_DIR)
if(NOT ARMNN_FOUND)
  message(ERROR
    "Cannot find ARMNN library.")
endif()

#add TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR} 
  PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_PARSER nvparsers
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_PARSER})
MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
find_package_handle_standard_args(
  TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
if(NOT TENSORRT_FOUND)
  message(ERROR
    "Cannot find TensorRT library.")
endif()
	

list(APPEND CUDA_NVCC_FLAGS "-D_FORCE_INLINES -Xcompiler -fPIC")
CUDA_INCLUDE_DIRECTORIES(${CUDNN_INCLUDE_DIR} ${TENSORRT_INCLUDE_DIR})
CUDA_ADD_LIBRARY(pipeTrt STATIC ${PLUGIN_SOURCES})




target_include_directories(pipeTrt PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/include 
  ${CUDA_INCLUDE_DIRS} 
  ${TENSORRT_INCLUDE_DIR} 
  ${CUDNN_INCLUDE_DIR}
  ${ARMNN_INCLUDE_DIR}
  ${CAFFE_INCLUDE_DIR}
  ${BOOST_INCLUDE_DIR}
  ${CAFFE_PROTO_INCLUDE})

  include(FindProtobuf)
  find_package(Protobuf REQUIRED)
  include_directories(${PROTOBUF_INCLUDE_DIR})

target_link_libraries(pipeTrt ${TENSORRT_LIBRARY} ${ARMNN_LIBRARY} 
  ${CAFFE_LIBRARY_PATH}/libcaffe.so ${PROTOBUF_LIBRARY})
find_package(Threads REQUIRED)
if(THREADS_HAVE_PTHREAD_ARG)
  target_compile_options(pipeTrt PUBLIC "-pthread")
endif()
if(CMAKE_THREAD_LIBS_INIT)
  target_link_libraries(pipeTrt "${CMAKE_THREAD_LIBS_INIT}")
endif()


