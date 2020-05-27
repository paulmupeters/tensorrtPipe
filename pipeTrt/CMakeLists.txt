cmake_minimum_required(VERSION 3.0.0)
project(pipetrt VERSION 0.1.0)

set(CMAKE_BUILD_TYPE Debug)
#
# CUDA Configuration
#
find_package(CUDA REQUIRED)
set(CUDA_VERBOSE_BUILD ON)

set(CMAKE_INSTALL_RPATH ${COMPUTE_LIBRARY_ROOT}/build)

# Specify the cuda host compiler to use the same compiler as cmake.
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(PLUGIN_SOURCES 
    src/multiStream.cpp
    src/pipe.cpp
    src/parsePrototxt.cpp
    src/cpuEngine.cpp
)
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

find_library(ARM_LIBRARY arm_compute
  HINTS ${COMPUTE_LIBRARY_ROOT}
  PATH_SUFFIXES build)
find_library(ACL_CORE_LIBRARY arm_compute_core
  HINTS ${COMPUTE_LIBRARY_ROOT}
  PATH_SUFFIXES build)

find_library(ACL_GRAPH_LIBRARY arm_compute_graph
  HINTS ${COMPUTE_LIBRARY_ROOT}
  PATH_SUFFIXES build)

set(ACL_LIBRARY ${ARM_LIBRARY} ${ACL_CORE_LIBRARY} ${ACL_GRAPH_LIBRARY})

MESSAGE(STATUS "Found arm compute libs at ${ACL_LIBRARY}")



target_include_directories(pipeTrt PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/include 
  ${CUDA_INCLUDE_DIRS} 
  ${TENSORRT_INCLUDE_DIR} 
  ${CUDNN_INCLUDE_DIR}
  ${COMPUTE_LIBRARY_ROOT}
  ${COMPUTE_LIBRARY_ROOT}/include
  ${CAFFE_INCLUDE_DIR}
  ${BOOST_INCLUDE_DIR}
  ${CAFFE_PROTO_INCLUDE})

  include(FindProtobuf)
  find_package(Protobuf REQUIRED)
  include_directories(${PROTOBUF_INCLUDE_DIR})

target_link_libraries(pipeTrt ${TENSORRT_LIBRARY} ${ACL_LIBRARY}
  ${CAFFE_LIBRARY_PATH}/libcaffe.so ${PROTOBUF_LIBRARY})
find_package(Threads REQUIRED)
if(THREADS_HAVE_PTHREAD_ARG)
  target_compile_options(pipeTrt PUBLIC "-pthread")
endif()
if(CMAKE_THREAD_LIBS_INIT)
  target_link_libraries(pipeTrt "${CMAKE_THREAD_LIBS_INIT}")
endif()


