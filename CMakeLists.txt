cmake_minimum_required(VERSION 3.0.0)
project(pipetrt VERSION 0.1.0)

set(CMAKE_BUILD_TYPE Debug)


# add lib pipeTrt
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pipeTrt sample.out)
include_directories({$CMAKE_CURRENT_SOURCE_DIR}/pipeTrt)



add_executable(sample_mnist pipeMNISTTest.cpp)
target_link_libraries(sample_mnist pipeTrt)

install(TARGETS sample_mnist DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install/)