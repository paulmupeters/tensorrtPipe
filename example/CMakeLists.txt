add_executable(sample_mnist pipeMNISTTest.cpp prepData.cpp)

target_link_libraries(sample_mnist PRIVATE pipeTrt)