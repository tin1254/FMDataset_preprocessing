cmake_minimum_required(VERSION 2.8)
project(PREPROCESSING)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Debug)
ENDIF()

find_package(OpenCV 3.0 REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)


add_executable(preprocess preprocess.cpp)
target_link_libraries(preprocess ${OpenCV_LIBS} ${EIGEN3_LIBS})

