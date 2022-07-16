cmake_minimum_required(VERSION 2.8)

project(h_det)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(h_det src/tester.cpp src/detector.cpp src/detector.h)

target_link_libraries(h_det ${OpenCV_LIBS})