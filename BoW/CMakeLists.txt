cmake_minimum_required(VERSION 3.0)
project(BoW)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(BoW BoW.cpp)

target_link_libraries(BoW ${OpenCV_LIBS})