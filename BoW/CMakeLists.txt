cmake_minimum_required(VERSION 3.0)
project(cascade)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cascade cascade.cpp)

target_link_libraries(cascade ${OpenCV_LIBS})