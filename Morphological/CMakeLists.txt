cmake_minimum_required(VERSION 2.8)

project(morphological)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(morphological src/morphological.cpp)

target_link_libraries(morphological ${OpenCV_LIBS})