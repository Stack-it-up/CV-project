cmake_minimum_required(VERSION 2.8)

project(Segmentation)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS}
                    include)

add_library(kmeans include/kmeans.h src/kmeans.cpp)
set_target_properties(kmeans PROPERTIES LINKER_LANGUAGE CXX)

add_executable(main src/main.cpp)

target_link_libraries(kmeans ${OpenCV_LIBS})
target_link_libraries(main kmeans ${OpenCV_LIBS})