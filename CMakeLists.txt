cmake_minimum_required(VERSION 2.84)
project( Hand_Detect )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} lib )

set( CMAKE_CXX_STANDARD 14 )

add_library( util STATIC lib/Util.h src/lib/Util.cpp)
add_library( callbacks STATIC lib/callbacks.h src/lib/callbacks.cpp )
add_library( detector STATIC lib/detector.h src/lib/detector.cpp )

add_executable( detect src/detect.cpp )
add_executable( segment src/segment.cpp )
add_executable( create_decolored_samples src/create_decolored_samples.cpp )

target_link_libraries( detect util detector ${OpenCV_LIBS} )
target_link_libraries( segment util callbacks ${OpenCV_LIBS} )
target_link_libraries( create_decolored_samples ${OpenCV_LIBS} )
