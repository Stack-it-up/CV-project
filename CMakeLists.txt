cmake_minimum_required(VERSION 2.84)
project( Hand_Detect )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} lib )

set( CMAKE_CXX_STANDARD 14 )

add_library( util STATIC lib/util.h src/lib/util.cpp)
#add_library( segment STATIC lib/segment.cpp lib/segment.h )
#add_library( callbacks STATIC lib/callbacks.cpp lib/callbacks.h )

add_executable( detect src/detect.cpp )
add_executable( segment src/segment.cpp )

target_link_libraries( detect util ${OpenCV_LIBS} )
target_link_libraries( segment util ${OpenCV_LIBS} )
