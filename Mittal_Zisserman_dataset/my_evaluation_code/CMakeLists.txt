cmake_minimum_required(VERSION 2.84)
project( my_evaluation )
find_package( OpenCV REQUIRED )
include_directories( /usr/local/include/opencv4 )

set(CMAKE_CXX_STANDARD 17)

#add_library( util STATIC lib/util.h lib/util.cpp)
#add_library( segment STATIC lib/segment.cpp lib/segment.h )
#add_library( callbacks STATIC lib/callbacks.cpp lib/callbacks.h )

add_executable( my_evaluation_code my_evaluation.cpp )

target_link_libraries( my_evaluation_code ${OpenCV_LIBS} )
