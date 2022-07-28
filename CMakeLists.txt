cmake_minimum_required(VERSION 2.84)
project( Hand_Detect )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} lib )

set( CMAKE_CXX_STANDARD 17 )

add_library( util STATIC lib/Util.h src/lib/Util.cpp)
add_library( detector STATIC lib/detector.h src/lib/detector.cpp )
add_library( snakes STATIC lib/snakes.h src/lib/snakes.cpp )
add_library( detect STATIC src/detect.h src/detect.cpp )
add_library( segment STATIC src/segment.h src/segment.cpp )

#add_executable( detect src/detect.cpp )
#add_executable( segment src/segment.cpp )
add_executable( create_decolored_samples src/create_decolored_samples.cpp )
add_executable( hand_detect src/main.cpp )

target_link_libraries( hand_detect segment detect util detector ${OpenCV_LIBS} )
target_link_libraries( segment util snakes ${OpenCV_LIBS} )
target_link_libraries( create_decolored_samples ${OpenCV_LIBS} )
