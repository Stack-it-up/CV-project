cmake_minimum_required(VERSION 2.84)
project( Ensemble )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} include)

set( CMAKE_CXX_STANDARD 17 )

add_library(util STATIC include/Util.h src/Util.cpp)
add_library( callbacks STATIC include/callbacks.h src/callbacks.cpp )
add_library( snakes STATIC include/snakes.h src/snakes.cpp )
add_library(kmeans STATIC include/kmeans.h src/kmeans.cpp)

add_executable( main src/main.cpp )

target_link_libraries( main util callbacks snakes kmeans ${OpenCV_LIBS} )