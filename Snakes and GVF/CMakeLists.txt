cmake_minimum_required(VERSION 2.84)
project( snakes )
find_package( OpenCV REQUIRED )
include_directories( /usr/local/include/opencv4 )

set(CMAKE_CXX_STANDARD 17)

add_executable( snake snake.cpp )
add_executable( gvf gvf.cpp )

target_link_libraries( snake ${OpenCV_LIBS} )
target_link_libraries( gvf ${OpenCV_LIBS} )