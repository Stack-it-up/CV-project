cmake_minimum_required(VERSION 2.84)
project( snake_segment )
find_package( OpenCV REQUIRED )
include_directories( /usr/local/include/opencv4 )

set(CMAKE_CXX_STANDARD 17)

add_library( snakes STATIC snakes.h snakes.cpp)

add_executable( snake snake.cpp )

target_link_libraries( snake ${OpenCV_LIBS} snakes)