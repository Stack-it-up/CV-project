cmake_minimum_required(VERSION 3.21)
project(grabcutDemo)
find_package( OpenCV REQUIRED )
set(CMAKE_CXX_STANDARD 14)

add_executable(grabcutDemo main.cpp)
add_executable(meanshift meanshift.cpp)

add_library(util STATIC Util.h Util.cpp)

include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( grabcutDemo ${OpenCV_LIBS} )
target_link_libraries( meanshift ${OpenCV_LIBS} util)