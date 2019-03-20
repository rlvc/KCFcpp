cmake_minimum_required(VERSION 2.8)
project(FEALESS)

find_package(OpenCV REQUIRED)

if(NOT WIN32)
ADD_DEFINITIONS("-std=c++0x -O3")
endif(NOT WIN32)

include_directories(src) 
FILE(GLOB_RECURSE sourcefiles "src/*")
add_executable( LINEMOD_KCF ${sourcefiles} )
target_link_libraries( LINEMOD_KCF ${OpenCV_LIBS})




