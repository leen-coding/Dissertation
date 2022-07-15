cmake_minimum_required(VERSION 3.2)
set(OpenCV_DIR "/home/xurui/opencv-3.3.1/build")
project(sunglasses_process)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)

add_subdirectory(/home/xurui/lib/dlib-19.9 dlib_build)
add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS} dlib::dlib)

