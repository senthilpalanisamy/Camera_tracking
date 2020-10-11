cmake_minimum_required(VERSION 3.0)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(example)
find_package(OpenCV REQUIRED)  
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(object_tracking objectTracking.cpp)
target_link_libraries(object_tracking ${OpenCV_LIBS})

