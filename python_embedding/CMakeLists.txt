cmake_minimum_required(VERSION 3.0)
project(example)
# set(PYBIND11_PYTHON_VERSION 3.6)
set(PYBIND11_PYTHON_VERSION 3.6 CACHE STRING "")
find_package(OpenCV REQUIRED)  
include_directories(${OpenCV_INCLUDE_DIRS})


INCLUDE_DIRECTORIES(/usr/include/python3.6m/) 

find_package(pybind11 REQUIRED)  # or `add_subdirectory(pybind11)`
##add_subdirectory(/home/senthilpalanisamy/software/pybind11)

add_executable(test test.cpp)
add_executable(testImage testImage.cpp)
## pybind11_add_module(test test.cpp)
target_link_libraries(test PRIVATE pybind11::embed)
target_link_libraries(testImage PRIVATE pybind11::embed ${OpenCV_LIBS})
