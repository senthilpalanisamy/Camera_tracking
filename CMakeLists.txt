cmake_minimum_required(VERSION 3.0)
project(example)
# set(PYBIND11_PYTHON_VERSION 3.6)
set(PYBIND11_PYTHON_VERSION 3.6 CACHE STRING "")
find_package( OpenCV 4.0.1 REQUIRED ) 
 #find_package(OpenCV REQUIRED)  
 #include_directories(${OpenCV_INCLUDE_DIRS})


# INCLUDE_DIRECTORIES(/usr/include/python3.6m/) 
# INCLUDE_DIRECTORIES(/usr/include/python3.7m/)

# find_package(pybind11 REQUIRED)  # or 
add_subdirectory(third_party/pybind11)
##add_subdirectory(/home/senthilpalanisamy/software/pybind11)

file(GLOB SOURCE_FILES
      "./src/*.cpp"
     )

add_executable(main ${SOURCE_FILES})
target_include_directories(main PRIVATE ./include ${OpenCV_INCLUDE_DIRS})
#add_executable(testImage testImage.cpp)
## pybind11_add_module(test test.cpp)
target_link_libraries(main pybind11::embed ${OpenCV_LIBS})
#target_link_libraries(testImage pybind11::embed ${OpenCV_LIBS})
