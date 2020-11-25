cmake_minimum_required(VERSION 3.5.1)
project(example)
# set(PYBIND11_PYTHON_VERSION 3.6)
set(PYBIND11_PYTHON_VERSION 3.6 CACHE STRING "")
set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
find_package( OpenCV 4.0.1 REQUIRED ) 

# some variables for image capture
set(CLIB_EXEC output)
set(IMAGE_STITCH_EXEC image_output2)
SET(DRIVER_FLAGS "-DC_GNU64=400 -DOS_LINUX -no-pie")
SET(CMAKE_CXX_FLAGS  "${DRIVER_FLAGS}")
# set(CMAKE_CXX_COMPILER "/usr/bin/g++")

# puts all .cpp files inside src to the SOURCES variable
SET(CLIB_SOURCES src/simple_capture.cpp)
SET(IMG_STITCH_SRC src/image_stitching.cpp)


add_subdirectory(third_party/pybind11)

# file(GLOB SOURCE_FILES
#       "./src/*.cpp"
#      )
SET(SOURCE_FILES ./src/deeplabFrameInference.cpp ./src/main.cpp)

add_executable(main ${SOURCE_FILES})
target_include_directories(main PRIVATE ./include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(main pybind11::embed ${OpenCV_LIBS})


include_directories(/usr/local/xclib/inc/ /snap/gnome-3-34-1804/27/usr/include)


# compiles the files defined by SOURCES to generante the executable defined by EXEC
add_library(${CLIB_EXEC} ${CLIB_SOURCES})
add_executable(${IMAGE_STITCH_EXEC} ${IMG_STITCH_SRC})
add_executable(sample_capture ./src/record_samples.cpp ./src/record_samples_main.cpp)
target_include_directories(${IMAGE_STITCH_EXEC} PRIVATE ./include/)
target_include_directories(${CLIB_EXEC} PRIVATE ./include/)
target_include_directories(sample_capture PRIVATE ./include/)
add_executable(sequence_capture ./src/sequence_capture )
add_executable(back_ground_subt ./src/backGroundSubtraction.cpp ./src/record_samples.cpp ./src/simple_capture.cpp)
target_include_directories(back_ground_subt PRIVATE ./include/)

target_link_libraries(${CLIB_EXEC} ${OpenCV_LIBS} -ljsoncpp)
target_link_libraries(${CLIB_EXEC} /usr/local/xclib/lib/pxipl_x86_64.a  /usr/local/xclib/lib/xclib_x86_64.a m -pthread -ljsoncpp)
target_link_libraries(back_ground_subt ${OpenCV_LIBS} /usr/local/xclib/lib/pxipl_x86_64.a  /usr/local/xclib/lib/xclib_x86_64.a m -pthread -ljsoncpp)

target_link_libraries(sequence_capture ${OpenCV_LIBS} -ljsoncpp)
target_link_libraries(sequence_capture /usr/local/xclib/lib/pxipl_x86_64.a  /usr/local/xclib/lib/xclib_x86_64.a m -pthread -lstdc++fs -lstdc++fs -lstdc++fs -lstdc++fs -lstdc++fs -lstdc++fs -lstdc++fs -lstdc++fs -lstdc++fs -ljsoncpp)

target_link_libraries(sample_capture ${OpenCV_LIBS} ${CLIB_EXEC} -ljsoncpp)
target_link_libraries(${IMAGE_STITCH_EXEC} ${OpenCV_LIBS} ${CLIB_EXEC} -ljsoncpp)
