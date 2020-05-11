# only for cmake --version >= 3.5.1
cmake_minimum_required(VERSION 3.5.1)

# project name
project(image-stiching)

# creates the variable EXEC and sets it to hello


set(CLIB_EXEC output)
set(IMAGE_STITCH_EXEC image_output2)
set(CMAKE_BUILD_TYPE Debug)

find_package( OpenCV REQUIRED)


SET(DRIVER_FLAGS "-DC_GNU64=400 -DOS_LINUX -no-pie")
#set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_COMPILER "g++")

# set the C++14 standard
set(CMAKE_CXX_STANDARD 14)
#SET(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${DRIVER_FLAGS}")
SET(CMAKE_CXX_FLAGS  "${CMAKE_C_FLAGS} ${DRIVER_FLAGS}")


# I../includes
include_directories(/usr/local/xclib/inc/ /snap/gnome-3-34-1804/27/usr/include)

# puts all .cpp files inside src to the SOURCES variable
SET(CLIB_SOURCES src/simple_capture.cpp)
SET(IMG_STITCH_SRC src/image_stitching.cpp)

# compiles the files defined by SOURCES to generante the executable defined by EXEC
add_executable(${CLIB_EXEC} ${CLIB_SOURCES})
add_executable(${IMAGE_STITCH_EXEC} ${IMG_STITCH_SRC})
target_link_libraries(${CLIB_EXEC} /usr/local/xclib/lib/pxipl_x86_64.a  /usr/local/xclib/lib/xclib_x86_64.a m)
target_link_libraries(${CLIB_EXEC} ${OpenCV_LIBS})
target_link_libraries(${IMAGE_STITCH_EXEC} ${OpenCV_LIBS})
