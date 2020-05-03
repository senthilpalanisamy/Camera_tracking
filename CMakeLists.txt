# only for cmake --version >= 3.5.1
cmake_minimum_required(VERSION 3.5.1)

# project name
project(image-stiching)

# creates the variable EXEC and sets it to hello


set(EXEC output)

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
file(GLOB SOURCES src/*.cpp)

# compiles the files defined by SOURCES to generante the executable defined by EXEC
add_executable(${EXEC} ${SOURCES})
target_link_libraries(${EXEC} /usr/local/xclib/lib/pxipl_x86_64.a  /usr/local/xclib/lib/xclib_x86_64.a m)
target_link_libraries(${EXEC} ${OpenCV_LIBS})
