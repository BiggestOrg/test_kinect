# CMakeList.txt: 顶层 CMake 项目文件，在此处执行全局配置
# 并包含子项目。
#
cmake_minimum_required (VERSION 3.8)

project ("test_kinect")


include_directories($ENV{OPENNI2_INCLUDE64})
link_directories($ENV{OPENNI2_REDIST64})
link_directories($ENV{OPENNI2_LIB64})
message($ENV{OPENNI2_INCLUDE64})
message($ENV{OPENNI2_REDIST64})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})
find_package(PCL 1.9.0 REQUIRED COMPONENTS io visualization)
include_directories(${PCL_INCLUDE_DIRS})
#find_package(Eigen REQUIRED)
include_directories("C:\\Program Files\\PCL 1.9.0\\3rdParty\\Eigen\\eigen3")

# Find aruco. You have set the aruco_ROOT path with your environment.
set(ArUco_ROOT "C:/Program Files/aruco")
IF(NOT ArUco_ROOT)
    MESSAGE("ERROR: ArUco_ROOT must be set!")
ENDIF(NOT ArUco_ROOT)

FIND_PATH(ArUco_INCLUDE_DIRS DOC "Path to ArUco include directory." NAMES aruco/aruco.h PATHS ${ArUco_ROOT}/include)

FIND_LIBRARY(ArUco_LIBRARIES DOC "Absolute path to ArUco library."
    NAMES aruco312.lib
    PATHS ${ArUco_ROOT}/lib)

include_directories(${ArUco_INCLUDE_DIRS})

# 将源代码添加到此项目的可执行文件。
add_executable (test_kinect_camera "src/camera.cpp")
target_link_libraries(test_kinect_camera ${OpenCV_LIBRARIES} OpenNI2.lib)

add_executable (test_kinect_twokinect "src/twokinect.cpp")
target_link_libraries(test_kinect_twokinect ${OpenCV_LIBRARIES} OpenNI2.lib)

add_executable (test_kinect_twokinect_calibration "src/twokinect_calibration.cpp")
target_link_libraries(test_kinect_twokinect_calibration ${OpenCV_LIBRARIES} ${ArUco_LIBRARIES} OpenNI2.lib)

add_executable (test_kinect_twokinect_pcl "src/twokinect_pcl.cpp")
target_link_libraries(test_kinect_twokinect_pcl ${OpenCV_LIBRARIES} ${ArUco_LIBRARIES} ${PCL_LIBRARIES} OpenNI2.lib)


if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/Z7")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/INCREMENTAL:NO /DEBUG /OPT:REF /OPT:ICF")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4100 /wd4127 /wd4150 /wd4191 /wd4242 /wd4244 /wd4251 /wd4305 /wd4365 /wd4512 /wd4514 /wd4548 /wd4571 /wd4619 /wd4625 /wd4626 /wd4628 /wd4668 /wd4710 /wd4711 /wd4738 /wd4820 /wd4946 /wd4986 /wd4819 /wd4996 /wd4267")
    add_definitions(-DNDEBUG)
	add_definitions(-D_USE_MATH_DEFINES)
    add_definitions(-DNOMINMAX)
endif()