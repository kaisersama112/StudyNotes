
cmake_minimum_required(VERSION 3.20)
project(test_face)

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR "E:/opencv/opencv/build/x64/vc15/lib")
find_package(OpenCV REQUIRED)
#set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/opencv-mobile-2.4.13.7-windows-vs2019/x64")
#find_package(OpenCV REQUIRED)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found!")
else ()
    message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
endif ()
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR})
# ncnn
set(ncnn_DIR "${CMAKE_SOURCE_DIR}/ncnn-20240410-windows-vs2019-shared/x64/lib/cmake/ncnn")
find_package(ncnn REQUIRED)
include_directories(${ncnn_INCLUDE_DIRS})
# libheif
find_package(libheif REQUIRED)
set(libheif_LIBRARIES "heif")
include_directories(${libheif_INCLUDE_DIRS})
link_directories(${libheif_LIBRARY_DIRS})

## 包含 FFmpeg 头文件
#find_package(FFMPEG REQUIRED)
#include_directories(${FFMPEG_INCLUDE_DIRS})
#link_directories(${FFMPEG_LIBRARY_DIRS})


add_executable(${PROJECT_NAME}
        face_de/include/scrfd_kps.h
        face_de/include/arcface.h
        face_de/include/scr_arc_face.h
        face_de/src/arcface.cpp
        face_de/src/scrfd_kps.cpp
        face_de/src/scr_arc_face.cpp
        face_de/tests/test.cpp

)


# 设置编译选项以匹配运行时库设置
if (MSVC)
    target_link_libraries(
            ${PROJECT_NAME}
            ${OpenCV_LIBS}
            ${libheif_LIBRARIES}
            #${FFMPEG_LIBRARIES}
            ncnn

    )
else ()
    # Release 模式
    target_compile_options(
            ${PROJECT_NAME}
            ${OpenCV_LIBS}
            ${libheif_LIBRARIES}
            #${FFMPEG_LIBRARIES}
            ncnn

    )
endif ()


