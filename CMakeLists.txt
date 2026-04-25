cmake_minimum_required(VERSION 3.15)
project(IMAGEPROCESSINGPROJECT)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(ImageProcessingProject
    src/main.cpp
)

# Fedora's OpenCV needs gdal and pq, Windows/vcpkg does not
if(UNIX AND NOT APPLE)
    find_library(GDAL_LIB gdal)
    find_library(PQ_LIB pq)
    set(EXTRA_LIBS "")
    if(GDAL_LIB)
        list(APPEND EXTRA_LIBS ${GDAL_LIB})
    endif()
    if(PQ_LIB)
        list(APPEND EXTRA_LIBS ${PQ_LIB})
    endif()
else()
    set(EXTRA_LIBS "")
endif()

target_link_libraries(ImageProcessingProject
    ${OpenCV_LIBS}
    ${EXTRA_LIBS}
)