cmake_minimum_required(VERSION 2.0.0)

find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})

project(SportVideo)
find_package(OpenCV REQUIRED)
add_executable(detect main.cpp detection.cpp classification.cpp heatmap.cpp)
target_link_libraries(detect ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)


