cmake_minimum_required(VERSION 3.1)
project(lfFocusMaps LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(${PROJECT_NAME}_VERSION_MAJOR 0)
set(${PROJECT_NAME}_VERSION_MINOR 0)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeModules")

set(SOURCES 
src/lfLoader.cpp
src/main.cpp
src/interpolator.cu
src/libs/arguments/arguments.hpp
src/libs/loadingBar/loadingbar.hpp
src/libs/stb_image.h
src/libs/stb_image_write.h
)

find_package(glm REQUIRED)
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/src/libs)

add_executable(${PROJECT_NAME} ${SOURCES})
target_compile_options(${PROJECT_NAME} PRIVATE -Wall)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --diag_suppress=esa_on_defaulted_function_ignored>)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 75)
