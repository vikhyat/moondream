cmake_minimum_required(VERSION 3.3)
project(moondream_ggml_python)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

message(STATUS "Using CMake version " ${CMAKE_VERSION})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(../../core ../../core/build)
include_directories(../../core/include)

pybind11_add_module(
    moondream_ggml_ffi 
    ${CMAKE_CURRENT_LIST_DIR}/ffi.cpp
)

target_link_libraries(moondream_ggml_ffi PRIVATE moondream_ggml)
install(TARGETS moondream_ggml_ffi DESTINATION moondream_ggml_python)
