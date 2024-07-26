cmake_minimum_required(VERSION 3.3)
project(moondream_ggml)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

message(STATUS "Using CMake version " ${CMAKE_VERSION})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(../../core ../../core/build)
include_directories(../../core/include)

pybind11_add_module(
    cpp_ffi 
    ${CMAKE_CURRENT_LIST_DIR}/ffi.cpp
)

target_link_libraries(cpp_ffi PRIVATE moondream_ggml)
install(TARGETS cpp_ffi DESTINATION moondream_ggml)
