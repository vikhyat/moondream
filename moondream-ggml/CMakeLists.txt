cmake_minimum_required(VERSION 3.0)
project(moondream_ggml)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(./dependencies/ggml)

set(SOURCES src/main.cpp src/unicode_data.cpp src/unicode.cpp)

add_executable(moondream_ggml ${SOURCES})

target_link_libraries(moondream_ggml PRIVATE ggml)
target_include_directories(moondream_ggml PRIVATE ./dependencies/ggml/include)
target_compile_options(moondream_ggml PRIVATE -g)
