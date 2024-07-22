cmake_minimum_required(VERSION 3.0)
project(moondream_ggml)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(DEBUG_BUILD "Build with debug flags" OFF)

if(DEBUG_BUILD)
    set(CMAKE_BUILD_TYPE Debug)
    message("CMAKE_BUILD_TYPE=Debug")
else()
    set(CMAKE_BUILD_TYPE Release)
    message("CMAKE_BUILD_TYPE=Release")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build ggml as a static library" FORCE)
add_subdirectory(./dependencies/ggml)

set(SOURCES src/moondream.cpp src/unicode_data.cpp src/unicode.cpp)

# Library target.
add_library(moondream_ggml STATIC ${SOURCES})
target_link_libraries(moondream_ggml PRIVATE ggml)
target_include_directories(moondream_ggml PRIVATE ./dependencies/ggml/include)
# Define MOONDREAM_LIBRARY_BUILD in order to exclude main function.
target_compile_definitions(moondream_ggml PRIVATE MOONDREAM_LIBRARY_BUILD)

# Executable target.
add_executable(moondream_ggml_exe ${SOURCES})
target_link_libraries(moondream_ggml_exe PRIVATE ggml)
target_include_directories(moondream_ggml_exe PRIVATE ./dependencies/ggml/include)

if(DEBUG_BUILD)
    target_compile_options(moondream_ggml PRIVATE -g)
    target_compile_options(moondream_ggml_exe PRIVATE -g)
endif()
