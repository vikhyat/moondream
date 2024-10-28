cmake_minimum_required(VERSION 3.0)
project(moondream_ggml)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(DEBUG_BUILD "Build with debug flags" OFF)
option(MOONDREAM_SHARED_LIBS "Build shared libraries" OFF)
option(MOONDREAM_EXE "Build executable instead of library" OFF)

if(DEBUG_BUILD)
    set(CMAKE_BUILD_TYPE Debug)
    message("CMAKE_BUILD_TYPE=Debug")
else()
    set(CMAKE_BUILD_TYPE Release)
    message("CMAKE_BUILD_TYPE=Release")
endif()

if(NOT MOONDREAM_SHARED_LIBS)
    set(GGML_STATIC ON)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build ggml as a static library" FORCE)
endif()
add_subdirectory(./dependencies/ggml)

set(SOURCES 
    src/moondream.cpp
    src/lm.cpp
    src/mmproj.cpp
    src/helpers.cpp
    src/stb_image.cpp
    src/unicode_data.cpp
    src/unicode.cpp
)

if(MOONDREAM_EXE)
    add_executable(moondream_ggml ${SOURCES})
else()
    if(MOONDREAM_SHARED_LIBS)
        add_library(moondream_ggml SHARED ${SOURCES})
    else()
        add_library(moondream_ggml STATIC ${SOURCES})
    endif()
    # Define MOONDREAM_LIBRARY_BUILD in order to exclude main function.
    target_compile_definitions(moondream_ggml PRIVATE MOONDREAM_LIBRARY_BUILD)
endif()

target_link_libraries(moondream_ggml PRIVATE ggml)
target_include_directories(moondream_ggml PRIVATE ./dependencies/ggml/include)
target_include_directories(moondream_ggml PRIVATE ./dependencies/single_header/include)
