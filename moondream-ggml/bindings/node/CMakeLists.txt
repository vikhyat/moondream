cmake_minimum_required(VERSION 3.5)
cmake_policy(SET CMP0042 NEW)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(moondream_node)

add_definitions(-DNAPI_VERSION=7)
message(STATUS "Using CMake version " ${CMAKE_VERSION})

include_directories(${CMAKE_JS_INC})


# find_package(Python COMPONENTS Interpreter Development REQUIRED)
# find_package(pybind11 CONFIG REQUIRED)

# Build a shared library named after the project from the files in `src/`
file(GLOB SOURCE_FILES "src/*.cc" "src/*.h")

add_subdirectory(../../core ../../core/build)
include_directories(../../core/include)

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${CMAKE_JS_SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")

# Include N-API wrappers
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_SOURCE_DIR}/node_modules/node-addon-api")

target_link_libraries(${PROJECT_NAME} ${CMAKE_JS_LIB})
target_link_libraries(${PROJECT_NAME} PRIVATE moondream_ggml)

if(MSVC AND CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
  # Generate node.lib
  execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF} /out:${CMAKE_JS_NODELIB_TARGET} ${CMAKE_STATIC_LINKER_FLAGS})
endif()

# pybind11_add_module(
#     cpp_ffi
#     ${CMAKE_CURRENT_LIST_DIR}/ffi.cpp
# )

# target_link_libraries(cpp_ffi PRIVATE moondream_ggml)
# install(TARGETS cpp_ffi DESTINATION moondream_ggml)
