# CMake version
cmake_minimum_required(VERSION 3.8.2 FATAL_ERROR)

# Include cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")

# Declare project
project(CubbyDNN)

# Set output directories
set(DEFAULT_CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# Set enable output of compile commands during generation
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Includes
include_directories(Includes)
include_directories(Libraries)
include_directories(Libraries/doctest/doctest)

# Compile options
include(CMake/CompileOptions.cmake)

# Build type - Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Overrides
set(CMAKE_MACOSX_RPATH ON)

set(BLAZE_LIBRARY)

include(CheckIncludeFileCXX)

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    find_package(blaze)

    if( blaze_FOUND )
        add_library( blaze_target INTERFACE )
        target_link_libraries( blaze_target INTERFACE blaze::blaze )
        set(BLAZE_LIBRARY blaze_target)
    endif()
else()
    check_include_file_cxx("blaze/Blaze.h" blaze_FOUND)
    MESSAGE(STATUS ${CMAKE_REQUIRED_INCLUDES})
endif()

if(blaze_FOUND)
    add_compile_definitions(BLAZE_FOUND)
endif()

# Project modules
add_subdirectory(Sources/cubbydnn)
add_subdirectory(Tests/UnitTests)