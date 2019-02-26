# Target name
set(target CubbyDNN)

# Define
set(root_dir ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# Includes
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../Libraries
        ${CMAKE_CURRENT_SOURCE_DIR}/../../Includes
)

# Sources
file(GLOB header_dir
        ${root_dir}/Includes/cubbydnn)

file(GLOB_RECURSE headers
        ${header_dir}/*.hpp)

file(GLOB_RECURSE sources
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Build library
add_library(${target}
    ${sources}
        ../../Includes/cubbydnn/ExcecutionFunctions/Reticifiers.hpp
        ../../Includes/cubbydnn/ExcecutionFunctions/Sigmoids.hpp
        ../../Includes/cubbydnn/ExcecutionFunctions/SoftMax.hpp
        ../../Includes/cubbydnn/Arthimetics/Arithmetic.hpp Arthimetics/Arithmetic.cpp
        ../../Includes/cubbydnn/Types/bFloat16.hpp
        ../../Includes/cubbydnn/Types/Endian.hpp
        ../Types/bFloat16.cpp
        ../Types/TypeCast.cpp
        ../../Includes/cubbydnn/Types/TypeCast.hpp ../../Includes/cubbydnn/Types/Posit.hpp)

# Project options
set_target_properties(${target}
        PROPERTIES
        ${DEFAULT_PROJECT_OPTIONS}
        )

# Compile options
target_compile_options(${target}
        PRIVATE

        PUBLIC
        ${DEFAULT_COMPILE_OPTIONS}

        INTERFACE
        )

target_link_libraries(${target}
        PRIVATE

        PUBLIC
        ${DEFAULT_LINKER_OPTIONS}
        ${DEFAULT_LIBRARIES}

        INTERFACE
        )

# Install
install(TARGETS ${target} DESTINATION lib)
install(DIRECTORY ${header_dir} DESTINATION include)