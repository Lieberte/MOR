cmake_minimum_required(VERSION 3.20)

set(VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "" FORCE)

if(CMAKE_GENERATOR MATCHES "Visual Studio")
    set(CMAKE_GENERATOR_PLATFORM "x64" CACHE STRING "" FORCE)
endif()

if(DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
else()
    get_filename_component(VCPKG_TOOLCHAIN_PATH 
        "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
        ABSOLUTE
    )
    if(EXISTS "${VCPKG_TOOLCHAIN_PATH}")
        set(CMAKE_TOOLCHAIN_FILE "${VCPKG_TOOLCHAIN_PATH}"
            CACHE STRING "Vcpkg toolchain file")
    endif()
endif()

project(MOR 
    VERSION 1.0.0
    DESCRIPTION "C++ based Model Order Reduction framework"
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_subdirectory(core)

