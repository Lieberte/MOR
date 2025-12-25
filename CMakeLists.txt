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

# C++ standard settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generate compile_commands.json for better IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Build options
option(BUILD_TESTING "Build tests" OFF)
option(BUILD_EXAMPLES "Build examples" OFF)

# Find required packages
find_package(MPI REQUIRED COMPONENTS C CXX)

# Print configuration summary
message(STATUS "MOR Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  MPI found: ${MPI_FOUND}")
if(MPI_FOUND)
    message(STATUS "  MPI C compiler: ${MPI_C_COMPILER}")
    message(STATUS "  MPI CXX compiler: ${MPI_CXX_COMPILER}")
endif()

# Add subdirectories
add_subdirectory(core)

# Optional modules (add as they are implemented)
# add_subdirectory(mor)
# add_subdirectory(preprocessing)
# add_subdirectory(postprocessing)
# add_subdirectory(dataio)
# add_subdirectory(visualization)
# add_subdirectory(hyperreduction)
# add_subdirectory(interpolation)
# add_subdirectory(sampling)

# Installation configuration
include(GNUInstallDirs)

install(TARGETS core
    EXPORT MORTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY core/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

install(EXPORT MORTargets
    FILE MORTargets.cmake
    NAMESPACE MOR::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MOR
)

# Enable testing if requested
if(BUILD_TESTING)
    enable_testing()
    # add_subdirectory(tests)
endif()

