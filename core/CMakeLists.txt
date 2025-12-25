# Find required packages
find_package(MPI REQUIRED COMPONENTS C CXX)
find_package(Eigen3 CONFIG QUIET)

# Collect source files
file(GLOB_RECURSE CORE_SOURCES "src/*.cpp")

# Create library
if(CORE_SOURCES)
    add_library(core STATIC ${CORE_SOURCES})
    set(CORE_SCOPE PUBLIC)
    message(STATUS "Building core library with source files")
else()
    add_library(core INTERFACE)
    set(CORE_SCOPE INTERFACE)
    message(STATUS "Creating core interface library (header-only)")
endif()

# Set target properties
set_target_properties(core PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON
)

# Include directories
target_include_directories(core ${CORE_SCOPE}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/framework/core>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/framework/core>
)

# C++ standard
target_compile_features(core ${CORE_SCOPE} cxx_std_17)

# Link libraries
target_link_libraries(core ${CORE_SCOPE}
    MPI::MPI_C
    MPI::MPI_CXX
)

if(Eigen3_FOUND)
    target_link_libraries(core ${CORE_SCOPE} Eigen3::Eigen)
    message(STATUS "  Eigen3 found: ${Eigen3_VERSION}")
else()
    message(STATUS "  Eigen3 not found (optional)")
endif()
