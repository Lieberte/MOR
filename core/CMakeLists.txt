find_package(MPI REQUIRED COMPONENTS C CXX)
find_package(Eigen3 CONFIG QUIET)

file(GLOB_RECURSE CORE_SOURCES "src/*.cpp")

if(CORE_SOURCES)
    add_library(core STATIC ${CORE_SOURCES})
    set(CORE_SCOPE PUBLIC)
    message(STATUS "Building core library with source files")
else()
    add_library(core INTERFACE)
    set(CORE_SCOPE INTERFACE)
    message(STATUS "Creating core interface library (header-only)")
endif()

target_include_directories(core ${CORE_SCOPE}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(core ${CORE_SCOPE} cxx_std_17)

target_link_libraries(core ${CORE_SCOPE}
    MPI::MPI_C
    MPI::MPI_CXX
)

if(Eigen3_FOUND)
    target_link_libraries(core ${CORE_SCOPE} Eigen3::Eigen)
endif()
