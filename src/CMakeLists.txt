# Find OpenMP first
find_package(OpenMP REQUIRED)

# cpp/CMakeLists.txt
add_library(sph_core STATIC
    solver/solver.cpp
    utils/vtk_writer.cpp
    operators/PressureViscousForce.cpp
)

target_include_directories(sph_core PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/solver
    ${CMAKE_CURRENT_SOURCE_DIR}/integrator
    ${CMAKE_CURRENT_SOURCE_DIR}/neighbor
    ${CMAKE_CURRENT_SOURCE_DIR}/sph
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/operators
)

# Link OpenMP to sph_core (THIS IS THE FIX)
target_link_libraries(sph_core PUBLIC OpenMP::OpenMP_CXX)

# pybind extension target
pybind11_add_module(SimplePH
    bindings.cpp
)

# Link sph_core and OpenMP in python extension
target_link_libraries(SimplePH PRIVATE sph_core OpenMP::OpenMP_CXX)

# optional: place built .so into top-level python/ directory
set_target_properties(SimplePH PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python
)