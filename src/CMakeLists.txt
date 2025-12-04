# cpp/CMakeLists.txt
add_library(sph_core STATIC
    solver/solver.cpp
    utils/vtk_writer.cpp
)

target_include_directories(sph_core PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/solver
    ${CMAKE_CURRENT_SOURCE_DIR}/integrator
    ${CMAKE_CURRENT_SOURCE_DIR}/neighbor
    ${CMAKE_CURRENT_SOURCE_DIR}/sph
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

# pybind extension target
pybind11_add_module(SimplePH
    bindings.cpp
)

# link your core C++ into the python extension
target_link_libraries(SimplePH PRIVATE sph_core)

# optional: place built .so into top-level python/ directory
set_target_properties(SimplePH PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python
)