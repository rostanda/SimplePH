cmake_minimum_required(VERSION 3.14)
project(SimplePH LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Find pybind11 from the active Python venv (pip installation) ---
# Use PYBIND11_DIR if explicitly set, otherwise find it via python3 -m pybind11
if(NOT DEFINED PYBIND11_DIR)
    execute_process(
        COMMAND python3 -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYBIND11_RESULT
    )
    if(PYBIND11_RESULT EQUAL 0)
        set(pybind11_DIR "${PYBIND11_CMAKE_DIR}")
    endif()
endif()
message(STATUS "Using pybind11 from: ${pybind11_DIR}")

find_package(pybind11 REQUIRED)

add_subdirectory(src)