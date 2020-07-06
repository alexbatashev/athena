add_subdirectory(${PROJECT_SOURCE_DIR}/contrib/pybind11
                 ${PROJECT_BINARY_DIR}/contrib/pybind11)

pybind11_add_module(polar-python context.cpp)
target_link_libraries(polar-python PRIVATE athena)
