function(add_athena_sycl_library target_name modifier export_name export_header_name)
    set(source_list ${ARGN})

    if(ATHENA_SYCL_COMPILER MATCHES "computecpp")
      include(FindComputeCpp)
      find_package(ComputeCpp REQUIRED)

      add_library(${target_name} ${modifier} ${source_list})
      add_sycl_to_target(
        TARGET ${target_name}
        SOURCES ${source_list}
      )
      target_compile_definitions(${target_name} PRIVATE -DUSES_COMPUTECPP)
    endif()
    
    if (UNIX)
        target_compile_options(${target_name} PRIVATE -fPIC -Werror)
    endif ()

    configure_file(${CMAKE_SOURCE_DIR}/CMakeModules/export.h.in
            ${CMAKE_BINARY_DIR}/export/athena/${export_header_name})
    target_include_directories(${target_name} PUBLIC
            $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/export>)
    target_compile_definitions(${target_name} PRIVATE -D${target_name}_EXPORT)

    set_target_properties(${target_name} PROPERTIES
                BUILD_RPATH "${CMAKE_BUILD_RPATH};${PROJECT_BINARY_DIR}/lib"
                LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib 
                RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
endfunction()
