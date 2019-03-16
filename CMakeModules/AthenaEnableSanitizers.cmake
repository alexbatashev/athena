function(enable_sanitizers)
    set(SANITIZERS_LIST "")
    if ("${USE_SANITIZERS}" STREQUAL "seq")
        if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
            message(FATAL_ERROR "Sanitizers are not supported in MSVC")
        elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
            set(SANITIZERS_LIST "undefined,address")
        else ()
            set(SANITIZERS_LIST "undefined,address,leak")
        endif ()
    elseif ("${USE_SANITIZERS}" STREQUAL "par")
        if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
            message(FATAL_ERROR "Sanitizers are not supported in MSVC")
        else ()
            set(SANITIZERS_LIST "undefined,thread")
        endif ()
    endif ()

    if (NOT SANITIZERS_LIST STREQUAL "")
        if (CMAKE_BUILD_TYPE STREQUAL "Debug")
            message(STATUS "Enabled sanitizers: ${SANITIZERS_LIST}")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=${SANITIZERS_LIST}" PARENT_SCOPE)
        else ()
            message(WARNING "Sanitizers are only available in Debug builds")
        endif ()
    endif ()
endfunction()