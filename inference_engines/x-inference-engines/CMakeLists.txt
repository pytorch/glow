cmake_minimum_required(VERSION 3.0)
project(x-infer)

option(LIB_ONLY "Only build the library" OFF)
option(BUILD_FOR_DYNAMIC_LINKAGE "Whether to build for dynamic or static bundle linkage" ON)
option(LINK_LIBS_STATICALLY "Whether to link in all the required libraries statically" ON)
option(ENABLE_PERF_MONITORING "Whether to enable performance monitoring" OFF)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

if(LINK_LIBS_STATICALLY)
    message(WARNING "Static linking may cause issues with dlopen on target")
    message("Will statically link executable")
    set(TARGET_COMPILE_FLAGS "-static"
    )
    set(TARGET_LINKER_FLAGS "-static"
    )
else()
    set(TARGET_COMPILE_FLAGS )
endif()

if (DEFINED TARGET_OPTIONS)
    set(TARGET_COMPILE_FLAGS "${TARGET_COMPILE_FLAGS} ${TARGET_OPTIONS}")
endif()

include_directories(${PROJECT_SOURCE_DIR})

set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${TARGET_COMPILE_FLAGS})
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TARGET_COMPILE_FLAGS})
set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} ${TARGET_LINKER_FLAGS})

if(ENABLE_PERF_MONITORING)
    message("Will enable performance monitoring")
    message(WARNING "Performance monitoring is support only on Linux")
    message(WARNING "If you're running in a VM, performance monitoring will be "
                    "broken, and will likely break inference.")
    add_definitions(-DENABLE_PERF_MONITORING)
endif()

add_library(xinfer STATIC x_inference_lib.c x_perf_monitor.c)

if(LIB_ONLY)
    message("Will only build the library")
else()
    message("Will build the library and the executable")
    if (BUILD_FOR_DYNAMIC_LINKAGE)
        message("Will build the executable for dynamic bundle linkage")
        set(OBJS )
        add_definitions(-DX_USE_DYNAMIC)
    else()
        if(NOT DEFINED LINKED_BUNDLE)
            message(FATAL_ERROR "When building with static bundle linkage, "
                                "bundle must be specified with "
                                "-DLINKED_BUNDLE=")
        endif()
        if (NOT DEFINED LINKED_MODEL_NAME)
            message(FATAL_ERROR "When building with static bundle linkage, "
                                "model name must be specified with "
                                "-DLINKED_MODEL_NAME=")
        endif()

        add_definitions(-DX_MODEL_NAME=${LINKED_MODEL_NAME})
        set(OBJS ${LINKED_BUNDLE})
        set_source_files_properties(${OBJS} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    endif()
    link_directories(${LIBRARY_OUTPUT_PATH})
    add_executable(x-infer main.c ${OBJS})
    target_link_libraries(x-infer xinfer dl m)
endif()
