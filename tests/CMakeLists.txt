
find_package(Threads REQUIRED)

include(ExternalProject)
ExternalProject_Add(googletest
                    SOURCE_DIR
                      ${CMAKE_CURRENT_SOURCE_DIR}/googletest
                    CMAKE_ARGS
                      -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                      -DBUILD_SHARED_LIBS=NO
                    PREFIX
                      ${CMAKE_CURRENT_BINARY_DIR}/googletest
                    BUILD_BYPRODUCTS
                      <INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX}
                      <INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gtest_main${CMAKE_STATIC_LIBRARY_SUFFIX}
                    STEP_TARGETS
                      install)
ExternalProject_Get_Property(googletest source_dir install_dir)

add_library(gtest
            STATIC IMPORTED GLOBAL)
set_target_properties(gtest
                      PROPERTIES
                        IMPORTED_LOCATION
                          ${install_dir}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX}
                        IMPORTED_LINK_INTERFACE_LIBRARIES
                          Threads::Threads
                        INTERFACE_INCLUDE_DIRECTORIES
                          ${source_dir}/googletest/include)
add_dependencies(gtest googletest-install)

add_library(gtest_main
            STATIC IMPORTED GLOBAL)
set_target_properties(gtest_main
                      PROPERTIES
                        IMPORTED_LOCATION
                          ${install_dir}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gtest_main${CMAKE_STATIC_LIBRARY_SUFFIX}
                        IMPORTED_LINK_INTERFACE_LIBRARIES
                          Threads::Threads
                        INTERFACE_INCLUDE_DIRECTORIES
                          ${source_dir}/googletest/include)
add_dependencies(gtest_main googletest-install)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${GLOW_BINARY_DIR}/tests/)
add_subdirectory(unittests)

