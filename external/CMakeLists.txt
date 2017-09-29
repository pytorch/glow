
# NOTE: default to disconnected updates, we dont want to access the network when
# building if possible
set(EP_UPDATE_DISCONNECTED 1)

include(ExternalProject)

find_package(Threads REQUIRED)

# NOTE: we use the exact commit to avoid fetching sources after the initial sync
ExternalProject_Add(llvm
                    SOURCE_DIR
                      ${CMAKE_SOURCE_DIR}/external/llvm
                    BINARY_DIR
                      ${CMAKE_BINARY_DIR}/external/llvm
                    GIT_REPOSITORY
                      https://github.com/llvm-mirror/llvm.git
                    GIT_SHALLOW
                      1
                    GIT_TAG
                      release_50
                    CMAKE_ARGS
                      -DCMAKE_BUILD_TYPE=RelWithDebInfo
                      -DBUILD_SHARED_LIBS=NO
                      -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                      -DLLVM_TARGETS_TO_BUILD=X86
                      -DLLVM_BUILD_TOOLS=NO
                      -DLLVM_BUILD_UTILS=NO
                      -DLLVM_BUILD_RUNTIMES=NO
                      -DLLVM_BUILD_RUNTIME=NO
                      -DLLVM_INCLUDE_EXAMPLES=NO
                      -DLLVM_INCLUDE_GO_TESTS=NO
                      -DLLVM_ENABLE_OCAMLDOC=NO
                    BUILD_BYPRODUCTS
                      <INSTALL_DIR>/include
                      <INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMSupport${CMAKE_STATIC_LIBRARY_SUFFIX}
                    UPDATE_COMMAND
                      ""
                    STEP_TARGETS
                      update build install)
ExternalProject_Get_Property(llvm source_dir install_dir)

# NOTE: this is a workaround for the fact that the sources may not be downloaded
# just yet.  CMake does not like the addition of INTERFACE_INCLUDE_DIRECTORIES
# without the directory existing.  Just create the location which will be
# populated during the installation.
file(MAKE_DIRECTORY ${install_dir}/include)

add_library(LLVMSupport
            STATIC IMPORTED GLOBAL)
set_target_properties(LLVMSupport
                      PROPERTIES
                        IMPORTED_LOCATION
                          ${install_dir}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}LLVMSupport${CMAKE_STATIC_LIBRARY_SUFFIX}
                        INTERFACE_INCLUDE_DIRECTORIES
                          ${install_dir}/include)
add_dependencies(LLVMSupport llvm-install)

ExternalProject_Add(googletest
                    SOURCE_DIR
                      ${CMAKE_SOURCE_DIR}/external/googletest
                    BINARY_DIR
                      ${CMAKE_BINARY_DIR}/external/googletest
                    GIT_REPOSITORY
                      https://github.com/google/googletest.git
                    GIT_SHALLOW
                      1
                    GIT_TAG
                      ec44c6c1675c25b9827aacd08c02433cccde7780
                    CMAKE_ARGS
                      -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                      -DBUILD_SHARED_LIBS=NO
                    BUILD_BYPRODUCTS
                      <INSTALL_DIR>/include
                      <INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX}
                      <INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gtest_main${CMAKE_STATIC_LIBRARY_SUFFIX}
                    UPDATE_COMMAND
                      ""
                    STEP_TARGETS
                      update build install)
ExternalProject_Get_Property(googletest source_dir install_dir)

# NOTE: this is a workaround for the fact that the sources may not be downloaded
# just yet.  CMake does not like the addition of INTERFACE_INCLUDE_DIRECTORIES
# without the directory existing.  Just create the location which will be
# populated during the installation.
file(MAKE_DIRECTORY ${install_dir}/include)

add_library(gtest
            STATIC IMPORTED GLOBAL)
set_target_properties(gtest
                      PROPERTIES
                        IMPORTED_LOCATION
                          ${install_dir}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX}
                        IMPORTED_LINK_INTERFACE_LIBRARIES
                          Threads::Threads
                        INTERFACE_INCLUDE_DIRECTORIES
                          ${install_dir}/include)
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
                          ${install_dir}/include)
add_dependencies(gtest_main googletest-install)

