
include(ExternalProject)

# NOTE: default to disconnected updates, we don't want to access the network
# when building if possible
set(EP_UPDATE_DISCONNECTED 1)

set_property(DIRECTORY PROPERTY EP_BASE Dependencies)

ExternalProject_Add(llvm
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
                    UPDATE_COMMAND
                      ""
                    STEP_TARGETS
                      update build install)
ExternalProject_Get_Property(llvm install_dir)

ExternalProject_Add(Glow
                    CMAKE_ARGS
                      -DBUILD_DEPENDENCIES=NO
                      -DCMAKE_PREFIX_PATH=${install_dir}
                    DEPENDS
                      llvm
                    SOURCE_DIR
                      ${CMAKE_SOURCE_DIR}
                    UPDATE_COMMAND
                      ""
                    INSTALL_COMMAND
                      ""
                    STEP_TARGETS
                      update build)

