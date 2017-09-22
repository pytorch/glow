
include(ExternalProject)

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
                      81029f142231bde8e119becda112a2173f1459c9
                    CMAKE_ARGS
                      -DCMAKE_BUILD_TYPE=RelWithDebInfo
                      -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                      -DLLVM_TARGETS_TO_BUILD=X86
                      -DLLVM_BUILD_TOOLS=NO
                      -DLLVM_BUILD_UTILS=NO
                      -DLLVM_BUILD_RUNTIMES=NO
                      -DLLVM_BUILD_RUNTIME=NO
                      -DLLVM_INCLUDE_EXAMPLES=NO
                      -DLLVM_INCLUDE_GO_TESTS=NO
                      -DLLVM_ENABLE_OCAMLDOC=NO
                    STEP_TARGETS
                      build update
                    EXCLUDE_FROM_ALL
                      1)

