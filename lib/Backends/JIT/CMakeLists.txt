find_program(CLANG_EXECUTABLE NAMES clang DOC "Path to the clang front-end.")

if(NOT CLANG_EXECUTABLE)
  message(FATAL_ERROR "clang is required to build the JIT library.")
endif()

set(LIBJITBC ${GLOW_BINARY_DIR}/libjit.bc)
set(LIBJITSRC ${CMAKE_CURRENT_SOURCE_DIR}/libjit.c)

add_custom_command(OUTPUT
                   ${LIBJITBC}
                   COMMAND ${CLANG_EXECUTABLE} -emit-llvm -o ${LIBJITBC} -c ${LIBJITSRC}
                   DEPENDS ${LIBJITSRC}
                   COMMENT "Clang: Generating runtime bytecode library." VERBATIM)
add_library(JIT
            JIT.cpp)

target_link_libraries(JIT
                      PRIVATE
                        Optimizer
                        Base
                        Graph
                        LLVMCore)

add_custom_target(
  generate-libjitbc
  DEPENDS ${LIBJITBC}
  )
add_dependencies(JIT generate-libjitbc)

