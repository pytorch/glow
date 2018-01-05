add_library(JIT
            JIT.cpp)

set(LIBJITBC ${GLOW_BINARY_DIR}/libjit.bc)
set(LIBJITSRC ${CMAKE_CURRENT_SOURCE_DIR}/libjit.c)

add_custom_command(OUTPUT
                   ${LIBJITBC}
                   COMMAND clang -emit-llvm -o ${LIBJITBC} -c ${LIBJITSRC}
                   DEPENDS ${LIBJITSRC}
                   COMMENT "Clang: Generating runtime bytecode library." VERBATIM)

target_link_libraries(JIT
                      PRIVATE
                        Optimizer
                        Base
                        Graph)

add_custom_target(
    generate-libjitbc
    DEPENDS ${LIBJITBC}
    )
add_dependencies(JIT generate-libjitbc)
