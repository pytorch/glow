
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
add_subdirectory(Base)
add_subdirectory(ExecutionEngine)
add_subdirectory(Graph)
add_subdirectory(IR)
add_subdirectory(Importer)
add_subdirectory(Interpreter)
add_subdirectory(Optimizer)
add_subdirectory(Support)

