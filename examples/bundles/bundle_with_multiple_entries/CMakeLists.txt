set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${GLOW_BINARY_DIR}/bundles)

# Output directories for regular and quantized outputs.
set(BUNDLE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/bundle_with_multiple_entries)

add_custom_target(BundleWithMultipleEntriesDir ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${BUNDLE_OUTPUT_DIRECTORY}
)

# Final Executables.
# =================
# Regular.
add_executable(bundle_with_multiple_entries $<TARGET_OBJECTS:bundle_with_multiple_entriesMain>)
set_target_properties(bundle_with_multiple_entries PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BUNDLE_OUTPUT_DIRECTORY})
target_link_libraries(bundle_with_multiple_entries ${BUNDLE_OUTPUT_DIRECTORY}/testBundle.o png)
add_dependencies(bundle_with_multiple_entries bundle_with_multiple_entriesMain bundle_with_multiple_entriesNet)

add_executable(bundle_with_multiple_entriesBundleSaver BundleSaver.cpp)
set_target_properties(bundle_with_multiple_entriesBundleSaver PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BUNDLE_OUTPUT_DIRECTORY})
target_link_libraries(bundle_with_multiple_entriesBundleSaver
                        PRIVATE
                          Backends
                          Graph
                          Support)
# Glow Bundles.
# ============
# Regular Bundle.
add_custom_command(
  OUTPUT
    ${BUNDLE_OUTPUT_DIRECTORY}/testBundle.o
  COMMAND
    ${BUNDLE_OUTPUT_DIRECTORY}/bundle_with_multiple_entriesBundleSaver -emit-bundle ${BUNDLE_OUTPUT_DIRECTORY} -bundle-api=dynamic -g
  DEPENDS
    bundle_with_multiple_entriesBundleSaver BundleWithMultipleEntriesDir
)
add_custom_target(bundle_with_multiple_entriesNet DEPENDS ${BUNDLE_OUTPUT_DIRECTORY}/testBundle.o)

# Other.
# =====
# Driver program with main function for regular bundle.
add_library(bundle_with_multiple_entriesMain OBJECT main.cpp)
target_compile_options(bundle_with_multiple_entriesMain PRIVATE -std=c++11 -g)
target_include_directories(bundle_with_multiple_entriesMain PUBLIC ${BUNDLE_OUTPUT_DIRECTORY})
add_dependencies(bundle_with_multiple_entriesMain bundle_with_multiple_entriesNet)
