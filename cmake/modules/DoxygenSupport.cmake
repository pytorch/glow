option(BUILD_DOCS "Build doxygen documentation" OFF)

if(BUILD_DOCS)
  # check if Doxygen is installed
  find_package(Doxygen)
  if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/docs)
      file(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/docs)
    endif()

    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs)
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    add_custom_target(doc_doxygen ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating C++ API documentation with Doxygen"
        VERBATIM)

  else()
    message(FATAL_ERROR "Doxygen needs to be installed to generate the documentation")
  endif()
endif()
