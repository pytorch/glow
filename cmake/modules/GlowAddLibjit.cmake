# Copyright (c) Glow Contributors. See CONTRIBUTORS file.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#===============================================================================
# DESCRIPTION:
#   CMake utility function used to build a Glow LIBJIT library for an LLVM based
#   backend. The byproducts of this function are:
#   - LLVM bitcode library in binary format (e.g. "libjit_cpu.bc") stored in the
#     "<glow_build_folder>/libjit" folder.
#   - LLVM bitcode library in text format (e.g. "libjit_cpu.inc") stored in the
#     "<glow_build_folder>/glow/libjit" folder.
#
# ARGUMENTS:
#   NAME            LIBJIT name (e.g. "libjit_cpu"). (MANDATORY) 
#   SOURCE_FILES    List of source files (absolute paths). (MANDATORY)
#   OBJECT_FILES    List of additional object files (LLVM bitcode) which should
#                   be linked to the library. (OPTIONAL)
#   DEPENDS         List of optional (extra) dependencies for the LIBJIT apart
#                   from the source/object files. (OPTIONAL)
#   COMPILE_OPTIONS List of compile options. (MANDATORY)
#   CLANG_BIN       Clang compiler program used for compilation. If not given
#                   then clang++ is searched for and used by default. (OPTIONAL)
#   LLVM_LINK_BIN   LLVM linker program used for linking. If not given then it
#                   is searched for by default in the system. (OPTIONAL)
#
# OPTIONS:
#   COMPILE_ONLY    If used then the source files are only compiled to object
#                   files without linking into a library. (OPTIONAL)
#
# RETURN:
#  When function returns it will define in the PARENT SCOPE (caller scope) the
#  following:
#    <NAME>_OBJECT_FILES  List of absolute paths of all the LIBJIT object files.
#    <NAME>_BINARY_FILE   The full absolute path of the LIBJIT binary file.
#    <NAME>_INCLUDE_FILE  The full absolute path of the LIBJIT include file.
#    <NAME>_TARGET        Custom target which can be used as dependency for
#                         other targets/libraries (e.g. for the Backend which
#                         depends on and is using this LIBJIT), for example:
#                           add_dependencies(BackendTarget libjit_cpu_TARGET)
#===============================================================================
function(glow_add_libjit)

  # Parse arguments (arguments are concatenated with a "LIBJIT_" prefix).
  set(options OPTIONAL COMPILE_ONLY)
  set(oneValueArgs NAME CLANG_BIN LLVM_LINK_BIN)
  set(multiValueArgs SOURCE_FILES OBJECT_FILES COMPILE_OPTIONS DEPENDS)
  cmake_parse_arguments(LIBJIT "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT LIBJIT_NAME)
    message(FATAL_ERROR "NAME argument is mandatory!")
  endif()

  if (NOT LIBJIT_SOURCE_FILES)
    message(FATAL_ERROR "SOURCE_FILES argument is mandatory!")
  endif()

  if (NOT LIBJIT_COMPILE_OPTIONS)
    message(FATAL_ERROR "COMPILE_OPTIONS argument is mandatory!")
  endif()

  # Find clang++ if not explicitly given.
  if (NOT LIBJIT_CLANG_BIN)
    if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
      set(LIBJIT_CLANG_BIN ${CMAKE_CXX_COMPILER})
    else()
      find_program(LIBJIT_CLANG_BIN clang++)
    endif()
    if(NOT LIBJIT_CLANG_BIN)
      message(FATAL_ERROR "Unable to find clang++ to build LIBJIT!")
    endif()
  endif()

  # Find llvm-link if not explicitly given.
  if(NOT LIBJIT_LLVM_LINK_BIN)
    if(MSVC)
      set(LIBJIT_LLVM_LINK_BIN ${LLVM_BINARY_DIR}/$(Configuration)/bin/llvm-link)
    else()
      find_program(LIBJIT_LLVM_LINK_BIN NAMES llvm-link)
      if(NOT EXISTS ${LIBJIT_LLVM_LINK_BIN})
        set(LIBJIT_LLVM_LINK_BIN ${LLVM_BINARY_DIR}/bin/llvm-link)
      endif()
    endif()
  endif()
  if(NOT EXISTS ${LIBJIT_LLVM_LINK_BIN} AND NOT MSVC)
    message(FATAL_ERROR "Unable to find llvm-link to build LIBJIT!")
  endif()

  # Initialize object files.
  if(NOT LIBJIT_OBJECT_FILES)
    set(LIBJIT_OBJECT_FILES)
  endif()

  # Initialize dependency targets.
  if(NOT LIBJIT_DEPENDS)
    set(LIBJIT_DEPENDS)
  endif()

  # LIBJIT common compile options.
  set(LIBJIT_COMMON_COMPILE_OPTIONS
    # Clang option to emit LLVM IR bitcode.
    -emit-llvm
    # Clang option for Windows to NOT generate the "#pragma detect_mismatch"
    # metadata in the "llvm.linker.options" to avoid incompatibilities between
    # linked objects.
    -fno-autolink
  )

  # LIBJIT output binary directory.
  set(LIBJIT_BINARY_DIR ${GLOW_BINARY_DIR}/libjit)
  file(MAKE_DIRECTORY ${LIBJIT_BINARY_DIR})

  # LIBJIT output include directory.
  set(LIBJIT_INCLUDE_DIR ${GLOW_BINARY_DIR}/glow/libjit)
  file(MAKE_DIRECTORY ${LIBJIT_INCLUDE_DIR})

  # Directory path to store object files.
  set(LIBJIT_OBJECT_DIR ${LIBJIT_BINARY_DIR}/${LIBJIT_NAME}_obj)
  file(MAKE_DIRECTORY ${LIBJIT_OBJECT_DIR})

  # LIBJIT binary/include file path.
  set(LIBJIT_BINARY_FILE ${LIBJIT_BINARY_DIR}/${LIBJIT_NAME}.bc)
  set(LIBJIT_INCLUDE_FILE ${LIBJIT_INCLUDE_DIR}/${LIBJIT_NAME}.inc)

  # Print status.
  message(STATUS "Adding LIBJIT library: ${LIBJIT_NAME}.")

  # Compile all source files to LLVM bitcode.  
  foreach(src_file ${LIBJIT_SOURCE_FILES})
    get_filename_component(src_file_name ${src_file} NAME_WE)
    set(obj_file ${LIBJIT_OBJECT_DIR}/${src_file_name}${CMAKE_C_OUTPUT_EXTENSION})
    add_custom_command(
      OUTPUT  ${obj_file}
      COMMAND ${LIBJIT_CLANG_BIN} -c ${src_file} -o ${obj_file} ${LIBJIT_COMMON_COMPILE_OPTIONS} ${LIBJIT_COMPILE_OPTIONS} 
      DEPENDS ${src_file}
      WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    )
    list(APPEND LIBJIT_OBJECT_FILES ${obj_file})
  endforeach()

  # Propagate the list of object files to parent scope.
  set("${LIBJIT_NAME}_OBJECT_FILES" ${LIBJIT_OBJECT_FILES} PARENT_SCOPE)

  if (NOT LIBJIT_COMPILE_ONLY)

    # Link all object files into LIBJIT.
    add_custom_command(
      OUTPUT ${LIBJIT_BINARY_FILE}
      COMMAND ${LIBJIT_LLVM_LINK_BIN} ${LIBJIT_OBJECT_FILES} -o ${LIBJIT_BINARY_FILE}
      DEPENDS ${LIBJIT_SOURCE_FILES} ${LIBJIT_OBJECT_FILES} ${LIBJIT_DEPENDS}
      WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    )

    # Serialize LIBJIT into a text file.
    add_custom_command(
      OUTPUT ${LIBJIT_INCLUDE_FILE}
      COMMAND include-bin "${LIBJIT_BINARY_FILE}" "${LIBJIT_INCLUDE_FILE}"
      DEPENDS ${LIBJIT_BINARY_FILE}
      WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    )

    # Add custom LIBJIT target.
    set(LIBJIT_TARGET_NAME "${LIBJIT_NAME}_TARGET")
    add_custom_target(${LIBJIT_TARGET_NAME}
      DEPENDS ${LIBJIT_INCLUDE_FILE}
      WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    )

    # Propagate variables to the parent scope.
    set("${LIBJIT_NAME}_BINARY_FILE" ${LIBJIT_BINARY_FILE} PARENT_SCOPE)
    set("${LIBJIT_NAME}_INCLUDE_FILE" ${LIBJIT_INCLUDE_FILE} PARENT_SCOPE)

  endif()

endfunction()
