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
#   CMake utility function used to serialize as C/C++ binary data arrays all the
#   given input files into a given output directory and generate a C/C++ include
#   file which defines a std::vector variable with buffer references defined as
#   llvm::MemoryBufferRef. The output file is generated while running CMake but
#   the serialization is done during the build. All the given files must have
#   unique names and the names (without extension) must be valid C identifiers.
#
# ARGUMENTS:
#   INP_FILES       List with input file paths (absolute paths). (MANDATORY)
#   OUT_DIR         Output directory where files are serialized. (MANDATORY)
#   OUT_FILE        Output C/C++ file path. (absolute path). (MANDATORY)
#   OUT_VAR         The name of the vector variable from the output file which
#                   holds the references for all the memory buffers. (MANDATORY)
#   OUT_TARGET      The name of the custom output target. This target will be
#                   used as dependency for other targets using the byproducts of
#                   the serialization. (MANDATORY)
#===============================================================================
function(glow_serialize)

  # Parse arguments (arguments are concatenated with an "ARG_" prefix).
  set(oneValueArgs OUT_DIR OUT_FILE OUT_VAR OUT_TARGET)
  set(multiValueArgs INP_FILES)
  cmake_parse_arguments("ARG" "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # Check function parameters.
  # We do not check "ARG_INP_FILES" since it is allowed to be empty.

  if (NOT ARG_OUT_DIR)
    message(FATAL_ERROR "OUT_DIR argument is mandatory!")
  endif()

  if (NOT ARG_OUT_FILE)
    message(FATAL_ERROR "OUT_FILE argument is mandatory!")
  endif()

  if (NOT ARG_OUT_VAR)
    message(FATAL_ERROR "OUT_VAR argument is mandatory!")
  endif()

  if (NOT ARG_OUT_TARGET)
    message(FATAL_ERROR "OUT_TARGET argument is mandatory!")
  endif()

  # Make output directory to serialize files.
  file(MAKE_DIRECTORY ${ARG_OUT_DIR})

  # List with serialized files.
  set(OUT_FILES)

  # Iterate all the input files.
  file(WRITE ${ARG_OUT_FILE} "// Auto-generated file. Do not edit!\n\n")
  foreach(inp_file ${ARG_INP_FILES})

    # Get input file path fields.
    get_filename_component(file_name ${inp_file} NAME)
    get_filename_component(file_name_we ${inp_file} NAME_WE)
    message(STATUS "Serializing file: ${inp_file}")

    # Write C/C++ code for file serialization.
    file(APPEND ${ARG_OUT_FILE} "static const unsigned char ${file_name_we}_data[] = {\n")
    file(APPEND ${ARG_OUT_FILE} "  #include \"${file_name}.inc\"\n")
    file(APPEND ${ARG_OUT_FILE} "};\n\n")

    # Output file path.
    set(out_file "${ARG_OUT_DIR}/${file_name}.inc")
    list(APPEND OUT_FILES ${out_file})

    # Add command to serialize file to text.
    add_custom_command(
      OUTPUT ${out_file}
      COMMAND include-bin "${inp_file}" "${out_file}"
      DEPENDS ${inp_file}
      WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    )

  endforeach()

  # Create vector with buffers defined as llvm::MemoryBufferRef objects.
  file(APPEND ${ARG_OUT_FILE} "#define MEM_BUFF_REF(name, data) llvm::MemoryBufferRef(llvm::StringRef(reinterpret_cast<const char *>(data), sizeof(data)), name)\n\n")
  file(APPEND ${ARG_OUT_FILE} "static const std::vector<llvm::MemoryBufferRef> ${ARG_OUT_VAR} = {\n")
  foreach(inp_file ${ARG_INP_FILES})
    get_filename_component(file_name ${inp_file} NAME)
    get_filename_component(file_name_we ${inp_file} NAME_WE)
    file(APPEND ${ARG_OUT_FILE} "  MEM_BUFF_REF(\"${file_name}\", ${file_name_we}_data),\n")
  endforeach()
  file(APPEND ${ARG_OUT_FILE} "};\n\n")
  file(APPEND ${ARG_OUT_FILE} "#undef MEM_BUFF_REF\n")

  # Add custom target for serialization.
  add_custom_target(
    ${ARG_OUT_TARGET}
    DEPENDS ${OUT_FILES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  )

endfunction()
