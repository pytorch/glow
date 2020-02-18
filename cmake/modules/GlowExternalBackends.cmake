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

# A function to add a test to be driven through the 'check' target.
# Unlike the 'test' target, the 'check' target rebuilds the executables
# before invoking the tests.

# Returns the name of the backend enabling.
MACRO(getBackendEnableVariable result backend_dir_name)
  string(TOUPPER "GLOW_WITH_${backend_dir_name}" ${result})
ENDMACRO()


# Macro to be called at top level.
MACRO(ExternalBackendsInit)
# External backends
set(EXTERNAL_BACKENDS_DIR ${GLOW_SOURCE_DIR}/externalbackends)
include_directories(${EXTERNAL_BACKENDS_DIR})
getSubDirList(SUBDIRS ${EXTERNAL_BACKENDS_DIR})

FOREACH(child ${SUBDIRS})
  # Add an option for the backend. The backend is enabled by default.
  # The user can disable it by setting the right variable to OFF.
  getBackendEnableVariable(backend_enable_variable ${child})
  option(${backend_enable_variable} "Build the ${child} backend" ON)
  # define BACKEND_ROOT_<upper case backend name>
  string(TOUPPER "BACKEND_ROOT_${child}" BACKEND_ROOT_NAME)
  set(${BACKEND_ROOT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/externalbackends/${child})

  # Verbosing
  message(STATUS "Detected external backend '${child}'")
  message(STATUS " -> Backend '${child}' can be disabled by setting ${backend_enable_variable}=OFF")
  message(STATUS " -> Backend '${child}' specific variables:")
  message(STATUS "    - ${backend_enable_variable} = ${${backend_enable_variable}}")
  message(STATUS "    - ${BACKEND_ROOT_NAME} = ${${BACKEND_ROOT_NAME}}")
  if (${backend_enable_variable})
      message(STATUS " -> Backend '${child}' ENABLED")
      add_definitions(-D${backend_enable_variable}=1)
  else()
      message(STATUS " -> Backend '${child}' DISABLED")
  endif()

  # Handle the backend only when activated
  if (${backend_enable_variable})
      # If the backend has a global CMakefile, include it.
     if(EXISTS "${EXTERNAL_BACKENDS_DIR}/${child}/CMakeLists.txt")
        include("${EXTERNAL_BACKENDS_DIR}/${child}/CMakeLists.txt")
     else()
        message(STATUS "External backend '${child}' has no global CMakeLists.txt")
     endif()
  endif()
ENDFOREACH()
ENDMACRO()

# Macro to register external backends.
MACRO(ExternalBackendsRegister)
getSubDirList(SUBDIRS ${GLOW_SOURCE_DIR}/externalbackends)
FOREACH(child ${SUBDIRS})
  getBackendEnableVariable(backend_enable_variable ${child})
  # Handle the backend only when activated
  if (${backend_enable_variable})
    # If the backend has a 'Backends' sub-directory, add it.
    if(EXISTS ${EXTERNAL_BACKENDS_DIR}/${child}/Backends)
      message("Adding external ${child} backend.")
      set(EXTERNAL_BACKEND_NAME ${child}Backend)
      add_subdirectory(${EXTERNAL_BACKENDS_DIR}/${child}/Backends EXT_${EXTERNAL_BACKEND_NAME})
    else()
      message(FATAL_ERROR "External backend '${child}' has no 'Backends' sub-directory (${EXTERNAL_BACKENDS_DIR}/${child}/Backends)")
    endif()
  endif()
ENDFOREACH()
ENDMACRO()

# Macro to register backend specific nodes and instructions.
MACRO(ExternalBackendsClassGen)
set(ClassGen_Include_DIR ${GLOW_BINARY_DIR}/glow)
getSubDirList(SUBDIRS ${GLOW_SOURCE_DIR}/externalbackends)
FOREACH(child ${SUBDIRS})
  getBackendEnableVariable(backend_enable_variable ${child})
  # Handle the backend only when activated
  if (${backend_enable_variable})
    # If the backend has a 'ClassGen' sub-directory, add it.
    set(backend_classgen_DIR "${EXTERNAL_BACKENDS_DIR}/${child}/ClassGen")
    if(EXISTS ${EXTERNAL_BACKENDS_DIR}/${child}/ClassGen)
       add_subdirectory(${EXTERNAL_BACKENDS_DIR}/${child}/ClassGen EXT_${child})

       # Check for header files with custom node definitions in this subdirectory.
       file(GLOB backend_specific_nodes
            RELATIVE "${backend_classgen_DIR}"
            "${backend_classgen_DIR}/*SpecificNodes.h")
       # Include these header files into NodeGenIncludes.h.
       foreach(include_file ${backend_specific_nodes})
           file(APPEND "${ClassGen_Include_DIR}/NodeGenIncludes.h"
                       "#include \"${EXTERNAL_BACKENDS_DIR}/${child}/ClassGen/${include_file}\"\n")
       endforeach()
       # Check for header files with custom instruction definitions in this subdirectory.
       file(GLOB backend_specific_instrs
            RELATIVE "${backend_classgen_DIR}"
            "${backend_classgen_DIR}/*SpecificInstrs.h")
       # Include these header files into InstrGenIncludes.h.
       foreach(include_file ${backend_specific_instrs})
           file(APPEND "${ClassGen_Include_DIR}/InstrGenIncludes.h"
                       "#include \"${EXTERNAL_BACKENDS_DIR}/${child}/ClassGen/${include_file}\"\n")
       endforeach()
    else()
      message(STATUS "External backend '${child}' has no 'ClassGen' sub-directory")
    endif()
  endif()
ENDFOREACH()
ENDMACRO()

# Macro to add backend specific ONNX model writers.
MACRO(ExternalBackendsCollectONNXModelWriters)
set(Exporter_Include_DIR ${GLOW_BINARY_DIR}/glow)
getSubDirList(SUBDIRS ${GLOW_SOURCE_DIR}/externalbackends)
FOREACH(backend ${SUBDIRS})
  getBackendEnableVariable(backend_enable_variable ${backend})
  # Handle the backend only when activated
  if (${backend_enable_variable})
    message(STATUS "Check backend ${backend} for ONNXModelWriters")
    set(backend_ONNX_DIR "${EXTERNAL_BACKENDS_DIR}/${backend}")
    # Check for ONNXModelWriters in the current backend subdirectory.
    file(GLOB backend_specific_onnx_model_writers
            RELATIVE "${backend_ONNX_DIR}"
            "${backend_ONNX_DIR}/*ONNXModelWriter.cpp")
    # Include these files into ONNXModelWriterIncludes.h.
    foreach(onnx_model_writer ${backend_specific_onnx_model_writers})
           file(APPEND "${Exporter_Include_DIR}/ONNXModelWriterIncludes.h"
                       "#include \"${EXTERNAL_BACKENDS_DIR}/${backend}/ONNX/${onnx_model_writer}\"\n")
    endforeach()
  endif()
ENDFOREACH()
ENDMACRO()

# Macro to add external backends tests.
MACRO(ExternalBackendsTest)
getSubDirList(SUBDIRS ${GLOW_SOURCE_DIR}/externalbackends)
FOREACH(child ${SUBDIRS})
  getBackendEnableVariable(backend_enable_variable ${child})
  # Handle the backend only when activated
  if (${backend_enable_variable})
    # If the backend has a 'Tests' sub-directory, add it.
    if(EXISTS "${EXTERNAL_BACKENDS_DIR}/${child}/Tests")
      add_subdirectory("${EXTERNAL_BACKENDS_DIR}/${child}/Tests" EXT_${child})
    else()
      message(STATUS "External backend '${child}' has no 'Tests' sub-directory")
    endif()
  endif()
ENDFOREACH()
ENDMACRO()
