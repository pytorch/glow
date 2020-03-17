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

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type selected, default to Debug")
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type (default Debug)" FORCE)
endif()

if(NOT ENABLE_TESTING)
  message(STATUS "Tests not explicitly enabled/disabled, default to enable")
  set(ENABLE_TESTING YES CACHE STRING "" FORCE)
endif()

if(MSVC OR CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Wall /EHsc- /GR- /bigobj")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Oy /Od")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE}")
  set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_RELEASE}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:4194304")
else()
  include(CheckCXXCompilerFlag)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wnon-virtual-dtor")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -O0")
  CHECK_CXX_COMPILER_FLAG("-Wno-psabi" HAS_W_NO_PSABI)
  if(HAS_W_NO_PSABI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-psabi")
  endif()
  if(NOT ((CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64 OR CMAKE_SYSTEM_PROCESSOR STREQUAL armv7)
     AND CMAKE_CXX_COMPILER_ID STREQUAL Clang))
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -march=native")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -march=native")
  endif()
endif()

# Specify whole archive flag for SRC library.
# This is useful when static registration is performed in the SRC library and we'd like to ensure
# linker does not clean unused objects.
function(make_whole_archive DST SRC)
  get_target_property(__src_target_type ${SRC} TYPE)

  # Depending on the type of the source library, we will set up the
  # link command for the specific SRC library.
  if (${__src_target_type} STREQUAL "STATIC_LIBRARY")
    if(APPLE)
      target_link_libraries(
          ${DST} INTERFACE -Wl,-force_load,$<TARGET_FILE:${SRC}>)
    elseif(MSVC)
      # In MSVC, we will add whole archive in default.
      target_link_libraries(
          ${DST} INTERFACE -WHOLEARCHIVE:$<TARGET_FILE:${SRC}>)
    else()
      # Assume everything else is like gcc
      target_link_libraries(${DST} INTERFACE
        "-Wl,--whole-archive $<TARGET_FILE:${SRC}> -Wl,--no-whole-archive")
      set_target_properties(${DST} PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")
    endif()
  else()
    target_link_libraries(${DST} INTERFACE ${SRC})
  endif()
endfunction()

# Return in result the name of curdir sub-directories
MACRO(getSubDirList result curdir)
  file(GLOB children RELATIVE ${curdir} ${curdir}/[^\.]*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
      LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()
