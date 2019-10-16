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

# From https://github.com/pytorch/pytorch/blob/master/cmake/public/glog.cmake

# We will try to use the config mode first, and then manual find.
find_package(glog CONFIG QUIET)
if (NOT TARGET glog::glog)
  find_package(glog MODULE QUIET)
endif()

if (TARGET glog::glog)
  message(STATUS "Found glog with new-style glog target.")
elseif(GLOG_FOUND)
  message(
      STATUS
      "Found glog with old-style glog starget. Glog never shipped "
      "old style glog targets, so somewhere in your cmake path there might "
      "be a custom Findglog.cmake file that got triggered. We will make a "
      "best effort to create the new style glog target for you.")
  add_library(glog::glog UNKNOWN IMPORTED)
  set_property(
      TARGET glog::glog PROPERTY IMPORTED_LOCATION ${GLOG_LIBRARY})
  set_property(
      TARGET glog::glog PROPERTY INTERFACE_INCLUDE_DIRECTORIES
      ${GLOG_INCLUDE_DIR})
else()
  message(STATUS "Cannot find glog automatically. Using legacy find.")

  # - Try to find Glog
  #
  # The following variables are optionally searched for defaults
  #  GLOG_ROOT_DIR: Base directory where all GLOG components are found
  #
  # The following are set after configuration is done:
  #  GLOG_FOUND
  #  GLOG_INCLUDE_DIRS
  #  GLOG_LIBRARIES
  #  GLOG_LIBRARYRARY_DIRS

  include(FindPackageHandleStandardArgs)
  set(GLOG_ROOT_DIR "" CACHE PATH "Folder contains Google glog")
  if(NOT WIN32)
      find_path(GLOG_INCLUDE_DIR glog/logging.h
          PATHS ${GLOG_ROOT_DIR})
  endif()

  find_library(GLOG_LIBRARY glog
      PATHS ${GLOG_ROOT_DIR}
      PATH_SUFFIXES lib lib64)

  find_package_handle_standard_args(glog DEFAULT_MSG GLOG_INCLUDE_DIR GLOG_LIBRARY)

  if(GLOG_FOUND)
    message(STATUS
        "Found glog (include: ${GLOG_INCLUDE_DIR}, "
        "library: ${GLOG_LIBRARY})")
    add_library(glog::glog UNKNOWN IMPORTED)
    set_property(
        TARGET glog::glog PROPERTY IMPORTED_LOCATION ${GLOG_LIBRARY})
    set_property(
        TARGET glog::glog PROPERTY INTERFACE_INCLUDE_DIRECTORIES
        ${GLOG_INCLUDE_DIR})
  endif()
endif()

# After above, we should have the glog::glog target now.
if (NOT TARGET glog::glog)
  message(FATAL_ERROR "glog cannot be found.")
endif()