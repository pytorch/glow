
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type selected, default to Debug")
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type (default Debug)" FORCE)
endif()

if(NOT ENABLE_TESTING)
  message(STATUS "Tests not explicitly enabled/disabled, default to enable")
  set(ENABLE_TESTING YES CACHE STRING "" FORCE)
endif()

