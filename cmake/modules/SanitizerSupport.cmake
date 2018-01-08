
set(GLOW_USE_SANITIZER "" CACHE STRING
    "Define the sanitizer used to build binaries and tests.")

if(GLOW_USE_SANITIZER)
  # TODO(compnerd) ensure that the compiler supports these options before adding
  # them.  At the moment, assume that this will just be used with a GNU
  # compatible driver and that the options are spelt correctly in light of that.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")

  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1")
  elseif(NOT CMAKE_BUILD_TYPE MATCHES "Debug" AND
         NOT CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gline-tables-only")
  endif()

  if(GLOW_USE_SANITIZER STREQUAL "Address")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
  elseif(GLOW_USE_SANITIZER MATCHES "Memory(WithOrigins)?")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=memory")
    if(GLOW_USE_SANITIZER STREQUAL "MemoryWithOrigins")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize-memory-track-origins") 
    endif()
  elseif(GLOW_USE_SANITIZER STREQUAL "Undefined")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-sanitize-recover=all")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-sanitize=vptr,function") 
  elseif(GLOW_USE_SANITIZER STREQUAL "Thread")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
  elseif(GLOW_USE_SANITIZER STREQUAL "Address;Undefined" OR
         GLOW_USE_SANITIZER STREQUAL "Undefined;Address")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address,undefined")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-sanitize=vptr,function")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-sanitize-recover=all")
  elseif(GLOW_USE_SANITIZER STREQUAL "Leaks")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak")
  else()
    message(FATAL_ERROR "unsupported value of GLOW_USE_SANITIZER: ${GLOW_USE_SANITIZER}")
  endif()
endif()

