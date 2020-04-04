add_library(Base
              Tensor.cpp
              TensorSerialization.cpp
              Type.cpp
              IO.cpp
              TaggedList.cpp)

target_link_libraries(Base
                      PUBLIC
                        Support
                        glog::glog)

if(PNG_FOUND)
  target_sources(Base PRIVATE Image.cpp)
  target_include_directories(Base
                             PRIVATE
                               ${PNG_INCLUDE_DIR})
  target_link_libraries(Base
                        PRIVATE
                          ${PNG_LIBRARY})
  # Link z.lib if exists. For some targets (MSVC) it might not
  # be required because the PNG library already contains z.lib.
  find_library(Z_LIB z)
  if(Z_LIB_FOUND)
    target_link_libraries(Base
                          PRIVATE
                            ${Z_LIB})
  endif()
endif()

add_dependencies(Base AutoGen)
