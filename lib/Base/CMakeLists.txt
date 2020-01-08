add_library(Base
              Tensor.cpp
              Type.cpp
              IO.cpp
              TaggedList.cpp
              SharedMutex.cpp)

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
endif()

add_dependencies(Base AutoGen)
