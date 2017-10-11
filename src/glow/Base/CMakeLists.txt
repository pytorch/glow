
add_library(Base
              Train.cpp
              Image.cpp)
target_link_libraries(Base
                      PUBLIC
                        IR)
if(PNG_FOUND)
  target_compile_definitions(Base
                             PRIVATE
                               WITH_PNG=1)
  target_include_directories(Base
                             PRIVATE
                               ${PNG_INCLUDE_DIR})
  target_link_libraries(Base
                        PRIVATE
                          ${PNG_LIBRARY})
endif()

