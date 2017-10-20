
add_library(Base
              ${CMAKE_SOURCE_DIR}/include/glow/Base/Image.h
              ${CMAKE_SOURCE_DIR}/include/glow/Base/Tensor.h
              ${CMAKE_SOURCE_DIR}/include/glow/Base/Train.h
              ${CMAKE_SOURCE_DIR}/include/glow/Base/Traits.h
              ${CMAKE_SOURCE_DIR}/include/glow/Base/Type.h

              ${CMAKE_SOURCE_DIR}/src/glow/Base/Train.cpp
              ${CMAKE_SOURCE_DIR}/src/glow/Base/Type.cpp
              ${CMAKE_SOURCE_DIR}/src/glow/Base/Image.cpp)
target_link_libraries(Base
                      PUBLIC
                        Support)
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
