add_library(Base
              Tensor.cpp
              Type.cpp
              IO.cpp)

target_link_libraries(Base
                      PUBLIC
                        Support)
if(PNG_FOUND)
  target_sources(Base PRIVATE Image.cpp)
  target_compile_definitions(Base
                             PRIVATE
                               WITH_PNG=1)
  target_include_directories(Base
                             PRIVATE
                               ${PNG_INCLUDE_DIR})
  target_link_libraries(Base
                        PRIVATE
                          ${PNG_LIBRARY})
message(${PNG_LIBRARY})
message(${PNG_INCLUDE_DIR})
endif()
