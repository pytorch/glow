
add_library(Network
              Train.cpp
              Image.cpp
              Network.cpp
              Node.cpp
              Nodes.cpp)
target_link_libraries(Network
                      PUBLIC
                        IR)
if(PNG_FOUND)
  target_compile_definitions(Network
                             PRIVATE
                               WITH_PNG=1)
  target_include_directories(Network
                             PRIVATE
                               ${PNG_INCLUDE_DIR})
  target_link_libraries(Network
                        PRIVATE
                          ${PNG_LIBRARY})
endif()

