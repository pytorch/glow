add_library(Example Example.cpp)
target_link_libraries(Example
  PRIVATE
  Base Backend Graph)
set(linked_backends ${linked_backends} Example PARENT_SCOPE)
