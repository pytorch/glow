add_library(Extractor
              Extractor.cpp)

target_link_libraries(Extractor
                      PRIVATE
                        Graph)