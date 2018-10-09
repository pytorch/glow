add_library(Converter
            FunctionConverter.cpp)

target_link_libraries(Converter
                      PRIVATE
                        Base
                        Graph)
