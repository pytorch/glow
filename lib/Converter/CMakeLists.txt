add_library(Converter
            Float16Converter.cpp
            FunctionConverter.cpp
            TypeAToTypeBFunctionConverter.cpp)

target_link_libraries(Converter
                      PRIVATE
                        Base
                        Graph)
