add_library(Converter
            Float16Converter.cpp
            FusedRowwiseConverter.cpp
            FunctionConverter.cpp
            TypeAToTypeBFunctionConverter.cpp)

target_link_libraries(Converter
                      PRIVATE
                        Base
                        Graph)
