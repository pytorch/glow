add_library(Converter
            FunctionConverter.cpp
            TypeAToTypeBFunctionConverter.cpp)

target_link_libraries(Converter
                      PRIVATE
                        Base
                        Graph
                        ExecutionEngine)
