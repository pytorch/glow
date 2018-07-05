add_library(Quantization
              Profile.cpp
              Serialization.cpp
              Quantization.cpp)

target_link_libraries(Quantization
                      PRIVATE
                        Graph
                        ExecutionEngine
			QuantizationBase)
