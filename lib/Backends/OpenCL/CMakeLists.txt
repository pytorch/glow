add_library(OpenCL
            OpenCL.cpp
            Transforms.cpp)

target_link_libraries(OpenCL
                      PRIVATE
                      Base
                      Graph
                      CodeGen
                      IR)

target_link_libraries(OpenCL
                      PRIVATE
                      OpenCL::OpenCL)
