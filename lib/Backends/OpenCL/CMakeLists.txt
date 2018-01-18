add_library(OpenCL
            OpenCL.cpp)

target_link_libraries(OpenCL
                      PRIVATE
                      Base
                      Graph
                      CodeGen)

target_link_libraries(OpenCL
                      PRIVATE
                      OpenCL::OpenCL)

