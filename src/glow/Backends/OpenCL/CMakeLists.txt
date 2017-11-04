find_package(OpenCL REQUIRED )

add_library(OpenCL
            OpenCL.cpp)

target_link_libraries(OpenCL
                      PRIVATE
                      Base
                      Graph
                      OpenCL::OpenCL)

