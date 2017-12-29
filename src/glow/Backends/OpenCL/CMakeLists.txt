add_library(OpenCL
            OpenCL.cpp)

target_link_libraries(OpenCL
                      PRIVATE
                      Base
                      Graph
                      CodeGen)

if(OpenCL_FOUND)
  target_compile_definitions(OpenCL
                             PRIVATE
                             WITH_OPENCL=1)

  target_link_libraries(OpenCL
                        PRIVATE
                        OpenCL::OpenCL)
endif()

