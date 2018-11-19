add_executable(png2bin
  png2bin.cpp)

target_link_libraries(png2bin
                      PRIVATE
                        Base
                        Quantization
                        Support)
