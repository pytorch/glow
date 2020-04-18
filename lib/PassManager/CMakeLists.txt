add_library(PassManager
              PassConfig.cpp
              PassManager.cpp)

target_link_libraries(PassManager
                      PRIVATE
                        Base
                        Support
                        glog::glog)
