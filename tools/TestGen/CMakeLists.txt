
add_executable(BackendTestGen
                 BackendTestGen.cpp
                 InferFuncBuilder.cpp)

target_link_libraries(BackendTestGen
                      PUBLIC
                        Support)
