add_executable(resnet-training
                 main.cpp)
target_link_libraries(resnet-training
                      PRIVATE
                        ExecutionEngine
                        Graph
                        Importer
                        GraphOptimizer)
