
add_library(Optimizer
            Optimizer.cpp
           )
target_link_libraries(Optimizer
                      PRIVATE
                      IR
                     )

