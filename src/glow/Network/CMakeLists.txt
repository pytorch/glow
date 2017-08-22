if (PNG_FOUND)
  include_directories(${PNG_INCLUDE_DIR})
endif (PNG_FOUND)

add_library (Network
             ./Train.cpp
             ./Image.cpp
             ./Network.cpp
             ./Node.cpp
             ./Nodes.cpp
            )


