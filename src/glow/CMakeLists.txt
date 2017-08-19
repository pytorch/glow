if (PROTOBUF_FOUND)
  include_directories(${PROTOBUF_INCLUDE_DIRS})
endif (PROTOBUF_FOUND)

if (PNG_FOUND)
  include_directories(${PNG_INCLUDE_DIR})
endif (PNG_FOUND)

add_library (libGlow
                       ./Train.cpp
                       ./Image.cpp
                       ./Network.cpp
                       ./Node.cpp
                       ./Nodes.cpp
                       ./Random.cpp
                       ${header_files}
                       ${PROTO_SRCS}
                       ${PROTO_HDRS})


