# Copy the models to the build directory.

file(GLOB files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} caffe2Models/*.pbtxt)
foreach(filename ${files})
  configure_file(${filename} ${CMAKE_CURRENT_BINARY_DIR}/${filename} COPYONLY)
endforeach(filename)
