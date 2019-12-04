# Copy the models to the build directory.

file(GLOB files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} caffe2Models/*.pbtxt)
foreach(filename ${files})
  configure_file(${filename} ${CMAKE_CURRENT_BINARY_DIR}/${filename} COPYONLY)
endforeach(filename)

file(GLOB files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} onnxModels/*.onnxtxt)
foreach(filename ${files})
  configure_file(${filename} ${CMAKE_CURRENT_BINARY_DIR}/${filename} COPYONLY)
endforeach(filename)

# In case of using 32b dimensions, overwrite the models with ones that do not
# try to use 64b as an index type.
if (TENSOR_DIMS_32_BITS)
  file(GLOB files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} onnxModels/dim32/*.onnxtxt)
  foreach(filename ${files})
    get_filename_component(basefn ${filename} NAME)
    configure_file(${filename} ${CMAKE_CURRENT_BINARY_DIR}/onnxModels/${basefn} COPYONLY)
  endforeach(filename)
endif ()
