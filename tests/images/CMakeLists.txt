# Copy the images and test script to the build directory.

configure_file(run.sh ${CMAKE_CURRENT_BINARY_DIR}/run.sh COPYONLY)

file(GLOB files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} imagenet/*.png)
foreach(filename ${files})
  configure_file(${filename} ${CMAKE_CURRENT_BINARY_DIR}/${filename} COPYONLY)
endforeach(filename)

file(GLOB files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} mnist/*.png)
foreach(filename ${files})
  configure_file(${filename} ${CMAKE_CURRENT_BINARY_DIR}/${filename} COPYONLY)
endforeach(filename)
