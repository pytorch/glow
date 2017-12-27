# Copy the images and test script to the build directory.

foreach (filename "cat_285.png" "dog_207.png" "zebra_340.png" "run.sh")
  configure_file(${filename} ${CMAKE_CURRENT_BINARY_DIR}/${filename} COPYONLY)
endforeach(filename)

