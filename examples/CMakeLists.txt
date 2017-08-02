
add_executable(cifar10 cifar10.cpp ${header_files})
target_link_libraries (cifar10 LINK_PUBLIC ${LinkLibs})

add_executable(mnist mnist.cpp ${header_files})
target_link_libraries (mnist LINK_PUBLIC ${LinkLibs})

