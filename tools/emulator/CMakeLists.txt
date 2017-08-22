add_executable(emulator emulator.cpp ${header_files})
target_link_libraries (emulator LINK_PUBLIC ${LinkLibs})

