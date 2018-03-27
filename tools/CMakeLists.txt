add_subdirectory(ClassGen)
add_subdirectory(emulator)
if(PNG_FOUND)
  add_subdirectory(loader)
endif()

