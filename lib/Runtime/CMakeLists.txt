add_subdirectory(Provisioner)
add_subdirectory(Executor)
add_subdirectory(HostManager)

add_library(Runtime
  ErrorReporter.cpp
  DeviceHealthMonitor.cpp
  DeferredWeightLoader.cpp
  StatsExporter.cpp)
target_link_libraries(Runtime
  PRIVATE
  Support)
