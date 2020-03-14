add_executable(image-classifier
  Loader.cpp
  ImageClassifier.cpp
  ExecutorCore.cpp
  ExecutorCoreHelperFunctions.cpp)

target_link_libraries(image-classifier
                      PRIVATE
                        Backends
                        Base
                        Converter
                        ExecutionEngine
                        HostManager
                        Graph
                        Importer
                        GraphOptimizer
                        Quantization
                        LLVMSupport)

add_executable(text-translator
  Loader.cpp
  TextTranslator.cpp)

target_link_libraries(text-translator
                      PRIVATE
                        Backends
                        Base
                        Converter
                        Graph
                        HostManager
                        Importer
                        ExecutionEngine
                        GraphOptimizer
                        Quantization
                        LLVMSupport)

add_executable(model-runner
  Loader.cpp
  ModelRunner.cpp)

target_link_libraries(model-runner
                      PRIVATE
                        Backends
                        Base
                        Converter
                        Graph
                        HostManager
                        Importer
                        ExecutionEngine
                        GraphOptimizer
                        Quantization
                        LLVMSupport)

add_executable(x-model-builder
  Loader.cpp
  XModelBuilder.cpp)

target_link_libraries(x-model-builder
                      PRIVATE
                        Backends
                        Base
                        Converter
                        Graph
                        HostManager
                        Importer
                        ExecutionEngine
                        GraphOptimizer
                        Quantization
                        LLVMSupport)

add_executable(model-compiler
  Loader.cpp
  ModelCompiler.cpp)

target_link_libraries(model-compiler
                      PRIVATE
                        Backends
                        Base
                        Converter
                        Graph
                        HostManager
                        Importer
                        ExecutionEngine
                        GraphOptimizer
                        Quantization
                        LLVMSupport)

add_executable(model-profiler
  Loader.cpp
  LoaderUtils.cpp
  ModelProfiler.cpp)

target_link_libraries(model-profiler
                      PRIVATE
                        Backends
                        Base
                        Converter
                        Graph
                        HostManager
                        Importer
                        ExecutionEngine
                        GraphOptimizer
                        Quantization
                        LLVMSupport)

add_executable(model-tuner
  Loader.cpp
  LoaderUtils.cpp
  ModelTuner.cpp)

target_link_libraries(model-tuner
                      PRIVATE
                        Backends
                        Base
                        Converter
                        Graph
                        HostManager
                        Importer
                        ExecutionEngine
                        GraphOptimizer
                        Quantization
                        LLVMSupport)
