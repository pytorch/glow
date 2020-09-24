
# This definition is due to older buggy C++11 cstdint implementations which do
# not import the printf format macros (used at least by DimType.h) without it
# in place. Since the definition doesn't harm the correct implementations,
# we'll define it explicitly here without a preprocessor test.
#
# See: https://github.com/pytorch/glow/issues/3929 and
#      https://sourceware.org/bugzilla/show_bug.cgi?id=15366
add_definitions("-D__STDC_FORMAT_MACROS")

add_custom_target(AutoGen)

add_subdirectory(Backend)
add_subdirectory(Backends)
add_subdirectory(Base)
add_subdirectory(CodeGen)
add_subdirectory(Converter)
add_subdirectory(ExecutionContext)
add_subdirectory(ExecutionEngine)
add_subdirectory(Exporter)
add_subdirectory(Flags)
add_subdirectory(Graph)
add_subdirectory(IR)
add_subdirectory(Importer)
if(GLOW_WITH_LLVMIRCODEGEN)
    add_subdirectory(LLVMIRCodeGen)
endif()
add_subdirectory(Optimizer)
add_subdirectory(Quantization)
add_subdirectory(Runtime)
add_subdirectory(Support)
add_subdirectory(Onnxifi)
add_subdirectory(Partitioner)
add_subdirectory(PassManager)
