#include "glow/Importer/ModelLoaderPrecisionConfiguration.h"
#include "glow/Support/Support.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace yaml {

/// Mapping for ModelLoaderPrecisionConfiguration yaml serializer.
template <> struct MappingTraits<glow::ModelLoaderPrecisionConfiguration> {
  static void mapping(IO &io, glow::ModelLoaderPrecisionConfiguration &info) {
    io.mapRequired("FP16NodeInstanceNames", info.fp16OpInstanceNames);
  }
};

} // end namespace yaml
} // end namespace llvm

namespace glow {

llvm::cl::OptionCategory loaderPrecisionCat("ModelLoader Precision Options");

llvm::cl::opt<std::string> loadModelLoaderPrecisionFileOpt(
    "node-precision-info",
    llvm::cl::desc("Load model loader precision file which contains\n"
                   "instances output names to be executed in FP16\n"
                   "Currently supported only for ONNX models"),
    llvm::cl::value_desc("precision_info.yaml"),
    llvm::cl::cat(loaderPrecisionCat));

void setModelLoaderPrecisionOpt(llvm::StringRef fileName) {
  loadModelLoaderPrecisionFileOpt = fileName;
}

bool isModelLoaderPrecisionOptEnabled() {
  return !loadModelLoaderPrecisionFileOpt.empty();
}

Expected<ModelLoaderPrecisionConfiguration>
deserializeModelLoaderPrecisionInfosFromYaml() {
  ModelLoaderPrecisionConfiguration modelLoaderPrecsionConfig;

  llvm::StringRef fileName = loadModelLoaderPrecisionFileOpt;

  RETURN_ERR_IF_NOT(llvm::sys::fs::exists(fileName),
                    "Could not find file with name: " + fileName.str());

  // Open YAML input stream.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> text =
      llvm::MemoryBuffer::getFileAsStream(fileName);

  RETURN_ERR_IF_NOT(!text.getError(),
                    "Unable to open file with name: " + fileName.str());

  std::unique_ptr<llvm::MemoryBuffer> buffer = std::move(*text);
  llvm::yaml::Input yin(buffer->getBuffer());

  // Error message in case of incorrect precision info format.
  std::string ErrMsg =
      strFormat("Error reading YAML file '%s'!", fileName.data());

  // Read profiling info.
  yin >> modelLoaderPrecsionConfig;
  RETURN_ERR_IF_NOT(!yin.error(), ErrMsg);
  return modelLoaderPrecsionConfig;
}

} // namespace glow
