#ifndef GLOW_IMPORTER_MODELLOADERPRECISIONCONFIGURATION_H
#define GLOW_IMPORTER_MODELLOADERPRECISIONCONFIGURATION_H

#include "glow/Support/Error.h"

#include "llvm/ADT/StringRef.h"

#include <vector>

namespace glow {
/// Holds info about mixed precision details which can be used across model
/// loaders
struct ModelLoaderPrecisionConfiguration {
  /// Used during operator loading while constructing glow graph to keep the
  /// precision of specified operator names to FP16 (i.e. quantization
  /// conversion is skipped and FP16 conversion is done for any node kinds
  /// found here). This creates a graph where some nodes execute in quantized
  /// or FP32 precision and remaining in FP16 precision. If the node kind
  /// specified via its name is unsupported by the backend in FP16 precision
  /// it will throw an exception. Node instances intended to run in FP16 will
  /// be in yaml file as list which can be mapped directly to a vector of
  /// string, therefore parsing will be faster.
  std::vector<std::string> fp16OpInstanceNames;
};

/// Sets model loader precision profile option with YAML \p fileName.
void setModelLoaderPrecisionOpt(llvm::StringRef fileName);

/// Check if node precision info file is provided.
bool isModelLoaderPrecisionOptEnabled();

/// Deserialize Model loader precision info from the YAML file found at path
/// \ref loadModelLoaderPrecisionFileOpt.
Expected<ModelLoaderPrecisionConfiguration>
deserializeModelLoaderPrecisionInfosFromYaml();
} // namespace glow

#endif // GLOW_IMPORTER_MODELLOADERPRECISIONCONFIGURATION_H
