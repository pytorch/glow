#ifndef GLOW_TOOLS_LOADER_H
#define GLOW_TOOLS_LOADER_H

#include "llvm/Support/CommandLine.h"

enum class ImageNormalizationMode;

namespace opts {
extern llvm::cl::list<std::string> InputImageFilenames;
extern llvm::cl::opt<std::string> NetDescFilename;
extern llvm::cl::opt<std::string> NetWeightFilename;
extern llvm::cl::opt<std::string> NetDirectory;
extern llvm::cl::opt<std::string> DumpGraphDAGFile;
extern llvm::cl::opt<bool> DumpGraph;
extern llvm::cl::opt<std::string> DumpIRDAGFile;
extern llvm::cl::opt<bool> DumpIR;
extern llvm::cl::opt<ImageNormalizationMode> ImageMode; // interesting
extern llvm::cl::opt<bool> Verbose;
extern llvm::cl::opt<bool> Timer;
} // namespace opts

#endif
