// Copyright 2004-present Facebook. All Rights Reserved.

#include "glow/Flags/Flags.h"
#include "glow/fb/fx_nnpi_importer/Utils.h"
#include "glow/lib/Backends/NNPI/DebugMacros.h"
#include "glow/lib/Backends/NNPI/FXIRImporter.h"
#include "glow/lib/Backends/NNPI/NNPICompiledFunction.h"

using namespace glow;

NNPICompiledFunction::NNPICompiledFunction(
    const folly::dynamic &FXIR, const std::string &submod,
    const llvm::StringMap<const void *> &constants, Module *glowModule)
    : CompiledFunction(
          utils::createRuntimeBundle(FXIR, submod, constants, glowModule)),
      compilationOptions_({}) {
  std::memset(&config_, 0, sizeof(config_));
  std::memset(&devNetConfig_, 0, sizeof(devNetConfig_));
}

Error NNPICompiledFunction::compileFX(
    const folly::dynamic &FXIR, const std::string &submod,
    const llvm::StringMap<const void *> &constants,
    const BackendOptions &opts) {
  BackendOptions newOpts = opts;
  compilationOptions_ = NNPICompilationOptions(newOpts.backendSpecificOpts);

  if (compilationOptions_.compileOutputPostfix) {
    compilationFileName_ =
        compilationOptions_.compiledFile.get() + "_" + submod;
  } else {
    compilationFileName_ = compilationOptions_.compiledFile.get();
  }
  LOG_IF_NOT_RETURN_LLVMERROR(
      compilationFileName_.length() < NNPI_MAX_STRING_LEN, "Bad filename");

  FXNNPIImporter importer(compilationOptions_);
  network_ = importer.importFunction(FXIR, submod, constants);

  LOG_IF_INVALID_HANDLE_RETURN_LLVMERROR(network_, "Failed to import function");
  // Setting the network name.
  std::string networkName = compilationFileName_;
  if (compilationFileName_.empty()) {
    networkName = submod;
  }
  ASSERT_LOG_NNPI_ERROR(nnpiNetworkSetName(network_, networkName.c_str()),
                        "Failed to set NNPI network name");

  // Apply optimizations.
  NNPIOptimizationConfig optConf;
  std::memset(&optConf, 0, sizeof(NNPIOptimizationConfig));
  optConf.lstmReconstruction = 1;
  optConf.reorderTransposeConvert = 1;
  if (!compilationOptions_.disableConstFolding) {
    optConf.constantFolding = 1;
  }

  DBG_MEM_USAGE("NNPICompiledFunction call optimize <<");
  LOG_NNPI_IF_ERROR_RETURN_LLVMERROR(nnpiNetworkOptimize(network_, &optConf),
                                     "Failed NNPI API Optimize");
  DBG_MEM_USAGE("NNPICompiledFunction call get compilation config <<");
  LOG_NNPI_IF_ERROR_RETURN_LLVMERROR(nnpiGetDefaultCompilationConfig(&config_),
                                     "Failed NNPI API Read Config");

  RETURN_IF_ERR(updateCompilationConfigFromOptions(compilationOptions_));
  // Collect input/output names.
  {
    size_t numInputs, numOutputs;
    NNPIObjectName name;
    NNPITensorDesc desc;
    LOG_NNPI_IF_ERROR_RETURN_LLVMERROR(
        nnpiNetworkGetInputNum(network_, &numInputs),
        "Failed to query NNPI network inputs");
    for (size_t i = 0; i < numInputs; i++) {
      LOG_NNPI_IF_ERROR_RETURN_LLVMERROR(
          nnpiNetworkGetInputDesc(network_, i, name, &desc),
          "Failed to query NNPI network inputs");
      inputNames_.push_back(name);
    }
    LOG_NNPI_IF_ERROR_RETURN_LLVMERROR(
        nnpiNetworkGetOutputNum(network_, &numOutputs),
        "Failed to query NNPI network outputs");
    for (size_t i = 0; i < numOutputs; i++) {
      LOG_NNPI_IF_ERROR_RETURN_LLVMERROR(
          nnpiNetworkGetOutputDesc(network_, i, name, &desc),
          "Failed to query NNPI network outputs");
      outputNames_.push_back(name);
    }
  }

  if (compilationOptions_.useIceT || compilationOptions_.inferOnDevice) {
    if (compilationFileName_.empty()) // Compile to memory.
    {
      NNPIStream outFileStream;
      outFileStream.userData = &compiledStream_;
      outFileStream.readCallback = nullptr;
      outFileStream.seekCallback = nullptr;
      outFileStream.writeCallback = [](const void *ptr, uint64_t size,
                                       uint64_t count,
                                       void *userData) -> uint64_t {
        DBG_MEM_USAGE("NNPICompiledFunction before appending: "
                      << ((size * count) / 1024));
        BlockStream *ss = reinterpret_cast<BlockStream *>(userData);
        size_t wSize = ss->write(static_cast<const char *>(ptr), size * count);
        if (wSize < size * count) {
          return 0;
        } else {
          DBG_MEM_USAGE("NNPICompiledFunction stream appended: "
                        << ((size * count) / 1024) << " KB current size: "
                        << ss->getSize() / 1024 << " KB ");
          return size * count;
        }
      };
      DBG_MEM_USAGE("NNPICompiledFunction call get compile <<");
      LOG_NNPI_IF_ERROR_RETURN_LLVMERROR(
          nnpiNetworkCompileToStream(network_, &config_, &outFileStream, NULL),
          "Failed NNPI Compile");

      DBG_MEM_USAGE("NNPICompiledFunction done compile <<");
    } else // Compile to file.
    {
      LOG_NNPI_IF_ERROR_RETURN_LLVMERROR(
          nnpiNetworkCompileToFile(network_, &config_,
                                   compilationFileName_.c_str(), NULL),
          "Failed NNPI Compile");
    }

    // Update compilation info after NNPI compilation.
    if (compilationOptions_.dumpCompilationInfo ||
        compilationOptions_.lightCompilation ||
        flags::DumpBackendSpecificIRJSON) {
      if (!updateCompilationInfo()) {
        // Only issuing a warning (soft fail)
        LOG(WARNING) << "Failed to update NNPI compilation info";
      } else if (compilationOptions_.dumpCompilationInfo) {
        LONG_LOG(INFO, compilationInfo_.dump(networkName));
      }
    }

    if (compilationOptions_.inferOnDevice) {
      DBG_MEM_USAGE("NNPICompiledFunction destroy network");
      // NNPINetwork is not needed anymore on the inferfence api path.
      // Once the complied stream is loaded, query on the network can be done
      // using the host network instead.
      LOG_NNPI_IF_ERROR(nnpiNetworkDestroy(network_),
                        "Failed NNPI Network Destroy");
      network_ = NNPI_INVALID_NNPIHANDLE;
      DBG_MEM_USAGE("NNPICompiledFunction destroy network done");
    }
  }

  // Update device network config.
  devNetConfig_ = parseDeviceNetworkConfig(compilationOptions_);

  return Error::success();
}
