/*
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "NNPICompiledFunction.h"
#include "DebugMacros.h"
#include "Importer.h"
#include "NNPI.h"
#include "NNPIOptions.h"
#include "nnpi_transformer.h"

#include "glow/Backend/BackendUtils.h"

using namespace glow;

Error NNPICompiledFunction::updateCompilationConfigFromOptions(
    NNPICompilationOptions &compilationOptions) {
  if (compilationOptions.showVars) {
    LOG(INFO) << compilationOptions.dumpStatus();
  }
  if (!compilationOptions.customDspKernelsFile.get().empty()) {
    std::strncpy(config_.customDspKernelsFile,
                 compilationOptions.customDspKernelsFile.get().c_str(),
                 sizeof(config_.customDspKernelsFile));
  }

  // Handle device version.
  if (compilationOptions.deviceVersion > 0) {
    switch (compilationOptions.deviceVersion) {
    case 1:
      config_.deviceType = NNPI_1000_A;
      break;
    case 2:
      config_.deviceType = NNPI_1000_B;
      break;
    case 3:
      config_.deviceType = NNPI_1000_C;
      break;
    default:
      LOG_IF_NOT_RETURN_LLVMERROR(
          false, "INVALID NNPI_DEVICE_VERSION, valid values are 1,2,3");
    }
  }

  if (compilationOptions.iceCores > 0) {
    config_.numCoresToUse = static_cast<uint32_t>(compilationOptions.iceCores);
  }
  if (!compilationOptions.debugCompileConfigFile.get().empty()) {
    strncpy(config_.debugConfigFile,
            compilationOptions.debugCompileConfigFile.get().c_str(),
            sizeof(config_.debugConfigFile));
  }
  return Error::success();
}

Error NNPICompiledFunction::compile(Function *F, const BackendOptions &opts) {
  BackendOptions newOpts = opts;
  if (opts.backendHints.executionUnits) {
    LOG(INFO) << " Setting backendSpecificOpts cores for network: "
              << F->getName().str() << " to "
              << std::to_string(opts.backendHints.executionUnits) << " cores ";

    newOpts.backendSpecificOpts["IceCores"] =
        std::to_string(opts.backendHints.executionUnits);
  }

  compilationOptions_ = NNPICompilationOptions(newOpts.backendSpecificOpts);
  NNPIImporter importer(compilationOptions_);
  network_ = importer.importFunction(F, newOpts);
  LOG_INVALID_HANDLE_RETURN_LLVMERROR(network_, "Failed to import function");

  // Apply optimizations.
  NNPIOptimizationConfig optConf;
  std::memset(&optConf, 0, sizeof(NNPIOptimizationConfig));
  optConf.lstmReconstruction = 1;
  optConf.reorderTransposeConvert = 1;
  DBG_MEM_USAGE("NNPICompiledFunction call optimize <<");
  LOG_NNPI_ERROR_RETURN_LLVMERROR(nnpiNetworkOptimize(network_, &optConf),
                                  "Failed NNPI API Optimize");
  DBG_MEM_USAGE("NNPICompiledFunction call get compilation config <<");
  LOG_NNPI_ERROR_RETURN_LLVMERROR(nnpiGetDefaultCompilationConfig(&config_),
                                  "Failed NNPI API Read Config");

  auto error = updateCompilationConfigFromOptions(compilationOptions_);
  if (error) {
    return error;
  }

  if (compilationOptions_.useIceT || compilationOptions_.inferOnDevice) {
    compilationFileName_ = compilationOptions_.compiledFile.get();
    LOG_IF_NOT_RETURN_LLVMERROR(
        compilationFileName_.length() < NNPI_MAX_STRING_LEN, "Bad filename");

    if (compilationFileName_.empty()) // Compile to memory.
    {
      NNPIStream outFileStream;
      outFileStream.userData = &compiledStream_;
      outFileStream.readCallback = NULL;
      outFileStream.seekCallback = NULL;
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
      LOG_NNPI_ERROR_RETURN_LLVMERROR(
          nnpiNetworkCompileToStream(network_, &config_, &outFileStream, NULL),
          "Failed NNPI Compile");

      DBG_MEM_USAGE("NNPICompiledFunction done compile <<");
    } else // Compile to file.
    {
      LOG_NNPI_ERROR_RETURN_LLVMERROR(
          nnpiNetworkCompileToFile(network_, &config_,
                                   compilationFileName_.c_str(), NULL),
          "Failed NNPI Compile");
    }
    if (compilationOptions_.inferOnDevice) {
      DBG_MEM_USAGE("NNPICompiledFunction destroy network");
      // NNPINetwork is not needed anymore on the inferfence api path.
      // Once the complied stream is loaded, query on the network can be done
      // using the host network instead.
      LOG_NNPI_ERROR(nnpiNetworkDestroy(network_),
                     "Failed NNPI Network Destroy");
      network_ = NNPI_INVALID_NNPIHANDLE;
      DBG_MEM_USAGE("NNPICompiledFunction destroy network done");
    }
  }

  // Determine and save what inputs can be treated as partial. Need to do this
  // while we still have access to F.
  for (auto const &P : F->getParent()->getPlaceholders()) {
    if (!usedInFunction(P, F)) {
      continue;
    }
    if (allowsPartialInput(P, F)) {
      partialInputs_.insert(P);
    }
    if (P->isStatic()) {
      staticInputs_.insert(P);
    }
  }
  return Error::success();
}

NNPICompiledFunction::~NNPICompiledFunction() {
  if (network_ != NNPI_INVALID_NNPIHANDLE) {
    LOG_NNPI_ERROR(nnpiNetworkDestroy(network_), "Failed NNPI Network Destroy");
  }
}

BlockStream &NNPICompiledFunction::lockCompiledStream() {
  DBG_MEM_USAGE("lockCompiledStream");
  compiledStreamMutex_.lock();
  return compiledStream_;
}

void NNPICompiledFunction::unlockCompiledStream() {
  DBG_MEM_USAGE("unlockCompiledStream");
  compiledStream_.resetRead();
  compiledStreamMutex_.unlock();
}

void NNPICompiledFunction::freeCompilationResources() {
  DBG_MEM_USAGE("[Before] freeCompilationResources ");
  lockCompiledStream();
  compiledStream_.releaseMemory();
  unlockCompiledStream();
  DBG_MEM_USAGE("[After] freeCompilationResources ");
}
