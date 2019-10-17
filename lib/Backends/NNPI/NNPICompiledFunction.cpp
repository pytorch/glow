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
#include "nnpi_transformer.h"

using namespace glow;

Error NNPICompiledFunction::compile(Function *F, const BackendOptions &opts) {
  NNPIImporter importer;
  network_ = importer.importFunction(F, opts);
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

  // Parse compilation config options.
  auto customDspLib = opts.backendSpecificOpts.find("NNPICustomDSPLib");
  if (customDspLib != opts.backendSpecificOpts.end()) {
    std::strncpy(config_.customDspKernelsFile, customDspLib->second.c_str(),
                 sizeof(config_.customDspKernelsFile));
  }

  // Handle device version.
  std::string deviceVersion = EnvDeviceVersion();
  if (deviceVersion.empty() &&
      opts.backendSpecificOpts.count("NNPIDeviceVersion")) {
    deviceVersion = opts.backendSpecificOpts.at("NNPIDeviceVersion");
  }
  if (!deviceVersion.empty()) {
    if (deviceVersion.compare("1") == 0) {
      config_.deviceType = NNPI_DEVICE_M2_A;
    } else if (deviceVersion.compare("2") == 0) {
      config_.deviceType = NNPI_DEVICE_M2_B;
    } else if (deviceVersion.compare("3") == 0) {
      config_.deviceType = NNPI_DEVICE_M2_C;
    } else {
      LOG_IF_NOT_RETURN_LLVMERROR(
          false, "INVALID NNPI_DEVICE_VERSION, valid values are 1,2,3");
    }
  }

  if (UseIceT() || UseInferenceAPI()) {
    auto filename = ICETFilename();
    LOG_IF_NOT_RETURN_LLVMERROR(filename.length() < NNPI_MAX_STRING_LEN,
                                "Bad filename");

    if (filename.empty()) // Compile to memory.
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
          nnpiNetworkCompileToFile(network_, &config_, filename.c_str(), NULL),
          "Failed NNPI Compile");
    }
  }
  return Error::success();
}

NNPICompiledFunction::~NNPICompiledFunction() {
  LOG_NNPI_ERROR(nnpiNetworkDestroy(network_), "Failed NNPI Network Destroy");
}

BlockStream &NNPICompiledFunction::lockCompiledStream() {
  DBG_MEM_USAGE("lockCompiledStream");
  compiledStreamMutex_.lock();
  return compiledStream_;
}

void NNPICompiledFunction::unlockCompiledStream() {
  DBG_MEM_USAGE("unlockCompiledStream");
  compiledStream_.reset();
  compiledStreamMutex_.unlock();
}

void NNPICompiledFunction::freeCompilationResources() {
  DBG_MEM_USAGE("[Before] freeCompilationResources ");
  lockCompiledStream();
  compiledStream_.releaseMemory();
  unlockCompiledStream();
  DBG_MEM_USAGE("[After] freeCompilationResources ");
}
