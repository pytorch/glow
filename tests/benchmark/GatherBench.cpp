/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
 *
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
#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <future>
#include <random>
#include <string>

#include "Bench.h"

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

using namespace glow;

/*
 * This class implements an Gather microbenchmark.
 *
 * Microbenchmarks are generally useful for understanding performance
 * through targeted experiementation and are not representative of
 * end-to-end workloads.
 */

struct GatherParam {
  dim_t numReps;
  dim_t numAsyncLaunches;
  std::string backendStr;
  std::string devId;
  dim_t numIndices;
  dim_t numTableEntries;
  dim_t numElementsPerRow;
  dim_t numGatherNodes;
  bool isSorted;
  ElemKind dtype;
};

std::string getGatherDescription(GatherParam param) {
  std::string GatherStr = std::string("Gather");

  return strFormat("%s_%zu_%zu_%zu", GatherStr.c_str(),
                   (size_t)param.numIndices, (size_t)param.numTableEntries,
                   (size_t)param.numElementsPerRow);
}

class GatherBench : public Benchmark {
  std::unique_ptr<runtime::HostManager> hostManager_;
  std::vector<std::unique_ptr<ExecutionContext>> contexts_;
  std::vector<std::vector<Tensor>> indicesReal_;
  dim_t asyncLaunchSize_;
  std::string backendStr_;
  std::vector<GatherParam> params_;
  std::string devId_;

public:
  GatherBench(dim_t asyncLaunchSize_, std::string backendStr_,
              std::vector<GatherParam> params_,
              std::string devId_ = std::string(""))
      : asyncLaunchSize_(asyncLaunchSize_), backendStr_(backendStr_),
        params_(params_), devId_(devId_) {}

  double countGatherInputGbytes(GatherParam param) const {

    dim_t elementSize = 2;
    if (param.dtype == ElemKind::FloatTy) {
      elementSize = 4;
    }

    // Embedding data.
    double input_gbytes = 0.0;
    input_gbytes += (param.numGatherNodes * param.numIndices *
                     (param.numElementsPerRow * elementSize)) /
                    1e9;

    // + Indices.
    input_gbytes +=
        (param.numGatherNodes * param.numIndices * sizeof(int32_t)) / 1e9;

    return input_gbytes;
  }

  void addGatherNode(std::unique_ptr<Module> &mod, Function *fn,
                     GatherParam param) {

    // Input date is Non-quantized and Constant.
    Constant *dataConstant = nullptr;
    Tensor dataConstantTensor(param.dtype,
                              {param.numTableEntries, param.numElementsPerRow});
    if (param.dtype == ElemKind::FloatTy) {
      dataConstantTensor.getHandle<float>().clear(1.0f);
    } else {
      dataConstantTensor.getHandle<float16_t>().clear(1.0f);
    }
    dataConstant = mod->createConstant("GatherData", dataConstantTensor);

    auto *indices = mod->createPlaceholder(ElemKind::Int32ITy,
                                           {param.numIndices}, "indices",
                                           /* isTrainable */ false);

    for (dim_t i = 0; i < asyncLaunchSize_; i++) {

      // Create and sort indices.
      Tensor indicesReal(ElemKind::Int32ITy, {param.numIndices});
      indicesReal.getHandle<int32_t>().randomize(0, param.numTableEntries - 1,
                                                 mod->getPRNG());
      // Sort each segment.
      if (param.isSorted) {
        int32_t *indicesRealPtr = (int32_t *)indicesReal.getUnsafePtr();
        std::sort(indicesRealPtr, indicesRealPtr + param.numIndices);
      }
      indicesReal_[i].push_back(std::move(indicesReal));

      Tensor indicesPartial(indicesReal_[i].back().getUnsafePtr(),
                            indices->getType(),
                            indicesReal_[i].back().getSizeInBytes());

      contexts_[i]->getPlaceholderBindings()->insert(indices,
                                                     std::move(indicesPartial));

    } // i

    // Create Gather node, then slice it and then save node.
    Node *R = nullptr;
    R = fn->createGather(getGatherDescription(param), dataConstant, indices, 0);
    SliceNode *SN;
    SN = fn->createSlice("slice", R, {0, 0}, {1, param.numElementsPerRow});

    SaveNode *S = nullptr;
    S = fn->createSave("save", SN);

    // For each context, add output bindings.
    for (dim_t i = 0; i < asyncLaunchSize_; i++) {
      contexts_[i]->getPlaceholderBindings()->allocate(S->getPlaceholder());
    }
  }

  void setup() override {

    // Create execution contexts here.
    for (dim_t i = 0; i < asyncLaunchSize_; i++) {
      std::unique_ptr<ExecutionContext> context(new ExecutionContext);
      contexts_.push_back(std::move(context));
    }

    // Setup host manager.
    std::vector<std::unique_ptr<runtime::DeviceConfig>> configs;
    auto config = glow::make_unique<runtime::DeviceConfig>(backendStr_.c_str());
    if (!devId_.empty()) {
      config->parameters["DeviceID"] = devId_.c_str();
    }
    configs.push_back(std::move(config));
    hostManager_ = glow::make_unique<runtime::HostManager>(std::move(configs));

    // Create a function.
    std::unique_ptr<Module> mod(new Module);
    auto fn = mod->createFunction("singleNode");

    // Keep tensors around so they aren't deleted.
    indicesReal_.resize(asyncLaunchSize_);

    // Add Gather nodes.
    for (auto &param : params_) {
      for (int i = 0; i < param.numGatherNodes; i++) {
        addGatherNode(mod, fn, param);
      }
    }

    fn->dumpDAG("gatherbench.dot");
    CompilationContext ctx;
    EXIT_ON_ERR(hostManager_->addNetwork(std::move(mod), ctx));
  }

  void run() override {
    std::vector<std::unique_ptr<ExecutionContext>> localContexts(
        asyncLaunchSize_);
    std::vector<std::promise<void>> promises(asyncLaunchSize_);
    std::vector<std::future<void>> futures;

    // Launch a number of independent requests.
    int i = 0;
    for (auto &promise : promises) {
      futures.push_back(promise.get_future());
      hostManager_->runNetwork(
          "singleNode", std::move(contexts_[i]),
          [&localContexts, &promise,
           i](runtime::RunIdentifierTy, Error err,
              std::unique_ptr<ExecutionContext> contextPtr) {
            EXIT_ON_ERR(std::move(err));
            localContexts[i] = std::move(contextPtr);
            promise.set_value();
          });
      i++;
    }
    for (auto &fut : futures) {
      fut.wait();
    }
    for (dim_t j = 0; j < asyncLaunchSize_; j++) {
      contexts_[j] = std::move(localContexts[j]);
    }
  }

  void teardown() override {}

  double inputgbytes() const {
    double total_in = 0.0;
    for (auto &param : params_) {
      total_in += countGatherInputGbytes(param);
    }
    return total_in;
  }
};

// Index of arguments.
#define DEVICE_ID 10

GatherParam parseArgs(int argc, char *argv[]) {
  GatherParam param;
  param.numIndices = atoi(argv[1]);
  param.numTableEntries = atoi(argv[2]);
  param.numElementsPerRow = atoi(argv[3]);
  param.numReps = atoi(argv[4]);
  param.numAsyncLaunches = atoi(argv[5]);
  param.numGatherNodes = atoi(argv[6]);
  printf("numIndices %zu\n", (size_t)param.numIndices);
  printf("numTableEntries %zu\n", (size_t)param.numTableEntries);
  printf("numElementsPerRow %zu\n", (size_t)param.numElementsPerRow);
  printf("numReps %zu\n", (size_t)param.numReps);
  printf("numAsyncLaunches %zu\n", (size_t)param.numAsyncLaunches);
  printf("numGatherNodes %zu\n", (size_t)param.numGatherNodes);
  printf("sortedStr %s\n", argv[7]);
  if (std::string(argv[7]) == "Sorted") {
    param.isSorted = true;
  } else if (std::string(argv[7]) == "Unsorted") {
    param.isSorted = false;
  } else {
    llvm_unreachable("Invalid sortedStr");
  }
  printf("backendStr %s\n", argv[8]);
  param.backendStr = std::string(argv[8]);
  printf("dtypeStr %s\n", argv[9]);
  if (std::string(argv[9]) == "Float16") {
    param.dtype = ElemKind::Float16Ty;
  } else if (std::string(argv[9]) == "Float32") {
    param.dtype = ElemKind::FloatTy;
  } else {
    llvm_unreachable("Invalid dtype");
  }
  if (argc > DEVICE_ID) {
    printf("devId %s\n", argv[DEVICE_ID]);
    param.devId = std::string(argv[DEVICE_ID]);
  } else {
    param.devId = std::string("");
  }
  printf("\n\n");
  return param;
}

int main(int argc, char *argv[]) {

  printf("Gather Microbenchmark\n");
  printf("Usage: GatherBench numIndices(Int) "
         "numTableEntries(Int) "
         "numElementsPerRow(int) numReps(Int) "
         "numAsyncLaunches(Int) numGatherNodes(Int) "
         "sortedStr(\"Sorted\"|\"Unsorted\") backendStr(String) "
         "dtypeStr(\"Float16\"|\"Float32\") "
         "dev_id(Int)\n");
  printf("\n");

  std::vector<GatherParam> params;
  std::string runHeader;
  std::string runPrefix;

  // Using a config file.
  if (argc == 2) {
    auto fname = std::string(argv[1]);
    std::ifstream fin(fname.c_str());
    if (!fin) {
      std::cout << "Could not open file: " << fname << std::endl;
      exit(0);
    }
    std::string line;
    while (getline(fin, line)) {
      std::array<char, 1024> buf;
      char *saveptr = nullptr;
      std::vector<char *> argVec;
      strcpy(buf.data(), line.c_str());
      char *ptr = strtok_r(buf.data(), " ", &saveptr);
      while (ptr != nullptr) {
        argVec.push_back(ptr);
        ptr = strtok_r(nullptr, " ", &saveptr);
      }
      GatherParam param = parseArgs(argVec.size(), argVec.data());
      params.push_back(param);
      runHeader = std::string("_,benchName,_,filename");
      runPrefix = std::string(strFormat("GatherBench,SW,%s", fname.c_str()));
    }
  }
  // Using command line.
  else if (argc == 10 || argc == 11) {
    GatherParam param = parseArgs(argc, argv);
    params.push_back(param);

    runHeader =
        std::string("_,benchName,_numIndices,"
                    "numTableEntries,"
                    "numElementsPerRow,numReps,numAsyncLaunches,numGatherNodes,"
                    "sorted,backendStr,dtypeStr");
    runPrefix = std::string(
        strFormat("GatherBench,SW,%zu,%zu,%zu,%zu,%zu,%zu,%s,%s,%s",
                  (size_t)param.numIndices, (size_t)param.numTableEntries,
                  (size_t)param.numElementsPerRow, (size_t)param.numReps,
                  (size_t)param.numAsyncLaunches, (size_t)param.numGatherNodes,
                  argv[7], argv[8], argv[9]));
  } else {
    llvm_unreachable("Invalid command line");
  }

  GatherParam param = params.front();
  GatherBench b(param.numAsyncLaunches, param.backendStr, params, param.devId);
  auto times = bench(&b, param.numReps);

  printf("%s,runtime, gbytesPerSec\n", runHeader.c_str());
  for (auto t : times) {
    printf("BenchResult,%s,%f,%f\n", runPrefix.c_str(),
           t / param.numAsyncLaunches,
           b.inputgbytes() * param.numAsyncLaunches / t);
  }
  double min = *(std::min_element(times.begin(), times.end()));
  dim_t midElt = times.size() / 2;
  std::nth_element(times.begin(), times.begin() + midElt, times.end());
  double median = times[midElt];
  double medianRuntime = median / ((double)param.numAsyncLaunches);
  double minRuntime = min / ((double)param.numAsyncLaunches);
  printf("%s,medianRuntime,minRuntime,"
         "medianGbytesPerSec,maxGbytesPerSec\n",
         runHeader.c_str());
  printf("BenchSummary,%s,%f,%f,%f,%f\n", runPrefix.c_str(), medianRuntime,
         minRuntime, b.inputgbytes() / medianRuntime,
         b.inputgbytes() / minRuntime);
}
