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
#include "CommonSLSEB.h"

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"

using namespace glow;

/*
 * This class implements an SLS microbenchmark. There are a number of
 * parallel FusedRowwiseQuantizedSparseLengthsWeightedSum,
 * FusedRowwiseQuantizedSparseLengthsSum, SparseLengthsWeightedSum, or
 * SparseLengthsSum nodes which are created.
 *
 * Microbenchmarks are generally useful for understanding performance
 * through targeted experiementation and are not representative of
 * end-to-end workloads.
 */

llvm::cl::OptionCategory SLSBenchCat("SLSBench Category");
llvm::cl::opt<bool> dumpOnnx("dump_onnx",
                             llvm::cl::desc("dump onnx text format for model"),
                             llvm::cl::Optional, llvm::cl::init(false),
                             llvm::cl::cat(SLSBenchCat));

class SLSBench : public Benchmark {
  std::unique_ptr<runtime::HostManager> hostManager_;
  std::vector<std::unique_ptr<ExecutionContext>> contexts_;
  std::vector<std::vector<Tensor>> indicesReal_;
  std::vector<std::vector<Tensor>> weightsReal_;
  dim_t batchSize_;
  dim_t asyncLaunchSize_;
  std::string backendStr_;
  std::vector<benchmark::SLSParam> params_;
  bool convertFusedToFP32_;
  std::string devId_;

public:
  SLSBench(dim_t batchSize_, dim_t asyncLaunchSize_, std::string backendStr_,
           std::vector<benchmark::SLSParam> params_, bool convertFusedToFP32,
           std::string devId_ = std::string(""))
      : batchSize_(batchSize_), asyncLaunchSize_(asyncLaunchSize_),
        backendStr_(backendStr_), params_(params_),
        convertFusedToFP32_(convertFusedToFP32), devId_(devId_) {}

  void setup() override {
    benchmark::setupSLS(batchSize_, asyncLaunchSize_, backendStr_, devId_,
                        convertFusedToFP32_, hostManager_, contexts_,
                        indicesReal_, weightsReal_, params_, dumpOnnx,
                        benchmark::SLS_BENCH);
  }

  void run() override {
    benchmark::runSLS(asyncLaunchSize_, hostManager_, contexts_);
  }

  void teardown() override {}

  double gbytes() const {
    double total = 0.0;
    for (auto &param : params_) {
      total += benchmark::countSLSGbytes(param, batchSize_);
    }
    return total;
  }
};

int main(int argc, char *argv[]) {

  std::string runPrefix, runHeader;
  std::vector<benchmark::SLSParam> params;

  std::tie(params, runPrefix, runHeader) =
      benchmark::preMain(argc, argv, benchmark::SLS_BENCH);

  benchmark::SLSParam param = params.front();
  SLSBench b(param.batchSize, param.numAsyncLaunches, param.backendStr, params,
             param.convertFusedToFP32, param.devId);
  auto times = bench(&b, param.numReps);
  benchmark::printSummary(runPrefix, runHeader, param, times, b.gbytes());
}
