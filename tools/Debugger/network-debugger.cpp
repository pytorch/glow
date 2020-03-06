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

/*
 *  This is a debugging tool that can be used to detect errors in a test
 * backend. it works by loading a model from a protobuf file, compile this model
 * on two backends; a reference backend (Interpreter) and a a test backend. It
 * compares the results from running these networks on a layer by layer fashion
 * to detect which layer generated wrong results.
 *
 *  Sample run line:
 *  ./network-debuger --model function_0.zip --inputs input_0.onnx -backend=CPU
 *
 *  The tool will print to screen when a faulty layer is detected. Input and
 * output tensors for the layers will be printed from the reference network to
 * disk.
 *
 */

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Exporter/ONNXModelWriter.h"
#include "glow/Importer/ONNXModelLoader.h"
#include "glow/Support/Debug.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Signals.h"

#include "NetworkComparator.h"
#include "glow/Converter/Float16Converter.h"

#include <string>

using namespace glow;

namespace {
llvm::cl::OptionCategory debuggerTestCat("Debugger Category");

llvm::cl::opt<std::string> modelPathOpt("model", llvm::cl::desc("Input models"),
                                        llvm::cl::value_desc("modelPath"),
                                        llvm::cl::Required,
                                        llvm::cl::cat(debuggerTestCat));
llvm::cl::list<std::string> inputsOpt("inputs", llvm::cl::desc("Inputs"),
                                      llvm::cl::value_desc("Inputs"),
                                      llvm::cl::Required, llvm::cl::OneOrMore,
                                      llvm::cl::cat(debuggerTestCat));
llvm::cl::opt<std::string>
    testBackend("backend",
                llvm::cl::desc("Backend to use, e.g. Interpreter, CPU, NNPI:"),
                llvm::cl::init("Interpreter"), llvm::cl::cat(debuggerTestCat));

llvm::cl::opt<std::string> comparatorType(
    "comparator",
    llvm::cl::desc("The type of comparator to use, Recursive or Intermediate"),
    llvm::cl::init("Intermediate"), llvm::cl::cat(debuggerTestCat));

llvm::cl::opt<float> numericCmpThreshold(
    "threshold", llvm::cl::desc("Threshold for tensor numeric comparison"),
    llvm::cl::Optional, llvm::cl::init(1e-5), llvm::cl::cat(debuggerTestCat));

llvm::cl::opt<bool> dumpTensors(
    "dump_tensors",
    llvm::cl::desc("Dump input(s)\\output(s) of an errant layer to files."),
    llvm::cl::Optional, llvm::cl::init(false), llvm::cl::cat(debuggerTestCat));

llvm::cl::opt<bool>
    globalFp16Opt("glow_global_fp16",
                  llvm::cl::desc("Enable fp16 lowering for all ops on the net"),
                  llvm::cl::Optional, llvm::cl::cat(debuggerTestCat));

llvm::cl::opt<bool> fuseScaleOffsetFp16Opt(
    "glow_global_fused_scale_offset_fp16",
    llvm::cl::desc(
        "Enable fp16 lowering for all op inputs using fused scale/offset"),
    llvm::cl::Optional, llvm::cl::cat(debuggerTestCat));

llvm::cl::opt<bool>
    ClipFp16Opt("glow_clip_fp16",
                llvm::cl::desc("Force glow to clip fp16 values to min/max"),
                llvm::cl::Optional, llvm::cl::cat(debuggerTestCat));

llvm::cl::opt<bool>
    forceFP16AccumSLSOpt("glow_global_force_sls_fp16_accum",
                         llvm::cl::desc("Force FP16 accumulation for SLS ops"),
                         llvm::cl::Optional, llvm::cl::cat(debuggerTestCat));

#define DEBUG_TYPE "verifier"

void parseCommandLine(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      " Network Debugger tool, part of the Glow compiler\n\n"
      "Glow is a compiler for neural network accelerators.\n");
}

// Utility functions:
void convertNetwork(Function *F) {
  CompilationContext cctx;
  PrecisionConfiguration precConfig;
  if (globalFp16Opt) {
    precConfig.convertToFP16 = globalFp16Opt;
  }
  if (fuseScaleOffsetFp16Opt) {
    precConfig.convertFusedToFP16 = fuseScaleOffsetFp16Opt;
  }
  if (ClipFp16Opt) {
    precConfig.clipFP16 = ClipFp16Opt;
  }
  if (forceFP16AccumSLSOpt) {
    precConfig.forceFP16AccumSLS = true;
  }
  // Convert Network to fp16.
  if (globalFp16Opt || fuseScaleOffsetFp16Opt || ClipFp16Opt ||
      forceFP16AccumSLSOpt)
    glow::convertFunctionToFloat16(F, precConfig);

  // Optimize network after conversion to remove unneeded converts.
  glow::optimize(F, CompilationMode::Infer);
}

/// Whether the ONNX loader loaded a model that was exporting with custom Glow
/// ops. This should be in sync with exporting of inputs and so is saved for use
/// with fillPlaceholders().
bool usingGlowCustomOps = false;

void loadModelIntoFunc(Function *F) {
  Error err = Error::empty();
  {
    ONNXModelLoader onnxLD(modelPathOpt, {}, {}, *F, &err, /*zipMode*/ true);
    usingGlowCustomOps = onnxLD.usingGlowCustomOps();
  }
  CHECK(!ERR_TO_BOOL(std::move(err)))
      << "ONNXModelLoader failed to load model: " << modelPathOpt;
  convertNetwork(F);
}

bool run() {
  LOG(INFO) << "Comparing the " << testBackend
            << " backend against the Interpreter "
               "(reference backend)";
  Module mod;
  bool allPass = true;
  loadModelIntoFunc(mod.createFunction("test"));
  // Create a Comparator based on the type.
  // NetworkComparatorBase *netCompare;
  std::unique_ptr<NetworkComparatorBase> netCompare;
  if (comparatorType == "comparatorType") {
    netCompare.reset(new IntermediateLayerComparator(
        mod, "Interpreter", testBackend, numericCmpThreshold, dumpTensors));
  } else {
    netCompare.reset(new RecursiveLayerComparator(
        mod, "Interpreter", testBackend, numericCmpThreshold, dumpTensors));
  }
  PlaceholderBindings inputBindings;
  inputBindings.allocate(mod.getPlaceholders());
  for (size_t idx = 0; idx != inputsOpt.size(); idx++) {
    fillPlaceholders(inputsOpt[idx], &inputBindings,
                     /* partialTensorPayloads */ nullptr, usingGlowCustomOps);
    // Test the network with these inputs
    allPass &= netCompare->verify(&inputBindings);
    inputBindings.clear();
  }
  return allPass;
}
} // namespace

int main(int argc, char **argv) {
  parseCommandLine(argc, argv);
  if (run()) {
    LOG(INFO) << "All layers match with no errors\n";
    return 0;
  } else {
    LOG(ERROR) << "Errors found!";
    return -1;
  }
}
