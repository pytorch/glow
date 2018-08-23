/**
 * Copyright (c) 2017-present, Facebook, Inc.
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

#include "Loader.h"

#include "glow/Graph/Nodes.h"
#include "glow/Importer/Caffe2.h"
#include "glow/Importer/ONNX.h"

#include "llvm/Support/raw_ostream.h"

#include <memory>

using namespace glow;

int main(int argc, char **argv) {
  // The loader verifies/initializes command line parameters, and initializes
  // the ExecutionEngine and Function.
  Loader loader(argc, argv);

  // Create the model based on the input net, and get SaveNode for the output.
  std::unique_ptr<ProtobufLoader> LD;
  if (!loader.getCaffe2NetDescFilename().empty()) {
    LD.reset(new caffe2ModelLoader(loader.getCaffe2NetDescFilename(),
                                   loader.getCaffe2NetWeightFilename(), {}, {},
                                   *loader.getFunction()));
  } else {
    LD.reset(new ONNXModelLoader(loader.getOnnxModelFilename(), {}, {},
                                 *loader.getFunction()));
  }
  Variable *output = LD->getSingleOutput();

  // Compile the model, and perform quantization/emit a bundle/dump debug info
  // if requested from command line.
  loader.compile();

  // If in bundle mode, do not run inference.
  if (!emittingBundle()) {
    loader.runInference({}, {});

    llvm::outs() << "Model: " << loader.getFunction()->getName() << "\n";

    // Print out the result of output operator.
    output->getPayload().getHandle().dump();
  }

  return 0;
}
