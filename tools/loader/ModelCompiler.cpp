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

#include "Loader.h"

#include "glow/Importer/Caffe2ModelLoader.h"
#include "glow/Importer/ONNXModelLoader.h"

using namespace glow;

int main(int argc, char **argv) {

  // Verify/initialize command line parameters, and then loader initializes
  // the ExecutionEngine and Function.
  parseCommandLine(argc, argv);

  // Initialize loader.
  Loader loader;

  // Emit bundle flag should be true.
  CHECK(emittingBundle())
      << "Bundle output directory not provided. Use the -emit-bundle option!";

  // Create the model based on the input model format.
  std::unique_ptr<ProtobufLoader> LD;
  if (!loader.getCaffe2NetDescFilename().empty()) {
    // For Caffe2 format the input placeholder names/types must be provided
    // explicitly. Get model input names and types.
    std::vector<std::string> inputNames;
    std::vector<Type> inputTypes;
    Loader::getModelInputs(inputNames, inputTypes);
    std::vector<const char *> inputNameRefs;
    std::vector<TypeRef> inputTypeRefs;
    for (size_t idx = 0, e = inputNames.size(); idx < e; idx++) {
      inputNameRefs.push_back(inputNames[idx].c_str());
      inputTypeRefs.push_back(&inputTypes[idx]);
    }
    LD.reset(new Caffe2ModelLoader(
        loader.getCaffe2NetDescFilename(), loader.getCaffe2NetWeightFilename(),
        inputNameRefs, inputTypeRefs, *loader.getFunction()));
  } else {
    // For ONNX format the input placeholders names/types are
    // derived automatically.
    LD.reset(new ONNXModelLoader(loader.getOnnxModelFilename(), {}, {},
                                 *loader.getFunction()));
  }

  // Compile the model and generate the bundle.
  CompilationContext ctx;
  loader.compile(ctx);

  return 0;
}
