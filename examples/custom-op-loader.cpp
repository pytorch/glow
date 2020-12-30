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

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/CustomOpData.h"
#include "glow/Graph/OpRepository.h"
#include "glow/Importer/ONNXModelLoader.h"
#include "glow/Support/Error.h"

using namespace glow;

int main() {
  glow::PlaceholderBindings bindings;
  glow::ExecutionEngine EE;
  auto &mod = EE.getModule();
  auto *F = mod.createFunction("mlp_custom");

  std::vector<ParamInfo> paramInfo;
  std::vector<NodeIOInfo> inputIOs;
  std::vector<NodeIOInfo> outputIOs;
  std::vector<ImplementationInfo> implInfo;

  ParamInfo alphaInfo("alpha", CustomOpDataType::DTFloat32,
                      true /* isScalar */);
  ParamInfo betaInfo("beta", CustomOpDataType::DTFloat32, true /* isScalar */);
  paramInfo.push_back(alphaInfo);
  paramInfo.push_back(betaInfo);

  // Populate parameter information for the node.
  inputIOs.push_back(NodeIOInfo("input", 1));
  outputIOs.push_back(NodeIOInfo("reluOut", 1));

  // funcLib should have the path to shared library with
  // registeration functions and implementation functions for interpreter.
  // File tests/unittests/CustomReluImpl.cpp can be compiled into
  // custom_relu.so.
  std::string funcLib = "custom_relu.so";
  implInfo.push_back(ImplementationInfo("Interpreter", "customReluExecute",
                                        (void *)(&funcLib)));
  // Register operation in OpRepository.
  OperationInfo reluOpInfo("CustomRelu", "OpDomain", paramInfo, inputIOs,
                           outputIOs, implInfo, funcLib);
  EXIT_ON_ERR(OpRepository::get()->registerOperation(reluOpInfo));

  // Load and compile LeNet MNIST model.
  glow::ONNXModelLoader loader("mlp_custom.onnx", {}, {}, *F);
  EE.compile(glow::CompilationMode::Infer);

  // Get input and output placeholders.
  auto *input = llvm::cast<glow::Placeholder>(
      EXIT_ON_ERR(loader.getNodeValueByName("x")));
  auto *output = EXIT_ON_ERR(loader.getSingleOutput());

  glow::Tensor batch(mod.uniqueType(glow::ElemKind::FloatTy, {3, 3}));
  auto handle = batch.getHandle<float>();
  handle.randomize(-5.0, 5.0, mod.getPRNG());

  // Allocate memory for input and bind it to the placeholders.
  bindings.allocate(mod.getPlaceholders());
  glow::updateInputPlaceholders(bindings, {input}, {&batch});

  // Perform inference.
  EE.run(bindings);

  // Read output and find argmax.
  auto out = bindings.get(output)->getHandle<float>();
  out.dump();

  return 0;
}
