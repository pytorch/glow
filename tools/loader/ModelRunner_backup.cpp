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
#include "glow/Importer/Caffe2ModelLoader.h"
#include "glow/Importer/ONNXModelLoader.h"

#include "llvm/Support/raw_ostream.h"

#include <memory>

using namespace glow;

int main(int argc, char **argv) {
  parseCommandLine(argc, argv);
  Loader loader;
  Type inputType(ElemKind::FloatTy, {1, 2});
  PlaceholderBindings bindings{};

  std::unique_ptr<ProtobufLoader> LD;
  LD.reset(new ONNXModelLoader(loader.getOnnxModelFilename(), {"in"}, {&inputType},
                                *loader.getFunction()));

  bindings.allocate(loader.getModule()->getPlaceholders());
  loader.compile(bindings);

  /*** Initialize input. ***/
  Placeholder *inputPH = llvm::cast<glow::Placeholder>(EXIT_ON_ERR(LD->getNodeValueByName("in")));
  Tensor *inputT = bindings.get(inputPH);
  for (auto iter = inputT->getHandle().begin(); iter != inputT->getHandle().end(); ++iter)
    *iter = 0.0;

  loader.runInference(bindings);

  return 0;
}
