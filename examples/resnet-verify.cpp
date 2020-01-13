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
#include "glow/Base/Image.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Hook.h"
#include "glow/Importer/Caffe2ModelLoader.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Format.h"

using namespace glow;

const char inputName[] = "gpu_0/data";

class Tester {
  PlaceholderBindings bindings, inferBindings;
  ExecutionEngine EEI;
  std::unique_ptr<Module> mod;
  Function *F;
  TypeRef inputType;
  Placeholder *input;
  Placeholder *output;

public:
  explicit Tester(llvm::StringRef backendName)
      : EEI(backendName), mod(new Module), F(mod->createFunction("resnet50")),
        inputType(mod->uniqueType(ElemKind::FloatTy, {1, 3, 224, 224})) {
    // Load and compile ResNet-50.
    Caffe2ModelLoader loader("resnet50/predict_net.pb", "resnet50/init_net.pb",
                             {inputName}, {inputType}, *F);
    input = llvm::cast<Placeholder>(
        EXIT_ON_ERR(loader.getNodeValueByName(inputName)));
    output = EXIT_ON_ERR(loader.getSingleOutput());
  }

  void bindInput(Tensor *batch) {
    // Allocate memory for input and bind it to the placeholders.
    bindings.allocate(mod->getPlaceholders());
    updateInputPlaceholders(bindings, {input}, {batch});
  }

  TypeRef getInputType() const { return inputType; }

  Function *getFunction() const { return F; }

  std::list<Tensor *> hookAndRun(llvm::StringRef name) {
    EEI.setBackendName(EEI.getBackendName());
    inferBindings.clear();
    auto modI = &EEI.getModule();
    auto *FI = modI->createFunction("resnet50");
    Caffe2ModelLoader loader(
        "resnet50/predict_net.pb", "resnet50/init_net.pb", {inputName},
        {mod->uniqueType(ElemKind::FloatTy, {1, 3, 224, 224})}, *FI);
    auto hook = hookNode(FI, name);
    inferBindings.allocate(modI->getPlaceholders());
    for (auto PH : bindings.pairs()) {
      auto iPH = inferBindings.getPlaceholderByName(PH.first->getName());
      inferBindings.get(iPH)->assign(PH.second);
    }

    std::list<Tensor *> outs;
    for (const auto &P : hook.outputs) {
      outs.emplace_back(inferBindings.get(P));
    }

    auto fName = hook.function->getName();
    EEI.compile(CompilationMode::Infer);
    EEI.run(inferBindings, fName);
    return outs;
  }
};

/// Compare layer-by-layer execution of ResNet on two backends.
int main() {
  Tester interp{"Interpreter"};
  Tester cpu{"CPU"};

  // Read an example PNG and add it to an input batch.
  auto image = readPngImageAndPreprocess(
      "tests/images/imagenet/cat_285.png", ImageNormalizationMode::k0to1,
      ImageChannelOrder::BGR, ImageLayout::NCHW, imagenetNormMean,
      imagenetNormStd);
  Tensor batch(interp.getInputType());
  batch.getHandle<float>().insertSlice(image, 0);

  interp.bindInput(&batch);
  cpu.bindInput(&batch);

  for (auto const &node : interp.getFunction()->getNodes()) {
    if (llvm::isa<SaveNode>(&node)) {
      continue;
    }
    llvm::errs() << "Verifying layer: " << node.getName() << "\n";
    auto interpOuts = interp.hookAndRun(node.getName());
    auto cpuOuts = cpu.hookAndRun(node.getName());

    if (interpOuts.size() == cpuOuts.size()) {
      auto interpOutIt = interpOuts.begin(), interpOutEnd = interpOuts.end();
      auto cpuOutIt = cpuOuts.begin(), cpuOutEnd = cpuOuts.end();

      while (interpOutIt != interpOutEnd && cpuOutIt != cpuOutEnd) {
        auto *interpOut = *interpOutIt;
        auto *cpuOut = *cpuOutIt;

        if (!interpOut->isEqual(*cpuOut)) {
          llvm::errs() << "Results differ\n";
          dumpImpl(interpOut);
          dumpImpl(cpuOut);
        }

        ++interpOutIt;
        ++cpuOutIt;
      }
    } else {
      llvm::errs()
          << "Backends produced different number of results using hook at "
          << node.getName() << "\n";
    }
  }

  return 0;
}
