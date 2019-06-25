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
#include "glow/Base/Image.h"
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Hook.h"
#include "glow/Importer/Caffe2ModelLoader.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Format.h"

using namespace glow;

const char inputName[] = "gpu_0/data";

class Tester {
  PlaceholderBindings bindings;
  ExecutionEngine EE;
  Module &mod;
  Function *F;
  TypeRef inputType;
  Placeholder *input;
  Placeholder *output;

public:
  Tester(llvm::StringRef backendName)
      : EE(backendName), mod(EE.getModule()), F(mod.createFunction("resnet50")),
        inputType(mod.uniqueType(ElemKind::FloatTy, {1, 3, 224, 224})) {
    // Load and compile ResNet-50.
    Caffe2ModelLoader loader("resnet50/predict_net.pb", "resnet50/init_net.pb",
                             {inputName}, {inputType}, *F);
    input = llvm::cast<Placeholder>(
        EXIT_ON_ERR(loader.getNodeValueByName(inputName)));
    output = EXIT_ON_ERR(loader.getSingleOutput());
  }

  void bindInput(Tensor *batch) {
    // Allocate memory for input and bind it to the placeholders.
    bindings.allocate(mod.getPlaceholders());
    updateInputPlaceholders(bindings, {input}, {batch});
  }

  TypeRef getInputType() const { return inputType; }

  Function *getFunction() const { return F; }

  Tensor *hookAndRun(llvm::StringRef name) {
    auto hook = hookOutput(F, name);
    auto *out = bindings.allocate(hook.output);
    EE.compile(CompilationMode::Infer, hook.function);
    EE.run(bindings);
    mod.eraseFunction(hook.function);
    return out;
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
    auto *interpOut = interp.hookAndRun(node.getName());
    auto *cpuOut = cpu.hookAndRun(node.getName());
    if (!interpOut->isEqual(*cpuOut)) {
      llvm::errs() << "Results differ\n";
      dumpImpl(interpOut);
      dumpImpl(cpuOut);
      auto IH = interpOut->getHandle<>();
      auto CH = cpuOut->getHandle<>();
      for (size_t i = 0, e = interpOut->size(); i < e; i++) {
        auto diff = std::abs(IH.raw(i) - CH.raw(i));
        if (diff > 0.0001) {
          llvm::errs() << llvm::format(
              "Index: %zu, Interp: %f, CPU: %f, diff: %f\n", i, IH.raw(i),
              CH.raw(i), diff);
        }
      }
    }
  }

  return 0;
}
