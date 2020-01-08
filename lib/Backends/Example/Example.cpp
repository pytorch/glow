#include "glow/Backend/Backend.h"
#include "glow/Support/Error.h"

using namespace glow;

class ExampleBackend final : public Backend {
public:
  ExampleBackend() = default;
  ~ExampleBackend() override = default;
  std::string getBackendName() const override { return getName(); }
  static std::string getName() { return "Example"; }

  Expected<std::unique_ptr<CompiledFunction>>
  compile(Function *F, const BackendOptions &opts) const override;

  bool isOpSupported(const NodeInfo &NI) const override;
  bool shouldLower(const Node *N) const override;
};

class ExampleFunction final : public CompiledFunction {
public:
  ExampleFunction(runtime::RuntimeBundle &&bundle, Placeholder *data,
                  Placeholder *indices, Placeholder *lengths,
                  Placeholder *result)
      : CompiledFunction(std::move(bundle)), data_(data), indices_(indices),
        lengths_(lengths), result_(result) {}

  Error execute(ExecutionContext *context) override;
  std::string getCompileBackendName() const override {
    return ExampleBackend::getName();
  }

private:
  Placeholder *data_;
  Placeholder *indices_;
  Placeholder *lengths_;
  Placeholder *result_;
};

Expected<std::unique_ptr<CompiledFunction>>
ExampleBackend::compile(Function *F, const BackendOptions &opts) const {
  Placeholder *data;
  Placeholder *indices;
  Placeholder *lengths;
  Placeholder *result;
  for (auto &N : F->getNodes()) {
    if (auto *SLS = llvm::dyn_cast<SparseLengthsSumNode>(&N)) {
      data = llvm::cast<Placeholder>(SLS->getData());
      indices = llvm::cast<Placeholder>(SLS->getIndices());
      lengths = llvm::cast<Placeholder>(SLS->getLengths());
    } else if (auto *save = llvm::dyn_cast<SaveNode>(&N)) {
      result = llvm::cast<Placeholder>(save->getOutput());
    }
  }
  return llvm::make_unique<ExampleFunction>(runtime::RuntimeBundle::create(*F),
                                            data, indices, lengths, result);
}

bool ExampleBackend::isOpSupported(const NodeInfo &NI) const {
  switch (NI.getKind()) {
  case Kinded::Kind::SparseLengthsSumNodeKind:
  case Kinded::Kind::SaveNodeKind:
    return true;
  default:
    return false;
  }
}

bool ExampleBackend::shouldLower(const Node *N) const {
  return N->getKind() != Kinded::Kind::SparseLengthsSumNodeKind;
}

void sparseLengthsSum(Tensor *data, Tensor *indices, Tensor *lengths,
                      Tensor *out) {
  auto IH = indices->getHandle<sdim_t>();
  auto LH = lengths->getHandle<int32_t>();

  size_t segments = lengths->dims()[0];
  size_t totalLength = 0;
  for (size_t i = 0; i < segments; i++) {
    totalLength += LH.raw(i);
  }
  assert(totalLength <= indices->dims()[0] &&
         "sum(Lengths) must be equal to len(Indices)");

  size_t lineSize = data->size() / data->dims()[0];

  auto DH = data->getHandle<float>();
  auto OH = out->getHandle<float>();

  size_t curIdx = 0;
  for (size_t i = 0; i < segments; i++) {
    for (size_t j = 0, e = LH.raw(i); j < e; j++) {
      size_t offsetIn = IH.raw(curIdx++) * lineSize;
      size_t offsetOut = i * lineSize;
      for (size_t k = 0; k < lineSize; k++)
        OH.raw(offsetOut++) += DH.raw(offsetIn++);
    }
  }
}

Error ExampleFunction::execute(ExecutionContext *context) {
  auto *bindings = context->getPlaceholderBindings();
  auto *data = bindings->get(data_);
  auto *indices = bindings->get(indices_);
  auto *lengths = bindings->get(lengths_);
  auto *result = bindings->get(result_);
  sparseLengthsSum(data, indices, lengths, result);
  return Error::success();
}

REGISTER_GLOW_BACKEND_FACTORY(ExampleFactory, ExampleBackend);
