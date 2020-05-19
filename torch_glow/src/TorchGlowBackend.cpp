// Copyright 2004-present Facebook. All Rights Reserved.
#include "TorchGlowBackend.h"
#include "Registration.h"
#include <torch/csrc/jit/ir/node_hashing.h>
#include <torch/csrc/jit/passes/freeze_module.h>

namespace glow {

static c10::ScalarType scalarTypeFromString(const std::string str) {
  if (str == "float") {
    return c10::ScalarType::Float;
  } else {
    throw std::invalid_argument("Invalid type");
  }
}

static std::vector<glow::InputMeta>
parseMethodCompileSpec(const c10::ivalue::Tuple method_spec) {
  // method_spec format:
  // backend_name(string) , input#0(tuple) , input#1(tuple) ...
  // Where:
  // input#k := scalar_type, dim#0, dim#1 ....
  std::string glowBackend = method_spec.elements()[0].toStringRef();
  setHostManager(glowBackend);
  std::vector<glow::InputMeta> inputMeta;
  for (int i = 1; i < method_spec.elements().size(); ++i) {
    auto input_spec = method_spec.elements()[i].toTuple();
    c10::ScalarType st =
        scalarTypeFromString(input_spec->elements()[0].toStringRef());
    std::vector<glow::dim_t> dims;
    for (auto e = ++(input_spec->elements().begin());
         e != input_spec->elements().end(); e++) {
      dims.emplace_back(e->toInt());
    }
    inputMeta.emplace_back(st, std::move(dims));
  }

  return inputMeta;
}

c10::IValue
TorchGlowBackend::preprocess(c10::IValue mod,
                             c10::impl::GenericDict method_compile_spec) {
  torch::jit::Module m = mod.toModule();
  m.eval();
  return torch::jit::freeze_module(m)._ivalue();
}

c10::impl::GenericDict
TorchGlowBackend::compile(c10::IValue processed,
                          c10::impl::GenericDict method_compile_spec) {
  auto module = processed.toModule();

  auto handles = c10::Dict<std::string, int64_t>();

  // Compile each method
  int64_t key = 0;
  for (const auto &method : module.get_methods()) {
    auto g = method.function().graph();
    // Remove "self" input
    CHECK(g->block()->inputs()[0]->uses().empty())
        << "self must have no uses in order to lower to Glow.";
    g->block()->eraseInput(0);

    // Create a corresponding runner and store {handle, runner} pair.
    glow::getPyTorchLoaderSettings().preCompilePyTorchModule = true;
    std::unique_ptr<CachingGraphRunner> runner =
        std::make_unique<glow::CachingGraphRunner>(
            g, glow::getHostManager(), getBackendName().c_str(),
            glow::getPyTorchLoaderSettings());

    // Find and parse method_compile_spec
    c10::impl::GenericDict::iterator spec =
        method_compile_spec.find(method.name());
    CHECK(spec != method_compile_spec.end())
        << "Could not find corresponding method_compile_spec for method: "
        << method.name();
    c10::IValue methodSpec = spec->value();
    c10::intrusive_ptr<c10::ivalue::Tuple> tup;
    try {
      tup = methodSpec.toTuple();
    } catch (std::exception e) {
      throw std::invalid_argument(
          "method_copmile_spec does not match a tuple type.");
    }
    std::vector<glow::InputMeta> inputMeta = parseMethodCompileSpec(*tup);

    // Compile
    auto e = runner->warmCache(inputMeta);
    CHECK(!(bool)e) << ERR_TO_STRING(std::move(e));

    // Bakcend is created on each to_backend call --> use simple consecutive
    // keys for methods.
    handleToRunnerMap_.emplace(key, std::move(runner));
    handles.insert(method.name(), key++);
  }
  return c10::impl::toGenericDict(handles);
}

c10::impl::GenericList
TorchGlowBackend::execute(c10::IValue handle, c10::impl::GenericList inputs) {
  torch::jit::Stack stack;
  for (const auto &i : inputs) {
    torch::jit::push(stack, i);
  }
  auto it = handleToRunnerMap_.find(handle.toInt());
  Error err = glow::ErrorEmpty();
  if (it != handleToRunnerMap_.end()) {
    err = it->second->runOnly(stack);
  } else {
    throw std::out_of_range("Could not find runner for handle " +
                            std::to_string(handle.toInt()));
  }

  if (static_cast<bool>(err)) {
    throw std::invalid_argument(ERR_TO_STRING(std::move(err)));
  }

  c10::List<at::Tensor> output_list;
  while (stack.size() > 0) {
    auto value = torch::jit::pop(stack);
    output_list.emplace_back(value.toTensor());
  }
  return c10::impl::toList(output_list);
}

} // namespace glow
