// Copyright 2004-present Facebook. All Rights Reserved.
#ifndef GLOW_TORCH_GLOW_BACKEND_H
#define GLOW_TORCH_GLOW_BACKEND_H

#include "CachingGraphRunner.h"
#include <torch/csrc/jit/backends/backend_interface.h>

namespace glow {

// Glow backend implementation to PyTorch backend
class TorchGlowBackend : public torch::jit::PyTorchBackendInterface {
public:
  TorchGlowBackend() {}
  ~TorchGlowBackend() override {}

  c10::IValue preprocess(c10::IValue mod,
                         c10::impl::GenericDict method_compile_spec) override;

  c10::impl::GenericDict
  compile(c10::IValue processed,
          c10::impl::GenericDict method_compile_spec) override;

  c10::impl::GenericList execute(c10::IValue handle,
                                 c10::impl::GenericList inputs) override;

private:
  std::unordered_map<int64_t, std::unique_ptr<CachingGraphRunner>>
      handleToRunnerMap_;
};

} // namespace glow
#endif // GLOW_TORCH_GLOW_BACKEND_H
