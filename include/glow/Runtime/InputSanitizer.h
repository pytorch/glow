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
#ifndef GLOW_RUNTIME_INPUTSANITIZER_H
#define GLOW_RUNTIME_INPUTSANITIZER_H

#include <memory>
#include <vector>

#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/Graph/PlaceholderBindings.h"
#include "glow/Support/Error.h"

namespace glow {
namespace runtime {

/*
 * Base abstract class for input sanitizers.
 * Each operator type can have its own specialization.
 */
class InputSanitizer {
public:
  virtual ~InputSanitizer() = default;
  virtual Error sanitize(const PlaceholderBindings &bindings) = 0;
  virtual std::string toString() = 0;
};

using InputSanitizerPtr = std::shared_ptr<InputSanitizer>;

class SparseLengthsSumInputSanitizer : public InputSanitizer {
public:
  SparseLengthsSumInputSanitizer(const size_t tableHeight,
                                 Placeholder *indicesPH, Placeholder *weightsPH,
                                 Placeholder *lengthsPH);

  Error sanitize(const PlaceholderBindings &bindings) override;

  std::string toString() override;

private:
  size_t tableHeight_{0};
  Placeholder *indicesPH_{nullptr};
  Placeholder *weightsPH_{nullptr};
  Placeholder *lengthsPH_{nullptr};
};

class EmbeddingBagInputSanitizer : public InputSanitizer {
public:
  EmbeddingBagInputSanitizer(size_t tableHeight, Placeholder *indicesPH,
                             Placeholder *weightsPH, Placeholder *offsetsPH);

  Error sanitize(const PlaceholderBindings &bindings) override;

  std::string toString() override;

private:
  size_t tableHeight_{0};
  Placeholder *indicesPH_{nullptr};
  Placeholder *weightsPH_{nullptr};
  Placeholder *offsetsPH_{nullptr};
};

//
// Public utility functions
//
std::vector<InputSanitizerPtr> getInputSanitizers(const Function &function);
Error sanitizeInputs(const std::vector<InputSanitizerPtr> &sanitizers,
                     const PlaceholderBindings &bindings);

} // namespace runtime
} // namespace glow

#endif // GLOW_RUNTIME_INPUTSANITIZER_H
