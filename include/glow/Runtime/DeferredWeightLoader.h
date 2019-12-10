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
#ifndef GLOW_RUNTIME_DEFERREDWEIGHTLOADER_H
#define GLOW_RUNTIME_DEFERREDWEIGHTLOADER_H

#include "glog/logging.h"

#include "glow/Base/Tensor.h"
#include "glow/Support/Error.h"
#include "glow/Support/Register.h"
#include "glow/Support/Support.h"

namespace glow {
namespace runtime {

/// A base class for deferred weight loaders. This allows for large weights to
/// be skipped during compilation and loaded after compilation one at a time.
class DeferredWeightLoader {
public:
  /// Loads the next weight, returns an Error indicating success/failure. Frees
  /// any resources used by the current deferred weight.
  virtual Error loadNextWeight() = 0;

  /// Accepts a void * \p loaderObject with is passed in from the interface
  /// library, e.g. deferredBlobReader in onnxifi.
  virtual Error setSrc(void *loaderObject) = 0;

  /// Accepts a map from string to Type \p info. This is used by the loader when
  /// converted the loaded weight into a glow Tensor.
  virtual void setTypeInfo(std::map<std::string, Type> info) = 0;

  /// Gets the name of the currently loaded weight.
  virtual std::string getName() = 0;

  /// Gets the Tensor for the current weight.
  virtual Tensor *getTensor() = 0;

  virtual ~DeferredWeightLoader() = default;
};

class DeferredWeightLoaderRegistry final {
public:
  void registerLoader(DeferredWeightLoader *loader);
  DeferredWeightLoader *getLoader();

private:
  DeferredWeightLoader *loader_{nullptr};
};

/// Global singleton.
DeferredWeightLoaderRegistry *DeferredLoader();

} // namespace runtime
} // namespace glow

#endif // GLOW_RUNTIME_DEFERREDWEIGHTLOADER_H
