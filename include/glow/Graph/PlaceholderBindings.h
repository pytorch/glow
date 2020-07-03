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
#ifndef GLOW_BASE_PLACEHOLDERBINDINGS_H
#define GLOW_BASE_PLACEHOLDERBINDINGS_H

#include "glow/ExecutionContext/TraceEvents.h"
#include "glow/Graph/Graph.h"
#include "llvm/ADT/ArrayRef.h"

#include <list>
#include <unordered_map>

namespace glow {

class Tensor;
class Placeholder;

/// This class provides a mapping between some graph nodes, which are a symbolic
/// representation of some computation, and concrete tensors that represent the
/// inputs and outputs to the graph. The PlaceholderBindings owns the tensors
/// and the graph uses these values as runtime. This is useful for the
/// multi-threaded execution of code, where each thread has a different
/// execution context. The difference between this class and a regular map is
/// that the PlaceholderBindings owns the Tensors (not only the pointers) and
/// manages their lifetime.
class PlaceholderBindings final {
public:
  /// Maps placeholders to the tensors that back them.
  using PlaceholderMap = std::unordered_map<Placeholder *, Tensor>;
  using PlaceholderMapIterator = PlaceholderMap::iterator;

private:
  /// Maps Placeholders to Tensors.
  PlaceholderMap map_;

public:
  /// \returns true if \p A and \p B contain the same Placeholders mapped to
  /// equivalent Tensors. \p allowedError is used when comparing each
  /// Placeholder's backing payload data.
  static bool compare(const PlaceholderBindings *A,
                      const PlaceholderBindings *B,
                      float allowedError = 0.0001);

  /// \returns the tensor that corresponds to Placeholder \p P or Null if the
  /// tensor is not found.
  Tensor *get(Placeholder *P);
  const Tensor *get(Placeholder *P) const;

  /// \returns the Placeholder named \name or null of the Placeholder is not
  /// found. Note that this uses a linear search path. If you want to seatch by
  /// name more quickly, consider building a map yourself.
  Placeholder *getPlaceholderByNameSlow(llvm::StringRef name) const;

  /// Inserts the Placeholder-Tensor pair. This takes ownership of the Tensor.
  PlaceholderMapIterator insert(Placeholder *P, Tensor &&T);

  /// Copy values from this PlaceholderBindings to another, \p dst, by \p name.
  /// This is useful when trained weights need to be transferred between
  /// bindings of two different modules.
  void copyToTarget(llvm::StringRef name, PlaceholderBindings &dst);

  /// Transfer all trainable weights to target PlaceholderBindings \p dst.
  void copyTrainableWeightsTo(PlaceholderBindings &dst);

  /// Allocates a tensor to back the placeholder \p P. The new tensor has the
  /// type of P.
  Tensor *allocate(Placeholder *P);

  /// Allocates zero-initialized backing tensors to all placeholders in \p lst
  /// that are not currently allocated in the bindings.
  /// \returns the number of tensors that were allocated.
  unsigned allocate(const std::list<Placeholder *> &lst);

  /// \returns the first placeholder in \p list that is not allocated by this
  /// bindings. This method returns null if all placeholders in the list are
  /// allocated.
  Placeholder *getFirstUnallocated(const std::list<Placeholder *> &lst) const;

  /// \returns True if \p P is a registered Placeholder.
  size_t count(Placeholder *P) const;

  /// Deletes all tensors and clears the mapping between Placeholders and
  /// tensors.
  void clear();

  /// Removes the Tensor backing Placeholder \p P;
  /// \p P must be a valid Placeholder registered in the bindings.
  void erase(Placeholder *P);

  /// Removes the existing Tensor backing Placeholder \p P; Bind \p T to \P.
  /// \p P must be a valid Placeholder registered in the bindings.
  void update(Placeholder *P, Tensor &&T);

  /// \returns a copy of the PlaceholderBindings, with each placeholder mapped
  /// to a new Tensor, with their own memory.
  PlaceholderBindings clone() const;

  /// \returns a copy of the PlaceholderBindings, with each placeholder mapped
  /// to a new Tensor, with their own memory. However instead of the current
  /// Placeholders in the current mapping, use the Placeholder with the same
  /// name found in \p newPHs.
  PlaceholderBindings clone(const PlaceholderList &newPHs) const;

  /// \returns the mapping between placeholder to tensors.
  PlaceholderMap &pairs() { return map_; }
  const PlaceholderMap &pairs() const { return map_; }

  /// \returns the size in bytes of allocated Tensors owned by
  /// PlaceholderBindings.
  uint64_t getDataSize() const;

  /// Copies all Device Resident Tensors back to the host.
  void ensureOnHost() {
    for (auto &ph : map_) {
      ph.second.ensureOnHost();
    }
  }

  PlaceholderBindings() = default;

  /// Construct the PlaceholderBindings with an initial mapping between \p
  /// placeholders and \p inputs;
  PlaceholderBindings(llvm::ArrayRef<Placeholder *> placeholders,
                      llvm::ArrayRef<Tensor *> inputs);

  PlaceholderBindings(PlaceholderBindings &&other)
      : map_(std::move(other.map_)) {}

  ~PlaceholderBindings() { clear(); };

  // Don't copy this class around.
  PlaceholderBindings(const PlaceholderBindings &other) = delete;
  PlaceholderBindings &operator=(const PlaceholderBindings &other) = delete;
};

} // namespace glow

#endif // GLOW_BASE_PLACEHOLDERBINDINGS_H
