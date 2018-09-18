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
#ifndef GLOW_BASE_CONTEXT_H
#define GLOW_BASE_CONTEXT_H

#include "llvm/ADT/ArrayRef.h"

#include <list>
#include <unordered_map>

namespace glow {

class Tensor;
class Placeholder;

/// This class provides a mapping between some graph nodes, which are a symbolic
/// representation of some computation, and concrete tensors that represent the
/// inputs and outputs to the graph. The context owns the tensors and the graph
/// uses these values as runtime. This is useful for the multi-threaded
/// execution of code, where each thread has a different execution context. The
/// difference between this class and a regular map is that the Context owns the
/// Tensors (not only the pointers) and manages their lifetime.
class Context final {
public:
  /// Maps placeholders to the tensors that back them.
  using PlaceholderMap = std::unordered_map<Placeholder *, Tensor *>;

private:
  /// Maps Placeholders to Tensors.
  PlaceholderMap map_;

public:
  /// \returns the tensor that corresponds to Placeholder \p P or Null if the
  /// tensor is not found.
  Tensor *get(Placeholder *P) const;

  /// Inserts the Placeholder-Tensor pair.
  void insert(Placeholder *P, Tensor &&T);

  /// Allocates a tensor to back the placeholder \p P. The new tensor has the
  /// type of P.
  Tensor *allocate(Placeholder *P);

  /// Allocates zero-initialized backing tensors to all placeholders in \p lst
  /// that are not currently allocated in the context.
  /// \returns the number of tensors that were allocated.
  unsigned allocate(std::list<Placeholder *> &lst);

  /// \returns the first placeholder in \p list that is not allocated by this
  /// context. This method returns null if all placeholders in the list are
  /// allocated.
  Placeholder *getFirstUnallocated(std::list<Placeholder *> &lst) const;

  /// \returns True if \p P is a registered Placeholder.
  size_t count(Placeholder *P) const;

  /// Deletes all tensors and clears the mapping between Placeholders and
  /// tensors.
  void clear();

  /// \returns the mapping between placeholder to tensors.
  const PlaceholderMap &pairs() const { return map_; }

  Context() = default;

  /// Construct the Context with an initial mapping between \p placeholders
  /// and \p inputs;
  Context(llvm::ArrayRef<Placeholder *> placeholders,
          llvm::ArrayRef<Tensor *> inputs);

  ~Context() { clear(); };

  // Don't copy or move this class around.
  Context(const Context &other) = delete;
  Context(Context &&other) = delete;
  Context &operator=(const Context &other) = delete;
  Context &operator=(Context &&other) = delete;
};

} // namespace glow

#endif // GLOW_BASE_CONTEXT_H
