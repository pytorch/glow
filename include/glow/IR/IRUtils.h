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
#ifndef GLOW_IR_IRUTILS_H
#define GLOW_IR_IRUTILS_H

#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"

#include <iterator>

namespace glow {

/// \returns true if the value \v is a tensor view.
bool isTensorView(Value *v);

/// \returns the offset for \p TVI into the underlying alloc activation.
size_t calculateTensorViewOffset(const TensorViewInst *TVI);

/// A helper class to iterate over all uses of a given Value.
/// It also recursively iterates over uses of any tensorview
/// instructions aliasing this value.
/// Iteration is performed by means of const iterators.
class ValueUses {
  /// The value whose uses are to be iterated over.
  const Value *v_;

public:
  /// The actual iterator implementation.
  template <bool is_const_iter = true>
  class ValueUsesIterator
      : public std::iterator<
            std::forward_iterator_tag,
            typename std::conditional<is_const_iter, const Use, Use>::type> {
    friend ValueUses;
    using BASE = std::iterator<
        std::forward_iterator_tag,
        typename std::conditional<is_const_iter, const Use, Use>::type>;
    using reference = typename BASE::reference;
    using UseList = std::list<Use>;
    using value =
        typename std::conditional<is_const_iter, const Value, Value>::type;
    using iterator =
        typename std::conditional<is_const_iter, UseList::const_iterator,
                                  UseList::iterator>::type;

  private:
    /// Set of values to iterate over. It includes the original value and
    /// eventually any tensor views directly or indirectly based on it.
    llvm::SmallVector<value *, 4> vals_;
    /// Current iterator.
    iterator it_;
    /// End of the use-list for the current value.
    iterator end_;
    /// Index of the value whose use-list is being iterated.
    int idx_;

    /// This constructor is used to create the begin iterator.
    ValueUsesIterator(value *v)
        : vals_{v}, it_(v->getUsers().begin()), end_(v->getUsers().end()),
          idx_(v->getUsers().begin() == v->getUsers().end() ? -1 : 0) {}

    /// This constructor is used to create the end iterator.
    ValueUsesIterator(value *v, int)
        : vals_{v}, it_(v->getUsers().end()), end_(v->getUsers().end()),
          idx_(-1) {}

  public:
    ValueUsesIterator &operator++() {
      if (it_ == end_) {
        llvm_unreachable("Cannot increment the end iterator");
      }
      ++it_;
      if (it_ != end_)
        return *this;
      for (; it_ == end_;) {
        // Reached the end of uses for the current value.
        // Try to iterate over another value if available.
        if (++idx_ >= (int)vals_.size()) {
          // Form the end iterator.
          *this = ValueUsesIterator{vals_[0], 1};
          return *this;
        }
        it_ = vals_[idx_]->getUsers().begin();
        end_ = vals_[idx_]->getUsers().end();
      }
      return *this;
    }

    reference operator*() {
      if (it_ == end_) {
        llvm_unreachable("An attempt to dereference the end iterator");
      }
      // If it is a tensorview instruction, add it is the set of
      // values to be procssed, because all uses of a tensorview
      // are considered to be uses of the tensorview's original
      // allocation.
      if (isTensorView(it_->get()) && vals_.back() != it_->get()) {
        vals_.push_back(it_->get());
      }
      return *it_;
    }

    bool operator!=(const ValueUsesIterator &Other) const {
      return idx_ != Other.idx_ || it_ != Other.it_ || end_ != Other.end_ ||
             vals_ != Other.vals_;
    }
  };

  using const_iterator = ValueUsesIterator<true>;

  ValueUses(const Value *v) : v_(v) {}
  ValueUses(ValueUses &VU) = default;
  const_iterator begin() const { return const_iterator{v_}; }
  const_iterator end() const { return const_iterator{v_, 1}; }
};

/// Get the allocation corrsponding to th value \p V. It can look through
/// tensorview instructions. \returns found allocation or nullptr.
Value *getAllocationOrigin(Value *V);

/// \returns peels off the layers of tensorviews from a value \p V.
Value *getOrigin(Value *V);

/// \returns the offset into the Value returned by getOrigin.
size_t getOriginOffset(Value *V);

/// \returns peels off the layers of tensorviews from a value \p V.
const Value *getOrigin(const Value *V);
} // namespace glow

#endif // GLOW_IR_IRUTILS_H
