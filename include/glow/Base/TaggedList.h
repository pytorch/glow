/**
 * Copyright (c) 2018-present, Facebook, Inc.
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
#ifndef GLOW_BASE_TAGGEDLIST_H
#define GLOW_BASE_TAGGEDLIST_H

#include <cinttypes>

namespace glow {

class TaggedListImpl;

/// Base class for nodes that can be inserted in a `TaggedList`.
class TaggedListNodeBase {
public:
  /// Is this the `end()` sentinel? All real nodes return false.
  bool isTaggedListSentinel() const { return nodeTag_ == 0; }

private:
  // Use long member names to avoid cluttering the namespace of sub-classes.
  TaggedListNodeBase *prevTaggedNode_ = nullptr;
  TaggedListNodeBase *nextTaggedNode_ = nullptr;
  uint32_t nodeTag_ = 0;

  friend class TaggedListImpl;
};

/// Template-free implementation of the `TaggedList`.
class TaggedListImpl {
  /// Insert `node` before `next`.
  void insert(TaggedListNodeBase *next, TaggedListNodeBase *node) {
    size_++;
    TaggedListNodeBase *prev = next->prevTaggedNode_;
    node->nextTaggedNode_ = next;
    node->prevTaggedNode_ = prev;
    next->prevTaggedNode_ = node;
    prev->nextTaggedNode_ = node;

    // Compute a tag half-way between prev and next.
    // Note that if `next` is the sentinel with tag 0, this will compute a tag
    // half-way between prev and UINT32_MAX. That is precisely what we want.
    uint32_t delta = (next->nodeTag_ - prev->nodeTag_) / 2;
    node->nodeTag_ = prev->nodeTag_ + delta;
    if (delta == 0)
      renumber(prev, node);
  }

  void remove(TaggedListNodeBase *node) {
    size_--;
    TaggedListNodeBase *prev = node->prevTaggedNode_;
    TaggedListNodeBase *next = node->nextTaggedNode_;
    prev->nextTaggedNode_ = next;
    next->prevTaggedNode_ = prev;
  }

private:
  void renumber(TaggedListNodeBase *lo, TaggedListNodeBase *hi);
  TaggedListNodeBase sentinel_;
  // Not a size_t because the list can't hold more than 2^32 nodes.
  uint32_t size_ = 0;
};

template <typename T> class TaggedList {
public:
  void insert();
  void remove();

private:
  TaggedListImpl impl_;
};

} // namespace glow
#endif // GLOW_BASE_TAGGEDLIST_H
