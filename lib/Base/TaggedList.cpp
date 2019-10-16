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

// The online list ordering algorithm is based on Bender et al., "Two Simplified
// Algorithms for Maintaining Order in a List."

#include "glow/Base/TaggedList.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cassert>

using namespace glow;
using namespace tagged_list_details;

// Fixed-point (0.32) factors for computing density limits at each tree level.
// The table is indexed by "size class" which is floor(log2(size)).
//
// Computed by this Python code:
//
//     for s in range(31):
//       f = 2 ** (s/31.0 - 1)
//       print(hex(int(f * 2**32)) + ',')
//
const uint32_t factorForSizeClass[32] = {
    0x80000000, 0x82e4ee78, 0x85da9dd7, 0x88e16f18, 0x8bf9c566, 0x8f24062b,
    0x9260991c, 0x95afe846, 0x9912601d, 0x9c886f86, 0xa01287ec, 0xa3b11d46,
    0xa764a62f, 0xab2d9bec, 0xaf0c7a83, 0xb301c0c6, 0xb70df068, 0xbb318e08,
    0xbf6d2145, 0xc3c134d0, 0xc82e567d, 0xccb51753, 0xd1560ba3, 0xd611cb16,
    0xdae8f0c7, 0xdfdc1b4d, 0xe4ebecdb, 0xea190b4a, 0xef642035, 0xf4cdd90f,
    0xfa56e732, 0xffffffff};

// Given two adjacent instruction with the same tag, renumber instructions
// such that the entire list is strictly increasing.
//
// It is assumed that the sub-list up to and including lo is monotonic, and so
// is the sub-list starting at hi.
void ListImpl::renumber(NodeBase *lo, NodeBase *hi) {
  assert(hi != &sentinel_);
  assert(lo->nodeTag_ == hi->nodeTag_);
  assert(lo->nextTaggedNode_ == hi);

  uint32_t count = 2; // maintained as count(lo..hi).
  NodeBase *insertionPoint = hi;

  // Start at tree level 0 which is the leaves.
  unsigned level = 0;
  // Inclusive tag range below the current tree node.
  uint32_t tagFloor = lo->nodeTag_;
  uint32_t tagCeil = lo->nodeTag_;
  // Maximum number of nodes at this level.
  uint32_t limit = 1;

  // 1.31 fixed-point factor to get the next level limit.
  unsigned sizeClass = llvm::Log2_32(size_);
  uint32_t factor = factorForSizeClass[sizeClass];

  // Increase the tree level until we're inside the limit.
  while (level < 32 && limit < count) {
    // Half the size of the next level;
    uint32_t half = UINT32_C(1) << level;

    // Advance level and limit.
    level++;
    limit = (uint64_t(2 * limit) * factor) >> 32;
    // Always round up. This is important at the very low levels.
    limit++;

    // The window doubles by moving either the floor or the ceiling.
    if (tagFloor & half) {
      // Move the floor down;
      tagFloor -= half;
      while (lo != &sentinel_ && lo->prevTaggedNode_->nodeTag_ >= tagFloor) {
        lo = lo->prevTaggedNode_;
        count++;
      }
    } else {
      // Move the ceiling up.
      tagCeil += half;
      while (hi->nextTaggedNode_ != &sentinel_ &&
             hi->nextTaggedNode_->nodeTag_ <= tagCeil) {
        hi = hi->nextTaggedNode_;
        count++;
      }
    }
  }

  // The window is now wide enough, so we can assign tags from the inclusive
  // range tagFloor..tagCeil to the instructions in the inclusive range
  // lo..hi.
  uint32_t dy = tagCeil - tagFloor;
  uint32_t dx = count - 1;

  // Make an even larger hold at the insertion point than is strictly
  // necessary. This anticipates future insertions near the same point.
  dx += std::min(dx / 8, dy - dx);
  uint32_t step = dy / dx;

  for (; lo != insertionPoint; lo = lo->nextTaggedNode_, tagFloor += step) {
    lo->nodeTag_ = tagFloor;
  }
  for (; hi != insertionPoint; hi = hi->prevTaggedNode_, tagCeil -= step) {
    hi->nodeTag_ = tagCeil;
  }
  hi->nodeTag_ = tagCeil;
}
