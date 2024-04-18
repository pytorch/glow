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
#ifndef GLOW_BASE_SLICERANGE_H
#define GLOW_BASE_SLICERANGE_H

#include "glow/Base/Type.h"
#include "glow/Graph/Graph.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace glow {

/// Dimension range representing a contiguous [start, stop) index interval with
/// start index included and stop index excluded, along some tensor dimension.
/// The indices are assumed to be 0 based (C indexing).
using DimRange = std::pair<dim_t, dim_t>;

/// Dimension paddings representing a virtual padding before/after a given
/// dimension range.
using DimPads = std::pair<dim_t, dim_t>;

/// Slice range utility class representing the ranges for all the dimensions
/// of a slice obtained by extraction from a larger tensor.
class SliceRange {

  /// Vector of ranges for all the dimensions of a slice.
  std::vector<DimRange> ranges_;

public:
  SliceRange() = default;

  /// Ctor.
  explicit SliceRange(std::vector<DimRange> ranges) { ranges_ = ranges; }

  /// Ctor.
  explicit SliceRange(TypeRef type) {
    for (auto size : type->dims()) {
      ranges_.emplace_back(0, size);
    }
  }

  /// Ctor.
  explicit SliceRange(const SliceNode *slice) {
    SliceRange sliceRange = SliceRange(slice->getResult().getType());
    auto start = slice->getStart();
    for (size_t dim = 0, dimEnd = start.size(); dim < dimEnd; ++dim) {
      sliceRange[dim].first += start[dim];
      sliceRange[dim].second += start[dim];
    }
    ranges_ = sliceRange.getRanges();
  }

  /// \returns the dimension ranges.
  llvm::ArrayRef<DimRange> getRanges() const { return ranges_; }

  /// \returns the start values of the dimension ranges.
  std::vector<dim_t> getStarts() const {
    std::vector<dim_t> starts(ranges_.size());
    for (size_t dim = 0, e = ranges_.size(); dim < e; ++dim) {
      starts[dim] = ranges_[dim].first;
    }
    return starts;
  }

  /// \returns the sizes of the dimension ranges.
  std::vector<dim_t> getSizes() const {
    std::vector<dim_t> sizes(ranges_.size());
    for (size_t dim = 0, e = ranges_.size(); dim < e; ++dim) {
      sizes[dim] = ranges_[dim].second - ranges_[dim].first;
    }
    return sizes;
  }

  /// \returns the number of dimensions.
  size_t getNumDims() const { return ranges_.size(); }

  /// \returns a mutable range for the given dimension \p dim.
  DimRange &operator[](size_t dim) {
    DCHECK_LT(dim, ranges_.size()) << "Invalid dimension!";
    return ranges_[dim];
  }

  /// \returns an immutable range for the given dimension \p dim.
  const DimRange &operator[](size_t dim) const {
    DCHECK_LT(dim, ranges_.size()) << "Invalid dimension!";
    return ranges_[dim];
  }

  /// \returns whether this slice range is equal to \p other.
  bool operator==(const SliceRange &other) const {
    auto rangesOther = other.getRanges();
    if (ranges_.size() != rangesOther.size()) {
      return false;
    }
    for (size_t dim = 0, e = ranges_.size(); dim < e; ++dim) {
      if (ranges_[dim] != rangesOther[dim]) {
        return false;
      }
    }
    return true;
  }

  /// \returns the range size along dimension \p dim.
  dim_t getDimSize(size_t dim) const {
    DCHECK_LT(dim, ranges_.size()) << "Invalid dimension!";
    return ranges_[dim].second - ranges_[dim].first;
  }

  /// \returns a slice range by extracting the dimension ranges between
  /// \p dimStart and \p dimStop (both included).
  SliceRange extractRanges(size_t dimStart, size_t dimStop) const {
    DCHECK_LT(dimStart, ranges_.size()) << "Invalid start dimension!";
    DCHECK_LT(dimStop, ranges_.size()) << "Invalid stop dimension!";
    DCHECK_LE(dimStart, dimStop) << "Invalid start/stop dimension!";
    std::vector<DimRange> dimRanges(ranges_.cbegin() + dimStart,
                                    ranges_.cbegin() + dimStop + 1);
    return SliceRange(dimRanges);
  }

  /// \returns a slice range by shuffling the dimension ranges using the
  /// indices \p shuffle. The flag \p invert allows optionally to invert
  /// the shuffle permutation before using it.
  SliceRange shuffleRanges(llvm::ArrayRef<size_t> shuffle,
                           bool invert = false) const {
    DCHECK_EQ(ranges_.size(), shuffle.size())
        << "Mismatch between ranges and shuffle sizes!";
    std::vector<DimRange> dimRanges(ranges_.size());
    for (size_t idx = 0, e = ranges_.size(); idx < e; ++idx) {
      size_t dimInp = invert ? idx : shuffle[idx];
      size_t dimOut = invert ? shuffle[idx] : idx;
      DCHECK_LT(dimInp, ranges_.size()) << "Invalid input shuffle index!";
      DCHECK_LT(dimOut, ranges_.size()) << "Invalid output shuffle index!";
      dimRanges[dimOut] = ranges_[dimInp];
    }
    return SliceRange(dimRanges);
  }

  /// \returns whether this slice range is empty.
  bool isEmpty() const {
    if (!ranges_.size()) {
      return true;
    }
    for (const auto &range : ranges_) {
      if (!(range.first < range.second)) {
        return true;
      }
    }
    return false;
  }

  /// \returns whether both ends of the range for a given dimension \p dim are
  /// aligned to \p align. For example the range [4, 8) is aligned to 4.
  bool isDimRangeAligned(size_t dim, dim_t align) const {
    DCHECK_LT(dim, ranges_.size()) << "Invalid dimension!";
    return (ranges_[dim].first % align == 0) &&
           (ranges_[dim].second % align == 0);
  }

  /// \returns whether this slice range is included by \p other.
  bool isIncludedBy(const SliceRange &other) const {
    auto rangesOther = other.getRanges();
    if (ranges_.size() != rangesOther.size()) {
      return false;
    }
    for (size_t dim = 0, e = ranges_.size(); dim < e; ++dim) {
      if (!((rangesOther[dim].first <= ranges_[dim].first) &&
            (ranges_[dim].second <= rangesOther[dim].second))) {
        return false;
      }
    }
    return true;
  }

  /// \returns a textual representation of this slice range.
  std::string toString() const {
    std::string storage;
    llvm::raw_string_ostream os(storage);
    for (size_t dim = 0, e = ranges_.size(); dim < e; ++dim) {
      os << getDimSize(dim) << "[" << ranges_[dim].first << ":"
         << ranges_[dim].second << ") ";
    }
    return os.str();
  }
};

} // namespace glow

#endif // GLOW_BASE_SLICERANGE_H
