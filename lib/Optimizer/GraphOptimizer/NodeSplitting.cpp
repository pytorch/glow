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

#include "glow/Optimizer/GraphOptimizer/NodeSplitting.h"

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace glow;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

namespace {
///===---------------------------------------------------------------------===//
///                                SliceRange
///===---------------------------------------------------------------------===//
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
} // namespace

///===---------------------------------------------------------------------===//
///                              SplitNodeOption
///===---------------------------------------------------------------------===//
size_t SplitNodeOption::getSplitDimIdx(dim_t splitDim) const {
  auto splitDimsIt = std::find(splitDims_.begin(), splitDims_.end(), splitDim);
  CHECK(splitDimsIt != splitDims_.end())
      << "Split dimension '" << splitDim
      << "' invalid! Not registered in this SplitNodeOption!";
  return std::distance(splitDims_.begin(), splitDimsIt);
}

std::vector<dim_t> SplitNodeByNumChunks::splitAlongDim(size_t dim,
                                                       dim_t dimSize) const {
  size_t dimIdx = getSplitDimIdx(dim);
  dim_t numChunks = numChunks_[dimIdx];
  CHECK((1 <= numChunks) && (numChunks <= dimSize))
      << "SplitNodeByNumChunks: Invalid number of chunks '" << numChunks
      << "' for splitting a dimension with size '" << dimSize << "'!";

  dim_t chunkDiv = dimSize / numChunks;
  dim_t chunkRem = dimSize % numChunks;

  // Small and big chunk sizes.
  dim_t numChunksBig = chunkRem;
  dim_t chunkSizeSmall = chunkDiv;
  dim_t chunkSizeBig = chunkDiv + 1;

  // Split dimension.
  std::vector<dim_t> chunkSizes(numChunks);
  for (size_t idx = 0, end = numChunks; idx < end; ++idx) {
    dim_t chunkSize = chunkSizeSmall;
    if (bigChunksFirst_ && (idx < numChunksBig)) {
      chunkSize = chunkSizeBig;
    }
    if ((!bigChunksFirst_ && (idx >= numChunks - numChunksBig))) {
      chunkSize = chunkSizeBig;
    }
    chunkSizes[idx] = chunkSize;
  }
  return chunkSizes;
}

std::vector<dim_t> SplitNodeByChunkSize::splitAlongDim(size_t dim,
                                                       dim_t dimSize) const {
  size_t dimIdx = getSplitDimIdx(dim);
  dim_t chunkSize = chunkSizes_[dimIdx];
  CHECK((1 <= chunkSize) && (chunkSize <= dimSize))
      << "SplitNodeByChunkSize: Invalid chunk size '" << chunkSize
      << "' for splitting a dimension with size '" << dimSize << "'!";

  dim_t chunkDiv = dimSize / chunkSize;
  dim_t chunkRem = dimSize % chunkSize;

  // Small and big chunk sizes.
  dim_t numChunks = chunkRem > 0 ? chunkDiv + 1 : chunkDiv;
  dim_t chunkSizeSmall = chunkRem > 0 ? chunkRem : chunkSize;
  dim_t chunkSizeBig = chunkSize;

  // Split dimension.
  std::vector<dim_t> chunkSizes(numChunks);
  for (size_t idx = 0, end = numChunks; idx < end; ++idx) {
    dim_t chunkSizeFinal = chunkSizeBig;
    if (bigChunksFirst_ && (idx == numChunks - 1)) {
      chunkSizeFinal = chunkSizeSmall;
    }
    if (!bigChunksFirst_ && (idx == 0)) {
      chunkSizeFinal = chunkSizeSmall;
    }
    chunkSizes[idx] = chunkSizeFinal;
  }
  return chunkSizes;
}

std::vector<dim_t> SplitNodeByChunkSizes::splitAlongDim(size_t dim,
                                                        dim_t dimSize) const {
  size_t dimIdx = getSplitDimIdx(dim);
  std::vector<dim_t> chunkSizes = chunkSizes_[dimIdx];
  size_t numChunks = chunkSizes.size();
  CHECK((1 <= numChunks) && (numChunks <= dimSize))
      << "SplitNodeByChunkSizes: Invalid number of sizes '" << numChunks
      << "' for splitting a dimension with size '" << dimSize << "'!";
  for (const auto &chunkSize : chunkSizes) {
    CHECK_GT(chunkSize, 0)
        << "SplitNodeByChunkSizes: Chunk size 0 is not allowed!";
  }
  return chunkSizes;
}

std::vector<dim_t> SplitNodeByChunkWeights::splitAlongDim(size_t dim,
                                                          dim_t dimSize) const {
  size_t dimIdx = getSplitDimIdx(dim);
  const std::vector<float> &chunkWeights = chunkWeights_[dimIdx];
  dim_t numChunks = chunkWeights.size();
  CHECK((1 <= numChunks) && (numChunks <= dimSize))
      << "SplitNodeByChunkWeights: Invalid number of weights '" << numChunks
      << "' for splitting a dimension with size '" << dimSize << "'!";

  // Verify that all the weights are positive and compute the weights sum.
  float chunkWeightsSum = 0;
  for (const auto &weight : chunkWeights) {
    CHECK_GT(weight, 0.f) << "SplitNodeByChunkWeights: Chunk weight '" << weight
                          << "' invalid! Should be strictly positive!";
    chunkWeightsSum += weight;
  }

  // Compute individual chunk sizes such that each chunk gets at least one unit
  // (empty chunks are NOT allowed). The total number of units is distributed
  // such that the error between the given chunk weights and the actual chunk
  // weights is minimized.
  std::vector<dim_t> chunkSizes(numChunks, 1);
  dim_t unitsRem = dimSize - numChunks;
  while (unitsRem > 0) {
    // Find chunk with maximum weight error.
    float weightErrMax = std::numeric_limits<float>::lowest();
    size_t weightErrMaxIdx = 0;
    for (size_t idx = 0; idx < numChunks; ++idx) {
      float weightVal =
          float(chunkSizes[idx]) / float(dimSize) * chunkWeightsSum;
      // We use a signed error here to starve those chunks for which the actual
      // weight surpassed the given weight.
      float weightErr = chunkWeights[idx] - weightVal;
      if (weightErr > weightErrMax) {
        weightErrMax = weightErr;
        weightErrMaxIdx = idx;
      }
    }
    // Distribute unit.
    chunkSizes[weightErrMaxIdx] += 1;
    unitsRem--;
  }
  return chunkSizes;
}

/// Utility function to split an array of slice ranges \p ranges along the given
/// dimension \p dim using the split option \p splitOption.
static std::vector<SliceRange>
splitSliceRanges(const std::vector<SliceRange> &ranges, size_t dim,
                 const SplitNodeOption *splitOption) {
  std::vector<SliceRange> outRanges;
  for (const auto &range : ranges) {

    // Split dimension.
    dim_t dimSize = range.getDimSize(dim);
    std::vector<dim_t> chunkSizes = splitOption->splitAlongDim(dim, dimSize);

    // Check for empty chunks.
    for (auto chunkSize : chunkSizes) {
      CHECK_GT(chunkSize, 0) << "Chunk size 0 is not allowed!";
    }

    // Check dimension splitting consistency.
    dim_t chunkSizesSum =
        std::accumulate(chunkSizes.begin(), chunkSizes.end(), (dim_t)0);
    CHECK_EQ(dimSize, chunkSizesSum)
        << "Inconsistent splitting of dimension " << dim << " with size "
        << dimSize << " into chunks with total size " << chunkSizesSum << "!";

    // Split current slice range.
    auto numChunks = chunkSizes.size();
    std::vector<SliceRange> splitRanges(numChunks, range);
    dim_t chunkStart = range[dim].first;
    for (size_t idx = 0; idx < numChunks; ++idx) {
      // Current chunk size.
      dim_t chunkSize = chunkSizes[idx];
      // Update chunk bounds.
      splitRanges[idx][dim].first = chunkStart;
      splitRanges[idx][dim].second = chunkStart + chunkSize;
      chunkStart += chunkSize;
    }
    CHECK_EQ(splitRanges.back()[dim].second, range[dim].second)
        << "Inconsistent splitting of SliceRange!";

    // Append split slice ranges.
    outRanges.insert(outRanges.end(), splitRanges.begin(), splitRanges.end());
  }
  return outRanges;
}

///===---------------------------------------------------------------------===//
///                            CheckedSliceRangeMap
///===---------------------------------------------------------------------===//
/// Definition of a checked slice range which provides an extra boolean flag to
/// inform whether the slice range is valid and allowed to be used.
using CheckedSliceRange = std::pair<bool, SliceRange>;

/// Definition of a functional mapping between two slice ranges with extra
/// information about whether the mapping is allowed to be used (valid).
using CheckedSliceRangeMap =
    std::function<CheckedSliceRange(const SliceRange &)>;

/// Identity checked slice range map to use for simple identity mappings.
CheckedSliceRange CheckedSliceRangeMapIdentity(const SliceRange &range) {
  return {true, range};
}

/// Definition of a pair with an operand index and a checked slice range map.
using OpIdxAndMap = std::pair<unsigned, CheckedSliceRangeMap>;

/// Utility function to verify that a given slice range \p map represents an
/// exact mapping from \p mapInputRanges to \p mapOutputRanges.
static bool isMappingExact(const CheckedSliceRangeMap &map,
                           const std::vector<SliceRange> &mapInputRanges,
                           const std::vector<SliceRange> &mapOutputRanges) {
  bool mapOk = true;
  DCHECK_EQ(mapInputRanges.size(), mapOutputRanges.size())
      << "Slice ranges length mismatch for CheckedSliceRangeMap verification!";
  for (size_t idx = 0, e = mapInputRanges.size(); idx < e; ++idx) {
    auto checkedSliceRange = map(mapInputRanges[idx]);
    mapOk = mapOk && checkedSliceRange.first;
    mapOk = mapOk && (checkedSliceRange.second == mapOutputRanges[idx]);
  }
  return mapOk;
}

/// Utility function to verify that a given slice range \p map when applied to
/// \p mapInputRanges produces ranges which are included in \p mapOutputRanges.
/// This is a weaker verification than \ref isMappingExact.
static bool isMappingIncluded(const CheckedSliceRangeMap &map,
                              const std::vector<SliceRange> &mapInputRanges,
                              const std::vector<SliceRange> &mapOutputRanges) {
  bool mapOk = true;
  DCHECK_EQ(mapInputRanges.size(), mapOutputRanges.size())
      << "Slice ranges length mismatch for CheckedSliceRangeMap verification!";
  for (size_t idx = 0, e = mapInputRanges.size(); idx < e; ++idx) {
    auto checkedSliceRange = map(mapInputRanges[idx]);
    mapOk = mapOk && checkedSliceRange.first;
    mapOk =
        mapOk && checkedSliceRange.second.isIncludedBy(mapOutputRanges[idx]);
  }
  return mapOk;
}

///===---------------------------------------------------------------------===//
///                           SplitNodeModifier
///===---------------------------------------------------------------------===//
/// Definition of a function which modifies the split node \p splitNode after it
/// was cloned from the original node \p origNode. The input slice ranges \p
/// inputSliceRanges and the output slice ranges \p outputSliceRanges are also
/// provided by the caller to provide extra context about how the split node was
/// obtained from the original node. This function is provided to the node
/// splitting procedure as a callback and provides the mechanism of modifying
/// the split node attributes in special situations, for example when the "Pads"
/// or "Group" node attributes must be changed when splitting Convolution nodes.
using SplitNodeModifier =
    std::function<void(const Node *origNode, Node *splitNode,
                       const std::vector<SliceRange> &inputSliceRanges,
                       const std::vector<SliceRange> &outputSliceRanges)>;

/// Definition of a "nop" split node modifier which does no modifications.
void SplitNodeModifierNop(const Node *origNode, Node *splitNode,
                          const std::vector<SliceRange> &inputSliceRanges,
                          const std::vector<SliceRange> &outputSliceRanges) {}

///===---------------------------------------------------------------------===//
///                            verifySplitParams
///===---------------------------------------------------------------------===//
/// List of nodes for which there is a weak mapping between input and output
/// and thus a weaker verification must be performed. Such an example is the
/// Conv2D/MaxPool node when using strides larger than 1 resulting in cases
/// where the output operand does not reference the input operand entirely.
static std::vector<Kinded::Kind> weakOutToInMappingNodeKinds = {
    Kinded::Kind::ConvolutionNodeKind,
    Kinded::Kind::MaxPoolNodeKind,
    Kinded::Kind::AvgPoolNodeKind,
};

/// Function to verify the split parameters.
static Error
verifySplitParams(const Node *node, dim_t splitOutputIdx,
                  const llvm::ArrayRef<size_t> &splitDims,
                  const llvm::ArrayRef<OpIdxAndMap> &inputIdxAndMaps,
                  const llvm::ArrayRef<OpIdxAndMap> &outputIdxAndMaps) {

  // Verify original node.
  if (!node->verify()) {
    llvm::errs() << node->toString() << "\n";
    return MAKE_ERR("Invalid node given to node splitting procedure!");
  }

  // Verify split dims.
  RETURN_ERR_IF_NOT(splitDims.size() > 0,
                    "Empty split dimensions for splitting node!");
  RETURN_ERR_IF_NOT(splitOutputIdx < node->getNumResults(),
                    "Invalid output index for splitting node!");
  for (size_t dim = 0; dim < splitDims.size() - 1; ++dim) {
    RETURN_ERR_IF_NOT(splitDims[dim] < splitDims[dim + 1],
                      "Invalid split dimensions for splitting node! Dimensions "
                      "should be given in ascending order e.g. {0,2,3}!");
  }
  for (const auto dim : splitDims) {
    RETURN_ERR_IF_NOT(dim < node->getType(splitOutputIdx)->dims().size(),
                      "Invalid split dimension for splitting node! Dimension "
                      "exceeds the split output tensor shape!");
  }

  // Verify all the input indices and maps were given.
  RETURN_ERR_IF_NOT(inputIdxAndMaps.size() == node->getNumInputs(),
                    "Invalid number of input maps for splitting node!");
  std::vector<bool> inputIdxMask(node->getNumInputs(), false);
  for (const auto &inputIdxMap : inputIdxAndMaps) {
    RETURN_ERR_IF_NOT(inputIdxMap.first < node->getNumInputs(),
                      "Invalid input index for input range map!");
    inputIdxMask[inputIdxMap.first] = true;
  }
  RETURN_ERR_IF_NOT(
      std::find(inputIdxMask.begin(), inputIdxMask.end(), false) ==
          inputIdxMask.end(),
      "Not all input indices and maps were provided for splitting node!");

  // Verify all the output indices and maps were given.
  RETURN_ERR_IF_NOT(outputIdxAndMaps.size() == node->getNumResults() - 1,
                    "Invalid number of output maps for splitting node!");
  std::vector<bool> outputIdxMask(node->getNumResults(), false);
  outputIdxMask[splitOutputIdx] = true;
  for (const auto &outputIdxMap : outputIdxAndMaps) {
    RETURN_ERR_IF_NOT(outputIdxMap.first < node->getNumResults(),
                      "Invalid output index for output range map!");
    outputIdxMask[outputIdxMap.first] = true;
  }
  RETURN_ERR_IF_NOT(
      std::find(outputIdxMask.begin(), outputIdxMask.end(), false) ==
          outputIdxMask.end(),
      "Not all output indices and maps were provided for splitting node!");

  // Get split output range.
  SliceRange splitOutputRange = SliceRange(node->getType(splitOutputIdx));

  // Verify the input slice range maps.
  for (const auto &inputIdxMap : inputIdxAndMaps) {
    SliceRange inputRange =
        SliceRange(node->getNthInput(inputIdxMap.first).getType());
    if (std::find(weakOutToInMappingNodeKinds.begin(),
                  weakOutToInMappingNodeKinds.end(),
                  node->getKind()) != weakOutToInMappingNodeKinds.end()) {
      // Verify weak mapping.
      RETURN_ERR_IF_NOT(isMappingIncluded(inputIdxMap.second,
                                          {splitOutputRange}, {inputRange}),
                        "Invalid input range map for splitting node!");
    } else {
      // Verify exact mapping.
      RETURN_ERR_IF_NOT(
          isMappingExact(inputIdxMap.second, {splitOutputRange}, {inputRange}),
          "Invalid input range map for splitting node!");
    }
  }

  // Verify the output slice range maps.
  for (const auto &outputIdxMap : outputIdxAndMaps) {
    SliceRange outputRange = SliceRange(node->getType(outputIdxMap.first));
    RETURN_ERR_IF_NOT(
        isMappingExact(outputIdxMap.second, {splitOutputRange}, {outputRange}),
        "Invalid output range map for splitting node!");
  }

  return Error::success();
}

///===---------------------------------------------------------------------===//
///                            verifySplitNodes
///===---------------------------------------------------------------------===//
/// Function to verify the split nodes.
static Expected<bool>
verifySplitNodes(const Node *node, dim_t splitOutputIdx,
                 const llvm::ArrayRef<SliceRange> &splitOutputSlices,
                 const llvm::ArrayRef<OpIdxAndMap> &inputIdxAndMaps,
                 const llvm::ArrayRef<OpIdxAndMap> &outputIdxAndMaps,
                 const SplitNodeConstraint *splitConstraint,
                 const SplitNodeModifier &splitNodeModifier) {

  // Create temporary nodes to make verifications without adding them to
  // the graph in order to avoid the pollution of the graph with nodes
  // which could be invalid or could not meet all the constraints and
  // hence be later removed from the graph.
  bool splitNodesCheck = true;
  std::vector<Node *> splitNodes;
  std::list<std::unique_ptr<SliceNode>> inputSliceNodes;
  std::list<Type> inputTypes;
  std::list<Type> outputTypes;
  for (const auto &splitOutputSlice : splitOutputSlices) {

    // Create clone to inherit all the inputs/members of the original node.
    Node *clone = node->clone();
    splitNodes.push_back(clone);

    // Detach clone from all the inputs of the original node.
    for (unsigned idx = 0, e = clone->getNumInputs(); idx < e; ++idx) {
      clone->setNthInput(idx, nullptr);
    }

    // Gather input slice ranges for the clone. The ranges are ordered
    // according to the input operand indices.
    std::vector<SliceRange> inputRanges(clone->getNumInputs());
    for (const auto &inputIdxMap : inputIdxAndMaps) {
      auto inputCheckedRange = inputIdxMap.second(splitOutputSlice);
      splitNodesCheck = splitNodesCheck && inputCheckedRange.first;
      splitNodesCheck = splitNodesCheck && !inputCheckedRange.second.isEmpty();
      inputRanges[inputIdxMap.first] = inputCheckedRange.second;
    }

    // Gather output slice ranges for the clone. The ranges are ordered
    // according to the output operand indices.
    std::vector<SliceRange> outputRanges(clone->getNumResults());
    outputRanges[splitOutputIdx] = splitOutputSlice;
    for (const auto &outputIdxMap : outputIdxAndMaps) {
      auto outputCheckedRange = outputIdxMap.second(splitOutputSlice);
      splitNodesCheck = splitNodesCheck && outputCheckedRange.first;
      splitNodesCheck = splitNodesCheck && !outputCheckedRange.second.isEmpty();
      outputRanges[outputIdxMap.first] = outputCheckedRange.second;
    }

    // Early break.
    if (!splitNodesCheck) {
      break;
    }

    // Set clone input types. Since a node does not own its input types and
    // the clone inherits the input types from the input nodes of the
    // original node we create here dummy input SliceNodes and attach them
    // to the clone in order to allow setting and checking the clone input
    // types without modifying the input types of the original node.
    for (const auto &inputIdxMap : inputIdxAndMaps) {
      auto &inputRange = inputRanges[inputIdxMap.first];
      auto inputType =
          Type::newShape(*(node->getNthInput(inputIdxMap.first).getType()),
                         inputRange.getSizes());
      inputTypes.push_back(inputType);
      inputSliceNodes.push_back(std::make_unique<SliceNode>(
          "inputSlice", &(inputTypes.back()),
          /* Input */ nullptr, inputRange.getStarts()));
      clone->setNthInput(inputIdxMap.first, inputSliceNodes.back().get());
    }

    // Set clone split output type. The original node output type is not
    // modified because the clone owns its output types.
    outputTypes.push_back(Type::newShape(*node->getType(splitOutputIdx),
                                         splitOutputSlice.getSizes()));
    clone->getNthResult(splitOutputIdx).setTypeUnsafe(&outputTypes.back());

    // Set clone output types. The original node output types are not
    // modified because the clone owns its output types.
    for (const auto &outputIdxMap : outputIdxAndMaps) {
      auto &outputRange = outputRanges[outputIdxMap.first];
      auto outputType = Type::newShape(*node->getType(outputIdxMap.first),
                                       outputRange.getSizes());
      outputTypes.push_back(outputType);
      clone->getNthResult(outputIdxMap.first)
          .setTypeUnsafe(&outputTypes.back());
    }

    // Modify clone.
    splitNodeModifier(node, clone, inputRanges, outputRanges);

    // Verify clone. If the clone is invalid at this point this means there
    // is a logic error in the splitting infrastructure (the input/output
    // maps are not checked properly or the split node modifier is flawed)
    // so we throw an error (not the same thing as returning false which is
    // intended for signaling that the splitting infrastructure correctly
    // identified an incorrect split configuration or the split configuration
    // is not accepted by the user constraints).
    if (!clone->verify()) {
      // Dump some extra error context.
      llvm::errs() << "Slice range description:\n";
      for (unsigned idx = 0, e = clone->getNumInputs(); idx < e; ++idx) {
        llvm::errs() << clone->getInputName(idx) << ": "
                     << inputRanges[idx].toString() << "\n";
      }
      for (unsigned idx = 0, e = clone->getNumResults(); idx < e; ++idx) {
        llvm::errs() << clone->getOutputName(idx).str() << ": "
                     << outputRanges[idx].toString() << "\n";
      }
      llvm::errs() << "Node description:\n";
      llvm::errs() << clone->toString() << "\n";
      return MAKE_ERR("Invalid node obtained during node splitting!");
    }

    // Early break.
    if (!splitNodesCheck) {
      break;
    }
  }

  // Check split nodes against user constraint (if any).
  if (splitConstraint) {
    splitNodesCheck = splitNodesCheck && (*splitConstraint)(node, splitNodes);
  }

  // Explicitly destroy the temporary nodes.
  for (auto *splitNode : splitNodes) {
    Node::destroyNode(splitNode);
  }

  return splitNodesCheck;
}

///===---------------------------------------------------------------------===//
///                            splitAndReplaceNode
///===---------------------------------------------------------------------===//
static Expected<std::vector<Node *>> splitAndReplaceNode(
    Node *node, const SplitNodeOption *splitOption,
    const SplitNodeConstraint *splitConstraint, dim_t splitOutputIdx,
    const llvm::ArrayRef<OpIdxAndMap> &inputIdxAndMaps,
    const llvm::ArrayRef<OpIdxAndMap> &outputIdxAndMaps = {},
    const SplitNodeModifier &splitNodeModifier = SplitNodeModifierNop) {

  // If the split output operand has no dimensions then return.
  if (node->getType(splitOutputIdx)->dims().empty()) {
    return std::vector<Node *>();
  }

  // The default split dims are all the dims of the split output operand.
  RETURN_ERR_IF_NOT(splitOutputIdx < node->getNumResults(),
                    "Invalid output index for splitting node!");
  std::vector<size_t> splitDims(node->getType(splitOutputIdx)->dims().size());
  std::iota(splitDims.begin(), splitDims.end(), 0);

  // Explicit split dims for this node.
  if (splitOption) {
    splitDims = splitOption->getSplitDims();
  }

  // Verify split parameters.
  RETURN_IF_ERR(verifySplitParams(node, splitOutputIdx, splitDims,
                                  inputIdxAndMaps, outputIdxAndMaps));

  // ------------------------------- Split output ------------------------------
  // Initialize the split output slices with the initial output range.
  SliceRange splitOutputRange = SliceRange(node->getType(splitOutputIdx));
  std::vector<SliceRange> splitOutputSlices = {splitOutputRange};

  // If a specific split option is given then we do a targeted splitting.
  // If no specific split option is given then we search a split configuration
  // which meets the constraint.
  if (splitOption) {

    // Split along all the given dimensions using the given option.
    for (size_t splitDim : splitDims) {
      splitOutputSlices =
          splitSliceRanges(splitOutputSlices, splitDim, splitOption);
    }

    // Verify split nodes.
    bool splitNodesCheck = true;
    ASSIGN_VALUE_OR_RETURN_ERR(
        splitNodesCheck,
        verifySplitNodes(node, splitOutputIdx, splitOutputSlices,
                         inputIdxAndMaps, outputIdxAndMaps, splitConstraint,
                         splitNodeModifier));

    // If split nodes are invalid then we do not perform any splitting.
    if (!splitNodesCheck) {
      return std::vector<Node *>();
    }

  } else {

    // When no split option is given a split constraint is mandatory.
    RETURN_ERR_IF_NOT(splitConstraint, "When a split option is not given then "
                                       "a split constraint must be given!");

    // Start searching of a split configuration which meets the constraint by
    // splitting along the given dimensions iteratively in smaller chunks.
    bool splitFound = false;
    for (size_t splitDim : splitDims) {

      dim_t splitDimSize = splitOutputRange.getDimSize(splitDim);
      std::vector<SliceRange> splitOutputSlicesTemp;
      for (dim_t dimNumChunks = 1; dimNumChunks <= splitDimSize;
           dimNumChunks++) {

        // Split along current dimension in the given number of chunks.
        auto splitOptionSearch =
            SplitNodeByNumChunks({splitDim}, {dimNumChunks});
        splitOutputSlicesTemp =
            splitSliceRanges(splitOutputSlices, splitDim, &splitOptionSearch);

        // Verify split nodes.
        bool splitNodesCheck = true;
        ASSIGN_VALUE_OR_RETURN_ERR(
            splitNodesCheck,
            verifySplitNodes(node, splitOutputIdx, splitOutputSlicesTemp,
                             inputIdxAndMaps, outputIdxAndMaps, splitConstraint,
                             splitNodeModifier));

        // If split is found we stop searching.
        if (splitNodesCheck) {
          splitFound = true;
          break;
        }
      }

      // Save the split output slices.
      splitOutputSlices = splitOutputSlicesTemp;

      // If split is found we stop searching.
      if (splitFound) {
        break;
      }
    }

    // If split is not found then we do not perform any splitting.
    if (!splitFound) {
      return std::vector<Node *>();
    }
  }

  // If with the current split parameters only one slice is obtained then no
  // splitting is required since the slice is the same as the original node.
  if (splitOutputSlices.size() == 1) {
    return std::vector<Node *>();
  }

  // -------------------------------- Split node -------------------------------
  // Get parent function.
  Function *F = node->getParent();
  RETURN_ERR_IF_NOT(F, "Cannot split a node without a parent Function!");

  // Allocate output tensors used for merging the partial output slices.
  std::vector<NodeValue> mergedOutputs(node->getNumResults());
  for (size_t outIdx = 0, outIdxEnd = node->getNumResults(); outIdx < outIdxEnd;
       outIdx++) {
    auto nodeName =
        node->getName().str() + ".TouchOutput" + std::to_string(outIdx);
    mergedOutputs[outIdx] = F->createTouch(nodeName, node->getType(outIdx));
  }

  // Create split nodes.
  std::vector<Node *> splitNodes(splitOutputSlices.size(), nullptr);
  for (size_t sliceIdx = 0, sliceIdxEnd = splitOutputSlices.size();
       sliceIdx < sliceIdxEnd; sliceIdx++) {

    // Current split output slice.
    const auto &splitOutputSlice = splitOutputSlices[sliceIdx];

    // Create clone to inherit all the inputs/members of the original node.
    Node *clone = node->clone();
    clone->setName(node->getName().str() + ".Split" + std::to_string(sliceIdx));

    // Gather final input slice ranges for the clone.
    std::vector<SliceRange> inputRanges(clone->getNumInputs());
    for (const auto &inputIdxMap : inputIdxAndMaps) {
      auto inputCheckedRange = inputIdxMap.second(splitOutputSlice);
      inputRanges[inputIdxMap.first] = inputCheckedRange.second;
    }

    // Gather final output slice ranges for the clone.
    std::vector<SliceRange> outputRanges(clone->getNumResults());
    outputRanges[splitOutputIdx] = splitOutputSlice;
    for (const auto &outputIdxMap : outputIdxAndMaps) {
      auto outputCheckedRange = outputIdxMap.second(splitOutputSlice);
      outputRanges[outputIdxMap.first] = outputCheckedRange.second;
    }

    // Create input Slice nodes.
    for (const auto &inputIdxMap : inputIdxAndMaps) {
      auto inputIdx = inputIdxMap.first;
      auto &inputRange = inputRanges[inputIdxMap.first];
      Type outTy = Type::newShape(*(node->getNthInput(inputIdx).getType()),
                                  inputRange.getSizes());
      auto nodeName =
          clone->getName().str() + ".SliceInput" + std::to_string(inputIdx);
      auto *inputSlice = F->createSlice(nodeName, node->getNthInput(inputIdx),
                                        inputRange.getStarts(), &outTy);
      clone->setNthInput(inputIdx, inputSlice);
    }

    // Set clone split output type. The original node output type is not
    // modified because the clone owns its output types.
    TypeRef splitOutputType = F->getParent()->uniqueTypeWithNewShape(
        node->getType(splitOutputIdx), splitOutputSlice.getSizes());
    clone->getNthResult(splitOutputIdx).setTypeUnsafe(splitOutputType);

    // Set clone output types. The original node output types are not
    // modified because the clone owns its output types.
    for (const auto &outputIdxMap : outputIdxAndMaps) {
      auto &outputRange = outputRanges[outputIdxMap.first];
      TypeRef outputType = F->getParent()->uniqueTypeWithNewShape(
          node->getType(outputIdxMap.first), outputRange.getSizes());
      clone->getNthResult(outputIdxMap.first).setTypeUnsafe(outputType);
    }

    // Modify clone.
    splitNodeModifier(node, clone, inputRanges, outputRanges);

    // Verify clone.
    RETURN_ERR_IF_NOT(clone->verify(),
                      "Invalid node obtained during node splitting!");

    // Add clone to function.
    F->addNode(clone);

    // Add clone to vector.
    splitNodes[sliceIdx] = clone;

    // Merge the partial outputs of this clone.
    for (size_t outIdx = 0, outIdxEnd = node->getNumResults();
         outIdx < outIdxEnd; outIdx++) {
      auto nodeName =
          clone->getName().str() + ".MergeOutput" + std::to_string(outIdx);
      mergedOutputs[outIdx] = F->createInsertTensor(
          nodeName, mergedOutputs[outIdx], clone->getNthResult(outIdx),
          outputRanges[outIdx].getStarts());
    }
  }

  // Replace all the node outputs with the merged outputs.
  for (size_t outIdx = 0, outIdxEnd = node->getNumResults(); outIdx < outIdxEnd;
       outIdx++) {
    node->getNthResult(outIdx).replaceAllUsesOfWith(mergedOutputs[outIdx]);
  }

  return splitNodes;
}

namespace {
///===---------------------------------------------------------------------===//
///                                    Conv
///===---------------------------------------------------------------------===//
/// Structure which contains a valid flag \p check, a dimension range \p range
/// and a dimension padding \p pads.
struct CheckedRangeAndPads {
  bool check{false};
  DimRange range;
  DimPads pads;
};

/// Structure which contains a valid flag \p check a dimension range \p range.
struct CheckedRange {
  bool check{false};
  DimRange range;
};
} // namespace

static CheckedRangeAndPads
getConvInputCheckedRangeAndPads(const DimRange &outputSliceRange,
                                const DimRange &inputRange, dim_t kernel,
                                dim_t stride, DimPads pads, dim_t dilation) {

  CHECK_LT(outputSliceRange.first, outputSliceRange.second)
      << "Invalid output slice range!";
  CHECK_LT(inputRange.first, inputRange.second) << "Invalid input range!";
  CHECK_EQ(inputRange.first, 0) << "Input range must start with 0!";
  CHECK_GE(kernel, 1) << "Invalid kernel size!";
  CHECK_GE(stride, 1) << "Invalid stride size!";
  CHECK_GE(dilation, 1) << "Invalid dilation size!";

  // Get padded input range.
  dim_t inputStartPadded = inputRange.first + pads.first;
  dim_t inputStopPadded = inputRange.second + pads.first;

  // Get padded input slice range.
  dim_t inputSliceStartPadded = outputSliceRange.first * stride;
  dim_t inputSliceStopPadded =
      (outputSliceRange.second - 1) * stride + dilation * (kernel - 1) + 1;

  // Verify input slice range bounds.
  dim_t inputSliceStopPaddedMax = pads.first + inputRange.second + pads.second;
  CHECK_LE(inputSliceStopPadded, inputSliceStopPaddedMax)
      << "Input slice range out of bounds!";

  // Get intersection.
  dim_t intersectStartPadded =
      std::max(inputStartPadded, inputSliceStartPadded);
  dim_t intersectStopPadded = std::min(inputStopPadded, inputSliceStopPadded);

  // Get checked input range.
  bool allowed = (intersectStartPadded < intersectStopPadded);
  dim_t inputSliceStart = intersectStartPadded - pads.first;
  dim_t inputSliceStop =
      intersectStopPadded >= pads.first ? intersectStopPadded - pads.first : 0;

  // Get start pad.
  dim_t inputSliceStartPad = 0;
  if (inputSliceStartPadded < inputStartPadded) {
    inputSliceStartPad = inputStartPadded - inputSliceStartPadded;
  }

  // Get stop pad.
  dim_t inputSliceStopPad = 0;
  if (inputSliceStopPadded > inputStopPadded) {
    inputSliceStopPad = inputSliceStopPadded - inputStopPadded;
  }

  DimRange inputSliceRange = {inputSliceStart, inputSliceStop};
  DimPads inputSlicePads = {inputSliceStartPad, inputSliceStopPad};
  return CheckedRangeAndPads{allowed, inputSliceRange, inputSlicePads};
}

static CheckedRange
getConvInputChannelCheckedRange(const DimRange &outputSliceRange,
                                const DimRange &inputRange, dim_t inputChannels,
                                dim_t outputChannels, dim_t group) {

  CHECK_EQ(inputChannels % group, 0)
      << "Input channels must be divisible by group!";
  CHECK_EQ(outputChannels % group, 0)
      << "Output channels must be divisible by group!";

  dim_t inputChannelsPerGroup = inputChannels / group;
  dim_t outputChannelsPerGroup = outputChannels / group;
  dim_t outputSliceChannels = SliceRange({outputSliceRange}).getDimSize(0);

  // Output slice range start/stop group index (inclusive).
  dim_t outputSliceRangeStartGroupIdx =
      outputSliceRange.first / outputChannelsPerGroup;
  dim_t outputSliceRangeStopGroupIdx =
      (outputSliceRange.second - 1) / outputChannelsPerGroup;

  bool allowed = false;
  if (outputSliceChannels <= outputChannelsPerGroup) {
    // If the output slice range spans fully or partially one group then both
    // ends of the range must be part of the same group.
    allowed = (outputSliceRangeStartGroupIdx == outputSliceRangeStopGroupIdx);
  } else {
    // If the output slice range spans multiple groups then both ends of the
    // range must be aligned to outputChannelsPerGroup.
    allowed = SliceRange({outputSliceRange})
                  .isDimRangeAligned(0, outputChannelsPerGroup);
  }

  // Compute input slice range as a multiple of groups.
  DimRange inputSliceRange;
  inputSliceRange.first = outputSliceRangeStartGroupIdx * inputChannelsPerGroup;
  inputSliceRange.second =
      (outputSliceRangeStopGroupIdx + 1) * inputChannelsPerGroup;
  return CheckedRange{allowed, inputSliceRange};
}

///===---------------------------------------------------------------------===//
///                                   Conv2D
///===---------------------------------------------------------------------===//
template <typename Shape>
static std::vector<OpIdxAndMap>
getConv2DInputIdxAndMaps(const ConvolutionNode *node) {

  ShapeHW kernels = ShapeHW(node->getKernels());
  ShapeHW strides = ShapeHW(node->getStrides());
  PaddingTLBR pads(node->getPads());
  unsigned_t group = node->getGroup();
  ShapeHW dilations = ShapeHW(node->getDilation());
  DimPads padsTB = {pads.top, pads.bottom};
  DimPads padsLR = {pads.left, pads.right};

  SliceRange inputRange = SliceRange(node->getInput().getType());
  SliceRange filterRange = SliceRange(node->getFilter().getType());
  SliceRange outputRange = SliceRange(node->getResult().getType());

  // Output slice to input slice range map.
  CheckedSliceRangeMap inputSliceRangeMap =
      [=](const SliceRange &outputSliceRange) -> CheckedSliceRange {
    // Get range and pads for dimension H.
    auto checkedRangeAndPadsH = getConvInputCheckedRangeAndPads(
        outputSliceRange[Shape::DimH], inputRange[Shape::DimH], kernels.height,
        strides.height, padsTB, dilations.height);

    // Get range and pads for dimension W.
    auto checkedRangeAndPadsW = getConvInputCheckedRangeAndPads(
        outputSliceRange[Shape::DimW], inputRange[Shape::DimW], kernels.width,
        strides.width, padsLR, dilations.width);

    // Get range for dimension C.
    auto checkedRangeC = getConvInputChannelCheckedRange(
        outputSliceRange[Shape::DimC], inputRange[Shape::DimC],
        inputRange.getDimSize(Shape::DimC), outputRange.getDimSize(Shape::DimC),
        group);

    std::vector<DimRange> inputDimRanges(4);
    inputDimRanges[Shape::DimN] = outputSliceRange[Shape::DimN];
    inputDimRanges[Shape::DimH] = checkedRangeAndPadsH.range;
    inputDimRanges[Shape::DimW] = checkedRangeAndPadsW.range;
    inputDimRanges[Shape::DimC] = checkedRangeC.range;
    bool allowed = checkedRangeAndPadsH.check && checkedRangeAndPadsW.check &&
                   checkedRangeC.check;
    return {allowed, SliceRange(inputDimRanges)};
  };

  // Output slice to filter slice range map.
  CheckedSliceRangeMap filterSliceRangeMap =
      [=](const SliceRange &outputSliceRange) -> CheckedSliceRange {
    std::vector<DimRange> filterDimRanges(4);
    filterDimRanges[Shape::DimN] = outputSliceRange[Shape::DimC];
    filterDimRanges[Shape::DimH] = filterRange[Shape::DimH];
    filterDimRanges[Shape::DimW] = filterRange[Shape::DimW];
    filterDimRanges[Shape::DimC] = filterRange[Shape::DimC];
    return {true, SliceRange(filterDimRanges)};
  };

  // Output slice to bias slice range map.
  CheckedSliceRangeMap biasSliceRangeMap =
      [=](const SliceRange &outputSliceRange) -> CheckedSliceRange {
    return {true, SliceRange({outputSliceRange[Shape::DimC]})};
  };

  // Return input indices and maps.
  return {{ConvolutionNode::InputIdx, inputSliceRangeMap},
          {ConvolutionNode::FilterIdx, filterSliceRangeMap},
          {ConvolutionNode::BiasIdx, biasSliceRangeMap}};
}

template <typename Shape>
void Conv2DSplitNodeModifier(const Node *origNode, Node *splitNode,
                             const std::vector<SliceRange> &inputSliceRanges,
                             const std::vector<SliceRange> &outputSliceRanges) {
  auto *convOrigNode = dyn_cast<ConvolutionNode>(origNode);
  auto *convSplitNode = dyn_cast<ConvolutionNode>(splitNode);
  if (!(convOrigNode && convSplitNode)) {
    return;
  }

  ShapeHW kernels = ShapeHW(convOrigNode->getKernels());
  ShapeHW strides = ShapeHW(convOrigNode->getStrides());
  PaddingTLBR pads(convOrigNode->getPads());
  ShapeHW dilations = ShapeHW(convOrigNode->getDilation());
  DimPads padsTB = {pads.top, pads.bottom};
  DimPads padsLR = {pads.left, pads.right};

  // Get paddings for split node.
  auto outputSliceRange = outputSliceRanges[ConvolutionNode::ResultIdx];
  auto inputRange = SliceRange(convOrigNode->getInput().getType());
  auto checkedRangeAndPadsH = getConvInputCheckedRangeAndPads(
      outputSliceRange[Shape::DimH], inputRange[Shape::DimH], kernels.height,
      strides.height, padsTB, dilations.height);
  auto checkedRangeAndPadsW = getConvInputCheckedRangeAndPads(
      outputSliceRange[Shape::DimW], inputRange[Shape::DimW], kernels.width,
      strides.width, padsLR, dilations.width);
  DimPads splitPadsTB = checkedRangeAndPadsH.pads;
  DimPads splitPadsLR = checkedRangeAndPadsW.pads;

  // Modify paddings for split node.
  std::vector<unsigned_t> splitPads = {
      static_cast<unsigned_t>(splitPadsTB.first),
      static_cast<unsigned_t>(splitPadsLR.first),
      static_cast<unsigned_t>(splitPadsTB.second),
      static_cast<unsigned_t>(splitPadsLR.second)};
  convSplitNode->setPads(splitPads);

  // Modify group for split node.
  dim_t outputChannels =
      SliceRange(convOrigNode->getType(ConvolutionNode::ResultIdx))
          .getDimSize(Shape::DimC);
  dim_t outputSliceChannels =
      SliceRange(convSplitNode->getType(ConvolutionNode::ResultIdx))
          .getDimSize(Shape::DimC);
  auto group = convOrigNode->getGroup();

  CHECK_EQ(outputChannels % group, 0)
      << "Output channels must be divisible by group!";
  dim_t outputChannelsPerGroup = outputChannels / group;

  if (outputSliceChannels <= outputChannelsPerGroup) {
    // If the output slice range spans fully or partially one group then we
    // set the group to 1.
    convSplitNode->setGroup(1);
  } else {
    // If the output slice range spans more than a group then it must span a
    // multiple of outputChannelsPerGroup.
    CHECK_EQ(outputSliceChannels % outputChannelsPerGroup, 0)
        << "Output slice channels must be divisible by the output channels per "
           "group!";
    dim_t splitGroup = outputSliceChannels / outputChannelsPerGroup;
    convSplitNode->setGroup(static_cast<unsigned_t>(splitGroup));
  }
}

///===---------------------------------------------------------------------===//
///                                    Pool
///===---------------------------------------------------------------------===//
static CheckedRangeAndPads
getPoolInputCheckedRangeAndPads(const DimRange &outputSliceRange,
                                const DimRange &inputRange, dim_t kernel,
                                dim_t stride, DimPads pads) {
  return getConvInputCheckedRangeAndPads(outputSliceRange, inputRange, kernel,
                                         stride, pads, /* dilation */ 1);
}

template <class PoolNode, typename Shape>
static std::vector<OpIdxAndMap> getPoolInputIdxAndMaps(const PoolNode *node) {

  ShapeHW kernels = ShapeHW(node->getKernels());
  ShapeHW strides = ShapeHW(node->getStrides());
  PaddingTLBR pads(node->getPads());
  DimPads padsTB = {pads.top, pads.bottom};
  DimPads padsLR = {pads.left, pads.right};

  // Output slice to input slice range map.
  SliceRange inputRange = SliceRange(node->getInput().getType());
  CheckedSliceRangeMap inputSliceRangeMap =
      [=](const SliceRange &outputSliceRange) -> CheckedSliceRange {
    // Get range and pads for dimension H.
    auto checkedRangeAndPadsH = getPoolInputCheckedRangeAndPads(
        outputSliceRange[Shape::DimH], inputRange[Shape::DimH], kernels.height,
        strides.height, padsTB);

    // Get range and pads for dimension W.
    auto checkedRangeAndPadsW = getPoolInputCheckedRangeAndPads(
        outputSliceRange[Shape::DimW], inputRange[Shape::DimW], kernels.width,
        strides.width, padsLR);

    std::vector<DimRange> inputDimRanges(4);
    inputDimRanges[Shape::DimN] = outputSliceRange[Shape::DimN];
    inputDimRanges[Shape::DimH] = checkedRangeAndPadsH.range;
    inputDimRanges[Shape::DimW] = checkedRangeAndPadsW.range;
    inputDimRanges[Shape::DimC] = outputSliceRange[Shape::DimC];
    bool allowed = checkedRangeAndPadsH.check && checkedRangeAndPadsW.check;
    return {allowed, SliceRange(inputDimRanges)};
  };

  // Return input index and map.
  return {{PoolNode::InputIdx, inputSliceRangeMap}};
}

template <class PoolNode, typename Shape>
void PoolSplitNodeModifier(const Node *origNode, Node *splitNode,
                           const std::vector<SliceRange> &inputSliceRanges,
                           const std::vector<SliceRange> &outputSliceRanges) {
  auto *poolOrigNode = dyn_cast<PoolNode>(origNode);
  auto *poolSplitNode = dyn_cast<PoolNode>(splitNode);
  if (!(poolOrigNode && poolSplitNode)) {
    return;
  }

  ShapeHW kernels = ShapeHW(poolOrigNode->getKernels());
  ShapeHW strides = ShapeHW(poolOrigNode->getStrides());
  PaddingTLBR pads(poolOrigNode->getPads());
  DimPads padsTB = {pads.top, pads.bottom};
  DimPads padsLR = {pads.left, pads.right};

  // Get paddings for split node.
  auto outputSliceRange = outputSliceRanges[PoolNode::ResultIdx];
  auto inputRange = SliceRange(poolOrigNode->getInput().getType());
  auto checkedRangeAndPadsH = getPoolInputCheckedRangeAndPads(
      outputSliceRange[Shape::DimH], inputRange[Shape::DimH], kernels.height,
      strides.height, padsTB);
  auto checkedRangeAndPadsW = getPoolInputCheckedRangeAndPads(
      outputSliceRange[Shape::DimW], inputRange[Shape::DimW], kernels.width,
      strides.width, padsLR);
  DimPads splitPadsTB = checkedRangeAndPadsH.pads;
  DimPads splitPadsLR = checkedRangeAndPadsW.pads;

  // Modify paddings for split node.
  std::vector<unsigned_t> splitPads = {
      static_cast<unsigned_t>(splitPadsTB.first),
      static_cast<unsigned_t>(splitPadsLR.first),
      static_cast<unsigned_t>(splitPadsTB.second),
      static_cast<unsigned_t>(splitPadsLR.second)};
  poolSplitNode->setPads(splitPads);
}

///===---------------------------------------------------------------------===//
///                               FullyConnected
///===---------------------------------------------------------------------===//
static std::vector<OpIdxAndMap>
getFullyConnectedInputIdxAndMaps(const FullyConnectedNode *node) {
  // Output slice to input slice range map.
  CheckedSliceRangeMap inputSliceRangeMap =
      [=](const SliceRange &outputSliceRange) -> CheckedSliceRange {
    SliceRange inputRange = SliceRange(node->getInput().getType());
    inputRange[ShapeHW::DimH] = outputSliceRange[ShapeHW::DimH];
    return {true, inputRange};
  };

  // Output slice to weights slice range map.
  CheckedSliceRangeMap weightsSliceRangeMap =
      [=](const SliceRange &outputSliceRange) -> CheckedSliceRange {
    SliceRange weightsRange = SliceRange(node->getWeights().getType());
    weightsRange[ShapeHW::DimW] = outputSliceRange[ShapeHW::DimW];
    return {true, weightsRange};
  };

  // Output slice to bias slice range map.
  CheckedSliceRangeMap biasSliceRangeMap =
      [=](const SliceRange &outputSliceRange) -> CheckedSliceRange {
    return {true, SliceRange({outputSliceRange[ShapeHW::DimW]})};
  };

  // Return input index and map.
  return {{FullyConnectedNode::InputIdx, inputSliceRangeMap},
          {FullyConnectedNode::WeightsIdx, weightsSliceRangeMap},
          {FullyConnectedNode::BiasIdx, biasSliceRangeMap}};
}

///===---------------------------------------------------------------------===//
///                                   MatMul
///===---------------------------------------------------------------------===//
static std::vector<OpIdxAndMap>
getMatMulInputIdxAndMaps(const MatMulNode *node) {
  // Output slice to LHS slice range map.
  CheckedSliceRangeMap lhsSliceRangeMap =
      [=](const SliceRange &outputSliceRange) -> CheckedSliceRange {
    SliceRange lhsRange = SliceRange(node->getLHS().getType());
    lhsRange[ShapeHW::DimH] = outputSliceRange[ShapeHW::DimH];
    return {true, lhsRange};
  };

  // Output slice to RHS slice range map.
  CheckedSliceRangeMap rhsSliceRangeMap =
      [=](const SliceRange &outputSliceRange) -> CheckedSliceRange {
    SliceRange rhsRange = SliceRange(node->getRHS().getType());
    rhsRange[ShapeHW::DimW] = outputSliceRange[ShapeHW::DimW];
    return {true, rhsRange};
  };

  // Return input index and map.
  return {{MatMulNode::LHSIdx, lhsSliceRangeMap},
          {MatMulNode::RHSIdx, rhsSliceRangeMap}};
}

///===---------------------------------------------------------------------===//
///                                 BatchMatMul
///===---------------------------------------------------------------------===//
static std::vector<OpIdxAndMap>
getBatchMatMulInputIdxAndMaps(const BatchMatMulNode *node) {
  // Output slice to LHS slice range map.
  CheckedSliceRangeMap lhsSliceRangeMap =
      [=](const SliceRange &outputSliceRange) -> CheckedSliceRange {
    SliceRange lhsRange = SliceRange(node->getLHS().getType());
    lhsRange[ShapeNHW::DimN] = outputSliceRange[ShapeNHW::DimN];
    lhsRange[ShapeNHW::DimH] = outputSliceRange[ShapeNHW::DimH];
    return {true, lhsRange};
  };

  // Output slice to RHS slice range map.
  CheckedSliceRangeMap rhsSliceRangeMap =
      [=](const SliceRange &outputSliceRange) -> CheckedSliceRange {
    SliceRange rhsRange = SliceRange(node->getRHS().getType());
    rhsRange[ShapeNHW::DimN] = outputSliceRange[ShapeNHW::DimN];
    rhsRange[ShapeNHW::DimW] = outputSliceRange[ShapeNHW::DimW];
    return {true, rhsRange};
  };

  // Return input index and map.
  return {{BatchMatMulNode::LHSIdx, lhsSliceRangeMap},
          {BatchMatMulNode::RHSIdx, rhsSliceRangeMap}};
}

///===---------------------------------------------------------------------===//
///                                 BatchedAdd
///===---------------------------------------------------------------------===//
static std::vector<OpIdxAndMap>
getBatchedAddInputIdxAndMaps(const BatchedAddNode *node) {

  // Output slice to Batch slice range map.
  CheckedSliceRangeMap batchSliceRangeMap =
      [=](const SliceRange &outputSliceRange) -> CheckedSliceRange {
    return {true, outputSliceRange};
  };

  // Output slice to Slice slice range map.
  CheckedSliceRangeMap sliceSliceRangeMap =
      [=](const SliceRange &outputSliceRange) -> CheckedSliceRange {
    size_t numOutDims = outputSliceRange.getNumDims();
    return {true, outputSliceRange.extractRanges(1, numOutDims - 1)};
  };

  // Return input index and map.
  return {{BatchedAddNode::BatchIdx, batchSliceRangeMap},
          {BatchedAddNode::SliceIdx, sliceSliceRangeMap}};
}

///===---------------------------------------------------------------------===//
///                                  Transpose
///===---------------------------------------------------------------------===//
static std::vector<OpIdxAndMap>
getTransposeInputIdxAndMaps(const TransposeNode *node) {

  // Transpose shuffle.
  std::vector<unsigned_t> nodeShuffle = node->getShuffle();
  std::vector<size_t> shuffle(nodeShuffle.size());
  for (size_t idx = 0, e = nodeShuffle.size(); idx < e; ++idx) {
    shuffle[idx] = static_cast<size_t>(nodeShuffle[idx]);
  }

  // Output slice to Input slice range map.
  CheckedSliceRangeMap inputSliceRangeMap =
      [=](const SliceRange &outputSliceRange) -> CheckedSliceRange {
    return {true, outputSliceRange.shuffleRanges(shuffle, /*invert*/ true)};
  };

  // Return input index and map.
  return {{TransposeNode::InputIdx, inputSliceRangeMap}};
}

///===---------------------------------------------------------------------===//
///                                  splitNode
///===---------------------------------------------------------------------===//
Expected<std::vector<Node *>>
glow::splitNode(Node *node, const SplitNodeOption *splitOption,
                const SplitNodeConstraint *splitConstraint) {

  // We can do the splitting if at least the option or the constraint is given.
  RETURN_ERR_IF_NOT(
      splitOption || splitConstraint,
      "At least the split option or the split constraint must be given!");

  switch (node->getKind()) {

  case Kinded::Kind::ConvolutionNodeKind: {
    return splitAndReplaceNode(
        node, splitOption, splitConstraint, ConvolutionNode::ResultIdx,
        getConv2DInputIdxAndMaps<ShapeNHWC>(dyn_cast<ConvolutionNode>(node)),
        {}, Conv2DSplitNodeModifier<ShapeNHWC>);
  }

  case Kinded::Kind::MaxPoolNodeKind: {
    // The current definition of the MaxPool node does not allow splitting
    // of the second output operand 'Argmax' which contains flattened
    // indices whose values will be altered if processed in smaller chunks.
    // We allow splitting only if the 'Argmax' node value has no users.
    if (node->getNthResult(MaxPoolNode::ArgmaxIdx).getNumUsers() != 0) {
      break;
    }
    return splitAndReplaceNode(
        node, splitOption, splitConstraint, MaxPoolNode::ResultIdx,
        getPoolInputIdxAndMaps<MaxPoolNode, ShapeNHWC>(
            dyn_cast<MaxPoolNode>(node)),
        {{MaxPoolNode::ArgmaxIdx, CheckedSliceRangeMapIdentity}},
        PoolSplitNodeModifier<MaxPoolNode, ShapeNHWC>);
  }

  case Kinded::Kind::AvgPoolNodeKind: {
    return splitAndReplaceNode(
        node, splitOption, splitConstraint, AvgPoolNode::ResultIdx,
        getPoolInputIdxAndMaps<AvgPoolNode, ShapeNHWC>(
            dyn_cast<AvgPoolNode>(node)),
        {}, PoolSplitNodeModifier<AvgPoolNode, ShapeNHWC>);
  }

  case Kinded::Kind::FullyConnectedNodeKind: {
    return splitAndReplaceNode(
        node, splitOption, splitConstraint, FullyConnectedNode::ResultIdx,
        getFullyConnectedInputIdxAndMaps(dyn_cast<FullyConnectedNode>(node)));
  }

  case Kinded::Kind::MatMulNodeKind: {
    return splitAndReplaceNode(
        node, splitOption, splitConstraint, MatMulNode::ResultIdx,
        getMatMulInputIdxAndMaps(dyn_cast<MatMulNode>(node)));
  }

  case Kinded::Kind::BatchMatMulNodeKind: {
    return splitAndReplaceNode(
        node, splitOption, splitConstraint, BatchMatMulNode::ResultIdx,
        getBatchMatMulInputIdxAndMaps(dyn_cast<BatchMatMulNode>(node)));
  }

  case Kinded::Kind::BatchedAddNodeKind: {
    return splitAndReplaceNode(
        node, splitOption, splitConstraint, BatchedAddNode::ResultIdx,
        getBatchedAddInputIdxAndMaps(dyn_cast<BatchedAddNode>(node)));
  }

  case Kinded::Kind::TransposeNodeKind: {
    return splitAndReplaceNode(
        node, splitOption, splitConstraint, TransposeNode::ResultIdx,
        getTransposeInputIdxAndMaps(dyn_cast<TransposeNode>(node)));
  }

  case Kinded::Kind::AddNodeKind:
  case Kinded::Kind::MulNodeKind:
  case Kinded::Kind::SubNodeKind:
  case Kinded::Kind::DivNodeKind:
  case Kinded::Kind::MaxNodeKind:
  case Kinded::Kind::MinNodeKind:
  case Kinded::Kind::CmpLTENodeKind:
  case Kinded::Kind::CmpLTNodeKind:
  case Kinded::Kind::CmpEQNodeKind:
  case Kinded::Kind::PowNodeKind: {
    DCHECK_EQ(node->getNumInputs(), 2) << "Binary operator invalid!";
    DCHECK_EQ(node->getNumResults(), 1) << "Binary operator invalid!";
    return splitAndReplaceNode(
        node, splitOption, splitConstraint, ArithmeticNode::ResultIdx,
        {{ArithmeticNode::LHSIdx, CheckedSliceRangeMapIdentity},
         {ArithmeticNode::RHSIdx, CheckedSliceRangeMapIdentity}});
  }

  case Kinded::Kind::ReluNodeKind:
  case Kinded::Kind::ClipNodeKind:
  case Kinded::Kind::TanhNodeKind:
  case Kinded::Kind::SigmoidNodeKind:
  case Kinded::Kind::LogNodeKind:
  case Kinded::Kind::ExpNodeKind:
  case Kinded::Kind::QuantizeNodeKind:
  case Kinded::Kind::RescaleQuantizedNodeKind:
  case Kinded::Kind::DequantizeNodeKind:
  case Kinded::Kind::ConvertToNodeKind: {
    DCHECK_EQ(node->getNumInputs(), 1) << "Unary operator invalid!";
    DCHECK_EQ(node->getNumResults(), 1) << "Unary operator invalid!";
    return splitAndReplaceNode(node, splitOption, splitConstraint,
                               /*splitOutputIdx*/ 0,
                               {{0, CheckedSliceRangeMapIdentity}});
  }

  default:
    VLOG(1) << "Splitting node type '" << node->getKindName()
            << "' is not supported!\n";
    break;
  }

  return std::vector<Node *>();
}

Expected<std::vector<Node *>>
glow::splitNode(Node *node, const SplitNodeOption &splitOption) {
  return splitNode(node, &splitOption, nullptr);
}

Expected<std::vector<Node *>>
glow::splitNode(Node *node, const SplitNodeConstraint &splitConstraint) {
  return splitNode(node, nullptr, &splitConstraint);
}

///===---------------------------------------------------------------------===//
///                                  splitNodes
///===---------------------------------------------------------------------===//
Expected<SplitNodeMap>
glow::splitNodes(Function *F, const SplitNodeOptionMap &splitOptionMap,
                 const SplitNodeConstraintMap &splitConstraintMap) {
  // Create split map.
  SplitNodeMap splitMap;

  // Since we will be transforming the original list of nodes, reverse iterate.
  auto &nodes = F->getNodes();
  for (auto it = nodes.rbegin(), e = nodes.rend(); it != e; it++) {
    Node *node = &*it;

    // Find explicit split option for current node (if any).
    const SplitNodeOption *splitOption = nullptr;
    auto splitOptionIt = splitOptionMap.find(node);
    if (splitOptionIt != splitOptionMap.end()) {
      splitOption = splitOptionIt->second;
    }

    // Find explicit split constraint for current node (if any).
    const SplitNodeConstraint *splitConstraint = nullptr;
    auto splitConstraintIt = splitConstraintMap.find(node);
    if (splitConstraintIt != splitConstraintMap.end()) {
      splitConstraint = splitConstraintIt->second;
    }

    // Split current node if at least the option or the constraint is given.
    if (splitOption || splitConstraint) {
      ASSIGN_VALUE_OR_RETURN_ERR(splitMap[node],
                                 splitNode(node, splitOption, splitConstraint));
    }
  }

  // Verify function after splitting nodes.
  RETURN_ERR_IF_NOT(F->verify(), "Function is not valid after node splitting!");
  return splitMap;
}

Expected<SplitNodeMap> glow::splitNodes(Function *F,
                                        const SplitNodeOption &splitOption) {
  // Since we will be transforming the original list of nodes, reverse iterate.
  SplitNodeMap splitMap;
  auto &nodes = F->getNodes();
  for (auto it = nodes.rbegin(), e = nodes.rend(); it != e; it++) {
    Node *node = &*it;
    const SplitNodeConstraint *splitConstraint = nullptr;
    ASSIGN_VALUE_OR_RETURN_ERR(splitMap[node],
                               splitNode(node, &splitOption, splitConstraint));
  }
  // Verify function after splitting nodes.
  RETURN_ERR_IF_NOT(F->verify(), "Function is not valid after node splitting!");
  return splitMap;
}

Expected<SplitNodeMap>
glow::splitNodes(Function *F, const SplitNodeConstraint &splitConstraint) {
  // Since we will be transforming the original list of nodes, reverse iterate.
  SplitNodeMap splitMap;
  auto &nodes = F->getNodes();
  for (auto it = nodes.rbegin(), e = nodes.rend(); it != e; it++) {
    Node *node = &*it;
    const SplitNodeOption *splitOption = nullptr;
    ASSIGN_VALUE_OR_RETURN_ERR(splitMap[node],
                               splitNode(node, splitOption, &splitConstraint));
  }
  // Verify function after splitting nodes.
  RETURN_ERR_IF_NOT(F->verify(), "Function is not valid after node splitting!");
  return splitMap;
}
