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
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace glow;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

///===---------------------------------------------------------------------===//
///                                SliceRange
///===---------------------------------------------------------------------===//
/// Dimension range representing a [start, stop] index interval (both ends are
/// included) along some tensor dimension.
using DimRange = std::pair<dim_t, dim_t>;

/// Dimension paddings representing a virtual padding before/after a given
/// dimension range.
using DimPads = std::pair<dim_t, dim_t>;

/// Slice range utility class representing the ranges for all the dimensions
/// of a slice obtained by extraction from a bigger tensor.
class SliceRange {

  /// Vector of ranges for all the dimensions of a slice.
  std::vector<DimRange> ranges_;

public:
  SliceRange() = default;

  /// Ctor.
  SliceRange(std::vector<dim_t> start, std::vector<dim_t> stop) {
    assert(start.size() <= max_tensor_dimensions &&
           "Maximum number of dimensions exceeded for SliceRange!");
    assert(start.size() == stop.size() &&
           "Invalid start/stop arrays for SliceRange!");
    for (size_t idx = 0, e = start.size(); idx < e; ++idx) {
      assert(start[idx] <= stop[idx] &&
             "Invalid start/stop arrays for SliceRange!");
      ranges_.emplace_back(start[idx], stop[idx]);
    }
  }

  /// Ctor.
  SliceRange(std::vector<DimRange> ranges) {
    assert(ranges.size() <= max_tensor_dimensions &&
           "Maximum number of dimensions exceeded for SliceRange!");
    for (auto range : ranges) {
      assert(range.first <= range.second && "Invalid ranges for SliceRange!");
    }
    ranges_ = ranges;
  }

  /// Ctor.
  SliceRange(TypeRef type) {
    assert(type->dims().size() <= max_tensor_dimensions &&
           "Maximum number of dimensions exceeded for SliceRange!");
    for (auto size : type->dims()) {
      ranges_.emplace_back(0, size - 1);
    }
  }

  /// Getter for ranges.
  std::vector<DimRange> getRanges() const { return ranges_; }

  /// Getter for dimensions sizes.
  std::vector<dim_t> getSizes() const {
    std::vector<dim_t> sizes;
    for (const auto &range : ranges_) {
      sizes.push_back(range.second - range.first + 1);
    }
    return sizes;
  }

  /// Getter for ranges start values.
  std::vector<dim_t> getStarts() const {
    std::vector<dim_t> starts;
    for (const auto &range : ranges_) {
      starts.push_back(range.first);
    }
    return starts;
  }

  /// Subscript operator for accessing a range for a given dimension.
  DimRange &operator[](size_t dim) {
    assert(dim < ranges_.size() && "Invalid dimension!");
    return ranges_[dim];
  }

  const DimRange &operator[](size_t dim) const {
    assert(dim < ranges_.size() && "Invalid dimension!");
    return ranges_[dim];
  }

  /// Equal operator for comparing range with another.
  bool operator==(const SliceRange &other) const {
    auto rangesOther = other.getRanges();
    if (ranges_.size() != rangesOther.size()) {
      return false;
    }
    for (size_t dim = 0; dim < ranges_.size(); dim++) {
      if ((ranges_[dim].first != rangesOther[dim].first) ||
          (ranges_[dim].second != rangesOther[dim].second)) {
        return false;
      }
    }
    return true;
  }

  /// Get slice range size along dimension \dim.
  dim_t getSize(size_t dim) const {
    assert(dim < ranges_.size() && "Invalid dimension!");
    return ranges_[dim].second - ranges_[dim].first + 1;
  }

  /// Get slice range total size.
  dim_t getSize() const {
    dim_t size = 1;
    for (size_t dim = 0, e = ranges_.size(); dim < e; dim++) {
      size *= getSize(dim);
    }
    return size;
  }

  /// Verify that both ends of a dimensions range are aligned to a given size.
  bool isDimRangeAligned(size_t dim, dim_t align) const {
    assert(dim < ranges_.size() && "Invalid dimension!");
    return (ranges_[dim].first % align == 0) &&
           ((ranges_[dim].second + 1) % align == 0);
  }
};

/// Function to split a slice range \p range along the dimension \p dim in the
/// given number of chunks \p numChunks. Since the split does not always result
/// in equal chunks, you can choose to list the bigger chunks first (with one
/// unit bigger than the others) by using the flag \p bigChunksFirst.
static std::vector<SliceRange>
splitAlongDimByNumChunks(const SliceRange &range, size_t dim, dim_t numChunks,
                         bool bigChunksFirst = true) {
  // Dimension range size used for splitting.
  dim_t rangeSize = range.getSize(dim);
  assert((1 <= numChunks) && (numChunks <= rangeSize) &&
         "Invalid number of chunks for splitting a SliceRange!");
  dim_t chunkRangeDiv = rangeSize / numChunks;
  dim_t chunkRangeRem = rangeSize % numChunks;

  // Small and big chunk sizes.
  dim_t numChunksBig = chunkRangeRem;
  dim_t chunkSizeSmall = chunkRangeDiv;
  dim_t chunkSizeBig = chunkRangeDiv + 1;

  // Perform splitting.
  std::vector<SliceRange> chunksRanges;
  dim_t chunkStart = range[dim].first;
  for (size_t idx = 0; idx < numChunks; idx++) {

    // Compute chunk size.
    dim_t chunkSizeCurr = chunkSizeSmall;
    if (bigChunksFirst && (idx < numChunksBig)) {
      chunkSizeCurr = chunkSizeBig;
    }
    if ((!bigChunksFirst && (idx >= numChunks - numChunksBig))) {
      chunkSizeCurr = chunkSizeBig;
    }

    // Insert chunk range.
    SliceRange chunk = range;
    chunk[dim].first = chunkStart;
    chunk[dim].second = chunkStart + chunkSizeCurr - 1;
    chunksRanges.emplace_back(chunk);
    chunkStart += chunkSizeCurr;
  }
  assert(chunkStart - 1 == range[dim].second &&
         "Inconsistent splitting of SliceRange!");
  return chunksRanges;
}

/// Function to split an array of slice ranges \p ranges along the dimension
/// \p dim in the given number of chunks \p numChunks. All the resulting arrays
/// of slice ranges are concatenated and returned. In each individual array of
/// slice ranges the bigger chunks will be listed first or not according to the
/// value of the boolean flag \p bigChunksFirst.
static std::vector<SliceRange>
splitAlongDimByNumChunks(const std::vector<SliceRange> &ranges, size_t dim,
                         dim_t numChunks, bool bigChunksFirst = true) {
  std::vector<SliceRange> outRanges;
  for (const auto &range : ranges) {
    auto splitRanges =
        splitAlongDimByNumChunks(range, dim, numChunks, bigChunksFirst);
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
/// information about whether the mapping is allowed to be used.
using CheckedSliceRangeMap =
    std::function<CheckedSliceRange(const SliceRange &)>;

/// Definition of a pair with an operand index and a checked slice range map.
using OpIdxAndMap = std::pair<unsigned, CheckedSliceRangeMap>;

/// Utility function to verify that a given slice range map \p map represents a
/// mapping between the input slice ranges \p ranges and the output slice ranges
/// \p mappedRanges.
static bool
verifyCheckedSliceRangeMap(const CheckedSliceRangeMap &map,
                           const std::vector<SliceRange> &ranges,
                           const std::vector<SliceRange> &mappedRanges) {
  bool mapOk = true;
  assert(ranges.size() == mappedRanges.size() &&
         "Slice ranges length mismatch for CheckedSliceRangeMap verification!");
  for (size_t idx = 0; idx < ranges.size(); idx++) {
    auto checkedSliceRange = map(ranges[idx]);
    mapOk = mapOk && checkedSliceRange.first;
    mapOk = mapOk && (checkedSliceRange.second == mappedRanges[idx]);
  }
  return mapOk;
}

///===---------------------------------------------------------------------===//
///                           SplitNodeModifier
///===---------------------------------------------------------------------===//
/// Definition of a function which modifies the split node \p splitNode after it
/// was cloned from the original node \p origNode. The input slice ranges \p
/// splitInputRanges and the output slice ranges \p splitOutputRanges are also
/// provided by the caller to provide extra context about how the split node was
/// obtained from the original node. This function is provided to the node
/// splitting procedure as a callback and provides the mechanism of modifying
/// the split node attributes in special situations, for example when the "Pads"
/// or "Group" node attributes must be changed when splitting Convolution nodes.
using SplitNodeModifier =
    std::function<void(const Node *origNode, Node *splitNode,
                       const std::vector<SliceRange> &splitInputRanges,
                       const std::vector<SliceRange> &splitOutputRanges)>;

/// Definition of a "nop" split node modifier when performs no modifications.
void SplitNodeModifierNop(const Node *origNode, Node *splitNode,
                          const std::vector<SliceRange> &splitInputRanges,
                          const std::vector<SliceRange> &splitOutputRanges) {}

///===---------------------------------------------------------------------===//
///                                Convolution
///===---------------------------------------------------------------------===//
static std::pair<DimRange, DimPads>
getConvInputDimRangeAndPads(const DimRange &outputSliceRange,
                            const DimRange &inputRange, dim_t kernel,
                            dim_t stride, DimPads pads, dim_t dilation) {

  assert(outputSliceRange.first <= outputSliceRange.second &&
         "Invalid output slice range!");
  assert(inputRange.first == 0 && "Input range is expected to start with 0!");
  assert(kernel >= 1 && "Invalid kernel size!");
  assert(stride >= 1 && "Invalid stride size!");
  assert(dilation >= 1 && "Invalid dilation size!");

  // Get padded input slice range start/stop indices.
  dim_t inputSliceStartPadded = outputSliceRange.first * stride + dilation * 0;
  dim_t inputSliceStopPadded =
      outputSliceRange.second * stride + dilation * (kernel - 1);

  // Get unpadded input slice range start/stop indices.
  dim_t inputSliceStart = (inputSliceStartPadded >= pads.first)
                              ? inputSliceStartPadded - pads.first
                              : 0;
  dim_t inputSliceStop = (inputSliceStopPadded >= pads.first)
                             ? inputSliceStopPadded - pads.first
                             : 0;
  inputSliceStart = (inputSliceStart <= inputRange.second) ? inputSliceStart
                                                           : inputRange.second;
  inputSliceStop = (inputSliceStop <= inputRange.second) ? inputSliceStop
                                                         : inputRange.second;

  // Get start pad.
  dim_t inputSliceStartPad = 0;
  if (inputSliceStartPadded < pads.first) {
    inputSliceStartPad = pads.first - inputSliceStartPadded;
  }

  // Get stop pad.
  dim_t inputSliceStopPad = 0;
  if (inputSliceStopPadded > pads.first + inputRange.second) {
    inputSliceStopPad = inputSliceStopPadded - (pads.first + inputRange.second);
  }

  return {{inputSliceStart, inputSliceStop},
          {inputSliceStartPad, inputSliceStopPad}};
}

static std::pair<DimRange, bool>
getConvInputChannelDimRange(const DimRange &outputSliceRange,
                            const DimRange &outputRange,
                            const DimRange &inputRange, dim_t group) {

  dim_t outputChannels = SliceRange({outputRange}).getSize(0);
  assert((outputChannels % group == 0) &&
         "Output channels must be divisible by group!");
  dim_t inputChannels = SliceRange({inputRange}).getSize(0);
  assert((inputChannels % group == 0) &&
         "Input channels must be divisible by group!");

  // Allow splitting the input channels only when the number of output channels
  // equals the convolution group (depthwise convolution). If not, the split is
  // allowed if the output channel slice is aligned to convolution group.
  dim_t filterChannels = inputChannels / group;
  DimRange inputDimRange;
  bool allowed = true;
  if (outputChannels == group) {
    inputDimRange.first = outputSliceRange.first * filterChannels;
    inputDimRange.second = (outputSliceRange.second + 1) * filterChannels - 1;
  } else {
    allowed = SliceRange({outputSliceRange}).isDimRangeAligned(0, group);
    inputDimRange = inputRange;
  }
  return {inputDimRange, allowed};
}

template <typename Shape>
static std::vector<OpIdxAndMap>
getConv2DInputIdxAndMaps(const ConvolutionNode *node) {

  ShapeHW kernels = ShapeHW(node->getKernels());
  ShapeHW strides = ShapeHW(node->getStrides());
  PaddingTLBR pads(node->getPads());
  auto dilation = node->getDilation();
  auto group = node->getGroup();
  DimPads padsTB = {pads.top, pads.bottom};
  DimPads padsLR = {pads.left, pads.right};

  SliceRange inputRange = SliceRange(node->getInput().getType());
  SliceRange filterRange = SliceRange(node->getFilter().getType());
  SliceRange outputRange = SliceRange(node->getResult().getType());

  // Output slice to input slice range map.
  CheckedSliceRangeMap inputSliceRangeMap =
      [=](const SliceRange &outputSliceRange) -> CheckedSliceRange {
    std::vector<DimRange> inputDimRanges(4);
    inputDimRanges[Shape::dimN] = outputSliceRange[Shape::dimN];
    inputDimRanges[Shape::dimH] =
        getConvInputDimRangeAndPads(outputSliceRange[Shape::dimH],
                                    inputRange[Shape::dimH], kernels.height,
                                    strides.height, padsTB, dilation)
            .first;
    inputDimRanges[Shape::dimW] =
        getConvInputDimRangeAndPads(outputSliceRange[Shape::dimW],
                                    inputRange[Shape::dimW], kernels.width,
                                    strides.width, padsLR, dilation)
            .first;
    auto inputDimRangeChecked = getConvInputChannelDimRange(
        outputSliceRange[Shape::dimC], outputRange[Shape::dimC],
        inputRange[Shape::dimC], group);
    inputDimRanges[Shape::dimC] = inputDimRangeChecked.first;
    bool allowed = inputDimRangeChecked.second;
    return {allowed, SliceRange(inputDimRanges)};
  };

  // Output slice to filter slice range map.
  CheckedSliceRangeMap filterSliceRangeMap =
      [=](const SliceRange &outputSliceRange) -> CheckedSliceRange {
    std::vector<DimRange> filterDimRanges(4);
    filterDimRanges[Shape::dimN] = outputSliceRange[Shape::dimC];
    filterDimRanges[Shape::dimH] = filterRange[Shape::dimH];
    filterDimRanges[Shape::dimW] = filterRange[Shape::dimW];
    filterDimRanges[Shape::dimC] = filterRange[Shape::dimC];
    return {true, SliceRange(filterDimRanges)};
  };

  // Output slice to bias slice range map.
  CheckedSliceRangeMap biasSliceRangeMap =
      [=](const SliceRange &outputSliceRange) -> CheckedSliceRange {
    return {true, SliceRange({outputSliceRange[Shape::dimC]})};
  };

  // Return input indices and maps.
  return {{ConvolutionNode::InputIdx, inputSliceRangeMap},
          {ConvolutionNode::FilterIdx, filterSliceRangeMap},
          {ConvolutionNode::BiasIdx, biasSliceRangeMap}};
}

template <typename Shape>
void Conv2DSplitNodeModifier(const Node *origNode, Node *splitNode,
                             const std::vector<SliceRange> &splitInputRanges,
                             const std::vector<SliceRange> &splitOutputRanges) {
  auto *convOrigNode = dyn_cast<ConvolutionNode>(origNode);
  auto *convSplitNode = dyn_cast<ConvolutionNode>(splitNode);
  if (!(convOrigNode && convSplitNode)) {
    return;
  }

  ShapeHW kernels = ShapeHW(convOrigNode->getKernels());
  ShapeHW strides = ShapeHW(convOrigNode->getStrides());
  PaddingTLBR pads(convOrigNode->getPads());
  auto dilation = convOrigNode->getDilation();
  DimPads padsTB = {pads.top, pads.bottom};
  DimPads padsLR = {pads.left, pads.right};

  // Get paddings for split node.
  auto splitOutputRange = splitOutputRanges[ConvolutionNode::ResultIdx];
  auto inputRange = SliceRange(convOrigNode->getInput().getType());
  DimRange convSplitPadsTB =
      getConvInputDimRangeAndPads(splitOutputRange[Shape::dimH],
                                  inputRange[Shape::dimH], kernels.height,
                                  strides.height, padsTB, dilation)
          .second;
  DimRange convSplitPadsLR =
      getConvInputDimRangeAndPads(splitOutputRange[Shape::dimW],
                                  inputRange[Shape::dimW], kernels.width,
                                  strides.width, padsLR, dilation)
          .second;

  // Modify paddings for split node.
  convSplitNode->setPads({static_cast<unsigned_t>(convSplitPadsTB.first),
                          static_cast<unsigned_t>(convSplitPadsLR.first),
                          static_cast<unsigned_t>(convSplitPadsTB.second),
                          static_cast<unsigned_t>(convSplitPadsLR.second)});

  // Modify group for split node.
  dim_t outputChannels =
      SliceRange(convOrigNode->getType(ConvolutionNode::ResultIdx))
          .getSize(Shape::dimC);
  dim_t outputSliceChannels =
      SliceRange(convSplitNode->getType(ConvolutionNode::ResultIdx))
          .getSize(Shape::dimC);
  auto group = convOrigNode->getGroup();
  if (outputChannels == group) {
    convSplitNode->setGroup(static_cast<unsigned_t>(outputSliceChannels));
  }
}

///===---------------------------------------------------------------------===//
///                                   Pooling
///===---------------------------------------------------------------------===//
/// Next coming ...

///===---------------------------------------------------------------------===//
///                            splitAndReplaceNode
///===---------------------------------------------------------------------===//
void splitAndReplaceNode(
    Function *F, Node *node, dim_t splitOutputIdx,
    const llvm::ArrayRef<size_t> splitDims,
    const llvm::ArrayRef<OpIdxAndMap> inputIdxAndMaps,
    const llvm::ArrayRef<OpIdxAndMap> outputIdxAndMaps,
    const llvm::ArrayRef<SplitNodeConstraint> constraints,
    const SplitNodeModifier &splitNodeModifier = SplitNodeModifierNop) {

  // If splitDims is empty then no splitting is performed.
  if (splitDims.size() == 0) {
    return;
  }

  // Verify split dims.
  assert(splitOutputIdx < node->getNumResults() &&
         "Invalid output index for splitting node!");
  for (size_t dimIdx = 0; dimIdx < splitDims.size() - 1; dimIdx++) {
    assert(splitDims[dimIdx] < splitDims[dimIdx + 1] &&
           "Invalid split dimensions for splitting node! The dimensions "
           "should be given in ascending order e.g. {0,2,3}!");
  }
  for (const auto dim : splitDims) {
    assert(dim < node->getNthResult(splitOutputIdx).getType()->dims().size() &&
           "Invalid split dimension for splitting node! The dimension exceeds "
           "the split output tensor shape!");
  }

  // Verify all the input indices and maps were given.
  assert(inputIdxAndMaps.size() == node->getNumInputs() &&
         "Invalid number of input maps for splitting node!");
  std::vector<bool> inputIdxMask(node->getNumInputs(), false);
  for (const auto &inputIdxMap : inputIdxAndMaps) {
    assert(inputIdxMap.first < node->getNumInputs() &&
           "Invalid input index for input range map!");
    inputIdxMask[inputIdxMap.first] = true;
  }
  assert(std::find(inputIdxMask.begin(), inputIdxMask.end(), false) ==
             inputIdxMask.end() &&
         "Not all input indices and maps were provided for splitting node!");

  // Verify all the output indices and maps were given.
  assert(outputIdxAndMaps.size() == node->getNumResults() - 1 &&
         "Invalid number of output maps for splitting node!");
  std::vector<bool> outputIdxMask(node->getNumResults(), false);
  outputIdxMask[splitOutputIdx] = true;
  for (const auto &outputIdxMap : outputIdxAndMaps) {
    assert(outputIdxMap.first < node->getNumResults() &&
           "Invalid input index for input range map!");
    outputIdxMask[outputIdxMap.first] = true;
  }
  assert(std::find(outputIdxMask.begin(), outputIdxMask.end(), false) ==
             outputIdxMask.end() &&
         "Not all output indices and maps were provided for splitting node!");

  // Get split output range.
  SliceRange splitOutputRange =
      SliceRange(node->getNthResult(splitOutputIdx).getType());

  // Verify the input slice range maps.
  for (const auto &inputIdxMap : inputIdxAndMaps) {
    SliceRange inputRange =
        SliceRange(node->getNthInput(inputIdxMap.first).getType());
    assert(verifyCheckedSliceRangeMap(inputIdxMap.second, {splitOutputRange},
                                      {inputRange}) &&
           "Invalid input range map for splitting node!");
  }

  // Verify the output slice range maps.
  for (const auto &outputIdxMap : outputIdxAndMaps) {
    SliceRange outputRange =
        SliceRange(node->getNthResult(outputIdxMap.first).getType());
    assert(verifyCheckedSliceRangeMap(outputIdxMap.second, {splitOutputRange},
                                      {outputRange}) &&
           "Invalid output range map for splitting node!");
  }

  // ------------------------ Search split configuration -----------------------
  // Initialize the split output slices.
  std::vector<SliceRange> splitOutputSlices = {splitOutputRange};

  // Start searching.
  bool splitFound = false;
  for (size_t splitIdx = 0; splitIdx < splitDims.size(); splitIdx++) {

    // Current dimension used for splitting.
    size_t splitDim = splitDims[splitIdx];
    dim_t splitDimSize = splitOutputRange.getSize(splitDim);

    // Split in more and more chunks along the current dimension.
    std::vector<SliceRange> splitOutputSlicesTemp;
    for (dim_t dimNumChunks = 1; dimNumChunks <= splitDimSize; dimNumChunks++) {

      // Split along current dimension in the given number of chunks.
      splitOutputSlicesTemp =
          splitAlongDimByNumChunks(splitOutputSlices, splitDim, dimNumChunks);

      // Create temporary nodes without adding them to the graph in order to
      // avoid the pollution of the graph with nodes which might not meet all
      // the constraints and might be later removed.
      bool constraintsCheck = true;
      for (const auto &splitOutputSlice : splitOutputSlicesTemp) {

        // Create clone to inherit all the inputs/members of the original node.
        Node *clone = node->clone();

        // Gather input slice ranges for the clone. The ranges are ordered
        // according to the input operand indices.
        std::vector<SliceRange> inputRanges(clone->getNumInputs());
        for (const auto &inputIdxMap : inputIdxAndMaps) {
          auto inputCheckedRange = inputIdxMap.second(splitOutputSlice);
          constraintsCheck = constraintsCheck && inputCheckedRange.first;
          inputRanges[inputIdxMap.first] = inputCheckedRange.second;
        }

        // Gather output slice ranges for the clone. The ranges are ordered
        // according to the output operand indices.
        std::vector<SliceRange> outputRanges(clone->getNumResults());
        outputRanges[splitOutputIdx] = splitOutputSlice;
        for (const auto &outputIdxMap : outputIdxAndMaps) {
          auto outputCheckedRange = outputIdxMap.second(splitOutputSlice);
          constraintsCheck = constraintsCheck && outputCheckedRange.first;
          outputRanges[outputIdxMap.first] = outputCheckedRange.second;
        }

        // Early break.
        if (!constraintsCheck) {
          break;
        }

        // Set clone input types. Since a node does not own its input types and
        // the clone inherits the input types from the input nodes of the
        // original node we create here dummy input SliceNodes and attach them
        // to the clone in order to allow setting and checking the clone input
        // types without modifying the input types of the original node.
        std::list<Type> inputTypes;
        std::list<std::unique_ptr<SliceNode>> inputSliceNodes;
        for (const auto &inputIdxMap : inputIdxAndMaps) {
          auto &inputRange = inputRanges[inputIdxMap.first];
          auto inputType =
              Type::newShape(*(node->getNthInput(inputIdxMap.first).getType()),
                             inputRange.getSizes());
          inputTypes.push_back(inputType);
          inputSliceNodes.push_back(std::make_unique<SliceNode>(
              "inputSlice", &(inputTypes.back()),
              node->getNthInput(inputIdxMap.first), inputRange.getStarts()));
          clone->setNthInput(inputIdxMap.first, inputSliceNodes.back().get());
        }

        // Set clone split output type. The original node output type is not
        // modified because the clone owns its output types.
        Type splitOutputType = Type::newShape(*node->getType(splitOutputIdx),
                                              splitOutputSlice.getSizes());
        clone->getNthResult(splitOutputIdx).setTypeUnsafe(&splitOutputType);

        // Set clone output types. The original node output types are not
        // modified because the clone owns its output types.
        std::list<Type> outputTypes;
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

        // Verify clone.
        assert(clone->verify() &&
               "Invalid node obtained during node splitting!");

        // Check clone against all constraints.
        SplitNodeContext splitCtx;
        splitCtx.origNode = node;
        splitCtx.splitNode = clone;
        splitCtx.numChunks = splitOutputSlicesTemp.size();
        for (const auto constraint : constraints) {
          constraintsCheck = constraintsCheck && constraint(splitCtx);
        }

        // Early break.
        if (!constraintsCheck) {
          break;
        }
      }

      // If all constraints are met we are done.
      if (constraintsCheck) {
        splitFound = true;
        break;
      }
    }

    // Save the split output slices.
    splitOutputSlices = splitOutputSlicesTemp;

    // If split is found we are done. If not, we continue splitting
    // along the following dimensions.
    if (splitFound) {
      break;
    }
  }

  // If no split configuration is found to meet all the constraints then we
  // do not perform any splitting.
  if (!splitFound) {
    return;
  }

  // ----------------------------- Perform splitting ---------------------------
  // Allocate output tensors used for merging the partial output slices.
  std::vector<NodeValue> mergedOutputs(node->getNumResults());
  for (size_t outIdx = 0; outIdx < node->getNumResults(); outIdx++) {
    auto nodeName =
        "touch." + node->getName().str() + "." + std::to_string(outIdx);
    mergedOutputs[outIdx] = F->createTouch(nodeName, node->getType(outIdx));
  }

  for (size_t sliceIdx = 0; sliceIdx < splitOutputSlices.size(); sliceIdx++) {

    // Current split output slice.
    const auto &splitOutputSlice = splitOutputSlices[sliceIdx];

    // Create clone to inherit all the inputs/members of the original node.
    Node *clone = node->clone();
    clone->setName(node->getName().str() + "." + std::to_string(sliceIdx));

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
      auto inputRange = inputRanges[inputIdxMap.first];
      Type outTy = Type::newShape(*(node->getNthInput(inputIdx).getType()),
                                  inputRange.getSizes());
      auto nodeName =
          node->getName().str() + ".slice." + std::to_string(inputIdx);
      auto *inputSlice = F->createSlice(nodeName, node->getNthInput(inputIdx),
                                        inputRange.getStarts(), &outTy);
      clone->setNthInput(inputIdx, inputSlice);
    }

    // Set clone split output type. The original node output type is not
    // modified because the clone owns its output types.
    TypeRef splitOutputType = F->getParent()->uniqueTypeWithNewShape(
        node->getType(splitOutputIdx), splitOutputSlice.getSizes());
    clone->getNthResult(splitOutputIdx).setTypeUnsafe(splitOutputType);

    // Modify clone.
    splitNodeModifier(node, clone, inputRanges, outputRanges);

    // Verify clone.
    assert(clone->verify() && "Invalid node obtained during node splitting!");

    // Add clone to the function.
    F->addNode(clone);

    // Merge the partial outputs of this clone.
    for (size_t outIdx = 0; outIdx < node->getNumResults(); outIdx++) {
      auto nodeName = node->getName().str() + ".insert." +
                      std::to_string(outIdx) + "." + std::to_string(sliceIdx);
      mergedOutputs[outIdx] = F->createInsertTensor(
          nodeName, mergedOutputs[outIdx], clone->getNthResult(outIdx),
          outputRanges[outIdx].getStarts());
    }
  }

  // Replace all the node outputs with the merged outputs.
  for (size_t outIdx = 0; outIdx < node->getNumResults(); outIdx++) {
    node->getNthResult(outIdx).replaceAllUsesOfWith(mergedOutputs[outIdx]);
  }
}

///===---------------------------------------------------------------------===//
///                          splitNodesWithConstraints
///===---------------------------------------------------------------------===//
Error glow::splitNodesWithConstraints(
    Function *F, const llvm::ArrayRef<SplitNodeConstraint> constraints) {

  // Since we will be transforming the original list of nodes, reverse iterate.
  auto &nodes = F->getNodes();
  for (auto it = nodes.rbegin(), e = nodes.rend(); it != e; it++) {
    Node *currNode = &*it;

    switch (currNode->getKind()) {
    case Kinded::Kind::ConvolutionNodeKind: {
      splitAndReplaceNode(F, currNode, ConvolutionNode::ResultIdx, {0, 1, 2, 3},
                          getConv2DInputIdxAndMaps<ShapeNHWC>(
                              dyn_cast<ConvolutionNode>(currNode)),
                          {}, constraints, Conv2DSplitNodeModifier<ShapeNHWC>);
      break;
    }
    default:
      VLOG(1) << "Spliting node '" << currNode->getKindName()
              << "' is not supported!\n";
      break;
    }
  }

  // Verify function after splitting nodes.
  RETURN_ERR_IF_NOT(F->verify(), "Function is not valid after node splitting!");

  return Error::success();
}
