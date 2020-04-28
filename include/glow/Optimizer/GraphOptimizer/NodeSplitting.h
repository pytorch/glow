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
#ifndef GLOW_OPTIMIZER_GRAPHOPTIMIZER_NODESPLITTING_H
#define GLOW_OPTIMIZER_GRAPHOPTIMIZER_NODESPLITTING_H

#include "glow/Base/Type.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Support/Error.h"

#include "llvm/ADT/ArrayRef.h"

#include <functional>

namespace glow {

/// Split node map type returned by the node splitting procedure consisting in
/// a mapping between the original node and the nodes it was split into.
using SplitNodeMap = std::unordered_map<Node *, std::vector<Node *>>;

///===---------------------------------------------------------------------===//
///                          splitNodesWithOptions
///===---------------------------------------------------------------------===//
/// Generic split node option as base type. All the split options refer to the
/// split procedure performed with respect to the tensor associated with the
/// output operand of a node. If a node has multiple output operands then the
/// option commonly refers to the first output operand.
class SplitNodeOption {

  /// All the dimensions of a tensor used for splitting in increasing order.
  std::vector<size_t> splitDims_;

public:
  /// Ctor.
  SplitNodeOption(std::vector<size_t> splitDims) : splitDims_(splitDims) {}

  /// Getter for split dims.
  std::vector<size_t> getSplitDims() const { return splitDims_; }

  /// Get index of \p splitDim in the \ref splitDims_ array.
  size_t getSplitDimIdx(dim_t splitDim) const;

  /// Split a given dimension size \p dimSize along the dimension \p dim.
  /// \returns the chunks sizes after splitting.
  virtual std::vector<dim_t> splitAlongDim(dim_t dimSize, size_t dim) const = 0;

  /// Dtor.
  virtual ~SplitNodeOption() = default;
};

/// Split option used for splitting a node along the dimensions \ref splitDims
/// with the given number of chunks \ref numChunks for each dimension. Since the
/// split does not always result in equal chunks, you can choose to start the
/// splitting with the bigger chunks first (one unit bigger than the others) by
/// using the flag \ref bigChunksFirst. The number of chunks for each dimension
/// must be lower than the respective dimension size, otherwise error is thrown.
class SplitNodeByNumChunks : public SplitNodeOption {

  /// Number of chunks for each split dimension.
  std::vector<dim_t> numChunks_;

  /// Whether to start splitting with bigger chunks first.
  bool bigChunksFirst_{true};

public:
  /// Ctor.
  SplitNodeByNumChunks(std::vector<size_t> splitDims,
                       std::vector<dim_t> numChunks, bool bigChunksFirst = true)
      : SplitNodeOption(splitDims), numChunks_(numChunks),
        bigChunksFirst_(bigChunksFirst) {
    CHECK(splitDims.size() == numChunks.size())
        << "Mistmatch between 'splitDims' and 'numChunks' array sizes!";
  }

  /// Split a given dimension size \p dimSize along the dimension \p dim.
  /// \returns the chunks sizes after splitting.
  std::vector<dim_t> splitAlongDim(dim_t dimSize, size_t dim) const override;
};

/// Split option used for splitting a node along the dimensions \ref splitDims
/// with the given chunk sizes \ref chunkSizes for each dimension such that each
/// chunk obtained during splitting has a size along a given split dimension
/// at most equal to the given chunk size along that respective dimension.
/// Since the split does not always result in equal chunks, you can choose to
/// start the splitting with the bigger chunks first by using the flag
/// \p bigChunksFirst. The chunk size for each dimension must be lower than the
/// respective dimension size, otherwise error is thrown.
class SplitNodeByChunkSize : public SplitNodeOption {

  /// Chunk size for each split dimension.
  std::vector<dim_t> chunkSizes_;

  /// Whether to start splitting with bigger chunks first.
  bool bigChunksFirst_{true};

public:
  /// Ctor.
  SplitNodeByChunkSize(std::vector<size_t> splitDims,
                       std::vector<dim_t> chunkSizes,
                       bool bigChunksFirst = true)
      : SplitNodeOption(splitDims), chunkSizes_(chunkSizes),
        bigChunksFirst_(bigChunksFirst) {
    CHECK(splitDims.size() == chunkSizes.size())
        << "Mistmatch between 'splitDims' and 'chunkSizes' array sizes!";
  }

  /// Split a given dimension size \p dimSize along the dimension \p dim.
  /// \returns the chunks sizes after splitting.
  std::vector<dim_t> splitAlongDim(dim_t dimSize, size_t dim) const override;
};

/// Split option used for splitting a node along the dimensions \ref splitDims
/// with the exact array of chunk sizes \ref chunkSizes for each dimension.
/// The chunk sizes must be strictly positive and the sum of the chunk sizes
/// for each dimension must be equal to the respective dimension size, otherwise
/// error is thrown.
class SplitNodeByChunkSizes : public SplitNodeOption {

  /// Array of chunk sizes for each split dimension.
  std::vector<std::vector<dim_t>> chunkSizes_;

public:
  /// Ctor.
  SplitNodeByChunkSizes(std::vector<size_t> splitDims,
                        std::vector<std::vector<dim_t>> chunkSizes)
      : SplitNodeOption(splitDims), chunkSizes_(chunkSizes) {
    CHECK(splitDims.size() == chunkSizes.size())
        << "Mistmatch between 'splitDims' and 'chunkSizes' array sizes!";
  }

  /// Split a given dimension size \p dimSize along the dimension \p dim.
  /// \returns the chunks sizes after splitting.
  std::vector<dim_t> splitAlongDim(dim_t dimSize, size_t dim) const override;
};

/// Split option used for splitting a node along the dimensions \ref splitDims
/// with the given chunk weights \ref chunkWeights for each dimension such that
/// the resulting chunk sizes after splitting will be proportional to the
/// chunk weights. The chunk weights must be strictly positive and are NOT
/// required to be normalized. The number of weights (number of chunks) for
/// each dimension must be lower than the respective dimension size, otherwise
/// error is thrown. For example when splitting a 1D tensor with size 10 along
/// the dimension 0 using the weights {1, 4} we obtain two slices with sizes
/// {2, 8}.
class SplitNodeByChunkWeights : public SplitNodeOption {

  /// Array of chunk weights for each split dimension.
  std::vector<std::vector<float>> chunkWeights_;

public:
  /// Ctor.
  SplitNodeByChunkWeights(std::vector<size_t> splitDims,
                          std::vector<std::vector<float>> chunkWeights)
      : SplitNodeOption(splitDims), chunkWeights_(chunkWeights) {
    CHECK(splitDims.size() == chunkWeights.size())
        << "Mistmatch between 'splitDims' and 'chunkWeights' array sizes!";
  }

  /// Split a given dimension size \p dimSize along the dimension \p dim.
  /// \returns the chunks sizes after splitting.
  std::vector<dim_t> splitAlongDim(dim_t dimSize, size_t dim) const override;
};

/// Split node option map provided to the node splitting procedure consisting in
/// a mapping between a node and the split options to be used for that node.
using SplitNodeOptionMap = llvm::DenseMap<Node *, SplitNodeOption *>;

///===---------------------------------------------------------------------===//
///                          splitNodesWithConstraints
///===---------------------------------------------------------------------===//
/// Context provided by the node splitting procedure for the user to define node
/// splitting constraints.
struct SplitNodeContext {
  /// Original node which was used for splitting. When this context is provided
  /// the original node is still part of the graph so the user can also check
  /// graph connexion properties (like whether the original node is followed or
  /// or preceded by another node, etc).
  const Node *origNode;

  /// Split nodes obtained by splitting the original node. When this context is
  /// provided, the split nodes are not yet part of the graph (orphans) and the
  /// nodes have only temporary SliceNodes attached to their inputs without any
  /// nodes attached to their outputs so you are only allowed to check the split
  /// nodes types, attributes, input/output operand types and also the offsets
  /// of the SliceNodes used for splitting the input operands.
  std::vector<Node *> splitNodes;
};

/// User defined constraint which verifies whether a given split configuration
/// given by the context \p ctx is allowed. \returns true if the configuration
/// is allowed (accepted by the user) and false otherwise. The node splitting
/// is performed only if all the constraints are met.
using SplitNodeConstraint = std::function<bool(const SplitNodeContext &ctx)>;

/// Particular split node constraint which requires that the total amount of
/// memory for each split node (the size for all the inputs/outputs) be less
/// than a maximum value (in bytes).
inline SplitNodeConstraint getMaxMemSplitNodeConstraint(unsigned maxMem) {
  return [=](const SplitNodeContext &ctx) -> bool {
    for (const auto *splitNode : ctx.splitNodes) {
      if (!(splitNode->getTotMemSize() <= maxMem)) {
        return false;
      }
    }
    return true;
  };
}

/// Function to split the node \p node using the given option \p splitOption
/// and the given split constraints \p splitConstraints. \returns a vector with
/// the nodes obtained after splitting \p node.
Expected<std::vector<Node *>> splitNodeWithConstraints(
    Node *node, const SplitNodeOption *splitOption,
    const llvm::ArrayRef<SplitNodeConstraint> &splitConstraints);

/// Function to split all the nodes from the function \p F for which there is
/// a split logic implemented using the given split option map \p splitOptionMap
/// and the given split constraints \p splitConstraints. \returns a split node
/// map for all the nodes which were actually split.
Expected<SplitNodeMap> splitNodesWithConstraints(
    Function *F, const SplitNodeOptionMap &splitOptionMap,
    const llvm::ArrayRef<SplitNodeConstraint> &splitConstraints);

} // namespace glow

#endif // GLOW_OPTIMIZER_GRAPHOPTIMIZER_NODESPLITTING_H
