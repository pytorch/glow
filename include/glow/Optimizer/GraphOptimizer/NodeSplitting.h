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

#include "glow/Base/SliceRange.h"
#include "glow/Base/Type.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Support/Error.h"

#include "llvm/ADT/ArrayRef.h"

#include <functional>

namespace glow {

/// Node output operand used for splitting. Currently we only support splitting
/// the 1st output operand of a node.
constexpr dim_t SplitNodeOutputIdx = 0;

///===---------------------------------------------------------------------===//
///                               SplitNodeOption
///===---------------------------------------------------------------------===//
/// Generic split node option as base type. All the split options refer to the
/// split procedure performed with respect to the tensor associated with the
/// output operand of a node. If a node has multiple output operands then the
/// option commonly refers to the first output operand.
class SplitNodeOption {
public:
  /// Discriminator for LLVM-style RTTI (for using dyn_cast<> et al).
  enum SplitNodeKind {
    SplitNodeBySliceRanges,
    SplitNodeByNumChunks,
    SplitNodeByChunkSize,
    SplitNodeByChunkSizes,
    SplitNodeByChunkWeights,
  };

private:
  const SplitNodeKind kind_;

public:
  /// Dtor.
  virtual ~SplitNodeOption() = default;
  SplitNodeKind getKind() const { return kind_; }
  SplitNodeOption(SplitNodeKind K) : kind_(K) {}
};

/// Orthogonal generic split node option.
class SplitNodeOptionOrthogonal : public SplitNodeOption {

  /// All the dimensions of a tensor used for splitting in increasing order.
  llvm::SmallVector<size_t, max_tensor_dimensions> splitDims_;

public:
  /// Ctor.
  SplitNodeOptionOrthogonal(SplitNodeOption::SplitNodeKind k,
                            llvm::ArrayRef<size_t> splitDims)
      : SplitNodeOption(k), splitDims_(splitDims.begin(), splitDims.end()) {}

  /// \returns the split dims.
  llvm::ArrayRef<size_t> getSplitDims() const { return splitDims_; }

  /// \returns the index of \p splitDim in the \ref splitDims_ array.
  size_t getSplitDimIdx(dim_t splitDim) const;

  /// Split a given dimension size \p dimSize along the dimension \p dim.
  /// \returns the chunks sizes after splitting.
  virtual std::vector<dim_t> splitAlongDim(size_t dim, dim_t dimSize) const = 0;

  /// Dtor.
  virtual ~SplitNodeOptionOrthogonal() = default;

  static bool classof(const SplitNodeOption *S) {
    return S->getKind() == SplitNodeKind::SplitNodeByNumChunks ||
           S->getKind() == SplitNodeKind::SplitNodeByChunkSize ||
           S->getKind() == SplitNodeKind::SplitNodeByChunkSizes ||
           S->getKind() == SplitNodeKind::SplitNodeByChunkWeights;
  }
};

/// Orthogonal split option for splitting a node along dimensions \ref splitDims
/// with the given number of chunks \ref numChunks for each dimension. Since the
/// split does not always result in equal chunks, you can choose to start the
/// splitting with the bigger chunks first (one unit bigger than the others) by
/// using the flag \ref bigChunksFirst. The number of chunks for each dimension
/// must be lower than the respective dimension size, otherwise error is thrown.
class SplitNodeByNumChunks : public SplitNodeOptionOrthogonal {

  /// Number of chunks for each split dimension.
  llvm::SmallVector<dim_t, max_tensor_dimensions> numChunks_;

  /// Whether to start splitting with bigger chunks first.
  bool bigChunksFirst_{true};

public:
  /// Ctor.
  SplitNodeByNumChunks(llvm::ArrayRef<size_t> splitDims,
                       llvm::ArrayRef<dim_t> numChunks,
                       bool bigChunksFirst = true)
      : SplitNodeOptionOrthogonal(SplitNodeKind::SplitNodeByNumChunks,
                                  splitDims),
        numChunks_(numChunks.begin(), numChunks.end()),
        bigChunksFirst_(bigChunksFirst) {
    CHECK_EQ(splitDims.size(), numChunks.size())
        << "Mismatch between 'splitDims' and 'numChunks' array sizes!";
  }

  /// Split a given dimension size \p dimSize along the dimension \p dim.
  /// \returns the chunks sizes after splitting.
  std::vector<dim_t> splitAlongDim(size_t dim, dim_t dimSize) const override;
};

/// Orthogonal split option for splitting a node along dimensions \ref splitDims
/// with the given chunk sizes \ref chunkSizes for each dimension such that each
/// chunk obtained during splitting has a size along a given split dimension
/// at most equal to the given chunk size along that respective dimension.
/// Since the split does not always result in equal chunks, you can choose to
/// start the splitting with the bigger chunks first by using the flag
/// \p bigChunksFirst. The chunk size for each dimension must be lower than the
/// respective dimension size, otherwise error is thrown.
class SplitNodeByChunkSize : public SplitNodeOptionOrthogonal {

  /// Chunk size for each split dimension.
  llvm::SmallVector<dim_t, max_tensor_dimensions> chunkSizes_;

  /// Whether to start splitting with bigger chunks first.
  bool bigChunksFirst_{true};

public:
  /// Ctor.
  SplitNodeByChunkSize(llvm::ArrayRef<size_t> splitDims,
                       llvm::ArrayRef<dim_t> chunkSizes,
                       bool bigChunksFirst = true)
      : SplitNodeOptionOrthogonal(SplitNodeKind::SplitNodeByChunkSize,
                                  splitDims),
        chunkSizes_(chunkSizes.begin(), chunkSizes.end()),
        bigChunksFirst_(bigChunksFirst) {
    CHECK_EQ(splitDims.size(), chunkSizes.size())
        << "Mismatch between 'splitDims' and 'chunkSizes' array sizes!";
  }

  /// Split a given dimension size \p dimSize along the dimension \p dim.
  /// \returns the chunks sizes after splitting.
  std::vector<dim_t> splitAlongDim(size_t dim, dim_t dimSize) const override;
};

/// Orthogonal split option for splitting a node along dimensions \ref splitDims
/// with the exact array of chunk sizes \ref chunkSizes for each dimension.
/// The chunk sizes must be strictly positive and the sum of the chunk sizes
/// for each dimension must be equal to the respective dimension size, otherwise
/// error is thrown.
class SplitNodeByChunkSizes : public SplitNodeOptionOrthogonal {

  /// Array of chunk sizes for each split dimension.
  llvm::SmallVector<std::vector<dim_t>, max_tensor_dimensions> chunkSizes_;

public:
  /// Ctor.
  SplitNodeByChunkSizes(llvm::ArrayRef<size_t> splitDims,
                        llvm::ArrayRef<std::vector<dim_t>> chunkSizes)
      : SplitNodeOptionOrthogonal(SplitNodeKind::SplitNodeByChunkSizes,
                                  splitDims),
        chunkSizes_(chunkSizes.begin(), chunkSizes.end()) {
    CHECK_EQ(splitDims.size(), chunkSizes.size())
        << "Mismatch between 'splitDims' and 'chunkSizes' array sizes!";
  }

  /// Split a given dimension size \p dimSize along the dimension \p dim.
  /// \returns the chunks sizes after splitting.
  std::vector<dim_t> splitAlongDim(size_t dim, dim_t dimSize) const override;
};

/// Orthogonal split option for splitting a node along dimensions \ref splitDims
/// with the given chunk weights \ref chunkWeights for each dimension such that
/// the resulting chunk sizes after splitting will be proportional to the
/// chunk weights. The chunk weights must be strictly positive and are NOT
/// required to be normalized. The number of weights (number of chunks) for
/// each dimension must be lower than the respective dimension size, otherwise
/// error is thrown. For example when splitting a 1D tensor with size 10 along
/// the dimension 0 using the weights {1, 4} we obtain two slices with sizes
/// {2, 8}.
class SplitNodeByChunkWeights : public SplitNodeOptionOrthogonal {

  /// Array of chunk weights for each split dimension.
  llvm::SmallVector<std::vector<float>, max_tensor_dimensions> chunkWeights_;

public:
  /// Ctor.
  SplitNodeByChunkWeights(llvm::ArrayRef<size_t> splitDims,
                          llvm::ArrayRef<std::vector<float>> chunkWeights)
      : SplitNodeOptionOrthogonal(SplitNodeKind::SplitNodeByChunkWeights,
                                  splitDims),
        chunkWeights_(chunkWeights.begin(), chunkWeights.end()) {
    CHECK_EQ(splitDims.size(), chunkWeights.size())
        << "Mismatch between 'splitDims' and 'chunkWeights' array sizes!";
  }

  /// Split a given dimension size \p dimSize along the dimension \p dim.
  /// \returns the chunks sizes after splitting.
  std::vector<dim_t> splitAlongDim(size_t dim, dim_t dimSize) const override;
};

/// Non-orthogonal split option based on raw slice ranges defined for the output
/// operand of a node. The slice ranges are allowed to overlap but must cover
/// the entire output operand of the node being split. This is the most flexible
/// and generic split option.
class SplitNodeBySliceRanges : public SplitNodeOption {

  /// Array of raw slice ranges.
  std::vector<SliceRange> sliceRanges_;

public:
  /// Ctor.
  SplitNodeBySliceRanges(llvm::ArrayRef<SliceRange> sliceRanges)
      : SplitNodeOption(SplitNodeKind::SplitNodeBySliceRanges),
        sliceRanges_(sliceRanges) {}

  /// \returns the raw slice ranges.
  llvm::ArrayRef<SliceRange> getSliceRanges() const { return sliceRanges_; }

  static bool classof(const SplitNodeOption *S) {
    return S->getKind() == SplitNodeKind::SplitNodeBySliceRanges;
  }
};

///===---------------------------------------------------------------------===//
///                             SplitNodeConstraint
///===---------------------------------------------------------------------===//
/// User defined constraint which verifies whether a given split configuration
/// is allowed. The context provided by the node splitting procedure for the
/// user to define a constraint is:
/// \p origNode - original node which was used for splitting. When this context
/// is provided the original node is still part of the graph so the user can
/// also check graph connection properties (like whether the original node is
/// followed or preceded by another node, etc).
/// \p splitNodes - the array of split nodes obtained by splitting the original
/// node. When this context is provided, the split nodes are not yet part of the
/// graph (orphans) and the nodes have only temporary SliceNodes attached to
/// their inputs without any nodes attached to their outputs so you are only
/// allowed to check the split nodes types, attributes, input/output operand
/// types and also the offsets of the SliceNodes used for splitting the input
/// operands of the original node.
/// \returns true if the configuration is accepted (allowed) by the user and
/// false otherwise. The splitting is done only if the constraint is verified.
using SplitNodeConstraint = std::function<bool(
    const Node *origNode, const std::vector<Node *> &splitNodes)>;

/// Particular split node constraint which requires that the total amount of
/// memory for each split node (the size for all the inputs/outputs) be less
/// than a maximum value (in bytes).
inline SplitNodeConstraint SplitNodeMaxMemConstraint(unsigned maxMem) {
  return
      [=](const Node *origNode, const std::vector<Node *> &splitNodes) -> bool {
        for (const auto *splitNode : splitNodes) {
          if (!(splitNode->getTotMemSize() <= maxMem)) {
            return false;
          }
        }
        return true;
      };
}

///===---------------------------------------------------------------------===//
///                                  splitNode
///===---------------------------------------------------------------------===//
/// Function to split the node \p node using the given option \p splitOption
/// and/or the given split constraint \p splitConstraint. At least the split
/// option or the split constraint must be given (not necessarily both), the one
/// which is not used must be set to null. If no option/constraint is provided
/// then an error is thrown. \returns a vector with the nodes obtained after
/// splitting \p node. The node is split only if there is a splitting logic
/// implemented for the node type, otherwise an empty vector is returned.
/// NOTE: The original node is deleted from the graph after being split. When
/// using this utility while iterating the nodes of a function make sure to
/// reverse iterate in order to not invalidate the iterator:
///   auto &nodes = F->getNodes();
///   for (auto it = nodes.rbegin(), e = nodes.rend(); it != e;) {
///     Node *node = &*(it++);
///     splitNode(node, ...);
///   }
Expected<std::vector<Node *>>
splitNode(Node *node, const SplitNodeOption *splitOption,
          const SplitNodeConstraint *splitConstraint);

/// Function to split the node \p node using the option \p splitOption.
/// \returns a vector with the nodes obtained after splitting \p node. The node
/// is split only if there is a splitting logic implemented for the node type,
/// otherwise an empty vector is returned.
/// NOTE: The original node is deleted from the graph after being split. When
/// using this utility while iterating the nodes of a function make sure to
/// reverse iterate in order to not invalidate the iterator:
///   auto &nodes = F->getNodes();
///   for (auto it = nodes.rbegin(), e = nodes.rend(); it != e;) {
///     Node *node = &*(it++);
///     splitNode(node, ...);
///   }
Expected<std::vector<Node *>> splitNode(Node *node,
                                        const SplitNodeOption &splitOption);

/// Function to split the node \p node using the constraint \p splitConstraint.
/// \returns a vector with the nodes obtained after splitting \p node. The node
/// is split only if there is a splitting logic implemented for the node type,
/// otherwise an empty vector is returned.
/// NOTE: The original node is deleted from the graph after being split. When
/// using this utility while iterating the nodes of a function make sure to
/// reverse iterate in order to not invalidate the iterator:
///   auto &nodes = F->getNodes();
///   for (auto it = nodes.rbegin(), e = nodes.rend(); it != e;) {
///     Node *node = &*(it++);
///     splitNode(node, ...);
///   }
Expected<std::vector<Node *>>
splitNode(Node *node, const SplitNodeConstraint &splitConstraint);

///===---------------------------------------------------------------------===//
///                                  splitNodes
///===---------------------------------------------------------------------===//
/// Split node option map provided to the node splitting procedure which
/// consists in a mapping between a node and the split option to be used
/// for that node.
using SplitNodeOptionMap = llvm::DenseMap<Node *, const SplitNodeOption *>;

/// Split node constraint map provided to the node splitting procedure which
/// consists in a mapping between a node and the split constraint to be used
/// for that node.
using SplitNodeConstraintMap =
    llvm::DenseMap<Node *, const SplitNodeConstraint *>;

/// Split node map returned by the node splitting procedure which consists in a
/// mapping between the original node and the vector of nodes it was split into.
/// If the node was not actually split then the vector is empty.
using SplitNodeMap = std::unordered_map<Node *, std::vector<Node *>>;

/// Function to split all the nodes from the function \p F for which there is
/// a split logic implemented using the given split option map \p splitOptionMap
/// and the given split constraint map \p splitConstraintMap. \returns a split
/// node map for those nodes which were actually split. This function does not
/// split those nodes for which no option or constraint was found in the maps.
/// NOTE: The original nodes are deleted from the graph after being split.
Expected<SplitNodeMap>
splitNodes(Function *F, const SplitNodeOptionMap &splitOptionMap,
           const SplitNodeConstraintMap &splitConstraintMap);

/// Function to split all the nodes from the function \p F for which there is
/// a split logic implemented using a common split option \p splitOption which
/// will be used for all the nodes. \returns a split node map for those nodes
/// which were actually split.
/// NOTE: The original nodes are deleted from the graph after being split.
Expected<SplitNodeMap> splitNodes(Function *F,
                                  const SplitNodeOption &splitOption);

/// Function to split all the nodes from the function \p F for which there is
/// a split logic implemented using a common split constraint \p splitConstraint
/// which will be used for all the nodes. \returns a split node map for those
/// nodes which were actually split.
/// NOTE: The original nodes are deleted from the graph after being split.
Expected<SplitNodeMap> splitNodes(Function *F,
                                  const SplitNodeConstraint &splitConstraint);

///===---------------------------------------------------------------------===//
///                            splitNodeRecursively
///===---------------------------------------------------------------------===//
/// Function to perform a recursive node splitting starting with \p node and
/// traversing the graph along the depth in order to split the parent nodes of
/// \p node up to a maximum depth of \p maxDepth. The splitting of the starting
/// node is based on the option \p splitOption. The splitting of the parent
/// nodes is based on a custom option which inherits the slicing boundaries from
/// the child node in order to create subgraphs which are completely parallel.
/// The constraint \p splitConstraint applies to all the nodes being split.
/// If \p singleUseOnly is given then the recursive split procedure stops when
/// encountering a node which has multiple uses.
/// The recursive split procedure stops either:
/// - when the maximum depth has been reached.
/// - when a node is encountered for which there is no split support.
/// - when a node is encountered for which the split constraint is not met.
/// \returns a split node map for those nodes which were actually split.
/// NOTE: The original nodes are deleted from the graph after being split. Do
/// not use this utility while iterating the nodes of a function.
Expected<SplitNodeMap>
splitNodeRecursively(Node *node, const SplitNodeOption *splitOption,
                     const SplitNodeConstraint *splitConstraint,
                     unsigned maxDepth, bool singleUseOnly = false);

/// Function to perform a recursive node splitting starting with \p node and
/// traversing the graph along the depth in order to split the parent nodes of
/// \p node up to a maximum depth of \p maxDepth. The splitting of the starting
/// node is based on the option \p splitOption. The splitting of the parent
/// nodes is based on a custom option which inherits the slicing boundaries from
/// the child node in order to create subgraphs which are completely parallel.
/// If \p singleUseOnly is given then the recursive split procedure stops when
/// encountering a node which has multiple uses.
/// \returns a split node map for those nodes which were actually split.
/// NOTE: The original nodes are deleted from the graph after being split. Do
/// not use this utility while iterating the nodes of a function.
Expected<SplitNodeMap> splitNodeRecursively(Node *node,
                                            const SplitNodeOption &splitOption,
                                            unsigned maxDepth,
                                            bool singleUseOnly = false);

/// Function to perform a recursive node splitting starting with \p node and
/// traversing the graph along the depth in order to split the parent nodes of
/// \p node up to a maximum depth of \p maxDepth.
/// The constraint \p splitConstraint applies to all the nodes being split.
/// If \p singleUseOnly is given then the recursive split procedure stops when
/// encountering a node which has multiple uses.
/// \returns a split node map for those nodes which were actually split.
/// NOTE: The original nodes are deleted from the graph after being split. Do
/// not use this utility while iterating the nodes of a function.
Expected<SplitNodeMap>
splitNodeRecursively(Node *node, const SplitNodeConstraint &splitConstraint,
                     unsigned maxDepth, bool singleUseOnly = false);

} // namespace glow

#endif // GLOW_OPTIMIZER_GRAPHOPTIMIZER_NODESPLITTING_H
