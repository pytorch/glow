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

#ifndef GLOW_QUANTIZATION_QUANTIZATION_H
#define GLOW_QUANTIZATION_QUANTIZATION_H

#include "glow/Graph/Graph.h"
#include "glow/Quantization/Base/Base.h"

#include <string>
#include <tuple>
#include <vector>

namespace glow {

class ExecutionEngine;

/// Tensor quantization parameters for a given node.
struct NodeQuantizationInfo {
  std::string nodeOutputName_;
  TensorQuantizationParams tensorQuantizationParams_;

  NodeQuantizationInfo() = default;
  NodeQuantizationInfo(const std::string &nodeOutputName,
                       const TensorQuantizationParams &tensorQuantizationParams)
      : nodeOutputName_(nodeOutputName),
        tensorQuantizationParams_(tensorQuantizationParams) {}

  float Scale() const { return tensorQuantizationParams_.scale; }
  int32_t Offset() const { return tensorQuantizationParams_.offset; }

  /// Get the full node output name based on the node name and output number.
  /// The following format is used: nodename:outputNumber
  static std::string generateNodeOutputName(const std::string &nodeName,
                                            unsigned outputNumber = 0) {
    return nodeName + ":" + std::to_string(outputNumber);
  }
};

/// Struct containing the output name string and node kind for use in the
/// LoweredInfoMap for keeping track of lowered node info.
struct NodeNameAndKind : public Named, public Kinded {
public:
  NodeNameAndKind(const NodeValue &NV)
      : Named(NodeQuantizationInfo::generateNodeOutputName(
            NV.getNode()->getName(), NV.getResNo())),
        Kinded(NV.getNode()->getKind()) {}
};

/// Overload < operator for NodeNameAndKind to allow for usage with std::set.
inline bool operator<(const NodeNameAndKind &x, const NodeNameAndKind &y) {
  return x.getName() < y.getName();
}

/// Overload == operator for NodeNameAndKind to allow for usage with std::set.
inline bool operator==(const NodeNameAndKind &x, const NodeNameAndKind &y) {
  return x.getName() == y.getName();
}

/// Used to keep track of the origin of lowered Nodes via output names as
/// determined by NodeQuantizationInfo::generateNodeOutputName(). For example if
/// some NodeValue X is lowered from some NodeValue Y, then the output name of X
/// is a key which maps to a set of names which contains the output name of Y.
using LoweredInfoMap = llvm::StringMap<std::set<NodeNameAndKind>>;

namespace quantization {

/// Generate NodeQuantizationInfo for all required nodes from function \p F
/// using the method specified by \p schema and target quantization precision \p
/// quantizationPrecision. Profiling values will be written into context \p
/// bindings. \p loweredMap maps from the NodeOutputName of a NodeValue which
/// was lowered to a vector of the original NodeOutputNames which it replaced;
/// this map is used to generate infos for the original unlowered NodeValues
/// which no longer exist in \p F.
std::vector<NodeQuantizationInfo> generateNodeQuantizationInfos(
    PlaceholderBindings &bindings, const Function *F,
    const LoweredInfoMap &loweredMap = {}, Schema schema = Schema::Asymmetric,
    ElemKind quantizationPrecision = ElemKind::Int8QTy);

/// Quantizes the function \p F into a new unoptimized partially quantized
/// function based on \p quantizationInfos and target quantization precision
/// \p quantizationPrecision. This method converts to integer as
/// many nodes as permitted by the backend \p EE. The new quantized function is
/// called \p newFuncName. If no name is given the method will generate a name.
/// This method clones original function \p F and caller is responsible for
/// cleaning up/erasing original function \p F if needed. Any nodes of kinds
/// contained in \p doNotQuantizeKinds will not be quantized, even if a profile
/// was gathered for them and the backend supports the quantized operation. If
/// \p enableRowwise is true, during quantization, all quantized FullyConnected
/// nodes will be converted to RowwiseQuantizedFullyConnected. \returns a new
/// quantized function.
Function *quantizeFunction(
    const ExecutionEngine &EE,
    llvm::ArrayRef<NodeQuantizationInfo> quantizationInfos,
    ElemKind quantizationPrecision, Function *F,
    const LoweredInfoMap &loweredMap = {}, llvm::StringRef newFuncName = "",
    const KindSet &doNotQuantizeKinds = {}, bool enableRowwise = false);

} // namespace quantization
} // namespace glow

#endif // GLOW_QUANTIZATION_QUANTIZATION_H
