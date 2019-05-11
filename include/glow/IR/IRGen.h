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
#ifndef GLOW_IR_IRGEN_H
#define GLOW_IR_IRGEN_H

#include "glow/Backend/Backend.h"
#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/IR/IR.h"
#include "glow/IR/IRBuilder.h"

#include <unordered_set>

//===----------------------------------------------------------------------===//
//              IRGen visitor - the code that generates the IR.
//===----------------------------------------------------------------------===//

namespace glow {

/// This class implements \p NodeWalker interface to translate from Node to
/// Instruction IR.
class IRGenVisitor : public NodeWalker {
private:
  using NodeValueToDestTy = std::unordered_map<NodeValue, Value *>;
  using NodeToInstrTy = std::unordered_map<Node *, Instruction *>;

  /// A set of visited nodes.
  std::unordered_set<Node *> visited_;
  /// Holds the mapping between graph nodes to the destination buffers.
  NodeValueToDestTy generatedNodeDest_;
  /// Holds the mapping between graph nodes and the lowered instructions. This
  /// map is used by instructions that want to inspect the generated
  /// instructions. For example, gradient instructions that look at operands
  /// that do not exist at the graph level. Not all variables are representible.
  NodeToInstrTy nodeToInstr_;

  /// The function that we are building.
  IRFunction *F_;
  /// The builder that adds instructions into the function.
  IRBuilder builder_;
  /// The Backend /p B is used for custom lowering of Node to Instruction IR.
  const Backend &B_;

public:
  bool shouldVisit(Node *parent, Node *N) override;

  explicit IRGenVisitor(IRFunction *M, const Backend &B)
      : F_(M), builder_(F_), B_(B) {}

  /// \returns the generated instruction for the node \p N.
  Value *valueForNode(NodeValue N);

  /// Saves the generated IR in \p v for the node \p N.
  void registerIR(NodeValue N, Value *v);

  /// Adds to Node \p N --> Instruction \p inst map.
  void setNodeToIR(Node *N, Instruction *inst);

  /// Return Instruction that is mapped to Node \p N.
  /// If mapping doesn't exists returns nullptr.
  Instruction *getNodeToIR(Node *N);

  void post(Node *parent, Node *N) override;

  IRBuilder *getBuilder() { return &builder_; }
};

} // namespace glow
#endif // GLOW_IR_IRGEN_H
