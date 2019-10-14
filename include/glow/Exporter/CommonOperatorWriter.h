/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
 *
 * Licensed under the Apache License, Version 2.0 (the "License.");
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

#ifndef GLOW_EXPORTER_COMMONOPERATORWRITER_H
#define GLOW_EXPORTER_COMMONOPERATORWRITER_H

#include "glow/Exporter/ProtobufWriter.h"

namespace glow {
/// Declares writer methods for all operators. Every writer method serializes
/// Glow node into the provided protobuf.
template <typename Traits> class CommonOperatorWriter : public ProtobufWriter {
protected:
  virtual ~CommonOperatorWriter() = default;

  /// Declare pure virtual methods, one per each node kind.
  /// Derived class must to implement all of it.
#define DEF_NODE(CLASS, NAME)                                                  \
  virtual Error write##NAME(const CLASS *node,                                 \
                            typename Traits::GraphProto &graph) = 0;
#include "glow/AutoGenNodes.def"

  /// Function invokes the correspondent virtual method according to \p node
  /// type to serialize node information into \p graph (protobuf), reports
  /// visited intermediate nodes through \p reporter, \returns Error.
  Error writeOperator(const Node *node, typename Traits::GraphProto &graph) {
    switch (node->getKind()) {
#define DEF_NODE(CLASS, NAME)                                                  \
  case glow::Kinded::Kind::CLASS##Kind:                                        \
    return write##NAME(llvm::cast<CLASS>(node), graph);
#include "glow/AutoGenNodes.def"
    default:
      llvm_unreachable(
          "Not reachable, values and instructions are not handled here");
      return Error::success();
    }
  }

  using ProtobufWriter::ProtobufWriter;
};
} // namespace glow

#endif // GLOW_EXPORTER_COMMONOPERATORWRITER_H
