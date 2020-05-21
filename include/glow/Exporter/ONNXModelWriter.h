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

#ifndef GLOW_EXPORTER_ONNXMODELWRITER_H
#define GLOW_EXPORTER_ONNXMODELWRITER_H

#include "glow/Graph/Graph.h"
#include "glow/Runtime/RuntimeTypes.h"

#include "onnx/onnx_pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <fstream>
#include <list>
#include <string>

namespace glow {

/// Writes ONNX models.
class ONNXModelWriter {
  // Declare shorter aliases.
  using GraphType = ONNX_NAMESPACE::GraphProto;
  using NodeType = ONNX_NAMESPACE::NodeProto;
  using TensorType = ONNX_NAMESPACE::TensorProto;
  using AttrType = ONNX_NAMESPACE::AttributeProto;
  using ValueInfoType = ONNX_NAMESPACE::ValueInfoProto;

  /// The graph that we are constructing.
  Function *F_;
  // ModelProto that we are writing to.
  ONNX_NAMESPACE::ModelProto modelProto_;
  // GraphProto that we are writing to.
  ONNX_NAMESPACE::GraphProto *graphProto_;
  /// Whether we use zip mode or not
  const bool zipMode_;
  /// Whether we use text mode or not
  const bool textMode_;
  /// Whether to include Constant (initializer) data in the exported proto.
  const bool includeConstantData_;
  /// Whether we are writing a DAG.
  const bool dagMode_;
  /// A dedicated list of initializers in case the tensors get too big and don't
  /// fit into the model.
  std::list<TensorType> initializers_;
  /// Holds all Functions from a DAG that are being written when in dagMode_.
  llvm::SmallSet<Function *, 6> functionsFromDAG_;
  /// Output file stream.
  std::ofstream ff_;

  /// Write out \p modeProto to \ref ff_ based on \p textMode.
  Error writeModel(const ::google::protobuf::Message &modelProto,
                   bool textMode = false);

  /// Writes tensor shape from placeholder \p PH into protpbuf \p valueProto.
  void tensorShapeFromPlaceholder(const Placeholder *PH,
                                  ValueInfoType *valueProto);

  /// Add an initializer. Depending on \ref zipMode_, it will add directly to
  /// the \p graph or to a separate list.
  TensorType *addInitializer(GraphType &graph);

  /// Write \p node to \p graph using custom Glow Nodes, exported via
  /// auto-generated export logic in NodeGen.
  Error writeGlowCustomOperator(const Node *node, GraphType &graph);

  /// Setup a new proto \ref modelProto_ and \ref graphProto_ with \p irVersion
  /// and \p opsetVersion. Also initialize output stream \ref ff_ pointing to
  /// \p modelFilename.
  Error setupNewProto(const std::string &modelFilename, const size_t irVersion,
                      const size_t opsetVersion);

  /// Write the current Function \ref F_ to \ref graphProto_. \returns if there
  /// was an issue during iteration or writing.
  Error writeFunction();

  /// Finalize the written function and write it out to \p filename. \returns if
  /// there is an error encountered.
  Error finalizeAndWriteProto(llvm::StringRef filename);

  /// Adds a metadata prop with \p key and \p val to \ref modelProto_.
  void addMetadataProp(const std::string &key, const std::string &val);

  /// Write out the Functions and metadata for all DAGNodes in \p postOrder
  /// given parent \p mod.
  Error writePartitionAndMetadataProps(
      Module &mod, llvm::ArrayRef<const runtime::DAGNode *> postOrder);

  /// \returns whether \p PH is an intermediate PH for the DAG being written
  /// (i.e. both input and an output for Functions in \ref functionsFromDAG_).
  bool isIntermediatePHForDAG(const Placeholder *PH);

public:
  /// Converts \p glowType to \p protoType.
  static typename TensorType::DataType convertType(const Type &glowType);
  /// Writes Glow tensor \p T to proto output \p out.  If \p includeData then
  /// the data from \p T will be included; otherwise only the type info and name
  /// will be.
  static void writeTensor(const Tensor &T, TensorType *out,
                          bool includeData = true);

  /// Creates an ONNX model writer to serialize \p F graph into file \p
  /// modelFilename, writing \p irVersion and \p opsetVersion.  If \p errPtr is
  /// not null then if an error occurs it will get assigned there otherwise if
  /// an error occurs it will abort. It also supports serialization with text
  /// format or binary format depending on \p textMode.  If \p zipMode is true,
  /// it will save weights into individual TensorProto file along with the model
  /// file and package them into a zip file. Modes are written using
  /// uto-generated export logic via NodeGen to export all Glow Nodes as is via
  /// custom ops, instead of trying to abide by the official ONNX ops. If \p
  /// includeConstantData then data for Constants will be serialized in the
  /// written model, otherwise it will be skipped (but initializers will still
  /// exist, they will just have no data).
  ONNXModelWriter(const std::string &modelFilename, Function &F,
                  size_t irVersion, size_t opsetVersion,
                  Error *errPtr = nullptr, bool textMode = false,
                  bool zipMode = false, bool includeConstantData = true);

  /// Creates an ONNX model writer to serialize \p dagList into file \p
  /// modelFilename, writing \p irVersion and \p opsetVersion. Each partition
  /// from \p dagList will be annotated with the name of the partition to the
  /// op.  If \p errPtr is not null then if an error occurs it will get assigned
  /// there otherwise if an error occurs it will abort. It also supports
  /// serialization with text format or binary format depending on \p
  /// textMode. If \p zipMode is true, it will save weights into individual
  /// TensorProto file along with the model file and package them into a zip
  /// file. If \p includeConstantData then data for Constants will be serialized
  /// in the written model, otherwise it will be skipped (but initializers will
  /// still exist, they will just have no data).
  ONNXModelWriter(const std::string &modelFilename, runtime::DAGListTy &dagList,
                  size_t irVersion, size_t opsetVersion,
                  Error *errPtr = nullptr, bool textMode = false,
                  bool zipMode = false, bool includeConstantData = true);
};

} // namespace glow

#endif // GLOW_EXPORTER_ONNXMODELWRITER_H
