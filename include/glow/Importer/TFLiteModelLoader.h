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

#ifndef GLOW_IMPORTER_TFLITEMODELLOADER_H
#define GLOW_IMPORTER_TFLITEMODELLOADER_H

#include "glow/Graph/Graph.h"
#include "schema_generated.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace glow {

/// Loads TensorFlowLite models.
class TFLiteModelLoader {

  /// TensorFlowLite model object.
  const tflite::Model *model_{nullptr};

  /// TensorFlowLite current graph object.
  const tflite::SubGraph *graph_{nullptr};

  /// TensorFlowLite model version.
  size_t modelVersion_;

  /// TensorFlowLite model description.
  std::string modelDescription_;

  /// The Glow function which is currently constructed from \ref graph_.
  Function *F_{nullptr};

  /// The Glow module containing the function(s) we are constructing.
  Module &mod_;

  /// Vector with node values ordered by their corresponding tensor positions
  /// in the original model. This vector contains only the node values (tensors)
  /// registered in the original model since only those are needed for chaining
  /// the model graph operators. Other node values created during graph loading
  /// will not be registered in this vector.
  std::vector<NodeValue> nodeValueByIndex_;

  /// \returns a tensor from the model using the index \p index.
  Expected<const tflite::Tensor *> getTensorByIndex(size_t index);

  /// \returns the name of the tensor \p tensor.
  std::string getTensorName(const tflite::Tensor *tensor);

  /// \returns the shape of the tensor \p tensor.
  Expected<std::vector<dim_t>> getTensorShape(const tflite::Tensor *tensor);

  /// \returns the element type of the tensor \p tensor.
  Expected<ElemKind> getTensorElemKind(const tflite::Tensor *tensor);

  /// \returns whether the tensor \p tensor is quantized or not.
  bool isTensorQuantized(const tflite::Tensor *tensor);

  /// \returns whether the tensor \p tensor is quantized per axis or not.
  bool isTensorPerAxisQuantized(const tflite::Tensor *tensor);

  /// \returns the scale quantization parameter of the tensor \p tensor.
  Expected<float> getTensorScale(const tflite::Tensor *tensor);

  /// \returns the offset quantization parameter of the tensor \p tensor.
  Expected<int32_t> getTensorOffset(const tflite::Tensor *tensor);

  /// \returns the scales quantization parameters of the tensor \p tensor.
  Expected<std::vector<float>> getTensorScales(const tflite::Tensor *tensor);

  /// \returns the offsets quantization parameters of the tensor \p tensor.
  Expected<std::vector<int32_t>> getTensorOffsets(const tflite::Tensor *tensor);

  /// \returns the type of the tensor \p tensor.
  Expected<Type> getTensorType(const tflite::Tensor *tensor);

  /// \returns the data pointer and the size of tensor \p tensor as a pair.
  Expected<std::pair<const char *, size_t>>
  getTensorDataAndSize(const tflite::Tensor *tensor);

  /// \returns the operator code of the operator \p op.
  Expected<tflite::BuiltinOperator> getOperatorCode(const tflite::Operator *op);

  /// \returns the operator version of the operator \p op.
  Expected<int32_t> getOperatorVersion(const tflite::Operator *op);

  /// \returns the operator type of the operator \p op.
  Expected<std::string> getOperatorType(const tflite::Operator *op);

  /// \returns the operator name of the operator \p op.
  Expected<std::string> getOperatorName(const tflite::Operator *op);

  /// \returns the tensor index of the input operand with index \p inputIdx
  /// of the operator \p op.
  Expected<size_t> getOperatorInputTensorIdx(const tflite::Operator *op,
                                             size_t inputIdx);

  /// \returns the tensor index of the output operand with index \p outputIdx
  /// of the operator \p op.
  Expected<size_t> getOperatorOutputTensorIdx(const tflite::Operator *op,
                                              size_t outputIdx);

  /// \returns whether the output operand with index \p outputIdx of the
  /// operator \p op is a final tensor (graph output placeholder).
  Expected<bool> isOperatorOutputFinalTensor(const tflite::Operator *op,
                                             size_t outputIdx);

  /// \returns Expected<NodeValue> if a node value is registered in the array
  /// \ref nodeValueByIndex_ with \p index (tensor index) and Error otherwise.
  Expected<NodeValue> getNodeValueByIndex(size_t index);

  /// Set a node value \p nodeValue using \p index (tensor index) in the array
  /// \ref nodeValueByIndex_. \returns Error if \p index is invalid.
  Error setNodeValueByIndex(size_t index, NodeValue nodeValue);

  /// \returns Expected<NodeValue> if an input node value with the given index
  /// \p index (operator level index that is 0 for 1st input node value, etc)
  /// is found for the operator \p op and Error otherwise.
  Expected<NodeValue> getInputNodeValue(const tflite::Operator *op,
                                        size_t inputIdx);

  /// Register the single output node value \p nodeValue for the operator \p op.
  Error setOutputNodeValue(const tflite::Operator *op, NodeValue nodeValue);

  /// Register multiple output node values \p nodeValues for the operator \p op.
  Error setOutputNodeValues(const tflite::Operator *op,
                            llvm::ArrayRef<NodeValue> nodeValues);

  /// \returns the output type for operator \p op with index \p outputIndex.
  Expected<TypeRef> getOutputType(const tflite::Operator *op,
                                  size_t outputIndex);

  /// Initialize the node value array \ref nodeValueByIndex_.
  void initializeNodeValues();

  /// Load the input placeholders of the current graph.
  Error loadInputPlaceholders();

  /// Load the constant weights of the current graph.
  Error loadConstants();

  /// Load the operators of the current graph.
  Error loadOperators();

  /// Save the output placeholders of the current graph.
  Error saveOutputPlaceholders();

  /// Add an activation function to the node value \p value using the activation
  /// type \p type. The node value is modified in-place.
  Error addActivation(NodeValue &value, tflite::ActivationFunctionType type);

  /// Local definition of a POD structure with operator meta information.
  struct OperatorInfo {
    std::string name;
    std::string type;
    size_t index;
    tflite::BuiltinOperator code;
    int32_t version;
  };

  /// Utility function to extend the error message \p errMsg with the operator
  /// context provided by \p opInfo. \returns the extended error message.
  const std::string opErrMsg(const OperatorInfo &opInfo,
                             const std::string &errMsg);

  /// \returns the value of axis given the operator info \p opInfo, the node
  /// value \p axis which stores the axis value and the node value \p value
  /// which the axis refers to which is used to wrap the axis value if negative.
  template <typename T>
  Expected<T> loadAxis(const OperatorInfo &opInfo, NodeValue axis,
                       NodeValue value);

  /// \returns the value of axes given the operator info \p opInfo, the node
  /// value \p axes which stores the axes values and the node value \p value
  /// which the axes refer to which is used to wrap the axes values if negative.
  template <typename T>
  Expected<std::vector<T>> loadAxes(const OperatorInfo &opInfo, NodeValue axes,
                                    NodeValue value);

  /// \returns the values stored in the node value \p value as a 1D array given
  /// the operator info \p opInfo. The node value \p value must be a Constant.
  template <typename T>
  Expected<std::vector<T>> loadArray(const OperatorInfo &opInfo,
                                     NodeValue value);

  /// Helper tool to verify whether the Conv2D or DepthwiseConv2D operator \p op
  /// with the operator info \p opInfo is quantized per axis. \returns true if
  /// the operator is quantized per axis and creates new graph constants by
  /// setting the pointers \p filterScalesC, \p filterOffsetsC, \p biasScalesC
  /// and \b biasOffsetsC and returns \p false otherwise.
  Expected<bool> isConv2DPerAxisQuantized(const tflite::Operator *op,
                                          const OperatorInfo &opInfo,
                                          Constant *&filterScalesC,
                                          Constant *&filterOffsetsC,
                                          Constant *&biasScalesC,
                                          Constant *&biasOffsetsC);

  /// Load the operator \p op into the current graph. \p opInfo provides meta
  /// information about \p op. \returns Error if operator cannot be loaded.
  Error loadOperator(const tflite::Operator *op, const OperatorInfo &opInfo);

  /// Load unary arithmetic operator.
  Error loadUnaryArithmetic(const tflite::Operator *op,
                            const OperatorInfo &opInfo);

  /// Load binary arithmetic operator.
  Error loadBinaryArithmetic(const tflite::Operator *op,
                             const OperatorInfo &opInfo);

  /// Load Pool2D operator (MaxPool2D or AvgPool2D).
  Error loadPool2D(const tflite::Operator *op, const OperatorInfo &opInfo);

  /// Load Concatenation operator.
  Error loadConcat(const tflite::Operator *op, const OperatorInfo &opInfo);

  /// Load Conv2D operator.
  Error loadConv2D(const tflite::Operator *op, const OperatorInfo &opInfo);

  /// Load DepthwiseConv2D operator.
  Error loadDepthwiseConv2D(const tflite::Operator *op,
                            const OperatorInfo &opInfo);

  /// Load FullyConnected operator.
  Error loadFullyConnected(const tflite::Operator *op,
                           const OperatorInfo &opInfo);

  /// Load Reshape operator.
  Error loadReshape(const tflite::Operator *op, const OperatorInfo &opInfo);

  /// Load Softmax operator.
  Error loadSoftmax(const tflite::Operator *op, const OperatorInfo &opInfo);

  /// Load Pad operator.
  Error loadPad(const tflite::Operator *op, const OperatorInfo &opInfo);

  /// Load Transpose operator.
  Error loadTranspose(const tflite::Operator *op, const OperatorInfo &opInfo);

  /// Load Reduce operator.
  Error loadReduce(const tflite::Operator *op, const OperatorInfo &opInfo);

  /// Load Split operator.
  Error loadSplit(const tflite::Operator *op, const OperatorInfo &opInfo);

  /// Load Arg operator (ArgMax or ArgMin).
  Error loadArg(const tflite::Operator *op, const OperatorInfo &opInfo);

  /// Load Slice operator.
  Error loadSlice(const tflite::Operator *op, const OperatorInfo &opInfo);

  /// Load Tile operator.
  Error loadTile(const tflite::Operator *op, const OperatorInfo &opInfo);

  /// Load Pack operator.
  Error loadPack(const tflite::Operator *op, const OperatorInfo &opInfo);

  /// Load Unpack operator.
  Error loadUnpack(const tflite::Operator *op, const OperatorInfo &opInfo);

public:
  /// \returns the TensorFlowLite model version.
  size_t getModelVersion() const { return modelVersion_; };

  /// \returns the TensorFlowLite model description.
  std::string getModelDescription() const { return modelDescription_; };

  /// Loads the TensorFlowLite model from the file \p modelFilename into the
  /// function \p F.
  TFLiteModelLoader(const std::string &modelFilename, Function *F);
};

} // namespace glow

#endif // GLOW_IMPORTER_TFLITEMODELLOADER_H
