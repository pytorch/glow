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

#ifndef GLOW_IMPORTER_COMMONOPERATORLOADER_H
#define GLOW_IMPORTER_COMMONOPERATORLOADER_H

#include "glow/Importer/ProtobufLoader.h"

#include "glow/Base/Tensor.h"
#include "glow/Graph/Graph.h"

#include "llvm/ADT/ArrayRef.h"

#include <functional>
#include <string>
#include <unordered_map>

namespace glow {

/// Contains loaders for operators, which are common to ONNX and Caffe2 formats.
/// Every loader method adds necessary nodes to property G_, which is inherited
/// from ProtobufLoader class, therefore modifying the class instance itself.
template <typename OpType, typename AttrType>
class CommonOperatorLoader : public ProtobufLoader {
protected:
  using ArgumentDictionaryTy =
      std::unordered_map<std::string, const AttrType *>;

  virtual llvm::Expected<bool> getBroadcast(const ArgumentDictionaryTy &dict) {
    return true;
  }

  void addNodeAsOutput(const OpType &op, Node *R) {
    for (int i = 0, e = op.output_size(); i < e; i++) {
      nodeValueByName_[op.output(i)] = NodeValue(R, i);
    }
  }

  /// Loads RELU operator, given its protobuf representation and parsed args.
  llvm::Error loadRelu(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    auto *R = G_.createRELU(opName, in);
    addNodeAsOutput(op, R);
    RETURN_SUCCESS();
  }

  llvm::Error loadSigmoid(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    auto *S = G_.createSigmoid(opName, in);
    addNodeAsOutput(op, S);
    RETURN_SUCCESS();
  }

  llvm::Error loadTanh(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    auto *T = G_.createTanh(opName, in);
    addNodeAsOutput(op, T);
    RETURN_SUCCESS();
  }

  llvm::Error loadShape(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));

    // This is statically known data, and so we create a Tensor for it and
    // register it in tensors_.
    auto *T = new Tensor(ElemKind::Int64ITy, {in.dims().size()});
    tensors_[opName] = T;
    T->template getHandle<int64_t>() =
        std::vector<int64_t>(in.dims().begin(), in.dims().end());

    if (auto resultOrErr = createAndRegisterConstant(opName, *T)) {
      RETURN_SUCCESS();
    } else {
      return resultOrErr.takeError();
    }
    RETURN_SUCCESS();
  }

  /// Loads Sqrt operator, given its protobuf representation and parsed args.
  llvm::Error loadSqrt(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    auto *R = G_.createPow(opName, in, 0.5f);
    addNodeAsOutput(op, R);
    RETURN_SUCCESS();
  }

  /// Loads Reciprocal operator, given its protobuf representation and parsed
  /// args.
  llvm::Error loadReciprocal(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    auto *R = G_.createPow(opName, in, -1.0f);
    addNodeAsOutput(op, R);
    RETURN_SUCCESS();
  }

  llvm::Error loadSum(const OpType &op, ArgumentDictionaryTy &dict) {
    if (op.input_size() == 1) {
      NodeValue in;
      ASSIGN_VALUE_OR_RETURN_ERR(
          in, getNodeValueOrCreateConstantByName(op.input(0)));
      addNodeAsOutput(op, in);
    } else if (op.input_size() == 2) {
      const std::string &opName = loadOperatorName(op);
      NodeValue in0;
      ASSIGN_VALUE_OR_RETURN_ERR(
          in0, getNodeValueOrCreateConstantByName(op.input(0)));
      NodeValue in1;
      ASSIGN_VALUE_OR_RETURN_ERR(
          in1, getNodeValueOrCreateConstantByName(op.input(1)));
      auto *node = G_.createAdd(opName, in0, in1);
      addNodeAsOutput(op, node);
    } else {
      const std::string &opName = loadOperatorName(op);
      const unsigned numInputs = op.input_size();
      llvm::SmallVector<NodeValue, 4> inputs;
      inputs.reserve(numInputs);
      for (unsigned i = 0; i < numInputs; i++) {
        NodeValue in;
        ASSIGN_VALUE_OR_RETURN_ERR(
            in, getNodeValueOrCreateConstantByName(op.input(i)));
        inputs.push_back(G_.createExpandDims(opName, in, {0}));
      }
      ConcatNode *concat = G_.createConcat(opName, inputs, /* axis */ 0);
      Node *node = G_.createBatchedReduceAdd(opName, concat, /* axis */ 0);
      addNodeAsOutput(op, node);
    }
    RETURN_SUCCESS();
  }

  llvm::Error loadSoftmax(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);

    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));

    // We do not do training right now on loaded protos. C2 and ONNX do not even
    // have an option for a selected input anyway. So I am creating this as a
    // placeholder which goes unused during inference.
    auto selected = G_.getParent()->createConstant(
        ElemKind::Int64ITy, {in.dims()[0], 1}, "selected");

    // ONNX allows shapes like <N x 10 x 1 x 1 >. Flatten the inputs to the
    // softmax function. This is similar to a bitcast operation.
    int axis = 1;
    if (dict.count("axis")) {
      ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(dict["axis"]));
    }

    auto *FN = G_.createFlatten("reshapeInput", in, axis);

    auto *SM = G_.createSoftMax(opName, FN, selected);

    // The output should have the same shape as the original input.
    auto origInDims = in.getType()->dims();
    auto *RN = G_.createReshape("reshapeOutput", SM, origInDims);
    addNodeAsOutput(op, RN);
    RETURN_SUCCESS();
  }

  llvm::Error loadLRN(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));

    size_t size;
    ASSIGN_VALUE_OR_RETURN_ERR(size, loadInt(dict["size"]));
    float alpha;
    ASSIGN_VALUE_OR_RETURN_ERR(alpha, loadFloat(dict["alpha"]));
    float beta;
    ASSIGN_VALUE_OR_RETURN_ERR(beta, loadFloat(dict["beta"]));
    float k;
    ASSIGN_VALUE_OR_RETURN_ERR(k, loadFloat(dict["bias"]));

    auto *tr = G_.createTranspose(opName, in, NCHW2NHWC);

    auto *node = G_.createLocalResponseNormalization(opName, tr, size / 2,
                                                     alpha, beta, k);

    auto *N = G_.createTranspose(opName, node, NHWC2NCHW);

    // LRN in Caffe2 has a scale_ output, but I believe it's unused for
    // inference. So explicitly only set output 0.
    nodeValueByName_[op.output(0)] = NodeValue(N, 0);
    RETURN_SUCCESS();
  }

  llvm::Error loadMinMax(llvm::StringRef typeName, const OpType &op,
                         ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in0;
    ASSIGN_VALUE_OR_RETURN_ERR(in0,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    NodeValue in1;
    ASSIGN_VALUE_OR_RETURN_ERR(in1,
                               getNodeValueOrCreateConstantByName(op.input(1)));

    Node *node = nullptr;
    if (typeName == "Min") {
      node = G_.createMin(opName, in0, in1);
    } else if (typeName == "Max") {
      node = G_.createMax(opName, in0, in1);
    } else {
      RETURN_ERR("Invalid min or max operator");
    }

    addNodeAsOutput(op, node);
    RETURN_SUCCESS();
  }

  llvm::Error loadBatchMatMul(const OpType &op, ArgumentDictionaryTy &dict,
                              bool isBatched) {
    const std::string &opName = loadOperatorName(op);
    NodeValue LHS;
    ASSIGN_VALUE_OR_RETURN_ERR(LHS,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    NodeValue RHS;
    ASSIGN_VALUE_OR_RETURN_ERR(RHS,
                               getNodeValueOrCreateConstantByName(op.input(1)));

    bool transLHS = false;
    if (dict.count("trans_a")) {
      int trans_a;
      ASSIGN_VALUE_OR_RETURN_ERR(trans_a, loadInt(dict["trans_a"]));
      transLHS = trans_a == 1;
    }

    (void)transLHS;
    RETURN_ERR_IF_NOT(!transLHS, "Don't support transpose lhs for now.");

    bool transRHS = false;
    if (dict.count("trans_b")) {
      int trans_b;
      ASSIGN_VALUE_OR_RETURN_ERR(trans_b, loadInt(dict["trans_b"]));
      transRHS = trans_b == 1;
    }

    if (transRHS) {
      // The semantic of the transpose in that context is:
      // swap the last two dimensions.
      unsigned_t nbDims = RHS.dims().size();
      RETURN_ERR_IF_NOT(nbDims >= 2, "C2 specs say rank of RHS must be >= 2");
      std::vector<unsigned_t> shuffle;
      unsigned_t i;
      for (i = 0; i < nbDims - 2; ++i) {
        shuffle.push_back(i);
      }
      shuffle.push_back(i + 1);
      shuffle.push_back(i);
      RHS = G_.createTranspose("RHS.transpose", RHS, shuffle);
    }

    Node *node = nullptr;

    // BatchMatMul sometimes is actually just a matmul, depending on dimensions
    // of inputs. Thus, only do batch matmul if LHS is 3-dimensional.
    if (isBatched && LHS.dims().size() == 3) {
      // BatchMatMul can be either multiplication of K matrices and another
      // K matrices, or broadcasted multiplication of K matrices and one other
      // matrix.
      if (RHS.dims().size() == 3) {
        node = G_.createParallelBatchMatMul(opName, LHS, RHS);
      } else {
        node = G_.createBroadcastedBatchMatMul(opName, LHS, RHS);
      }
    } else {
      node = G_.createMatMul(opName, LHS, RHS);
    }

    addNodeAsOutput(op, node);
    RETURN_SUCCESS();
  }

  llvm::Error loadArithmetic(llvm::StringRef typeName, const OpType &op,
                             ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in0;
    ASSIGN_VALUE_OR_RETURN_ERR(in0,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    NodeValue in1;
    ASSIGN_VALUE_OR_RETURN_ERR(in1,
                               getNodeValueOrCreateConstantByName(op.input(1)));

    bool broadcast;
    ASSIGN_VALUE_OR_RETURN_ERR(broadcast, getBroadcast(dict));

    Node *finalIn1 = nullptr;
    if (broadcast) {
      int axis = -1;
      if (dict.count("axis")) {
        ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(dict["axis"]));
      }

      // In ONNX, if axis == -1 then it sets the axis so that the
      // trailing-most dimensions are aligned like this.
      if (axis == -1) {
        axis = in0.dims().size() - in1.dims().size();
      }
      finalIn1 = G_.createBroadcast(opName, in1, in0.dims(), axis);
    } else {
      finalIn1 = in1;
    }

    Node *node = nullptr;
    if (typeName == "Mul") {
      node = G_.createMul(opName, in0, finalIn1);
    } else if (typeName == "Add") {
      node = G_.createAdd(opName, in0, finalIn1);
    } else if (typeName == "Sub") {
      node = G_.createSub(opName, in0, finalIn1);
    } else if (typeName == "Div") {
      node = G_.createDiv(opName, in0, finalIn1);
    } else {
      RETURN_ERR("Unsupported arithmetic typeName");
    }

    addNodeAsOutput(op, node);
    RETURN_SUCCESS();
  }

  llvm::Error loadSplit(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    size_t axis = 0;
    if (dict.count("axis")) {
      ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(dict["axis"]));
    }

    std::vector<size_t> split;
    if (dict.count("split"))
      split = getShape(dict["split"]);

    std::vector<Node *> outputs;
    G_.createSplit(opName, in, op.output_size(), axis, split, outputs);

    for (int i = 0, e = op.output_size(); i < e; i++) {
      // Each output from Split is a SliceNode which only has a single output,
      // so only use 0 here as the node value result.
      nodeValueByName_[op.output(i)] = NodeValue(outputs[i], 0);
    }
    RETURN_SUCCESS();
  }

  llvm::Expected<bool> loadReshape(const OpType &op,
                                   ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));

    // Get the requested shape from the model.
    // First look at input tensors, then at the "shape" attribute.
    std::vector<int64_t> requestedDims;
    if (op.input_size() > 1) {
      // Non-constant shape tensors are unsupported by Glow.
      if (!tensors_.count(op.input(1)))
        return false;
      Tensor *constShapeTensor;
      ASSIGN_VALUE_OR_RETURN_ERR(constShapeTensor,
                                 getTensorByName(op.input(1)));
      auto TH = constShapeTensor->getHandle<int64_t>();
      for (size_t i = 0, e = constShapeTensor->size(); i != e; i++) {
        requestedDims.push_back(TH.at({i}));
      }
    } else if (dict.count("shape")) {
      RETURN_ERR_IF_NOT(op.input_size() == 1,
                        "Cannot specify new shape by both argument and input.");
      std::vector<int64_t> protoDims = getShape<int64_t>(dict["shape"]);
      for (size_t i = 0, e = protoDims.size(); i != e; i++) {
        requestedDims.push_back(protoDims[i]);
      }
    } else {
      RETURN_ERR("Missing output shape information for the Reshape operator.");
    }

    // Compute the actual new shape
    ssize_t negOneIndex = -1;
    llvm::ArrayRef<size_t> inputDims = in.dims();
    std::vector<size_t> outputDims;
    int64_t dimProduct = 1;
    for (size_t i = 0, e = requestedDims.size(); i != e; i++) {
      int64_t newDim = requestedDims[i];
      if (newDim == 0) {
        // 0 means that corresponding input dimension should be propagated to
        // the output.
        newDim = inputDims[i];
      }
      if (newDim != -1) {
        dimProduct *= newDim;
        outputDims.push_back(newDim);
      } else {
        // -1 means that the corresponding dimension should be inferred
        // from all other dimensions, so that tensor size remains the same.
        RETURN_ERR_IF_NOT(negOneIndex < 0,
                          "At most one dimension of the new shape can be -1.");
        negOneIndex = (ssize_t)i;
        // The -1 case value is handled later.
        outputDims.push_back(0);
      }
    }
    if (negOneIndex >= 0) {
      outputDims[negOneIndex] = in.getType()->size() / dimProduct;
    }

    auto *node = G_.createReshape(opName, in, outputDims);

    // Caffe2 sometimes outputs old_shape which goes unused. We do not currently
    // support it, so explicitly only set the first output.
    nodeValueByName_[op.output(0)] = NodeValue(node, 0);
    return true;
  }

  llvm::Error loadTranspose(const OpType &op, ArgumentDictionaryTy &dict,
                            llvm::StringRef permArgName) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));

    // There is a difference between ONNX and Caffe2 specs for Transpose:
    // one contains permutation under name "perm", the other contains it under
    // argument name "axes". That's why the name is passed as a parameter.
    std::vector<unsigned_t> perm = getShape<unsigned_t>(dict[permArgName]);
    if (perm.empty()) {
      // Empty permutation argument means reversing axes order.
      size_t N = in.dims().size();
      for (int64_t i = N - 1; i >= 0; i--)
        perm.push_back(i);
    }

    auto *T = G_.createTranspose(opName, in, perm);

    addNodeAsOutput(op, T);
    RETURN_SUCCESS();
  }

  llvm::Error loadFlatten(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    int axis = 1;
    if (dict.count("axis")) {
      ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(dict["axis"]));
    }
    auto *node = G_.createFlatten(opName, in, axis);
    addNodeAsOutput(op, node);
    RETURN_SUCCESS();
  }

  llvm::Error loadIdentity(const OpType &op, ArgumentDictionaryTy &dict) {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    nodeValueByName_[op.output(0)] = NodeValue(in, 0);
    RETURN_SUCCESS();
  }

  llvm::Error loadTopK(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    unsigned_t k;
    ASSIGN_VALUE_OR_RETURN_ERR(k, loadInt(dict["k"]));

    int axis = -1;
    if (dict.count("axis")) {
      ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(dict["axis"]));
    }
    int lastDim = in.dims().size() - 1;
    if (axis == -1) {
      axis = lastDim;
    }

    RETURN_ERR_IF_NOT(axis == lastDim,
                      "Currently only support axis being last dimension.");

    auto *R = G_.createTopK(opName, in, k);
    addNodeAsOutput(op, R);
    RETURN_SUCCESS();
  }

  llvm::Error loadReduceMeanOrSum(llvm::StringRef typeName, const OpType &op,
                                  ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    auto axes = getShape(dict["axes"]);
    RETURN_ERR_IF_NOT(axes.size() == 1,
                      "Only supporting single reduction for now.");
    auto axis = axes[0];

    Node *node = nullptr;

    if (typeName == "ReduceMean") {
      node = G_.createBatchedReduceMean(opName, in, axis);
    } else {
      node = G_.createBatchedReduceAdd(opName, in, axis);
    }

    bool keepDims = true;
    if (dict.count("keepdims")) {
      int keepdims;
      ASSIGN_VALUE_OR_RETURN_ERR(keepdims, loadInt(dict["keepdims"]));
      keepDims = (bool)keepdims;
    }

    // Our batched reduce add does not keep the dim; reshape if necessary.
    if (keepDims) {
      std::vector<size_t> shape = node->dims(0);
      shape.insert(shape.begin() + axis, 1);
      node = G_.createReshape(opName, node, shape);
    }

    addNodeAsOutput(op, node);
    RETURN_SUCCESS();
  }

  llvm::Error loadBatchOneHot(const OpType &op) {
    const std::string &opName = loadOperatorName(op);
    NodeValue data;
    ASSIGN_VALUE_OR_RETURN_ERR(data,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    NodeValue lengths;
    ASSIGN_VALUE_OR_RETURN_ERR(lengths,
                               getNodeValueOrCreateConstantByName(op.input(1)));
    NodeValue values;
    ASSIGN_VALUE_OR_RETURN_ERR(values,
                               getNodeValueOrCreateConstantByName(op.input(2)));

    auto *node = G_.createBatchOneHot(opName, data, lengths, values);
    addNodeAsOutput(op, node);
    RETURN_SUCCESS();
  }

  llvm::Error loadSparseLengthsSum(const OpType &op) {
    NodeValue in0;
    ASSIGN_VALUE_OR_RETURN_ERR(in0,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    NodeValue in1;
    ASSIGN_VALUE_OR_RETURN_ERR(in1,
                               getNodeValueOrCreateConstantByName(op.input(1)));
    NodeValue in2;
    ASSIGN_VALUE_OR_RETURN_ERR(in2,
                               getNodeValueOrCreateConstantByName(op.input(2)));
    auto *node = G_.createSparseLengthsSum(loadOperatorName(op), in0, in1, in2);
    addNodeAsOutput(op, node);
    RETURN_SUCCESS();
  }

  llvm::Error loadSparseLengthsWeightedSum(const OpType &op) {
    NodeValue in0;
    ASSIGN_VALUE_OR_RETURN_ERR(in0,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    NodeValue in1;
    ASSIGN_VALUE_OR_RETURN_ERR(in1,
                               getNodeValueOrCreateConstantByName(op.input(1)));
    NodeValue in2;
    ASSIGN_VALUE_OR_RETURN_ERR(in2,
                               getNodeValueOrCreateConstantByName(op.input(2)));
    NodeValue in3;
    ASSIGN_VALUE_OR_RETURN_ERR(in3,
                               getNodeValueOrCreateConstantByName(op.input(3)));
    auto *node = G_.createSparseLengthsWeightedSum(loadOperatorName(op), in0,
                                                   in1, in2, in3);
    addNodeAsOutput(op, node);
    RETURN_SUCCESS();
  }

  llvm::Error loadLengthsToRanges(const OpType &op) {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    auto *node = G_.createLengthsToRanges(loadOperatorName(op), in);
    addNodeAsOutput(op, node);
    RETURN_SUCCESS();
  }

  llvm::Error loadBatchBoxCox(const OpType &op) {
    NodeValue data;
    ASSIGN_VALUE_OR_RETURN_ERR(data,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    NodeValue lambda1;
    ASSIGN_VALUE_OR_RETURN_ERR(lambda1,
                               getNodeValueOrCreateConstantByName(op.input(1)));
    NodeValue lambda2;
    ASSIGN_VALUE_OR_RETURN_ERR(lambda2,
                               getNodeValueOrCreateConstantByName(op.input(2)));
    auto *node =
        G_.createBatchBoxCox(loadOperatorName(op), data, lambda1, lambda2);
    addNodeAsOutput(op, node);
    RETURN_SUCCESS();
  }

  llvm::Error loadDotProduct(const OpType &op) {
    NodeValue X;
    ASSIGN_VALUE_OR_RETURN_ERR(X,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    NodeValue Y;
    ASSIGN_VALUE_OR_RETURN_ERR(Y,
                               getNodeValueOrCreateConstantByName(op.input(1)));
    auto *node = G_.createDotProduct(loadOperatorName(op), X, Y);
    addNodeAsOutput(op, node);
    RETURN_SUCCESS();
  }

  llvm::Error loadReplaceNaN(const OpType &op,
                             const ArgumentDictionaryTy &dict) {
    // Load the input and NaN replacement value:
    NodeValue input;
    ASSIGN_VALUE_OR_RETURN_ERR(input,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    auto valueIt = dict.find("value");
    float value = 0.0f;
    if (valueIt != dict.end()) {
      ASSIGN_VALUE_OR_RETURN_ERR(value, loadFloat(valueIt->second));
    }
    auto *node = G_.createReplaceNaN(loadOperatorName(op), input, value);
    addNodeAsOutput(op, node);
    RETURN_SUCCESS();
  }

  llvm::Error loadLengthsSum(const OpType &op) {
    const std::string &opName = loadOperatorName(op);
    NodeValue data;
    ASSIGN_VALUE_OR_RETURN_ERR(data,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    NodeValue lengths;
    ASSIGN_VALUE_OR_RETURN_ERR(lengths,
                               getNodeValueOrCreateConstantByName(op.input(1)));

    RETURN_ERR_IF_NOT(lengths.dims().size() == 1,
                      "Lengths must be a 1D vector.");

    auto *node = G_.createLengthsSum(opName, data, lengths);
    addNodeAsOutput(op, node);
    RETURN_SUCCESS();
  }

  llvm::Expected<bool> loadExpandDims(const OpType &op,
                                      const ArgumentDictionaryTy &dict) {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    auto dims = dict.find("dims");
    if (dims == dict.end()) {
      return false;
    }
    Node *node =
        G_.createExpandDims(loadOperatorName(op), in, getShape(dims->second));
    addNodeAsOutput(op, node);

    return true;
  }

  llvm::Error loadClip(const OpType &op, const ArgumentDictionaryTy &dict) {
    NodeValue in;
    ASSIGN_VALUE_OR_RETURN_ERR(in,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    float cmin = std::numeric_limits<float>::lowest();
    if (dict.count("min")) {
      ASSIGN_VALUE_OR_RETURN_ERR(cmin, loadFloat(dict.find("min")->second));
    }

    float cmax = std::numeric_limits<float>::max();
    if (dict.count("max")) {
      ASSIGN_VALUE_OR_RETURN_ERR(cmax, loadFloat(dict.find("max")->second));
    }

    auto *node = G_.createClip(loadOperatorName(op), in, cmin, cmax);
    addNodeAsOutput(op, node);
    RETURN_SUCCESS();
  }

  llvm::Expected<bool> loadSparseToDense(const OpType &op,
                                         const ArgumentDictionaryTy &dict) {
    if (op.input_size() != 3) {
      return false;
    }

    NodeValue indices;
    ASSIGN_VALUE_OR_RETURN_ERR(indices,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    NodeValue values;
    ASSIGN_VALUE_OR_RETURN_ERR(values,
                               getNodeValueOrCreateConstantByName(op.input(1)));
    NodeValue dataToInferDim;
    ASSIGN_VALUE_OR_RETURN_ERR(dataToInferDim,
                               getNodeValueOrCreateConstantByName(op.input(2)));

    auto *node = G_.createSparseToDense(loadOperatorName(op), indices, values,
                                        dataToInferDim);
    addNodeAsOutput(op, node);
    return true;
  }

  llvm::Expected<bool> loadGatherOps(const std::string &typeName,
                                     const OpType &op,
                                     const ArgumentDictionaryTy &dict) {
    if (typeName != "Gather" && typeName != "BatchGather") {
      return false;
    }

    NodeValue data;
    ASSIGN_VALUE_OR_RETURN_ERR(data,
                               getNodeValueOrCreateConstantByName(op.input(0)));
    NodeValue indices;
    ASSIGN_VALUE_OR_RETURN_ERR(indices,
                               getNodeValueOrCreateConstantByName(op.input(1)));
    size_t batchDims = typeName == "Gather" ? 0 : 1;

    if (dict.count("axis")) {
      int axis;
      ASSIGN_VALUE_OR_RETURN_ERR(axis, loadInt(dict.find("axis")->second));
      if (axis != 0 && axis != 1) {
        return false;
      }

      batchDims = axis;
    }

    Node *GN = G_.createGather(loadOperatorName(op), data, indices, batchDims);
    addNodeAsOutput(op, GN);
    return true;
  }

  using ProtobufLoader::ProtobufLoader;

  /// If operator type is supported, returns Expected<true> and creates new
  /// operator. Returns Operator<false> if operator type is not supported.
  /// Returns Error if an error occurred
  llvm::Expected<bool> tryLoadCommonOperator(llvm::StringRef typeName,
                                             const OpType &op,
                                             ArgumentDictionaryTy &dict) {
    if (typeName == "Relu") {
      RETURN_IF_ERR(loadRelu(op, dict));
      return true;
    }
    if (typeName == "Sigmoid") {
      RETURN_IF_ERR(loadSigmoid(op, dict));
      return true;
    }
    if (typeName == "Tanh") {
      RETURN_IF_ERR(loadTanh(op, dict));
      return true;
    }
    if (typeName == "Shape") {
      RETURN_IF_ERR(loadShape(op, dict));
      return true;
    }
    if (typeName == "Sqrt") {
      RETURN_IF_ERR(loadSqrt(op, dict));
      return true;
    }
    if (typeName == "Reciprocal") {
      RETURN_IF_ERR(loadReciprocal(op, dict));
      return true;
    }
    if (typeName == "Sum") {
      RETURN_IF_ERR(loadSum(op, dict));
      return true;
    }
    if (typeName == "Softmax") {
      RETURN_IF_ERR(loadSoftmax(op, dict));
      return true;
    }
    if (typeName == "LRN") {
      RETURN_IF_ERR(loadLRN(op, dict));
      return true;
    }
    if (typeName == "Min" || typeName == "Max") {
      RETURN_IF_ERR(loadMinMax(typeName, op, dict));
      return true;
    }
    if (typeName == "Mul" || typeName == "Add" || typeName == "Sub" ||
        typeName == "Div") {
      RETURN_IF_ERR(loadArithmetic(typeName, op, dict));
      return true;
    }
    if (typeName == "Split") {
      RETURN_IF_ERR(loadSplit(op, dict));
      return true;
    }
    if (typeName == "Reshape") {
      return loadReshape(op, dict);
    }
    if (typeName == "Flatten") {
      RETURN_IF_ERR(loadFlatten(op, dict));
      return true;
    }
    if (typeName == "Dropout") {
      // Save the identity operation. Note the second output (mask) for Dropout
      // in Caffe2 and ONNX is unused during inference, and our Dropout does not
      // currently implement it, thus we do not call addNodeAsOutput() here.
      RETURN_IF_ERR(loadIdentity(op, dict));
      return true;
    }
    if (typeName == "Identity") {
      RETURN_IF_ERR(loadIdentity(op, dict));
      return true;
    }
    if (typeName == "TopK") {
      RETURN_IF_ERR(loadTopK(op, dict));
      return true;
    }
    if (typeName == "ReduceMean" || typeName == "ReduceSum") {
      RETURN_IF_ERR(loadReduceMeanOrSum(typeName, op, dict));
      return true;
    }
    if (typeName == "BatchMatMul") {
      RETURN_IF_ERR(loadBatchMatMul(op, dict, true));
      return true;
    }
    if (typeName == "BatchOneHot") {
      RETURN_IF_ERR(loadBatchOneHot(op));
      return true;
    }
    if (typeName == "SparseLengthsSum") {
      RETURN_IF_ERR(loadSparseLengthsSum(op));
      return true;
    }
    if (typeName == "SparseLengthsWeightedSum") {
      RETURN_IF_ERR(loadSparseLengthsWeightedSum(op));
      return true;
    }
    if (typeName == "LengthsToRanges") {
      RETURN_IF_ERR(loadLengthsToRanges(op));
      return true;
    }
    if (typeName == "BatchBoxCox") {
      RETURN_IF_ERR(loadBatchBoxCox(op));
      return true;
    }
    if (typeName == "DotProduct") {
      RETURN_IF_ERR(loadDotProduct(op));
      return true;
    }
    if (typeName == "ReplaceNaN") {
      RETURN_IF_ERR(loadReplaceNaN(op, dict));
      return true;
    }
    if (typeName == "LengthsSum") {
      RETURN_IF_ERR(loadLengthsSum(op));
      return true;
    }
    if (typeName == "ExpandDims") {
      return loadExpandDims(op, dict);
    }

    if (typeName == "Clip") {
      RETURN_IF_ERR(loadClip(op, dict));
      return true;
    }

    if (typeName == "SparseToDense") {
      return loadSparseToDense(op, dict);
    }

    if (typeName == "Gather" || typeName == "BatchGather") {
      return loadGatherOps(typeName, op, dict);
    }

    return false;
  }
};

} // namespace glow

#endif // GLOW_IMPORTER_COMMONOPERATORLOADER_H
