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

  virtual bool getBroadcast(const ArgumentDictionaryTy &dict) { return true; }

  void addNodeAsOutput(const OpType &op, Node *R) {
    for (int i = 0, e = op.output_size(); i < e; i++) {
      nodeValueByName_[op.output(i)] = NodeValue(R, i);
    }
  }

  /// Loads RELU operator, given its protobuf representation and parsed args.
  void loadRelu(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    auto in = getNodeValueOrCreateVariableByName(op.input(0));
    auto *R = G_.createRELU(opName, in);
    addNodeAsOutput(op, R);
  }

  void loadSigmoid(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    auto in = getNodeValueOrCreateVariableByName(op.input(0));
    auto *S = G_.createSigmoid(opName, in);
    addNodeAsOutput(op, S);
  }

  void loadTanh(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    auto in = getNodeValueOrCreateVariableByName(op.input(0));
    auto *T = G_.createTanh(opName, in);
    addNodeAsOutput(op, T);
  }

  void loadShape(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    auto in = getNodeValueOrCreateVariableByName(op.input(0));

    // This is statically known data, and so we create a Tensor for it and
    // register it in tensors_.
    auto *T = new Tensor(ElemKind::IndexTy, {in.dims().size()});
    tensors_[opName] = T;
    T->template getHandle<size_t>() = in.dims();

    createAndRememberVariable(opName, *T, VisibilityKind::Private, false);
  }

  /// Loads Sqrt operator, given its protobuf representation and parsed args.
  void loadSqrt(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    auto in = getNodeValueOrCreateVariableByName(op.input(0));
    auto *R = G_.createPow(opName, in, 0.5f);
    addNodeAsOutput(op, R);
  }

  /// Loads Reciprocal operator, given its protobuf representation and parsed
  /// args.
  void loadReciprocal(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    auto in = getNodeValueOrCreateVariableByName(op.input(0));
    auto *R = G_.createPow(opName, in, -1.0f);
    addNodeAsOutput(op, R);
  }

  void loadSum(const OpType &op, ArgumentDictionaryTy &dict) {
    // TODO: support variadic arguments
    assert(op.input_size() == 2 && "Only Sum of 2 inputs is supported.");
    const std::string &opName = loadOperatorName(op);
    auto in0 = getNodeValueOrCreateVariableByName(op.input(0));
    auto in1 = getNodeValueOrCreateVariableByName(op.input(1));
    auto *node = G_.createAdd(opName, in0, in1);
    addNodeAsOutput(op, node);
  }

  void loadSoftmax(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);

    auto in = getNodeValueOrCreateVariableByName(op.input(0));

    // We do not do training right now on loaded protos. C2 and ONNX do not even
    // have an option for a selected input anyway. So I am creating this as a
    // placeholder which goes unused during inference.
    auto selected = G_.getParent()->createVariable(
        ElemKind::IndexTy, {in->dims(0)[0], 1}, "selected",
        VisibilityKind::Private, false);

    // ONNX allows shapes like <N x 10 x 1 x 1 >. Flatten the inputs to the
    // softmax function. This is similar to a bitcast operation.
    int axis = dict.count("axis") ? loadInt(dict["axis"]) : 1;
    auto *FN = G_.createFlatten("reshapeInput", in, axis);

    auto *SM = G_.createSoftMax(opName, FN, selected);

    // The output should have the same shape as the original input.
    auto origInDims = in.getType()->dims();
    auto *RN = G_.createReshape("reshapeOutput", SM, origInDims);
    addNodeAsOutput(op, RN);
  }

  void loadLRN(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    auto in = getNodeValueOrCreateVariableByName(op.input(0));

    size_t size = loadInt(dict["size"]);
    float alpha = loadFloat(dict["alpha"]);
    float beta = loadFloat(dict["beta"]);
    float k = loadFloat(dict["bias"]);

    auto *tr = G_.createTranspose(opName, in, NCHW2NHWC);

    auto *node = G_.createLocalResponseNormalization(opName, tr, size / 2,
                                                     alpha, beta, k);

    auto *N = G_.createTranspose(opName, node, NHWC2NCHW);

    // LRN in Caffe2 has a scale_ output, but I believe it's unused for
    // inference. So explicitly only set output 0.
    nodeValueByName_[op.output(0)] = NodeValue(N, 0);
  }

  void loadMinMax(llvm::StringRef typeName, const OpType &op,
                  ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    auto in0 = getNodeValueOrCreateVariableByName(op.input(0));
    auto in1 = getNodeValueOrCreateVariableByName(op.input(1));

    Node *node = nullptr;
    if (typeName == "Min") {
      node = G_.createMin(opName, in0, in1);
    } else if (typeName == "Max") {
      node = G_.createMax(opName, in0, in1);
    } else {
      llvm_unreachable("Invalid min or max operator");
    }

    addNodeAsOutput(op, node);
  }

  void loadArithmetic(llvm::StringRef typeName, const OpType &op,
                      ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    auto in0 = getNodeValueOrCreateVariableByName(op.input(0));
    auto in1 = getNodeValueOrCreateVariableByName(op.input(1));

    bool broadcast = getBroadcast(dict);

    Node *finalIn1 = nullptr;
    if (broadcast) {
      int axis = dict.count("axis") ? loadInt(dict["axis"]) : -1;
      // In ONNX, if axis == -1 then it sets the axis so that the
      // trailing-most dimensions are aligned like this.
      if (axis == -1) {
        axis = in0->dims(0).size() - in1->dims(0).size();
      }
      finalIn1 = G_.createBroadcast(opName, in1, in0->dims(0), axis);
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
      llvm_unreachable("Unsupported arithmetic typeName");
    }

    addNodeAsOutput(op, node);
  }

  void loadSplit(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    auto in = getNodeValueOrCreateVariableByName(op.input(0));
    size_t axis = dict.count("axis") ? loadInt(dict["axis"]) : 0;
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
  }

  void loadReshape(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    NodeValue in = getNodeValueOrCreateVariableByName(op.input(0));

    // Get the requested shape from the model.
    // First look at input tensors, then at the "shape" attribute.
    std::vector<int64_t> requestedDims;
    if (op.input_size() > 1) {
      Tensor *constShapeTensor = getTensorByName(op.input(1));
      auto TH = constShapeTensor->getHandle<size_t>();
      for (size_t i = 0, e = constShapeTensor->size(); i != e; i++) {
        requestedDims.push_back(TH.at({i}));
      }
    } else if (dict.count("shape")) {
      assert(op.input_size() == 1 &&
             "Cannot specify new shape by both argument and input.");
      std::vector<int64_t> protoDims = getShape<int64_t>(dict["shape"]);
      for (size_t i = 0, e = protoDims.size(); i != e; i++) {
        requestedDims.push_back(protoDims[i]);
      }
    } else {
      llvm_unreachable(
          "Missing output shape information for the Reshape operator.");
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
        assert(negOneIndex < 0 &&
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
  }

  void loadTranspose(const OpType &op, ArgumentDictionaryTy &dict,
                     llvm::StringRef permArgName) {
    const std::string &opName = loadOperatorName(op);
    auto in = getNodeValueOrCreateVariableByName(op.input(0));

    // There is a difference between ONNX and Caffe2 specs for Transpose:
    // one contains permutation under name "perm", the other contains it under
    // argument name "axes". That's why the name is passed as a parameter.
    std::vector<unsigned> perm = getShape<unsigned>(dict[permArgName]);
    if (perm.empty()) {
      // Empty permutation argument means reversing axes order.
      size_t N = in->dims(0).size();
      for (int64_t i = N - 1; i >= 0; i--)
        perm.push_back(i);
    }

    auto *T = G_.createTranspose(opName, in, perm);

    addNodeAsOutput(op, T);
  }

  void loadFlatten(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    auto in = getNodeValueOrCreateVariableByName(op.input(0));
    int axis = dict.count("axis") ? loadInt(dict["axis"]) : 1;
    auto *node = G_.createFlatten(opName, in, axis);
    addNodeAsOutput(op, node);
  }

  void loadDropout(const OpType &op, ArgumentDictionaryTy &dict) {
    auto in = getNodeValueOrCreateVariableByName(op.input(0));
    // Save the identity operation. Note the second output (mask) for Dropout in
    // Caffe2 and ONNX is unused during inference, and our Dropout does not
    // currently implement it, thus we do not call addNodeAsOutput() here.
    nodeValueByName_[op.output(0)] = NodeValue(in, 0);
  }

  void loadTopK(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    auto in = getNodeValueOrCreateVariableByName(op.input(0));
    auto k = loadInt(dict["k"]);

    int axis = dict.count("axis") ? loadInt(dict["axis"]) : -1;
    unsigned lastDim = in->dims(0).size() - 1;
    if (axis == -1) {
      axis = lastDim;
    }

    assert(axis == lastDim &&
           "Currently only support axis being last dimension.");

    auto *R = G_.createTopK(opName, in, k);
    addNodeAsOutput(op, R);
  }

  void loadReduceMeanOrSum(llvm::StringRef typeName, const OpType &op,
                           ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    auto in = getNodeValueOrCreateVariableByName(op.input(0));
    auto axes = getShape(dict["axes"]);
    assert(axes.size() == 1 && "Only supporting single reduction for now.");
    auto axis = axes[0];

    Node *node = nullptr;

    if (typeName == "ReduceMean") {
      node = G_.createBatchedReduceMean(opName, in, axis);
    } else {
      node = G_.createBatchedReduceAdd(opName, in, axis);
    }

    bool keepDims = dict.count("keepdims") ? loadInt(dict["keepdims"]) : true;

    // Our batched reduce add does not keep the dim; reshape if necessary.
    if (keepDims) {
      std::vector<size_t> shape = node->dims(0);
      shape.insert(shape.begin() + axis, 1);
      node = G_.createReshape(opName, node, shape);
    }

    addNodeAsOutput(op, node);
  }

  using ProtobufLoader::ProtobufLoader;

  /// If operator type is supported, returns true and creates new operator.
  /// Otherwise returns false.
  bool tryLoadCommonOperator(llvm::StringRef typeName, const OpType &op,
                             ArgumentDictionaryTy &dict) {
    if (typeName == "Relu") {
      loadRelu(op, dict);
      return true;
    }
    if (typeName == "Sigmoid") {
      loadSigmoid(op, dict);
      return true;
    }
    if (typeName == "Tanh") {
      loadTanh(op, dict);
      return true;
    }
    if (typeName == "Shape") {
      loadShape(op, dict);
      return true;
    }
    if (typeName == "Sqrt") {
      loadSqrt(op, dict);
      return true;
    }
    if (typeName == "Reciprocal") {
      loadReciprocal(op, dict);
      return true;
    }
    if (typeName == "Sum") {
      loadSum(op, dict);
      return true;
    }
    if (typeName == "Softmax") {
      loadSoftmax(op, dict);
      return true;
    }
    if (typeName == "LRN") {
      loadLRN(op, dict);
      return true;
    }
    if (typeName == "Min" || typeName == "Max") {
      loadMinMax(typeName, op, dict);
      return true;
    }
    if (typeName == "Mul" || typeName == "Add" || typeName == "Sub" ||
        typeName == "Div") {
      loadArithmetic(typeName, op, dict);
      return true;
    }
    if (typeName == "Split") {
      loadSplit(op, dict);
      return true;
    }
    if (typeName == "Reshape") {
      loadReshape(op, dict);
      return true;
    }
    if (typeName == "Flatten") {
      loadFlatten(op, dict);
      return true;
    }
    if (typeName == "Dropout") {
      loadDropout(op, dict);
      return true;
    }
    if (typeName == "TopK") {
      loadTopK(op, dict);
      return true;
    }
    if (typeName == "ReduceMean" || typeName == "ReduceSum") {
      loadReduceMeanOrSum(typeName, op, dict);
      return true;
    }
    return false;
  }
};

} // namespace glow

#endif // GLOW_IMPORTER_COMMONOPERATORLOADER_H
