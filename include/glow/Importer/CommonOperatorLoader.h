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

  void addNodeAsOutput(const OpType &op, Node *R) {
    for (int i = 0, e = op.output_size(); i < e; i++) {
      nodeByName_[op.output(i)] = R;
    }
  }

  /// Loads RELU operator, given its protobuf representation and parsed args.
  void loadRelu(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    auto *in = getOrCreateNodeByName(op.input(0));
    auto *R = G_.createRELU(opName, in);
    addNodeAsOutput(op, R);
  }

  void loadSum(const OpType &op, ArgumentDictionaryTy &dict) {
    // TODO: support variadic arguments
    assert(op.input_size() == 2 && "Only Sum of 2 inputs is supported.");
    const std::string &opName = loadOperatorName(op);
    auto *in0 = getOrCreateNodeByName(op.input(0));
    auto *in1 = getOrCreateNodeByName(op.input(1));
    auto *node = G_.createAdd(opName, in0, in1);
    addNodeAsOutput(op, node);
  }

  void loadSoftmax(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);

    auto *softmaxExpected = getOrCreateNodeByName("softmax_expected");

    Node *in = getOrCreateNodeByName(op.input(0));

    // ONNX allows shapes like <N x 10 x 1 x 1 >. Flatten the inputs to the
    // softmax function. This is similar to a bitcast operation.
    auto flatten = flattenCdr(in->getType()->dims());
    in = G_.createReshape("reshape", in, {flatten.first, flatten.second});

    auto *node = G_.createSoftMax(opName, in, softmaxExpected);
    addNodeAsOutput(op, node);
  }

  void loadFC(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    auto *in = getOrCreateNodeByName(op.input(0));
    // Load weights.
    Tensor *w = getTensorByName(op.input(1));
    Tensor *b = getTensorByName(op.input(2));

    // ONNX stores the transposed W matrix. In here we transpose W back.
    Tensor wtag;
    w->transpose(&wtag, {1, 0});

    auto W = G_.getParent()->addVar(
        new Variable("weights", VisibilityKind::Private, std::move(wtag)));
    auto B = G_.getParent()->addVar(
        new Variable("biases", VisibilityKind::Private, std::move(*b)));
    auto *FC = G_.createFullyConnected(opName, in, W, B);

    addNodeAsOutput(op, FC);
  }

  void loadLRN(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    auto *in = getOrCreateNodeByName(op.input(0));

    size_t size = loadInt(dict["size"]);
    float alpha = loadFloat(dict["alpha"]);
    float beta = loadFloat(dict["beta"]);
    float k = loadFloat(dict["bias"]);

    auto *tr = G_.createTranspose(opName, in, NCHW2NHWC);

    auto *node = G_.createLocalResponseNormalization(opName, tr, size / 2,
                                                     alpha, beta, k);

    auto *N = G_.createTranspose(opName, node, NHWC2NCHW);

    addNodeAsOutput(op, N);
  }

  void loadArithmetic(llvm::StringRef typeName, const OpType &op,
                      ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    auto *in0 = getOrCreateNodeByName(op.input(0));
    auto *in1 = getOrCreateNodeByName(op.input(1));

    int broadcast = loadInt(dict["broadcast"]);

    Node *finalIn1 = nullptr;
    if (broadcast == 1) {
      int axis = loadInt(dict["axis"]);
      // In ONNX, if axis == -1 then it sets the axis so that the
      // trailing-most dimensions are aligned like this.
      if (axis == -1) {
        axis = in0->dims().size() - in1->dims().size();
      }
      finalIn1 = G_.createBroadcast(opName, in1, in0->dims(), axis);
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
      assert(false && "Unsupported arithmetic typeName");
    }

    addNodeAsOutput(op, node);
  }

  void loadSplit(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    auto *in = getOrCreateNodeByName(op.input(0));
    size_t axis = dict.count("axis") ? loadInt(dict["axis"]) : 0;
    std::vector<size_t> split;
    if (dict.count("split"))
      split = getShape(dict["split"]);

    std::vector<Node *> outputs;
    G_.createSplit(opName, in, op.output_size(), axis, split, outputs);

    for (int i = 0, e = op.output_size(); i < e; i++) {
      nodeByName_[op.output(i)] = outputs[i];
    }
  }

  void loadReshape(const OpType &op, ArgumentDictionaryTy &dict) {
    const std::string &opName = loadOperatorName(op);
    auto *in = getOrCreateNodeByName(op.input(0));

    std::vector<size_t> newDim;
    if (dict.count("shape")) {
      std::vector<int64_t> protoDims = getShape<int64_t>(dict["shape"]);

      auto oldDim = in->dims();
      int64_t product = 1;
      for (size_t i = 0, e = protoDims.size(); i != e; i++) {
        if (protoDims[i] == 0)
          // shape[i] == 0 means that corresponding element should remain
          // the same.
          protoDims[i] = oldDim[i];
        if (protoDims[i] != -1)
          product *= protoDims[i];
      }
      for (size_t i = 0, e = protoDims.size(); i != e; i++) {
        if (protoDims[i] == -1)
          // shape[i] == -1 means that corresponding element should be inferred
          // from all other elements, so that Tensor size remains the same.
          protoDims[i] = in->getType()->size() / product;
        newDim.push_back(protoDims[i]);
      }
    } else {
      Tensor *T = getTensorByName(op.input(1));
      auto TH = T->getHandle<size_t>();
      for (size_t i = 0, e = T->size(); i != e; i++) {
        newDim.push_back(TH.raw(i));
      }
    }

    auto *node = G_.createReshape(opName, in, newDim);

    addNodeAsOutput(op, node);
  }

  void loadTranspose(const OpType &op, ArgumentDictionaryTy &dict,
                     llvm::StringRef permArgName) {
    const std::string &opName = loadOperatorName(op);
    auto *in = getOrCreateNodeByName(op.input(0));

    std::vector<unsigned> perm = getShape<unsigned>(dict[permArgName]);
    if (perm.empty()) {
      size_t N = in->dims().size();
      for (int64_t i = N - 1; i >= 0; i--)
        perm.push_back(i);
    }

    auto *T = G_.createTranspose(opName, in, perm);

    addNodeAsOutput(op, T);
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
    if (typeName == "Sum") {
      loadSum(op, dict);
      return true;
    }
    if (typeName == "Softmax") {
      loadSoftmax(op, dict);
      return true;
    }
    if (typeName == "FC") {
      loadFC(op, dict);
      return true;
    }
    if (typeName == "LRN") {
      loadLRN(op, dict);
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
    return false;
  }
};

} // namespace glow

#endif // GLOW_IMPORTER_COMMONOPERATORLOADER_H
