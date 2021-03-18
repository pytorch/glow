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

#include "InputMeta.h"

#include "glow/Support/Error.h"
#include "glow/Support/Support.h"

#include <folly/hash/Hash.h>

namespace glow {
std::string InputMeta::print() const {
  std::ostringstream oss;
  oss << "InputMeta(" << c10::toString(type) << ") [";
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << dims[i];
  }
  oss << "]";
  if (c10::isQIntType(type)) {
    oss << " scale: " << scale << " offset: " << offset;
  }
  return oss.str();
}

size_t InputMeta::hash() const {
  size_t h = static_cast<size_t>(type);
  for (const auto &d : dims) {
    h = folly::hash::hash_combine(h, d);
  }
  if (c10::isQIntType(type)) {
    h = folly::hash::hash_combine(h, scale);
    h = folly::hash::hash_combine(h, offset);
  }
  return h;
}

std::string InputMetaStack::print() const {
  std::ostringstream oss;
  oss << "InputMetaStack [";
  for (size_t i = 0; i < inputMetas.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << inputMetas[i].print();
  }
  oss << "]";
  return oss.str();
}

size_t InputMetaStack::hash() const {
  size_t h = 0;
  for (const auto &meta : inputMetas) {
    h = folly::hash::hash_combine(h, meta.hash());
  }
  return h;
}

size_t InputMetaStack::optimizedHash(int32_t nominalBatchIdx) const {
  size_t batchSize = 0;
  bool validInput = true;
  if (nominalBatchIdx < 0 || nominalBatchIdx >= inputMetas.size()) {
    validInput = false;
  } else {
    const auto &inputMeta = inputMetas[nominalBatchIdx];
    if (inputMeta.dims.size() == 0) {
      validInput = false;
    } else {
      batchSize = inputMeta.dims[0];
    }
  }
  if (validInput && batchSize != 0) {
    return batchSize;
  } else { // If input is not valid, fallback to default hash function.
    return hash();
  }
}

Expected<InputMetaStack>
inputMetaStackFromStack(const c10::ArrayRef<c10::IValue> &inputs,
                        bool ignoreNonTensors) {
  InputMetaStack metaStack;
  metaStack.inputMetas.reserve(inputs.size());
  for (const auto &input : inputs) {
    if (ignoreNonTensors && !input.isTensor()) {
      continue;
    }

    RETURN_ERR_IF_NOT(
        input.isTensor(),
        strFormat("Cannot create InputMeta from IValue of type %s",
                  input.tagKind().c_str()));
    const auto tensorInput = input.toTensor();
    if (tensorInput.is_quantized()) {
      RETURN_ERR_IF_NOT(tensorInput.qscheme() == at::kPerTensorAffine ||
                            tensorInput.qscheme() == at::kPerTensorSymmetric,
                        "Expect per_tensor quantization scheme");
      metaStack.inputMetas.emplace_back(
          tensorInput.scalar_type(), tensorInput.sizes(), tensorInput.q_scale(),
          tensorInput.q_zero_point());
    } else {
      metaStack.inputMetas.emplace_back(tensorInput.scalar_type(),
                                        tensorInput.sizes());
    }
  }
  return metaStack;
}

InputMetaStack loadInputMeta(const std::string &raw_data) {
  InputMetaStack metaStack;
  if (raw_data.empty()) {
    return {};
  }
  std::stringstream ss_raw(raw_data);

  std::string line;
  while (std::getline(ss_raw, line)) {
    std::vector<glow::sdim_t> dims;
    std::stringstream ss(line);
    ss.ignore();
    for (int i; ss >> i;) {
      dims.push_back(i);
      if (ss.peek() == ',' || ss.peek() == '[' || ss.peek() == ']') {
        ss.ignore();
      }
    }
    std::getline(ss_raw, line);
    c10::ScalarType t = static_cast<c10::ScalarType>(std::stoi(line));
    metaStack.inputMetas.emplace_back(t, std::move(dims));
  }
  return metaStack;
}

InputMetaStack
getInputMetas(const std::vector<c10::intrusive_ptr<InputSpec>> &inputSet) {
  InputMetaStack metaStack;
  for (const auto &inputSpec : inputSet) {
    std::vector<glow::sdim_t> dims;
    for (auto d : inputSpec->dims) {
      dims.emplace_back(static_cast<glow::sdim_t>(d));
    }

    metaStack.inputMetas.emplace_back(inputSpec->elem_type, std::move(dims),
                                      inputSpec->scale, inputSpec->offset);
  }
  return metaStack;
}
} // namespace glow
