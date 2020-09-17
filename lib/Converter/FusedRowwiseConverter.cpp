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

#include "glow/Converter/FusedRowwiseConverter.h"

#include "glow/Converter/TypeAToTypeBFunctionConverter.h"
#include "glow/Graph/Graph.h"

using namespace glow;

/// Helper to pass over all Nodes in \p F and look for data param of
/// FusedRowwiseSLWS with type UInt4FusedFP16QTy (if \p convertUInt4FP16 is
/// true) or UInt8FusedFP16QTy (if \p convertUInt8FP16 is true), and convert
/// them to UInt8FusedQTy.
static void convertFusedRowwiseQuantizedData(Function *F, bool convertUInt4FP16,
                                             bool convertUInt8FP16) {
  auto &mod = *F->getParent();

  // Iterate from original end to beginning to avoid processing new
  // ConvertToNodes added during the pass.
  auto nodeIt = F->getNodes().end();
  auto stopIt = F->getNodes().begin();
  do {
    --nodeIt;
    Node &node = *nodeIt;
    // Only convert FusedRowwiseQuantizedSparseLengthsWeightedSumNode;
    if (node.getKind() !=
        Kinded::Kind::FusedRowwiseQuantizedSparseLengthsWeightedSumNodeKind) {
      continue;
    }

    auto idx = FusedRowwiseQuantizedSparseLengthsWeightedSumNode::DataIdx;
    NodeValue data = node.getNthInput(idx);
    auto dataType = data.getType()->getElementType();
    if (dataType != ElemKind::UInt8FusedFP16QTy &&
        dataType != ElemKind::UInt4FusedFP16QTy) {
      continue;
    }

    if (dataType == ElemKind::UInt8FusedFP16QTy && !convertUInt8FP16) {
      continue;
    }

    if (dataType == ElemKind::UInt4FusedFP16QTy && !convertUInt4FP16) {
      continue;
    }

    const auto &shape = data.dims();
    assert(shape.size() == 2 && "FusedRowwise Tensor must be 2D.");

    // If a tensor is with ElemKind::UInt4FusedFP16QTy type, we need to double
    // its data column (i.e. from 4bit->8bit). Also, the scale/offset is change
    // from 2 bytes(float16) to 4 bytes(float32).
    const dim_t newCols =
        (shape[1] - 2 * sizeof(float16_t)) *
            (dataType == ElemKind::UInt4FusedFP16QTy ? 2 : 1) +
        2 * sizeof(float);
    auto OT =
        mod.uniqueType(ElemKind::UInt8FusedQTy, {shape[0], newCols}, 1.0, 0);
    ConvertToNode *CN =
        F->createConvertTo(data.getNode()->getName().str() + ".FP32", data, OT);
    node.setNthInput(idx, CN);
  } while (nodeIt != stopIt);
}

void glow::convertFunctionToFP32ScaleOffset(
    Function *F, const PrecisionConfiguration &precConfig) {
  bool convertUInt4FP16 = precConfig.convert4BitFusedTo8Bit;
  bool convertUInt8FP16 = precConfig.convert8BitFusedToFP32;
  DCHECK(convertUInt4FP16 || convertUInt8FP16)
      << "Expect to convert at least one of UInt4FusedFP16QTy or "
         "UInt8FusedFP16QTy.";
  convertFusedRowwiseQuantizedData(F, convertUInt4FP16, convertUInt8FP16);
}

void glow::convertFunctionIndicesToInt64(
    Function *F, const PrecisionConfiguration &precConfig) {
  DCHECK(precConfig.convertIndicesToInt64)
      << "Should enable indices conversion.";
  // Iterate from original end to beginning to avoid processing new
  // ConvertToNodes added during the pass.
  auto nodeIt = F->getNodes().end();
  auto stopIt = F->getNodes().begin();
  do {
    --nodeIt;
    Node &node = *nodeIt;
    // Only convert FusedRowwiseQuantizedSparseLengthsWeightedSumNode;
    if (node.getKind() !=
        Kinded::Kind::FusedRowwiseQuantizedSparseLengthsWeightedSumNodeKind) {
      continue;
    }
    auto idx = FusedRowwiseQuantizedSparseLengthsWeightedSumNode::IndicesIdx;
    NodeValue indices = node.getNthInput(idx);
    auto indicesType = indices.getType()->getElementType();
    if (indicesType == ElemKind::Int64ITy) {
      continue;
    }
    DCHECK(indicesType == ElemKind::Int32ITy) << "Indices must be Int32ITy.";
    ConvertToNode *CN =
        F->createConvertTo(indices.getNode()->getName().str() + ".Int64",
                           indices, ElemKind::Int64ITy);
    node.setNthInput(idx, CN);
  } while (nodeIt != stopIt);
}
