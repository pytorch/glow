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

#include "glow/Converter/Float16Converter.h"

#include "glow/Converter/TypeAToTypeBFunctionConverter.h"
#include "glow/Graph/Graph.h"

using namespace glow;

/// Helper to pass over all Nodes in \p F and look for inputs of UInt8FusedQTy,
/// and convert them to UInt8FusedFP16QTy. \p precConfig contains the
/// black/whitelist for skipping nodes for transformation.
static void
convertFusedRowwiseQuantizedInputs(Function *F,
                                   const PrecisionConfiguration &precConfig) {
  auto &mod = *F->getParent();

  // Iterate from original end to beginning to avoid processing new
  // ConvertToNodes added during the pass.
  auto nodeIt = F->getNodes().end();
  auto stopIt = F->getNodes().begin();
  do {
    --nodeIt;
    Node &node = *nodeIt;
    // Only convert allowed nodes based on black/whitelist.
    const bool inSet = precConfig.precisionModeKindSet.count(node.getKind());
    const bool allowConversion = precConfig.useSetAsWhitelist ? inSet : !inSet;
    if (!allowConversion) {
      continue;
    }

    // Now check if any inputs are UInt8FusedQTy, and convert them accordingly.
    for (unsigned idx = 0, end = node.getNumInputs(); idx != end; ++idx) {
      NodeValue input = node.getNthInput(idx);
      if (input.getElementType() != ElemKind::UInt8FusedQTy) {
        continue;
      }

      // Create the conversion using the same shape but without the extra space
      // needed for FP16 scale/offset instead of FP32.
      const auto &shape = input.dims();
      assert(shape.size() == 2 && "UInt8FusedQTy must be 2D.");
      const dim_t newCols = shape[1] - 2 * (sizeof(float) - sizeof(float16_t));
      auto OT = mod.uniqueType(ElemKind::UInt8FusedFP16QTy, {shape[0], newCols},
                               1.0, 0); // Dummy scale/offset.
      ConvertToNode *CN = F->createConvertTo(
          input.getNode()->getName().str() + ".FP16", input, OT);
      node.setNthInput(idx, CN);
    }
  } while (nodeIt != stopIt);
}

void glow::convertFunctionToFloat16(Function *F,
                                    const PrecisionConfiguration &precConfig) {
  DCHECK(precConfig.convertToFP16 || precConfig.convertFusedToFP16)
      << "Expected to convert at least one of FloatTy or UInt8FusedQTy.";

  // Convert FloatTy to Float16Ty.
  TypeAToTypeBFunctionConverter converter(*F, ElemKind::FloatTy,
                                          ElemKind::Float16Ty, precConfig);
  if (precConfig.convertToFP16) {
    converter.convert();

    // Storage nodes are not converted + clipped directly -- they need to be
    // converted via adding ConvertToNodes instead of directly setting their
    // types like the TypeAToTypeBFunctionConverter does.
    converter.convertAndClipStorage();
  }

  // Now we want to additionally convert all nodes with inputs in UInt8FusedQTy
  // to UInt8FusedFP16QTy. This does not fit cleanly into the
  // TypeAToTypeBFunctionConverter, so write a custom pass to do so.
  if (precConfig.convertFusedToFP16) {
    convertFusedRowwiseQuantizedInputs(F, precConfig);
  }
}
