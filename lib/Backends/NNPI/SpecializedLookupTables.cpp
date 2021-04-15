/*
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
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

#include "NNPI.h"

#include "glow/Graph/Graph.h"
#include "glow/Graph/Nodes.h"
#include "glow/Optimizer/GraphOptimizer/FunctionPassManager.h"

using namespace glow;

#if NNPI_MAJOR_VERSION >= 1 && NNPI_MINOR_VERSION >= 1

///
/// Initialize Gelu LUT using Tanh formula
///
/// @param lut pre allocated LUT table (empty)
/// @param rows number of raw in the LUT table
/// @param minInputValue minimum value in range
/// @param maxInputValue maximum value in range
///
/// @returns tuple<scale, offset>
///
static void initializeGeluLUTTanh(float *lut, size_t rows, float minInputValue,
                                  float maxInputValue) {
  double lookupTableStep = (maxInputValue - minInputValue) / (double)(rows - 3);
  for (int64_t i = 0; i < rows; i++) {
    double val = (minInputValue + lookupTableStep * (i - 1));
    double gval = val *
                  (1.0 + tanh(std::sqrt(2.0 / M_PI) *
                              (val + 0.044715 * val * val * val))) *
                  0.5;
    lut[i] = (float)(gval / 2.0);
  }
}

static Expected<NNPILookupTableNode *>
getGeluLUT(GeluNode *GN, size_t numLUTEntries, float minLookupTableValue,
           float maxLookupTableValue, const std::string &formulaType) {
  Function *F = GN->getParent();
  Module *mod = F->getParent();
  CHECK(GN->getInput().getElementType() == ElemKind::Float16Ty &&
        GN->getResult().getElementType() == ElemKind::Float16Ty);
  ElemKind k = ElemKind::Float16Ty;
  NNPILookupType lookupType = NNPILookupType::LOOKUP_QUADRATIC_INTERPOLATION;
  size_t lutSize = numLUTEntries + 3;

  std::vector<size_t> lookupTableTensorDims = {lutSize};
  auto inputTensorDims = GN->getInput().dims();

  // Create LUT.
  float lookupTableBuf[lutSize];

  if (formulaType == "tanh") {
    initializeGeluLUTTanh(lookupTableBuf, lutSize, minLookupTableValue,
                          maxLookupTableValue);
  } else {
    return MAKE_ERR(
        strFormat("Unsupported Gelu formula %s\n", formulaType.c_str()));
  }

  // Create lut tensor for the network.
  auto lutT = Tensor(k, lookupTableTensorDims);

  // Fill LUT tensor.
  for (size_t i = 0; i < lutT.size(); i++) {
    auto value = lookupTableBuf[i];
    lutT.getHandle<float16_t>().raw(i) = value;
  }

  auto *lutValues = mod->createConstant("lut", std::move(lutT));

  TypeRef geluOutputType = mod->uniqueType(k, inputTensorDims);

  // Using smaller LUT size because check in DL is strict.
  auto lutSizeForDeltaCalc = numLUTEntries;
  auto delta =
      (maxLookupTableValue - minLookupTableValue) / (lutSizeForDeltaCalc);

  // Create the LUT node and add to the network.
  auto *lutNode =
      F->addNode(new NNPILookupTableNode("gelu_lut",          // name
                                         geluOutputType,      // Result
                                         GN->getInput(),      // Input
                                         lutValues,           // LookupTable
                                         lookupType,          // LookupType
                                         minLookupTableValue, // LowerRange
                                         maxLookupTableValue, // UpperRange
                                         1.f,                 // UpperMulFactor
                                         0,                   // UpperOffset
                                         0.f,                 // LowerMulFactor
                                         0,                   // LowerOffset
                                         delta,               // Delta
                                         0                    // Bias
                                         ));
  return lutNode;
}

Expected<bool>
NNPIBackend::swapInSpecializedLUT(Function *F, CompilationContext &cctx) const {
  bool changed = false;
  for (auto &N : F->getNodes()) {
    if (auto *GN = llvm::dyn_cast<GeluNode>(&N)) {
      if (GN->getInput().getElementType() != ElemKind::Float16Ty ||
          GN->getResult().getElementType() != ElemKind::Float16Ty) {
        continue;
      }
      auto useLUTIt = cctx.backendOpts.backendSpecificOpts.find(
          std::string("NNPIUseGeluLUT"));
      if (useLUTIt == cctx.backendOpts.backendSpecificOpts.end() ||
          useLUTIt->second != "true") {
        continue;
      }
      // Set number of LUT entries
      size_t numEntries = 1024;
      auto numEntriesIt = cctx.backendOpts.backendSpecificOpts.find(
          std::string("NNPIGeluLUTNumEntries"));
      if (numEntriesIt != cctx.backendOpts.backendSpecificOpts.end()) {
        ASSIGN_VALUE_OR_RETURN_ERR(numEntries,
                                   getIntFromStr(numEntriesIt->second));
      }

      // Set min LUT entry
      float minLUTInput = -8.0;
      auto minLUTInputIt = cctx.backendOpts.backendSpecificOpts.find(
          std::string("NNPIGeluLUTMinInput"));
      if (minLUTInputIt != cctx.backendOpts.backendSpecificOpts.end()) {
        ASSIGN_VALUE_OR_RETURN_ERR(minLUTInput,
                                   getFloatFromStr(minLUTInputIt->second));
      }

      // Set max LUT entry
      float maxLUTInput = 8.0;
      auto maxLUTInputIt = cctx.backendOpts.backendSpecificOpts.find(
          std::string("NNPIGeluLUTMaxInput"));
      if (maxLUTInputIt != cctx.backendOpts.backendSpecificOpts.end()) {
        ASSIGN_VALUE_OR_RETURN_ERR(maxLUTInput,
                                   getFloatFromStr(maxLUTInputIt->second));
      }

      // Choose formula type
      std::string formulaType = "tanh";
      auto formulaTypeIt = cctx.backendOpts.backendSpecificOpts.find(
          std::string("NNPIGeluLUTFormula"));
      if (formulaTypeIt != cctx.backendOpts.backendSpecificOpts.end()) {
        formulaType = formulaTypeIt->second;
      }
      NNPILookupTableNode *newGN;
      ASSIGN_VALUE_OR_RETURN_ERR(newGN, getGeluLUT(GN, numEntries, minLUTInput,
                                                   maxLUTInput, formulaType));

      bool clipGeluVal = false;
      auto clipGeluValIt = cctx.backendOpts.backendSpecificOpts.find(
          std::string("NNPIGeluLUTEnableClip"));
      if (clipGeluValIt != cctx.backendOpts.backendSpecificOpts.end()) {
        clipGeluVal = (clipGeluValIt->second == "true");
      }

      if (newGN) {
        if (clipGeluVal) {
          auto *clp = F->createClip("gelu_clip", newGN, -5.0, 5.0);
          GN->getResult().replaceAllUsesOfWith(clp->getResult());
        } else {
          GN->getResult().replaceAllUsesOfWith(newGN->getResult());
        }
        changed = true;
      }
      continue;
    }
  }

  if (changed) {
    runDCEPass(F, cctx);
  }
  return changed;
}

#else

Expected<bool>
NNPIBackend::swapInSpecializedLUT(Function *F, CompilationContext &cctx) const {
  return false;
}

#endif
