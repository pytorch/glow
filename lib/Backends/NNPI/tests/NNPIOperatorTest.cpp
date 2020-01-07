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

#include "tests/unittests/BackendTestUtils.h"

using namespace glow;

std::set<std::string> glow::backendTestBlacklist = {
    "AdaptiveAvgPool/0",
    "AdaptiveAvgPoolNonSquare/0",
    "batchedReduceMinMultiAxis_Float/0",
    "batchedReduceMinMultiAxis_Int32/0",
    "batchedReduceMinMultiAxis_Int64/0",
    "Conv3DQuantizedTest_Int16_BiasInt16/0",
    "Conv3DQuantizedTest_Int16_BiasInt32/0",
    "Conv3DQuantizedTest_Int8_BiasInt32/0",
    "Conv3DQuantizedTest_Int8_BiasInt8/0",
    "ConvertFrom_Float16Ty_To_Int32ITy/0",
    "ConvertFrom_Float16Ty_To_Int32ITy_AndBack/0",
    "ConvertFrom_Float16Ty_To_Int64ITy/0",
    "ConvertFrom_Float16Ty_To_Int64ITy_AndBack/0",
    "ConvertFrom_Int32ITy_To_Float16Ty_AndBack/0",
    "ConvertFrom_Int64ITy_To_Float16Ty_AndBack/0",
    "ConvolutionDepth10_Int16_BiasInt16/0",
    "ConvolutionDepth10_Int16_BiasInt32/0",
    "ConvolutionDepth10_Int8_BiasInt32/0",
    "ConvolutionDepth10_Int8_BiasInt8/0",
    "convTest/0",
    "convTest_Float16/0",
    "EntropyLossTest/0",
    "FCGradientCheck/0",
    "FloatMaxPoolWithArgmax/0",
    "FloatMaxPoolWithArgmaxTransposed/0",
    "FP16AdaptiveAvgPool/0",
    "FullyConnected_Int16_BiasInt16/0",
    "FullyConnected_Int16_BiasInt32/0",
    "FullyConnected_Int8_BiasInt8/0",
    "FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16_"
    "NoFusedConvert/0",
    "GroupConv3D/0",
    "ChannelwiseQuantizedGroupConvolution/0",
    "insertTensorTest/0",
    "Int16ConvolutionDepth10/0",
    "Int16ConvolutionDepth8/0",
    "Int8AdaptiveAvgPool/0",
    "IntLookupTable/0",
    "LengthsSum/0",
    "LengthsToRanges/0",
    "ModuloInt32NoSignFollow/0",
    "ModuloInt32SignFollow/0",
    "ModuloInt64NoSignFollow/0",
    "ModuloInt64SignFollow/0",
    "NonCubicKernelConv3D/0",
    "NonCubicKernelConv3DQuantized/0",
    "NonCubicPaddingConv3D/0",
    "NonCubicStrideConv3D/0",
    "NonSquarePaddingAveragePool/0",
    "NonSquarePaddingMaxPool/0",
    "QuantizedMaxPoolWithArgmax/0",
    "QuantizedMaxPoolWithArgmaxTransposed/0",
    "ResizeNearest_Float/0",
    "ResizeNearest_Float16/0",
    "ResizeNearest_Int16/0",
    "ResizeNearest_Int32/0",
    "ResizeNearest_Int8/0",
    "rowwiseQuantizedFCTest_Int8_BiasInt32/0",
    "rowwiseQuantizedFCTest_Int8_BiasInt8/0",
    "RWQSLWSAllSame_Float16_AccumFP32/0",
    "ScatterAddNDimensionalDuplicatingIndices/0",
    "ScatterAddNDimensionalSimple/0",
    "ScatterAddQuantized/0",
    "ScatterData/0",
    "ScatterDataNDimensional/0",
    "ScatterDataNDimensionalSimple/0",
    "ScatterDataQuantized/0",
    "sliceReshape_Int32/0",
    "sliceVectors_Int32/0",
    "spaceToDepth_block2_Float/0",
    "spaceToDepth_block2_int8/0",
    "spaceToDepth_block3_Float/0",
    "spaceToDepth_block3_int8/0",
    "EmbeddingBag_1D_Float/0",
    "EmbeddingBag_1D_Float16/0",
    "EmbeddingBag_2D_Float/0",
    "EmbeddingBag_2D_Float16/0",
    "EmbeddingBagByteRowwiseOffsets_ConvertedFloat16_back_",
    "EmbeddingBagByteRowwiseOffsets_Float/0",
    "EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat/0",
    "EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat16/0",
    "EmbeddingBag_1D_Float_End_Offset/0",
    "EmbeddingBag_1D_Float16_End_Offset/0",
    "EmbeddingBag_2D_Float_End_Offset/0",
    "EmbeddingBag_2D_Float16_End_Offset/0",
    "EmbeddingBagByteRowwiseOffsets_ConvertedFloat16_back_",
    "EmbeddingBagByteRowwiseOffsets_Float_End_Offset/0",
    "EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat_End_Offset/0",
    "EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat16_End_Offset/0",
    "SparseToDense/0",
    "SparseToDenseMask1/0",
    "SparseToDenseMask2/0",
    "TanHSweep_Float16/0",
    "testQuantizedBatchAdd/0",
};

struct EmulatorOnlyTests {
  EmulatorOnlyTests() {
    // If USE_INF_API is set, we are running on real hardware, and need
    // to blacklist additional testcases.
    auto useInfAPI = getenv("USE_INF_API");
    if (useInfAPI && !strcmp(useInfAPI, "1")) {
      // N.B.: This insertion is defined to come after the initialization of
      // backendTestBlacklist because they are ordered within the same
      // translation unit (this source file).  Otherwise this technique would
      // be subject to the static initialization order fiasco.
      backendTestBlacklist.insert({
          "ArithAdd_int32_t/0",
          "ArithAdd_int64_t/0",
          "ArithMax_int32_t/0",
          "ArithMax_int64_t/0",
          "ArithMin_int32_t/0",
          "ArithMin_int64_t/0",
          "ArithMul_int32_t/0",
          "ArithMul_int64_t/0",
          "ArithSub_int32_t/0",
          "ArithSub_int64_t/0",
          "AvgPool/0",
          "BatchedGather/0",
          "batchedReduceMeanUsingAvgPool/0",
          "batchedReduceMin_Int32/0",
          "batchedReduceMin_Int64/0",
          "Bucketize/0",
          "CmpEQ/0",
          "ConcatTopK/0",
          "ConvertFrom_FloatTy_To_Int32ITy/0",
          "ConvertFrom_FloatTy_To_Int32ITy_AndBack/0",
          "ConvertFrom_FloatTy_To_Int64ITy/0",
          "ConvertFrom_FloatTy_To_Int64ITy_AndBack/0",
          "ConvertFrom_Int32ITy_To_Float16Ty/0",
          "ConvertFrom_Int32ITy_To_FloatTy/0",
          "ConvertFrom_Int32ITy_To_FloatTy_AndBack/0",
          "ConvertFrom_Int64ITy_To_Float16Ty/0",
          "ConvertFrom_Int64ITy_To_FloatTy/0",
          "ConvertFrom_Int64ITy_To_FloatTy_AndBack/0",
          "DilatedConvolution/0",
          "Exp/0",
          "FloatArgMaxKeepDim/0",
          "FloatArgMaxNoKeepDim/0",
          "FusedRowwiseQuantizedSLWSTwoColumn_Fused4Bit_Float16_AccumFloat16/0",
          "FusedRowwiseQuantizedSparseLengthsSum_Fused4Bit_Float16_"
          "AccumFloat16/0",
          "FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16_"
          "back_",
          "FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16_"
          "NoFusedConvert_FP32Accum/0",
          "to_back2/0",
          "GroupDilatedConvolution/0",
          "less_int32Cases/0",
          "less_int64Cases/0",
          "MaxPool/0",
          "NonSquareKernelAveragePool/0",
          "NonSquareKernelMaxPool/0",
          "NonSquareStrideAveragePool/0",
          "NonSquareStrideMaxPool/0",
          "SoftMax/0",
          "TopK/0",
          "TopK1/0",
          "TransposeIntoReshapeOptim/0",
      });
    }
  }
} emuTests;
