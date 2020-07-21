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

#include "TestBlacklist.h"
#include "tests/unittests/BackendTestUtils.h"

using namespace glow;

std::set<std::string> glow::backendTestBlacklist = {};
struct BlacklistInitializer {
  BlacklistInitializer() {
    const std::vector<std::pair<std::string, uint32_t>> testBlacklistedSetups =
        {
            {"add_int32/0", TestBlacklist::AnyDeviceHWEngine},
            {"add_int64/0", TestBlacklist::AnyDeviceHWEngine},
            {"batchedPairwiseDotProduct/0", TestBlacklist::AnyDeviceAnyEngine},
            {"batchedReduceMinMultiAxis_Float/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"batchedReduceMinMultiAxis_Int32/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"batchedReduceMinMultiAxis_Int64/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"BroadCastMax/0", TestBlacklist::AnyDeviceHWEngine},
            {"BroadCastMin/0", TestBlacklist::AnyDeviceHWEngine},
            {"Conv3DQuantizedTest_Int16_BiasInt16/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"Conv3DQuantizedTest_Int16_BiasInt32/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"Conv3DQuantizedTest_Int8_BiasInt8/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ConvertFrom_Float16Ty_To_Int32ITy/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ConvertFrom_Float16Ty_To_Int32ITy_AndBack/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ConvertFrom_Float16Ty_To_Int64ITy/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ConvertFrom_Float16Ty_To_Int64ITy_AndBack/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ConvertFrom_Int32ITy_To_Float16Ty_AndBack/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ConvertFrom_Int64ITy_To_Float16Ty_AndBack/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ConvolutionDepth10_Int16_BiasInt16/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ConvolutionDepth10_Int16_BiasInt32/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ConvolutionDepth10_Int8_BiasInt32/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ConvolutionDepth10_Int8_BiasInt8/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"convTest/0", TestBlacklist::AnyDeviceAnyEngine},
            {"convTest_Float16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"EntropyLossTest/0", TestBlacklist::AnyDeviceAnyEngine},
            {"FloatMaxPoolWithArgmax/0", TestBlacklist::AnyDeviceAnyEngine},
            {"FloatMaxPoolWithArgmaxTransposed/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"FullyConnected_Int16_BiasInt16/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"FullyConnected_Int16_BiasInt32/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"FullyConnected_Int8_BiasInt8/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16_"
             "NoFusedConvert/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"insertTensorTest/0", TestBlacklist::AnyDeviceAnyEngine},
            {"insertTensorTest3D/0", TestBlacklist::AnyDeviceAnyEngine},
            {"insertTensorCrossDimensions/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"insertTensorPartialSliceInnerDim/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"Int16ConvolutionDepth10/0", TestBlacklist::AnyDeviceAnyEngine},
            {"Int16ConvolutionDepth8/0", TestBlacklist::AnyDeviceAnyEngine},
            {"IntLookupTable/0", TestBlacklist::AnyDeviceAnyEngine},
            {"CumSum_Float/0", TestBlacklist::AnyDeviceAnyEngine},
            {"CumSum_Float16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"CumSum_Int32/0", TestBlacklist::AnyDeviceAnyEngine},
            {"CumSum_Int64/0", TestBlacklist::AnyDeviceAnyEngine},
            {"CumSum_Exclusive/0", TestBlacklist::AnyDeviceAnyEngine},
            {"CumSum_Reverse/0", TestBlacklist::AnyDeviceAnyEngine},
            {"CumSum_ExclusiveReverse/0", TestBlacklist::AnyDeviceAnyEngine},
            {"CumSum_WithZeroes/0", TestBlacklist::AnyDeviceAnyEngine},
            {"LayerNorm_Float/0", TestBlacklist::AnyDeviceHWEngine},
            {"LayerNorm_BFloat16/0", TestBlacklist::AnyDeviceHWEngine},
            {"LengthsSum/0", TestBlacklist::AnyDeviceAnyEngine},
            {"LengthsToRanges/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ModuloInt32NoSignFollow/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ModuloInt32SignFollow/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ModuloInt64NoSignFollow/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ModuloInt64SignFollow/0", TestBlacklist::AnyDeviceAnyEngine},
            {"mul_int32/0", TestBlacklist::AnyDeviceAnyEngine},
            {"mul_int64/0", TestBlacklist::AnyDeviceAnyEngine},
            {"NonCubicKernelConv3DQuantized/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"NonCubicPaddingConv3D/0", TestBlacklist::AnyDeviceAnyEngine},
            {"NonSquarePaddingAveragePool/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"NonSquarePaddingMaxPool/0", TestBlacklist::AnyDeviceAnyEngine},
            {"QuantizedMaxPoolWithArgmax/0", TestBlacklist::AnyDeviceAnyEngine},
            {"QuantizedMaxPoolWithArgmaxTransposed/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ResizeBilinear_Float/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ResizeBilinear_Float16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ResizeBilinear_Float16_outTy/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ResizeBilinear_Float_outTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ResizeBilinear_Int16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ResizeBilinear_Int16_outTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ResizeBilinear_Int32/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ResizeBilinear_Int32_outTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ResizeBilinear_Int8/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ResizeBilinear_Int8_outTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ResizeNearest_Float/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ResizeNearest_Float16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ResizeNearest_Float16_outTy/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ResizeNearest_Float_outTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ResizeNearest_Int16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ResizeNearest_Int16_outTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ResizeNearest_Int32/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ResizeNearest_Int32_outTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ResizeNearest_Int8/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ResizeNearest_Int8_outTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"rowwiseQuantizedFCTest_Int8_BiasInt32/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"rowwiseQuantizedFCTest_Int8_BiasInt8/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"rowwiseQuantizedFCTestAsymmetric_Int8_BiasFloat32/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"rowwiseQuantizedFCTestSymmetric_Int8_BiasFloat32/0",
             TestBlacklist::A0AnyEngine},
            {"rowwiseQuantizedFCTestSymmetric/0", TestBlacklist::A0AnyEngine},
            {"ScatterAddNDimensionalDuplicatingIndices/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ScatterAddNDimensionalSimple/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ScatterAddQuantized/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ScatterData/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ScatterDataNDimensional/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ScatterDataNDimensionalSimple/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ScatterDataQuantized/0", TestBlacklist::AnyDeviceAnyEngine},
            {"sliceReshape_Int32/0", TestBlacklist::AnyDeviceAnyEngine},
            {"sliceVectors_Int32I/0", TestBlacklist::AnyDeviceAnyEngine},
            {"sliceVectors_Int32Q/0", TestBlacklist::AnyDeviceAnyEngine},
            {"spaceToDepth_block2_Float/0", TestBlacklist::AnyDeviceAnyEngine},
            {"spaceToDepth_block2_int8/0", TestBlacklist::AnyDeviceAnyEngine},
            {"spaceToDepth_block3_Float/0", TestBlacklist::AnyDeviceAnyEngine},
            {"spaceToDepth_block3_int8/0", TestBlacklist::AnyDeviceAnyEngine},
            {"EmbeddingBag_1D_Float/0", TestBlacklist::AnyDeviceAnyEngine},
            {"EmbeddingBag_1D_Float16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"EmbeddingBag_2D_Float/0", TestBlacklist::AnyDeviceAnyEngine},
            {"EmbeddingBag_2D_Float16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"EmbeddingBagByteRowwiseOffsets_Float/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat16/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"EmbeddingBag4BitRowwiseOffsets_Float16/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"EmbeddingBag4BitRowwiseOffsets_Float16_AccumFloat/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"SparseToDense_Float/0", TestBlacklist::AnyDeviceAnyEngine},
            {"SparseToDense_Int64/0", TestBlacklist::AnyDeviceAnyEngine},
            {"SparseToDenseMask1/0", TestBlacklist::AnyDeviceAnyEngine},
            {"SparseToDenseMask2/0", TestBlacklist::AnyDeviceAnyEngine},
            {"TanHSweep_Float16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"testQuantizedBatchAdd_Int8/0", TestBlacklist::AnyDeviceAnyEngine},
            {"testQuantizedBatchAdd_Int32/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ArithAdd_int32_t/0", TestBlacklist::AnyDeviceHWEngine},
            {"ArithAdd_int64_t/0", TestBlacklist::AnyDeviceHWEngine},
            {"ArithMax_int32_t/0", TestBlacklist::AnyDeviceHWEngine},
            {"ArithMax_int64_t/0", TestBlacklist::AnyDeviceHWEngine},
            {"ArithMin_int32_t/0", TestBlacklist::AnyDeviceHWEngine},
            {"ArithMin_int64_t/0", TestBlacklist::AnyDeviceHWEngine},
            {"ArithMul_int32_t/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ArithMul_int64_t/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ArithSub_int32_t/0", TestBlacklist::AnyDeviceHWEngine},
            {"ArithSub_int64_t/0", TestBlacklist::AnyDeviceHWEngine},
            {"BatchedGather/0", TestBlacklist::AnyDeviceHWEngine},
            {"BatchNorm_Float/0", TestBlacklist::AnyDeviceHWEngine},
            {"batchedReduceMin_Int32/0", TestBlacklist::AnyDeviceHWEngine},
            {"batchedReduceMin_Int64/0", TestBlacklist::AnyDeviceHWEngine},
            {"batchedReduceMax_Int32/0", TestBlacklist::AnyDeviceHWEngine},
            {"batchedReduceMax_Int64/0", TestBlacklist::AnyDeviceHWEngine},
            {"Bucketize/0", TestBlacklist::AnyDeviceHWEngine},
            {"CmpEQ_Int32/0", TestBlacklist::AnyDeviceHWEngine},
            {"CmpEQ_Int64/0", TestBlacklist::AnyDeviceHWEngine},
            {"ConcatTopK/0", TestBlacklist::AnyDeviceHWEngine},
            {"ConvertFrom_FloatTy_To_BoolTy/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ConvertFrom_FloatTy_To_Int32ITy/0",
             TestBlacklist::AnyDeviceHWEngine},
            {"ConvertFrom_FloatTy_To_Int32ITy_AndBack/0",
             TestBlacklist::AnyDeviceHWEngine},
            {"ConvertFrom_FloatTy_To_Int64ITy/0",
             TestBlacklist::AnyDeviceHWEngine},
            {"ConvertFrom_FloatTy_To_Int64ITy_AndBack/0",
             TestBlacklist::AnyDeviceHWEngine},
            {"ConvertFrom_Int32ITy_To_Float16Ty/0",
             TestBlacklist::AnyDeviceHWEngine},
            {"ConvertFrom_Int32ITy_To_FloatTy/0",
             TestBlacklist::AnyDeviceHWEngine},
            {"ConvertFrom_Int32ITy_To_FloatTy_AndBack/0",
             TestBlacklist::AnyDeviceHWEngine},
            {"ConvertFrom_Int64ITy_To_Float16Ty/0",
             TestBlacklist::AnyDeviceHWEngine},
            {"ConvertFrom_Int64ITy_To_FloatTy/0",
             TestBlacklist::AnyDeviceHWEngine},
            {"ConvertFrom_Int64ITy_To_FloatTy_AndBack/0",
             TestBlacklist::AnyDeviceHWEngine},
            {"DequantizeFRWQ_Float/0", TestBlacklist::AnyDeviceAnyEngine},
            {"DequantizeFRWQ_Float16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"DilatedConvolution/0", TestBlacklist::AnyDeviceHWEngine},
            {"Exp/0", TestBlacklist::AnyDeviceHWEngine},
            {"Exp_BFloat16/0", TestBlacklist::AnyDeviceHWEngine},
            {"FloatArgMaxKeepDim/0", TestBlacklist::AnyDeviceHWEngine},
            {"FloatArgMaxNoKeepDim/0", TestBlacklist::AnyDeviceHWEngine},
            {"FloatArgMaxNoKeepDimWithAxis1/0",
             TestBlacklist::AnyDeviceHWEngine},
            {"FloatArgMaxNoKeepDimWithAxis2/0",
             TestBlacklist::AnyDeviceHWEngine},
            {"FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16_"
             "back_",
             TestBlacklist::AnyDeviceHWEngine},
            {"FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16_"
             "NoFusedConvert_FP32Accum/0",
             TestBlacklist::AnyDeviceHWEngine},
            {"to_back2/0", TestBlacklist::AnyDeviceHWEngine},
            {"GroupDilatedConvolution/0", TestBlacklist::AnyDeviceHWEngine},
            {"less_int32Cases/0", TestBlacklist::AnyDeviceHWEngine},
            {"less_int64Cases/0", TestBlacklist::AnyDeviceHWEngine},
            {"nms_center_point_box_with_gather_float/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"nms_v4_center_point_box_with_gather_float/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"nms_center_point_box_float/0", TestBlacklist::AnyDeviceAnyEngine},
            {"nms_v4_center_point_box_float/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"nms_flipped_coordinates_float/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"nms_identical_boxes_float/0", TestBlacklist::AnyDeviceAnyEngine},
            {"nms_limit_output_size_float/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"nms_single_box_float/0", TestBlacklist::AnyDeviceAnyEngine},
            {"nms_by_iou_float/0", TestBlacklist::AnyDeviceAnyEngine},
            {"nms_by_iou_and_scores_float/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"nms_two_batches_float/0", TestBlacklist::AnyDeviceAnyEngine},
            {"nms_two_classes_float/0", TestBlacklist::AnyDeviceAnyEngine},
            {"nms_two_boxes_float/0", TestBlacklist::AnyDeviceAnyEngine},
            {"nms_v4_center_point_box_float/0",
             TestBlacklist::AnyDeviceHWEngine},
            {"nms_v4_center_point_box_with_gather_float/0",
             TestBlacklist::AnyDeviceHWEngine},
            {"SoftMax/0", TestBlacklist::AnyDeviceHWEngine},
            {"TopK/0", TestBlacklist::AnyDeviceHWEngine},
            {"TopK1/0", TestBlacklist::AnyDeviceHWEngine},
            {"TopK1int32/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ConvTransposedAsymmetric/0", TestBlacklist::AnyDeviceAnyEngine},
            {"sanityConvTranspose/0", TestBlacklist::AnyDeviceAnyEngine},
            {"convTransposeCompareSimpleK8S1P0I3/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"convTransposeCompareSimpleK6S1P1I4/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"convTransposeConvolutionCompareSimpleK5S1P2I3/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ChannelwiseQuantizedConv3D/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ChannelwiseQuantizedConv2D_NonZero_FloatBias/0",
             TestBlacklist::AnyDeviceSWEngine},
            {"ChannelwiseQuantizedConv2D_NonZero_QuantizedBias/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ChannelwiseQuantizedConv2D_Int8_BiasInt8_TFT/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ChannelwiseQuantizedConv2D_Int8_BiasInt32_TFF/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ChannelwiseQuantizedConv2D_Int8_BiasInt32_TTF/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ChannelwiseQuantizedConv2D_Int8_BiasInt32_FFF/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ChannelwiseQuantizedConv2D_Int8_BiasInt32_FTF/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ChannelwiseQuantizedConv2D_Int8_BiasInt32_FTT/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ChannelwiseQuantizedConv2D_Int8_BiasInt8_TTT/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ChannelwiseQuantizedConv2D_Int8_BiasInt8_FTF/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ChannelwiseQuantizedConv2D_Int8_BiasInt8_FFT/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ChannelwiseQuantizedConv2D_Int8_BiasInt32_TTT/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ChannelwiseQuantizedConv2D_Int8_BiasInt8_TTF/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ChannelwiseQuantizedConv2D_Int8_BiasInt32_TFT/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ChannelwiseQuantizedConv2D_Int8_BiasInt32_FFT/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ChannelwiseQuantizedConv2D_Int8_BiasInt8_FTT/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ChannelwiseQuantizedConv2D_Int8_BiasInt32_FFT/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ChannelwiseQuantizedConv2D_Int8_BiasInt8/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ChannelwiseQuantizedConv2D_Int8_BiasInt32/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"Abs_FloatTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"Abs_Int8QTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"And/0", TestBlacklist::AnyDeviceAnyEngine},
            {"Ceil_FloatTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"Ceil_Int8QTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"CmpGTE_FloatTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"CmpGTE_Int32ITy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"CmpGTE_Int64ITy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"CmpGTE_Int8QTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"CmpGT_FloatTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"CmpGT_Int32ITy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"CmpGT_Int64ITy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"CmpGT_Int8QTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"CmpNEQ_FloatTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"CmpNEQ_Int32ITy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"CmpNEQ_Int64ITy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"CmpNEQ_Int8QTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"Cos_FloatTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"Cos_Int8QTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"FloatArgMinKeepDim/0", TestBlacklist::AnyDeviceAnyEngine},
            {"FloatArgMinNoKeepDim/0", TestBlacklist::AnyDeviceAnyEngine},
            {"FloatArgMinNoKeepDimWithAxis1/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"FloatArgMinNoKeepDimWithAxis2/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"Floor_FloatTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"Floor_Int8QTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"Neg_FloatTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"Neg_Int8QTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"Not/0", TestBlacklist::AnyDeviceAnyEngine},
            {"Or/0", TestBlacklist::AnyDeviceAnyEngine},
            {"QuantizedArgMinKeepDim/0", TestBlacklist::AnyDeviceAnyEngine},
            {"QuantizedArgMinNoKeepDim/0", TestBlacklist::AnyDeviceAnyEngine},
            {"Reciprocal_FloatTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"Reciprocal_Int8QTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"Round_FloatTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"Round_Int8QTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"Rsqrt_FloatTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"Rsqrt_Int8QTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"Sin_FloatTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"Sin_Int8QTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"Sqrt_FloatTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"Sqrt_Int8QTy/0", TestBlacklist::AnyDeviceAnyEngine},
            {"Xor/0", TestBlacklist::AnyDeviceAnyEngine},
            {"testBatchAdd_BFloat16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ArithMul_bfloat16_t/0", TestBlacklist::AnyDeviceHWEngine},
            {"batchedReduceAdd_BFloat16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"BatchBoxCox_Medium_BFloat16/0", TestBlacklist::AnyDeviceHWEngine},
            {"BatchOneHotDataBFloat16/0", TestBlacklist::AnyDeviceHWEngine},
            {"BFloat16AdaptiveAvgPool/0", TestBlacklist::AnyDeviceAnyEngine},
            {"BFloat16Max/0", TestBlacklist::AnyDeviceHWEngine},
            {"BFloat16Transpose2Dims/0", TestBlacklist::AnyDeviceHWEngine},
            {"ConvertFrom_BFloat16Ty_To_FloatTy/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ConvertFrom_BFloat16Ty_To_Int32ITy/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ConvertFrom_BFloat16Ty_To_Float16Ty_AndBack/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ConvertFrom_BFloat16Ty_To_Int32ITy_AndBack/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ConvertFrom_BFloat16Ty_To_Int64ITy/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ConvertFrom_BFloat16Ty_To_Int64ITy_AndBack/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ConvertFrom_BoolTy_To_BFloat16Ty/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ConvertFrom_FloatTy_To_BFloat16Ty/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ConvertFrom_Int32ITy_To_BFloat16Ty_AndBack/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ConvertFrom_Int64ITy_To_BFloat16Ty_AndBack/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"convTest_BFloat16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ExpandDims_BFloat16/0", TestBlacklist::AnyDeviceHWEngine},
            {"CumSum_BFloat16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"CumSum_Reverse_BFloat16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"less_bfloat16Cases/0", TestBlacklist::AnyDeviceHWEngine},
            {"Logit_BFloat16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"BFloat16Matmul/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ResizeBilinear_BFloat16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ResizeBilinear_BFloat16_outTy/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ResizeNearest_BFloat16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ResizeNearest_BFloat16_outTy/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"sliceReshape_BFloat16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"Split_BFloat16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"EmbeddingBag_1D_BFloat16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"EmbeddingBag_2D_BFloat16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"TanHSweep_BFloat16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ArithAdd_bfloat16_t/0", TestBlacklist::AnyDeviceHWEngine},
            {"ArithMax_bfloat16_t/0", TestBlacklist::AnyDeviceHWEngine},
            {"ArithMin_bfloat16_t/0", TestBlacklist::AnyDeviceHWEngine},
            {"ArithMul_bfloat16_t/0", TestBlacklist::AnyDeviceHWEngine},
            {"ArithSub_bfloat16_t/0", TestBlacklist::AnyDeviceHWEngine},
            {"BatchOneHotDataBFloat16/0", TestBlacklist::AnyDeviceHWEngine},
            {"ConvertFrom_Int32ITy_To_BFloat16Ty/0",
             TestBlacklist::AnyDeviceHWEngine},
            {"ConvertFrom_Int64ITy_To_BFloat16Ty/0",
             TestBlacklist::AnyDeviceHWEngine},
            {"batchedReduceAddWithAxis_BFloat16/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"Sigmoid_BFloat16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"SigmoidSweep_BFloat16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"BFloat16Max/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ExpandDims_BFloat16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"SparseLengthsWeightedSum_1D_BFloat16/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"BFloat16AvgPool/0", TestBlacklist::AnyDeviceAnyEngine},
            {"BFloat16Transpose2Dims/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ConvertFrom_BFloat16Ty_To_FloatTy_AndBack/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"SparseLengthsWeightedSum_2D_BFloat16/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"dotProduct2D_BFloat16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"BatchBoxCox_Small_BFloat16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ConvertFrom_Int64ITy_To_BFloat16Ty/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ConvertFrom_FloatTy_To_BFloat16Ty_AndBack/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"BFloat16SoftMax/0", TestBlacklist::AnyDeviceAnyEngine},
            {"BatchOneHotDataBFloat16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"BFloat16MaxPool/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ArithSub_bfloat16_t/0", TestBlacklist::AnyDeviceAnyEngine},
            {"concatVectors_BFloat16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"Flatten_BFloat16Ty/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ArithMin_bfloat16_t/0", TestBlacklist::AnyDeviceAnyEngine},
            {"BFloat16AvgPool3D/0", TestBlacklist::AnyDeviceAnyEngine},
            {"SparseLengthsSum_BFloat16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"Transpose3Dims_BFloat16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"BFloat16BatchAdd/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ConvertFrom_BFloat16Ty_To_BFloat16Ty/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"less_bfloat16Cases/0", TestBlacklist::AnyDeviceAnyEngine},
            {"BatchBoxCox_Medium_BFloat16/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"dotProduct1D_BFloat16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"BFloat16Add/0", TestBlacklist::AnyDeviceAnyEngine},
            {"batchedReduceZeroDimResult_BFloat16/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ConvertFrom_Float16Ty_To_BFloat16Ty/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"SparseLengthsSum_BFloat16_Int32/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ReluSimple_BFloat16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ConvertFrom_Int32ITy_To_BFloat16Ty/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"PReluSimple_BFloat16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"EmbeddingBag_1D_BFloat16_End_Offset/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"GatherRangesDataBFloat16IdxInt32/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"EmbeddingBag_2D_BFloat16_End_Offset/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"BatchBoxCox_Large_BFloat16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ArithMul_bfloat16_t/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ArithMax_bfloat16_t/0", TestBlacklist::AnyDeviceAnyEngine},
            {"Swish_BFloat16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"BFloat16Reshape/0", TestBlacklist::AnyDeviceAnyEngine},
            {"replaceNaN_BFloat16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"sliceConcatVectors_BFloat16/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"GatherDataBFloat16IdxInt64/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ConvertFrom_BFloat16Ty_To_BFloat16Ty_AndBack/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ConvertFrom_Float16Ty_To_BFloat16Ty_AndBack/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"concatVectorsRepeated_BFloat16/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"ConvertFrom_BFloat16Ty_To_Float16Ty/0",
             TestBlacklist::AnyDeviceAnyEngine},
            {"sliceVectors_BFloat16/0", TestBlacklist::AnyDeviceAnyEngine},
            {"BFloat16Splat/0", TestBlacklist::AnyDeviceAnyEngine},
            {"GatherDataBFloat16IdxInt32/0", TestBlacklist::AnyDeviceAnyEngine},
            {"ArithAdd_bfloat16_t/0", TestBlacklist::AnyDeviceAnyEngine},
        };
    TestBlacklist::prepareBlacklist(testBlacklistedSetups,
                                    backendTestBlacklist);
  }
} blacklistInitializer;
