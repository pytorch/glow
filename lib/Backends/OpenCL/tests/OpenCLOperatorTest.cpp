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
#include "tests/unittests/BackendTestUtils.h"

using namespace glow;

std::set<std::string> glow::backendTestBlacklist = {
    "less_int8/0",
    "less_floatCases/0",
    "less_float16Cases/0",
    "less_int64Cases/0",
    "less_float/0",
    "less_broadcast_float/0",
    "less_int32Cases/0",
    "BroadCastMax/0",
    "BroadCastMin/0",
    "where_2d_broadcast_x_y_i8/0",
    "where_2d_wise_i8/0",
    "where_2d_wise_float/0",
    "where_row_wise_float/0",
    "FloatArgMaxNoKeepDimWithAxis1/0",
    "FloatArgMaxNoKeepDimWithAxis2/0",
    "FloatArgMinNoKeepDimWithAxis1/0",
    "FloatArgMinNoKeepDimWithAxis2/0",
    "spaceToDepth_block3_int8/0",
    "spaceToDepth_block3_Float/0",
    "spaceToDepth_block2_int8/0",
    "spaceToDepth_block2_Float/0",
    "ResizeNearest_Float/0",
    "ResizeNearest_Float16/0",
    "ResizeNearest_Int8/0",
    "ResizeNearest_Int16/0",
    "ResizeNearest_Int32/0",
    "ResizeNearest_Float_outTy/0",
    "ResizeNearest_Float16_outTy/0",
    "ResizeNearest_Int8_outTy/0",
    "ResizeNearest_Int16_outTy/0",
    "ResizeNearest_Int32_outTy/0",
    "ResizeBilinear_Float/0",
    "ResizeBilinear_Float16/0",
    "ResizeBilinear_Int8/0",
    "ResizeBilinear_Int16/0",
    "ResizeBilinear_Int32/0",
    "ResizeBilinear_Float_outTy/0",
    "ResizeBilinear_Float16_outTy/0",
    "ResizeBilinear_Int8_outTy/0",
    "ResizeBilinear_Int16_outTy/0",
    "ResizeBilinear_Int32_outTy/0",
    "BoolReshape/0",
    "replaceNaN_Float/0",
    "replaceNaN_Float16/0",
    "log/0",
    "Logit_Float/0",
    "Logit_Float16/0",
    "CmpEQ_Int64/0",
    "CmpEQ_Int32/0",
    "FP16Add/0",
    "FP16Matmul/0",
    "batchedPairwiseDotProduct/0",
    "batchedReduceAdd_Float16/0",
    "batchedReduceAdd_5Dinput/0",
    "batchedReduceMin_Float/0",
    "batchedReduceMin_Int32/0",
    "batchedReduceMin_Int64/0",
    "batchedReduceProd_Float/0",
    "batchedReduceProd_Float16/0",
    "batchedReduceProd_BFloat16/0",
    "batchedReduceProd_Int32/0",
    "batchedReduceProd_Int64/0",
    "batchedReduceProd_Int8/0",
    "batchedReduceMinMultiAxis_Float/0",
    "batchedReduceMinMultiAxis_Int32/0",
    "batchedReduceMinMultiAxis_Int64/0",
    "batchedReduceMax_Float/0",
    "batchedReduceMax_Int32/0",
    "batchedReduceMax_Int64/0",
    "batchedReduceMaxMultiAxis_Float/0",
    "batchedReduceMaxMultiAxis_Int32/0",
    "batchedReduceMaxMultiAxis_Int64/0",
    "batchedReduceZeroDimResult_Float16/0",
    "batchedReduceZeroDimResult_Int8/0",
    "batchedReduceAddWithAxis_Float16/0",
    "batchedReduceAddWithAxis_Int8Q/0",
    "batchedReduceAddQuantized/0",
    "batchedReduceAddQuantizedWithAxis/0",
    "batchedReduceMeanQuantized/0",
    "batchedReduceMeanQuantizedWithAxis/0",
    "batchedReduceMeanUsingAvgPool/0",
    "batchedReduceMeanUsingAvgPoolQuantized/0",
    "Gelu_Float16/0",
    "ReluSimple_Float16/0",
    "PReluSimple_Float16/0",
    "FloatArgMaxKeepDim/0",
    "Float16ArgMaxKeepDim/0",
    "QuantizedArgMaxKeepDim/0",
    "FloatArgMaxNoKeepDim/0",
    "Float16ArgMaxNoKeepDim/0",
    "QuantizedArgMaxNoKeepDim/0",
    "FloatArgMinKeepDim/0",
    "QuantizedArgMinKeepDim/0",
    "FloatArgMinNoKeepDim/0",
    "QuantizedArgMinNoKeepDim/0",
    "ConcatTopK/0",
    "GatherDataFloatIdxInt32/0",
    "GatherDataFloat16IdxInt32/0",
    "GatherDataFloat16IdxInt64/0",
    "GatherDataInt8IdxInt32/0",
    "GatherDataInt8IdxInt64/0",
    "GatherDataNDFloat16IdxInt32/0",
    "GatherNDDataFloatIdxInt32/0",
    "GatherNDDataFloat16IdxInt32/0",
    "GatherNDDataFloat16IdxInt64/0",
    "GatherNDDataInt8IdxInt32/0",
    "GatherNDDataInt8IdxInt64/0",
    "GatherRangesDataInt64IdxInt32/0",
    "GatherRangesDataInt64IdxInt64/0",
    "GatherRangesDataFloatIdxInt32/0",
    "GatherRangesDataFloatIdxInt64/0",
    "GatherRangesDataFloat16IdxInt32/0",
    "GatherRangesDataFloat16IdxInt64/0",
    "GatherRangesDataInt8QIdxInt32/0",
    "GatherRangesDataInt8QIdxInt64/0",
    "FP16Transpose2Dims/0",
    "BoolTranspose2Dims/0",
    "Transpose3Dims_Float16/0",
    "TransposeIntoReshapeOptim/0",
    "Transpose6Dims/0",
    "GatherSizeT/0",
    "ScatterDataQuantized/0",
    "ScatterDataNDimensionalSimple/0",
    "ScatterDataNDimensional/0",
    "ScatterAddQuantized/0",
    "ScatterAddNDimensionalSimple/0",
    "ScatterAddNDimensionalDuplicatingIndices/0",
    "ArithAdd_int32_t/0",
    "ArithAdd_int64_t/0",
    "ArithAdd_float16_t/0",
    "ArithSub_int32_t/0",
    "ArithSub_int64_t/0",
    "ArithSub_float16_t/0",
    "ArithMul_int32_t/0",
    "ArithMul_int64_t/0",
    "ArithMul_float16_t/0",
    "ArithMax_int32_t/0",
    "ArithMax_int64_t/0",
    "ArithMax_float16_t/0",
    "ArithMin_int32_t/0",
    "ArithMin_int64_t/0",
    "ArithMin_float16_t/0",
    "ArithDiv_bfloat16_t/0",
    "ArithDiv_float16_t/0",
    "ArithDiv_int32_t/0",
    "ArithDiv_int64_t/0",
    "convTest_Float16/0",
    "EntropyLossTest/0",
    "FP16Max/0",
    "QuantizedCmpLTEAndSelect/0",
    "concatVectors_Int64/0",
    "concatVectors_Int32/0",
    "concatVectors_Bool/0",
    "concatVectors_Float16/0",
    "concatVectorsRepeated_Int64/0",
    "concatVectorsRepeated_Int32/0",
    "concatVectorsRepeated_Bool/0",
    "concatVectorsRepeated_Float16/0",
    "sliceReshape_Int32/0",
    "sliceVectors_Int32Q/0",
    "sliceVectors_Int32I/0",
    "sliceVectors_Int64/0",
    "sliceVectors_Float16/0",
    "sliceConcatVectors_Int64/0",
    "sliceConcatVectors_Float16/0",
    "ChannelShuffle/0",
    "ExpandDims_Float16/0",
    "Split_Float16/0",
    "Fp16Splat/0",
    "GroupConv3D/0",
    "NonCubicPaddingConv3D/0",
    "FP16AvgPool/0",
    "Int8AvgPool3D/0",
    "FP16AvgPool3D/0",
    "LSTMUnitFP16/0",
    "PyTorchLSTMFP16/0",
    "AdaptiveAvgPool/0",
    "FP16AdaptiveAvgPool/0",
    "Int8AdaptiveAvgPool/0",
    "AvgPoolCountExcludePads/0",
    "Int8AvgPoolCountExcludePads/0",
    "AdaptiveAvgPoolNonSquare/0",
    "FP16MaxPool/0",
    "QuantizedMaxPoolWithArgmax/0",
    "QuantizedMaxPoolWithArgmaxTransposed/0",
    "Exp/0",
    "nms_by_iou_and_scores_float/0",
    "nms_by_iou_float/0",
    "nms_center_point_box_float/0",
    "nms_center_point_box_with_gather_float/0",
    "nms_flipped_coordinates_float/0",
    "nms_identical_boxes_float/0",
    "nms_limit_output_size_float/0",
    "nms_single_box_float/0",
    "nms_two_batches_float/0",
    "nms_two_boxes_float/0",
    "nms_two_classes_float/0",
    "nms_v4_center_point_box_with_gather_float/0",
    "nms_v4_center_point_box_float/0",
    "NonCubicKernelConv3D/0",
    "NonCubicKernelConv3DQuantized/0",
    "NonSquareKernelAveragePool/0",
    "NonSquareKernelMaxPool/0",
    "NonCubicStrideConv3D/0",
    "NonSquareStrideAveragePool/0",
    "NonSquareStrideMaxPool/0",
    "FP16BatchAdd/0",
    "FP16BatchMul/0",
    "Sigmoid_Float16/0",
    "Swish_Float16/0",
    "Swish_Int8/0",
    "IntLookupTable/0",
    "testBatchAdd_Float16/0",
    "testBatchMul_Float16/0",
    "CumSum_Float/0",
    "CumSum_Float16/0",
    "CumSum_Int32/0",
    "CumSum_Int64/0",
    "CumSum_Exclusive/0",
    "CumSum_Reverse/0",
    "CumSum_ExclusiveReverse/0",
    "CumSum_WithZeroes/0",
    "LengthsSum/0",
    "SparseLengthsSum_Float_Int32/0",
    "SparseLengthsSum_Float16/0",
    "SparseLengthsSum_Float16_Int32/0",
    "SparseLengthsSumI8/0",
    "SparseLengthsWeightedSum_1D_Float16/0",
    "SparseLengthsWeightedSum_2D_Float16/0",
    "Embedding_Float/0",
    "Embedding_Float16/0",
    "Embedding_with_PadIdx/0",
    "Embedding_with_PadIdx_Float16/0",
    "EmbeddingBag_1D_Float/0",
    "EmbeddingBag_1D_Float16/0",
    "EmbeddingBag_2D_Float/0",
    "EmbeddingBag_2D_Float16/0",
    "EmbeddingBag_1D_Float_End_Offset/0",
    "EmbeddingBag_1D_Float16_End_Offset/0",
    "EmbeddingBag_2D_Float_End_Offset/0",
    "EmbeddingBag_2D_Float16_End_Offset/0",
    "EmbeddingBag_1D_Float_End_Offset_Partial/0",
    "EmbeddingBag_2D_Float_End_Offset_Partial/0",
    "SparseLengthsWeightedSumI8/0",
    "RowwiseQuantizedSparseLengthsWeightedSum_Float/0",
    "RowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat/0",
    "RowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat16/0",
    "RowwiseQuantizedSparseLengthsWeightedSum_Float_Int32/0",
    "RowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat_Int32/0",
    "RowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat16_Int32/0",
    "RowwiseQuantizedSparseLengthsSum_Float/0",
    "RowwiseQuantizedSparseLengthsSum_Float16_AccumFloat/0",
    "RowwiseQuantizedSparseLengthsSum_Float16_AccumFloat16/0",
    "FusedRowwiseQuantizedSparseLengthsWeightedSum_Float/0",
    "FusedRowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat/0",
    "FusedRowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat16/0",
    "FusedRowwiseQuantizedSparseLengthsWeightedSum_Float_Int32/0",
    "FusedRowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat_Int32/0",
    "FusedRowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat16_Int32/"
    "0",
    "FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16/0",
    "FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16_back_to_"
    "back/0",
    "FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16_back_to_"
    "back2/0",
    "EmbeddingBagByteRowwiseOffsets_Float/0",
    "EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat/0",
    "EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat16/0",
    "EmbeddingBagByteRowwiseOffsets_ConvertedFloat16/0",
    "EmbeddingBagByteRowwiseOffsets_Float_End_Offset/0",
    "EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat_End_Offset/0",
    "EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat16_End_Offset/0",
    "EmbeddingBagByteRowwiseOffsets_ConvertedFloat16_End_Offset/0",
    "EmbeddingBag4BitRowwiseOffsets_Float16/0",
    "EmbeddingBag4BitRowwiseOffsets_Float16_AccumFloat/0",
    "EmbeddingBag4BitRowwiseOffsets_Float16_HasEndOffset/0",
    "EmbeddingBag4BitRowwiseOffsets_Float16_HasEndOffset_AccumFloat/0",
    "EmbeddingBagByteRowwiseOffsets_Float_End_Offset_Partial/0",
    "EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat_End_Offset_Partial/0",
    "EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat16_End_Offset_Partial/0",
    "FusedRowwiseQuantizedSparseLengthsSum_Float/0",
    "FusedRowwiseQuantizedSparseLengthsSum_Float16_AccumFloat/0",
    "FusedRowwiseQuantizedSparseLengthsSum_Float16_AccumFloat16/0",
    "FusedRowwiseQuantizedSparseLengthsSum_Fused4Bit_Float16_AccumFloat16/0",
    "FusedRowwiseQuantizedSLWSTwoColumn_Float/0",
    "FusedRowwiseQuantizedSLWSTwoColumn_Float16_AccumFloat/0",
    "FusedRowwiseQuantizedSLWSTwoColumn_Float16_AccumFloat16/0",
    "FusedRowwiseQuantizedSLWSTwoColumn_Fused4Bit_Float16_AccumFloat16/0",
    "FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16_"
    "NoFusedConvert/0",
    "FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16_"
    "NoFusedConvert_FP32Accum/0",
    "SLWSTwoColumn_Float16_AccumFloat/0",
    "SLSWithZeroLengths/0",
    "SparseToDense_Float/0",
    "SparseToDense_Int64/0",
    "SparseToDenseMask1/0",
    "SparseToDenseMask2/0",
    "FP16Reshape/0",
    "sliceReshape_Float16/0",
    "Flatten_Float16Ty/0",
    "DivSizeT/0",
    "SigmoidCrossEntropyWithLogits/0",
    "Bucketize/0",
    "FP16SoftMax/0",
    "LengthsToRanges/0",
    "LengthsRangeFill/0",
    "BatchOneHotDataFloat/0",
    "BatchOneHotDataFloat16/0",
    "BatchOneHotDataInt64/0",
    "BatchOneHotDataInt32/0",
    "BatchOneHotDataInt8/0",
    "matmulQuantized_InterpCompareParClone/0",
    "ModuloInt64NoSignFollow/0",
    "ModuloInt64SignFollow/0",
    "ModuloInt32NoSignFollow/0",
    "ModuloInt32SignFollow/0",
    "dotProduct1D_Float16/0",
    "dotProduct2D_Float16/0",
    "dotProduct2D_Int8/0",
    "BatchBoxCox_Float/0",
    "BatchBoxCox_Large_Float16/0",
    "BatchBoxCox_Medium_Float16/0",
    "BatchBoxCox_Small_Float16/0",
    "ConvertFrom_FloatTy_To_Float16Ty/0",
    "ConvertFrom_FloatTy_To_Int32ITy/0",
    "ConvertFrom_FloatTy_To_Int64ITy/0",
    "ConvertFrom_FloatTy_To_BoolTy/0",
    "ConvertFrom_Float16Ty_To_FloatTy/0",
    "ConvertFrom_Float16Ty_To_Float16Ty/0",
    "ConvertFrom_Float16Ty_To_Int32ITy/0",
    "ConvertFrom_Float16Ty_To_Int64ITy/0",
    "ConvertFrom_Int32ITy_To_FloatTy/0",
    "ConvertFrom_Int32ITy_To_Float16Ty/0",
    "ConvertFrom_Int32ITy_To_Int32ITy/0",
    "ConvertFrom_Int32ITy_To_Int64ITy/0",
    "ConvertFrom_Int64ITy_To_FloatTy/0",
    "ConvertFrom_Int64ITy_To_Float16Ty/0",
    "ConvertFrom_Int64ITy_To_Int32ITy/0",
    "ConvertFrom_FloatTy_To_Float16Ty_AndBack/0",
    "ConvertFrom_FloatTy_To_Int32ITy_AndBack/0",
    "ConvertFrom_FloatTy_To_Int64ITy_AndBack/0",
    "ConvertFrom_Float16Ty_To_FloatTy_AndBack/0",
    "ConvertFrom_Float16Ty_To_Float16Ty_AndBack/0",
    "ConvertFrom_Float16Ty_To_Int32ITy_AndBack/0",
    "ConvertFrom_Float16Ty_To_Int64ITy_AndBack/0",
    "ConvertFrom_Int32ITy_To_FloatTy_AndBack/0",
    "ConvertFrom_Int32ITy_To_Float16Ty_AndBack/0",
    "ConvertFrom_Int32ITy_To_Int32ITy_AndBack/0",
    "ConvertFrom_Int32ITy_To_Int64ITy_AndBack/0",
    "ConvertFrom_Int64ITy_To_FloatTy_AndBack/0",
    "ConvertFrom_Int64ITy_To_Float16Ty_AndBack/0",
    "ConvertFrom_Int64ITy_To_Int32ITy_AndBack/0",
    "ConvertFrom_BoolTy_To_FloatTy/0",
    "ConvertFrom_BoolTy_To_Float16Ty/0",
    "ConvertFrom_BoolTy_To_Int32ITy/0",
    "BasicDivNetFloatVsInt8/0",
    "BasicAddNetFloatVsFloat16/0",
    "BasicSubNetFloatVsFloat16/0",
    "BasicMulNetFloatVsFloat16/0",
    "BasicDivNetFloatVsFloat16/0",
    "BasicMaxNetFloatVsFloat16/0",
    "BasicMinNetFloatVsFloat16/0",
    "Int16ConvolutionDepth10/0",
    "Int16ConvolutionDepth8/0",
    "FP16ConvolutionDepth10/0",
    "FP16ConvolutionDepth8/0",
    "FC_Float16/0",
    "Int8Tanh/0",
    "Tanh_Float16/0",
    "Exp_Float16/0",
    "Int8Log/0",
    "Int8Sigmoid/0",
    "rowwiseQuantizedFCTest/0",
    "rowwiseQuantizedFCTestSymmetric/0",
    "rowwiseQuantizedSLWSTest/0",
    "SLSAllZeroLengths_Float16/0",
    "FusedRWQSLSAllZeroLengths_Float/0",
    "FusedRWQSLSAllZeroLengths_Float16/0",
    "SigmoidSweep_Float16/0",
    "TanHSweep_Float16/0",
    "RepeatedSLSWithPartialTensors/0",
    "RepeatedSLWSWithPartialTensors/0",
    "GatherWithInt32PartialTensors/0",
    "GatherWithInt64PartialTensors/0",
    "ChannelwiseQuantizedConv2D_Int8_BiasInt8_FFT/0",
    "ChannelwiseQuantizedConv2D_Int8_BiasInt8_FTF/0",
    "ChannelwiseQuantizedConv2D_Int8_BiasInt8_FTT/0",
    "ChannelwiseQuantizedConv2D_Int8_BiasInt8_TFT/0",
    "ChannelwiseQuantizedConv2D_Int8_BiasInt8_TTF/0",
    "ChannelwiseQuantizedConv2D_Int8_BiasInt8_TTT/0",
    "ChannelwiseQuantizedConv2D_Int8_BiasInt32_FFF/0",
    "ChannelwiseQuantizedConv2D_Int8_BiasInt32_FFT/0",
    "ChannelwiseQuantizedConv2D_Int8_BiasInt32_FTF/0",
    "ChannelwiseQuantizedConv2D_Int8_BiasInt32_FTT/0",
    "ChannelwiseQuantizedConv2D_Int8_BiasInt32_TFF/0",
    "ChannelwiseQuantizedConv2D_Int8_BiasInt32_TFT/0",
    "ChannelwiseQuantizedConv2D_Int8_BiasInt32_TTF/0",
    "ChannelwiseQuantizedConv2D_Int8_BiasInt32_TTT/0",
    "ChannelwiseQuantizedConv2D_Int32Bias_SmallFilterData/0",
    "ChannelwiseQuantizedConv2D_Int32Bias_ZeroBiasData/0",
    "ChannelwiseQuantizedConv2D_Int8_BiasInt8/0",
    "ChannelwiseQuantizedConv2D_Int8_BiasInt32/0",
    "ChannelwiseQuantizedConv2D/0",
    "ChannelwiseQuantizedConv2D_NonZero_FloatBias/0",
    "ChannelwiseQuantizedConv2D_NonZero_QuantizedBias/0",
    "ChannelwiseQuantizedConv3D/0",
    "ParallelBatchMatMul_Float16/0",
    "RWQSLWSAllSame_Float16_AccumFP16/0",
    "RWQSLWSAllSame_Float16_AccumFP32/0",
    "sanityConvTranspose/0",
    "ConvTransposedAsymmetric/0",
    "ConvTransposedGroup/0",
    "convTransposeCompareSimpleK8S1P0I3/0",
    "convTransposeCompareSimpleK6S1P1I4/0",
    "convTransposeConvolutionCompareSimpleK5S1P2I3/0",
    "FusedRowwiseQuantizedSLWSAllLengthsOne_Float/0",
    "FusedRowwiseQuantizedSLWSAllLengthsOne_Float16_AccumFloat/0",
    "FusedRowwiseQuantizedSLWSAllLengthsOne_Float16_AccumFloat16/0",
    "FusedRowwiseQuantizedSLWSAllLengthsOne_Fused4Bit_Float16_AccumFloat16/0",
    "SLWSAllLengthsOne_Float16_AccumFloat/0",
    "TopK1int32/0",
    "mul_int32/0",
    "mul_int64/0",
    "add_int32/0",
    "add_int64/0",
    "FP16BatchNorm0D/0",
    "FP16BatchNorm1D/0",
    "FP16BatchNorm2D/0",
    "Int8BatchNorm2D/0",
    "FP16BatchNorm3D/0",
    "Int8BatchNorm3D/0",
    "LayerNorm_Float16/0",
    "LayerNorm_Int8/0",
    "DequantizeFRWQ_Float/0",
    "DequantizeFRWQ_Float16/0",
    "Not/0",
    "And/0",
    "Or/0",
    "Xor/0",
    "Abs_FloatTy/0",
    "Abs_Int8QTy/0",
    "Neg_FloatTy/0",
    "Neg_Int8QTy/0",
    "Floor_FloatTy/0",
    "Floor_Int8QTy/0",
    "Truncate_FloatTy/0",
    "Truncate_Float16Ty/0",
    "Truncate_Int8QTy/0",
    "Sign_FloatTy/0",
    "Sign_Int8QTy/0",
    "Ceil_FloatTy/0",
    "Ceil_Int8QTy/0",
    "Round_FloatTy/0",
    "Round_Int8QTy/0",
    "Sqrt_FloatTy/0",
    "Sqrt_Int8QTy/0",
    "Rsqrt_FloatTy/0",
    "Rsqrt_Int8QTy/0",
    "Reciprocal_FloatTy/0",
    "Reciprocal_Int8QTy/0",
    "Sin_FloatTy/0",
    "Sin_Int8QTy/0",
    "Cos_FloatTy/0",
    "Cos_Int8QTy/0",
    "Erf_FloatTy/0",
    "Erf_Int8QTy/0",
    "CmpNEQ_FloatTy/0",
    "CmpNEQ_Int8QTy/0",
    "CmpNEQ_Int32ITy/0",
    "CmpNEQ_Int64ITy/0",
    "CmpGT_FloatTy/0",
    "CmpGT_Int8QTy/0",
    "CmpGT_Int32ITy/0",
    "CmpGT_Int64ITy/0",
    "CmpGTE_FloatTy/0",
    "CmpGTE_Int8QTy/0",
    "CmpGTE_Int32ITy/0",
    "CmpGTE_Int64ITy/0",
    "rowwiseQuantizedFCTestAsymmetric_Int8_BiasFloat32/0",
    "rowwiseQuantizedFCTestSymmetric_Int8_BiasFloat32/0",
    "testBatchAdd_BFloat16/0",
    "testBatchMul_BFloat16/0",
    "ArithMul_bfloat16_t/0",
    "batchedReduceAdd_BFloat16/0",
    "BatchBoxCox_Medium_BFloat16/0",
    "BatchOneHotDataBFloat16/0",
    "BFloat16AdaptiveAvgPool/0",
    "BFloat16Max/0",
    "BFloat16Transpose2Dims/0",
    "ConvertFrom_BFloat16Ty_To_FloatTy/0",
    "ConvertFrom_BFloat16Ty_To_Int32ITy/0",
    "ConvertFrom_BFloat16Ty_To_Float16Ty_AndBack/0",
    "ConvertFrom_BFloat16Ty_To_Int32ITy_AndBack/0",
    "ConvertFrom_BFloat16Ty_To_Int64ITy/0",
    "ConvertFrom_BFloat16Ty_To_Int64ITy_AndBack/0",
    "ConvertFrom_BoolTy_To_BFloat16Ty/0",
    "ConvertFrom_FloatTy_To_BFloat16Ty/0",
    "ConvertFrom_Int32ITy_To_BFloat16Ty_AndBack/0",
    "ConvertFrom_Int64ITy_To_BFloat16Ty_AndBack/0",
    "convTest_BFloat16/0",
    "ExpandDims_BFloat16/0",
    "CumSum_BFloat16/0",
    "CumSum_Reverse_BFloat16/0",
    "less_bfloat16Cases/0",
    "Logit_BFloat16/0",
    "BFloat16Matmul/0",
    "ResizeBilinear_BFloat16/0",
    "ResizeBilinear_BFloat16_outTy/0",
    "ResizeNearest_BFloat16/0",
    "ResizeNearest_BFloat16_outTy/0",
    "sliceReshape_BFloat16/0",
    "Split_BFloat16/0",
    "EmbeddingBag_1D_BFloat16/0",
    "EmbeddingBag_2D_BFloat16/0",
    "TanHSweep_BFloat16/0",
    "ArithAdd_bfloat16_t/0",
    "ArithMax_bfloat16_t/0",
    "ArithMin_bfloat16_t/0",
    "ArithMul_bfloat16_t/0",
    "ArithSub_bfloat16_t/0",
    "BatchOneHotDataBFloat16/0",
    "ConvertFrom_Int32ITy_To_BFloat16Ty/0",
    "ConvertFrom_Int64ITy_To_BFloat16Ty/0",
    "batchedReduceAddWithAxis_BFloat16/0",
    "Sigmoid_BFloat16/0",
    "SigmoidSweep_BFloat16/0",
    "BFloat16Max/0",
    "ExpandDims_BFloat16/0",
    "SparseLengthsWeightedSum_1D_BFloat16/0",
    "BFloat16AvgPool/0",
    "BFloat16Transpose2Dims/0",
    "ConvertFrom_BFloat16Ty_To_FloatTy_AndBack/0",
    "SparseLengthsWeightedSum_2D_BFloat16/0",
    "dotProduct2D_BFloat16/0",
    "BatchBoxCox_Small_BFloat16/0",
    "ConvertFrom_Int64ITy_To_BFloat16Ty/0",
    "ConvertFrom_FloatTy_To_BFloat16Ty_AndBack/0",
    "BFloat16SoftMax/0",
    "BatchOneHotDataBFloat16/0",
    "BFloat16MaxPool/0",
    "ArithSub_bfloat16_t/0",
    "concatVectors_BFloat16/0",
    "Flatten_BFloat16Ty/0",
    "ArithMin_bfloat16_t/0",
    "BFloat16AvgPool3D/0",
    "SparseLengthsSum_BFloat16/0",
    "Transpose3Dims_BFloat16/0",
    "BFloat16BatchAdd/0",
    "BFloat16BatchMul/0",
    "ConvertFrom_BFloat16Ty_To_BFloat16Ty/0",
    "less_bfloat16Cases/0",
    "BatchBoxCox_Medium_BFloat16/0",
    "dotProduct1D_BFloat16/0",
    "BFloat16Add/0",
    "batchedReduceZeroDimResult_BFloat16/0",
    "ConvertFrom_Float16Ty_To_BFloat16Ty/0",
    "SparseLengthsSum_BFloat16_Int32/0",
    "ReluSimple_BFloat16/0",
    "ConvertFrom_Int32ITy_To_BFloat16Ty/0",
    "PReluSimple_BFloat16/0",
    "EmbeddingBag_1D_BFloat16_End_Offset/0",
    "GatherRangesDataBFloat16IdxInt32/0",
    "EmbeddingBag_2D_BFloat16_End_Offset/0",
    "BatchBoxCox_Large_BFloat16/0",
    "ArithMul_bfloat16_t/0",
    "ArithMax_bfloat16_t/0",
    "Swish_BFloat16/0",
    "BFloat16Reshape/0",
    "replaceNaN_BFloat16/0",
    "sliceConcatVectors_BFloat16/0",
    "GatherDataBFloat16IdxInt64/0",
    "ConvertFrom_BFloat16Ty_To_BFloat16Ty_AndBack/0",
    "ConvertFrom_Float16Ty_To_BFloat16Ty_AndBack/0",
    "concatVectorsRepeated_BFloat16/0",
    "ConvertFrom_BFloat16Ty_To_Float16Ty/0",
    "ConvertFusedToFusedFP16/0",
    "sliceVectors_BFloat16/0",
    "BFloat16Splat/0",
    "GatherDataBFloat16IdxInt32/0",
    "ArithAdd_bfloat16_t/0",
    "Exp_BFloat16/0",
    "Tanh_Float16/0",
    "TestFP32Accumulator/0",
    "LeakyRelu_Int8QTy/0",
    "RoiAlign/0",
    "RoiAlignWithAlignedCoordinates/0",
    "RoiAlignBatchIndexInBoxesTensor/0",
    "RoiAlignC2Batched/0",
    "RoiAlignRotatedBatchIndexInBoxesTensor/0",
    "FP16RoiAlign/0",
    "FP16RoiAlignWithAlignedCoordinates/0",
    "FP16RoiAlignBatchIndexInBoxesTensor/0",
    "FP16RoiAlignBatchIndexInBoxesTensorCompareToInterpreter/0",
    "FP16RoiAlignC2Batched/0",
    "FP16RoiAlignRotatedBatchIndexInBoxesTensor/0",
    "Asin_FloatTy/0",
    "Acos_FloatTy/0",
    "Atan_FloatTy/0",
    "Asin_Int8QTy/0",
    "Acos_Int8QTy/0",
    "Atan_Int8QTy/0",
    "PRelu_Int8/0",
    "BBoxTransform_Float/0",
    "BBoxTransform_Float16/0",
    "BBoxTransform_Rotated_Float/0",
    "BBoxTransform_Rotated_Float16/0",
    "BasicFloorDivNetFloatVsBFloat16/0",
    "BasicFloorDivNetFloatVsFloat16/0",
    "IntFloorDivBroadcast/0",
    "FloorDiv_FloatTy/0",
    "FloorDiv_Float16Ty/0",
    "FloorDiv_Int64ITy/0",
    "FloorDiv_Int32ITy/0",
    "FloorDiv_Int8QTy/0",
    "FloorDiv_Trunc_FloatTy/0",
    "FloorDiv_Trunc_Float16Ty/0",
    "FloorDiv_Trunc_Int64ITy/0",
    "FloorDiv_Trunc_Int32ITy/0",
    "FloorDiv_Trunc_Int8QTy/0",
    "VectorNorm_BFloat16/0",
    "VectorNorm_Float16/0",
    "VectorNorm_Float16Ty/0",
    "NonSquareDilationConvTranspose/0",
    "NonSquareDilationConv2D/0",
    "AvgPool2DLargePads_FloatTy_CountIncludePads/0",
    "AvgPool2DLargePads_FloatTy_CountExcludePads/0",
    "AvgPool2DLargePads_Int8QTy_CountIncludePads/0",
    "AvgPool2DLargePads_Int8QTy_CountExcludePads/0",
    "MaxPool2DLargePads_FloatTy/0",
    "MaxPool2DLargePads_Int8QTy/0",
    "Upsample_Nearest3D_Float/0",
    "Upsample_Nearest3D_Float16/0",
    "Upsample_Nearest3D_Int8/0",
    "Upsample_Nearest2D_Float/0",
    "Upsample_Nearest2D_Float16/0",
    "Upsample_Nearest2D_Int8/0",
    "Upsample_Nearest1D_Float/0",
    "Upsample_Nearest1D_Float16/0",
    "Upsample_Nearest1D_Int8/0",
    "batchedReduceAdd_Int32ITy/0",
};
