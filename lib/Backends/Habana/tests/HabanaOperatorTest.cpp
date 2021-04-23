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
    "AdaptiveAvgPool/0",
    "AdaptiveAvgPoolNonSquare/0",
    "ArithAdd_float16_t/0",
    "ArithAdd_int32_t/0",
    "ArithAdd_int64_t/0",
    "ArithMax_float16_t/0",
    "ArithMax_int32_t/0",
    "ArithMax_int64_t/0",
    "ArithMin_float16_t/0",
    "ArithMin_int32_t/0",
    "ArithMin_int64_t/0",
    "ArithMul_float16_t/0",
    "ArithMul_int32_t/0",
    "ArithMul_int64_t/0",
    "ArithSub_float16_t/0",
    "ArithSub_int32_t/0",
    "ArithSub_int64_t/0",
    "AvgPool/0",
    "BasicAddNetFloatVsFloat16/0",
    "BasicAddNetFloatVsInt8/0",
    "BasicDivNetFloatVsFloat16/0",
    "BasicDivNetFloatVsInt8/0",
    "BasicMaxNetFloatVsFloat16/0",
    "BasicMaxNetFloatVsInt8/0",
    "BasicMinNetFloatVsFloat16/0",
    "BasicMinNetFloatVsInt8/0",
    "BasicMulNetFloatVsFloat16/0",
    "BasicMulNetFloatVsInt8/0",
    "BasicSubNetFloatVsFloat16/0",
    "BasicSubNetFloatVsInt8/0",
    "BatchAdd/0",
    "BatchBoxCox_Large_Float16/0",
    "BatchBoxCox_Medium_Float16/0",
    "BatchBoxCox_Small_Float16/0",
    "BatchedGather/0",
    "batchedPairwiseDotProduct/0",
    "batchedReduceAdd_5Dinput/0",
    "batchedReduceAdd_Float16/0",
    "batchedReduceAddQuantized/0",
    "batchedReduceAddQuantizedWithAxis/0",
    "batchedReduceAddWithAxis_Float16/0",
    "batchedReduceAddWithAxis_Int8Q/0",
    "batchedReduceMeanQuantized/0",
    "batchedReduceMeanQuantizedWithAxis/0",
    "batchedReduceMeanUsingAvgPool/0",
    "batchedReduceMeanUsingAvgPoolQuantized/0",
    "batchedReduceMin_Float/0",
    "batchedReduceMin_Int32/0",
    "batchedReduceMin_Int64/0",
    "batchedReduceMin_Int8/0",
    "batchedReduceProd_Float/0",
    "batchedReduceProd_Float16/0",
    "batchedReduceProd_BFloat16/0",
    "batchedReduceProd_Int32/0",
    "batchedReduceProd_Int64/0",
    "batchedReduceProd_Int8/0",
    "batchedReduceMinMultiAxis_Float/0",
    "batchedReduceMinMultiAxis_Int32/0",
    "batchedReduceMinMultiAxis_Int64/0",
    "batchedReduceMinMultiAxis_Int8/0",
    "batchedReduceMax_Float/0",
    "batchedReduceMax_Int32/0",
    "batchedReduceMax_Int64/0",
    "batchedReduceMax_Int8/0",
    "batchedReduceMaxMultiAxis_Float/0",
    "batchedReduceMaxMultiAxis_Int32/0",
    "batchedReduceMaxMultiAxis_Int64/0",
    "batchedReduceMaxMultiAxis_Int8/0",
    "batchedReduceZeroDimResult_Float/0",
    "batchedReduceZeroDimResult_Float16/0",
    "batchedReduceZeroDimResult_Int8/0",
    "BatchOneHotDataFloat/0",
    "BatchOneHotDataFloat16/0",
    "BatchOneHotDataInt64/0",
    "BatchOneHotDataInt8/0",
    "BoolTranspose2Dims/0",
    "BoolReshape/0",
    "Bucketize/0",
    "Clip/0",
    "CmpEQ_Int32/0",
    "CmpEQ_Int64/0",
    "CmpLTE/0",
    "ConcatTopK/0",
    "concatVectors_Bool/0",
    "concatVectors_Float16/0",
    "concatVectorsRepeated_Bool/0",
    "concatVectorsRepeated_Float16/0",
    "concatVectorsRepeated_Int32/0",
    "concatVectorsRepeated_Int64/0",
    "Conv3DQuantizedTest_Int8_BiasInt32",
    "ConvertFrom_Float16Ty_To_Float16Ty/0",
    "ConvertFrom_Float16Ty_To_Float16Ty_AndBack/0",
    "ConvertFrom_Float16Ty_To_FloatTy/0",
    "ConvertFrom_Float16Ty_To_FloatTy_AndBack/0",
    "ConvertFrom_Float16Ty_To_Int32ITy/0",
    "ConvertFrom_Float16Ty_To_Int32ITy_AndBack/0",
    "ConvertFrom_Float16Ty_To_Int64ITy/0",
    "ConvertFrom_Float16Ty_To_Int64ITy_AndBack/0",
    "ConvertFrom_FloatTy_To_Float16Ty/0",
    "ConvertFrom_FloatTy_To_Float16Ty_AndBack/0",
    "ConvertFrom_FloatTy_To_Int32ITy/0",
    "ConvertFrom_FloatTy_To_Int32ITy_AndBack/0",
    "ConvertFrom_FloatTy_To_Int64ITy/0",
    "ConvertFrom_FloatTy_To_Int64ITy_AndBack/0",
    "ConvertFrom_Int32ITy_To_Float16Ty/0",
    "ConvertFrom_Int32ITy_To_Float16Ty_AndBack/0",
    "ConvertFrom_Int32ITy_To_FloatTy/0",
    "ConvertFrom_Int32ITy_To_FloatTy_AndBack/0",
    "ConvertFrom_Int32ITy_To_Int64ITy/0",
    "ConvertFrom_Int64ITy_To_Float16Ty/0",
    "ConvertFrom_Int64ITy_To_Float16Ty_AndBack/0",
    "ConvertFrom_Int64ITy_To_FloatTy/0",
    "ConvertFrom_Int64ITy_To_FloatTy_AndBack/0",
    "ConvertFrom_Int64ITy_To_Int32ITy/0",
    "ConvertFrom_Int64ITy_To_Int32ITy_AndBack/0",
    "ConvertFrom_BoolTy_To_FloatTy/0",
    "ConvertFrom_BoolTy_To_Float16Ty/0",
    "ConvertFusedToFusedFP16/0",
    "convTest/0",
    "convTest_Float16/0",
    "CumSum_Float/0",
    "CumSum_Float16/0",
    "CumSum_Int32/0",
    "CumSum_Int64/0",
    "CumSum_Exclusive/0",
    "CumSum_Reverse/0",
    "CumSum_ExclusiveReverse/0",
    "CumSum_WithZeroes/0",
    "DilatedConvolution/0",
    "DivSizeT/0",
    "dotProduct1D_Float16/0",
    "dotProduct1D_Int8/0",
    "dotProduct1D_Int8/0",
    "dotProduct2D_Float16/0",
    "dotProduct2D_Int8/0",
    "elementwiseLinear/0",
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
    "EntropyLossTest/0",
    "Exp/0",
    "ExpandDims_Float/0",
    "ExpandDims_Float16/0",
    "ExpandDims_Int8/0",
    "Exp_Float16/0",
    "FC_Float16/0",
    "Flatten_Float16Ty/0",
    "FloatArgMaxKeepDim/0",
    "FloatArgMaxNoKeepDim/0",
    "Float16ArgMaxKeepDim/0",
    "Float16ArgMaxNoKeepDim/0",
    "FloatMaxPoolWithArgmax/0",
    "FloatMaxPoolWithArgmaxTransposed/0",
    "FP16AdaptiveAvgPool/0",
    "FP16Add/0",
    "FP16AvgPool/0",
    "FP16BatchAdd/0",
    "FP16ConvolutionDepth10/0",
    "FP16ConvolutionDepth8/0",
    "FP16Matmul/0",
    "FP16Max/0",
    "FP16MaxPool/0",
    "FP16Reshape/0",
    "FP16SoftMax/0",
    "Fp16Splat/0",
    "FP16Transpose2Dims/0",
    "FusedRowwiseQuantizedSLWSTwoColumn_Float16_AccumFloat/0",
    "FusedRowwiseQuantizedSLWSTwoColumn_Float16_AccumFloat16/0",
    "FusedRowwiseQuantizedSLWSTwoColumn_Fused4Bit_Float16_AccumFloat16/0",
    "FusedRowwiseQuantizedSparseLengthsSum_Float16_AccumFloat/0",
    "FusedRowwiseQuantizedSparseLengthsSum_Float16_AccumFloat16/0",
    "FusedRowwiseQuantizedSparseLengthsSum_Fused4Bit_Float16_AccumFloat16/0",
    "FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16/0",
    "FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16_back_to_"
    "back/0",
    "FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16_back_to_"
    "back2/0",
    "FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16_"
    "NoFusedConvert/0",
    "FusedRowwiseQuantizedSparseLengthsWeightedSum_ConvertedFloat16_"
    "NoFusedConvert_FP32Accum/0",
    "FusedRowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat/0",
    "FusedRowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat16/0",
    "EmbeddingBagByteRowwiseOffsets_ConvertedFloat16/0",
    "EmbeddingBagByteRowwiseOffsets_Float/0",
    "EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat/0",
    "EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat16/0",
    "EmbeddingBagByteRowwiseOffsets_ConvertedFloat16_End_Offset/0",
    "EmbeddingBagByteRowwiseOffsets_Float_End_Offset/0",
    "EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat_End_Offset/0",
    "EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat16_End_Offset/0",
    "EmbeddingBag4BitRowwiseOffsets_Float16/0",
    "EmbeddingBag4BitRowwiseOffsets_Float16_HasEndOffset_AccumFloat/0",
    "EmbeddingBag4BitRowwiseOffsets_Float16_AccumFloat/0",
    "EmbeddingBag4BitRowwiseOffsets_Float16_HasEndOffset/0",
    "FusedRWQSLSAllZeroLengths_Float/0",
    "FusedRWQSLSAllZeroLengths_Float16/0",
    "GatherDataFloat16IdxInt32/0",
    "GatherDataFloat16IdxInt64/0",
    "GatherDataFloatIdxInt32/0",
    "GatherDataFloatIdxInt64/0",
    "GatherDataInt8IdxInt32/0",
    "GatherDataInt8IdxInt64/0",
    "GatherND_FloatTy_Int32ITy_Test1/0",
    "GatherND_FloatTy_Int32ITy_Test2/0",
    "GatherND_FloatTy_Int32ITy_Test3/0",
    "GatherND_FloatTy_Int32ITy_Test4/0",
    "GatherND_FloatTy_Int32ITy_Test5/0",
    "GatherND_FloatTy_Int64ITy_Test1/0",
    "GatherND_FloatTy_Int64ITy_Test2/0",
    "GatherND_FloatTy_Int64ITy_Test3/0",
    "GatherND_FloatTy_Int64ITy_Test4/0",
    "GatherND_FloatTy_Int64ITy_Test5/0",
    "GatherND_Float16Ty_Int32ITy_Test1/0",
    "GatherND_Float16Ty_Int32ITy_Test2/0",
    "GatherND_Float16Ty_Int32ITy_Test3/0",
    "GatherND_Float16Ty_Int32ITy_Test4/0",
    "GatherND_Float16Ty_Int32ITy_Test5/0",
    "GatherND_Float16Ty_Int64ITy_Test1/0",
    "GatherND_Float16Ty_Int64ITy_Test2/0",
    "GatherND_Float16Ty_Int64ITy_Test3/0",
    "GatherND_Float16Ty_Int64ITy_Test4/0",
    "GatherND_Float16Ty_Int64ITy_Test5/0",
    "GatherNDDataInt8IdxInt32/0",
    "GatherNDDataInt8IdxInt64/0",
    "GatherRangesDataFloat16IdxInt32/0",
    "GatherRangesDataFloat16IdxInt64/0",
    "GatherRangesDataFloatIdxInt32/0",
    "GatherRangesDataFloatIdxInt64/0",
    "GatherRangesDataInt64IdxInt32/0",
    "GatherRangesDataInt64IdxInt64/0",
    "GatherRangesDataInt8QIdxInt32/0",
    "GatherRangesDataInt8QIdxInt64/0",
    "GatherSizeT/0",
    "GatherWithInt32PartialTensors/0",
    "GatherWithInt64PartialTensors/0",
    "Gelu_Float/0",
    "Gelu_Float16/0",
    "GroupConv3D/0",
    "GroupConvolution/0",
    "GroupDilatedConvolution/0",
    "ChannelwiseQuantizedGroupConvolution/0",
    "ChannelwiseQuantizedGroupConvolution3D/0",
    "ChannelwiseQuantizedGroupConvolutionNonZero/0",
    "insertTensorTest/0",
    "insertTensorTest3D/0",
    "insertTensorCrossDimensions/0",
    "insertTensorPartialSliceInnerDim/0",
    "Int16ConvolutionDepth10/0",
    "Int16ConvolutionDepth8/0",
    "Int8AdaptiveAvgPool/0",
    "Int8ConvolutionDepth10/0",
    "Int8ConvolutionDepth10/0",
    "Int8ConvolutionDepth8/0",
    "Int8ConvolutionDepth8/0",
    "Int8Log/0",
    "Int8Sigmoid/0",
    "Int8Tanh/0",
    "IntBatchedArith/0",
    "IntLookupTable/0",
    "IntMatMul/0",
    "IntSplat/0",
    "LengthsToRanges/0",
    "less_broadcast_float/0",
    "less_float/0",
    "less_float16Cases/0",
    "less_floatCases/0",
    "less_int32Cases/0",
    "less_int64Cases/0",
    "less_int8/0",
    "Logit_Float/0",
    "Logit_Float16/0",
    "LSTMUnitFP16/0",
    "PyTorchLSTMFP16/0",
    "DynamicQuantizedFullyConnectedBasic/0",
    "DynamicQuantizedFullyConnectedStrongWeights/0",
    "DynamicRowwiseQuantizedFullyConnectedBasic/0",
    "matmulQuantized_InterpCompareParClone/0",
    "MaxPool/0",
    "ModuloInt32NoSignFollow/0",
    "ModuloInt32SignFollow/0",
    "ModuloInt64NoSignFollow/0",
    "ModuloInt64SignFollow/0",
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
    "NonCubicPaddingConv3D/0",
    "NonCubicStrideConv3D/0",
    "NonSquareKernelAveragePool/0",
    "NonSquareKernelConvolution/0",
    "NonSquareKernelMaxPool/0",
    "NonSquarePaddingAveragePool/0",
    "NonSquarePaddingConvolution/0",
    "NonSquarePaddingMaxPool/0",
    "NonSquareStrideAveragePool/0",
    "NonSquareStrideConvolution/0",
    "QuantizedTranspose/0",
    "ParallelBatchMatMul_Float16/0",
    "ParallelBatchMatMul_Int8/0",
    "pow/0",
    "PReluSimple_Float/0",
    "PReluSimple_Float16/0",
    "PRelu_Int8/0",
    "QuantizedArgMaxKeepDim/0",
    "QuantizedArgMaxNoKeepDim/0",
    "QuantizedArithmeticRescaled/0",
    "QuantizedArithmeticUnrescaled/0",
    "QuantizedCmpLTEAndSelect/0",
    "QuantizedMaxPoolWithArgmax/0",
    "QuantizedMaxPoolWithArgmaxTransposed/0",
    "QuantizedTopK/0",
    "ReluSimple_Float/0",
    "ReluSimple_Float16/0",
    "replaceNaN_Float/0",
    "replaceNaN_Float16/0",
    "Reshape/0",
    "ResizeNearest_Float/0",
    "ResizeNearest_Float16/0",
    "ResizeNearest_Int16/0",
    "ResizeNearest_Int32/0",
    "ResizeNearest_Int8/0",
    "rowwiseQuantizedFCTest/0",
    "rowwiseQuantizedFCTestSymmetric/0",
    "rowwiseQuantizedSLWSTest/0",
    "RowwiseQuantizedSparseLengthsSum_Float/0",
    "RowwiseQuantizedSparseLengthsSum_Float16_AccumFloat/0",
    "RowwiseQuantizedSparseLengthsSum_Float16_AccumFloat16/0",
    "RowwiseQuantizedSparseLengthsWeightedSum_Float/0",
    "RowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat/0",
    "RowwiseQuantizedSparseLengthsWeightedSum_Float16_AccumFloat16/0",
    "RWQSLWSAllSame_Float16_AccumFP16/0",
    "RWQSLWSAllSame_Float16_AccumFP32/0",
    "ScatterAddNDimensionalDuplicatingIndices/0",
    "ScatterAddNDimensionalSimple/0",
    "ScatterAddQuantized/0",
    "ScatterData/0",
    "ScatterDataCumulative/0",
    "ScatterDataNDimensional/0",
    "ScatterDataNDimensionalSimple/0",
    "ScatterDataQuantized/0",
    "Select/0",
    "SigmoidCrossEntropyWithLogits/0",
    "Sigmoid_Float16/0",
    "Swish_Float16/0",
    "SigmoidSweep_Float16/0",
    "TanHSweep_Float16/0",
    "simpleCmpSelectPredication/0",
    "sliceConcatVectors_Float16/0",
    "sliceConcatVectors_Int64/0",
    "sliceReshape_Float16/0",
    "sliceVectors_Float16/0",
    "sliceVectors_Int64/0",
    "SLWSTwoColumn_Float16_AccumFloat/0",
    "SLSAllZeroLengths_Float/0",
    "SLSAllZeroLengths_Float16/0",
    "SoftMax/0",
    "spaceToDepth_block2_Float/0",
    "spaceToDepth_block2_int8/0",
    "spaceToDepth_block3_Float/0",
    "spaceToDepth_block3_int8/0",
    "SparseLengthsSum_Float/0",
    "SparseLengthsSum_Float16/0",
    "SparseLengthsSumI8/0",
    "SparseLengthsWeightedSum_1D_Float/0",
    "SparseLengthsWeightedSum_1D_Float16/0",
    "SparseLengthsWeightedSumOffsets_1D_Float/0",
    "SparseLengthsWeightedSumOffsets_1D_Float16/0",
    "SparseLengthsWeightedSum_2D_Float16/0",
    "SparseLengthsWeightedSumOffsets_2D_Float/0",
    "SparseLengthsWeightedSumOffsets_2D_Float16/0",
    "SparseLengthsWeightedSumI8/0",
    "SparseToDense_Float/0",
    "SparseToDense_Int64/0",
    "SparseToDenseMask1/0",
    "SparseToDenseMask2/0",
    "SparseLabelSplit/0",
    "Split_Float16/0",
    "SqueezeExpand/0",
    "Tanh/0",
    "Tanh_Float16/0",
    "testBatchAdd_Float16/0",
    "testQuantizedBatchAdd_Int8/0",
    "testQuantizedBatchAdd_Int32/0",
    "TopK/0",
    "TopK1/0",
    "Transpose2Dims/0",
    "Transpose3Dims_Float16/0",
    "TransposeIntoReshapeOptim/0",
    "where_2d_broadcast_x_y_i8/0",
    "where_2d_wise_float/0",
    "where_2d_wise_i8/0",
    "where_element_wise_float/0",
    "where_row_wise_float/0",
    "sanityConvTranspose2OutCh/0",
    "sanityConvTranspose1OutCh/0",
    "sanityConvTransposeStrided/0",
    "sanityConvTransposePads/0",
    "convTransposeCompareSimpleK8S1P0I3/0",
    "convTransposeCompareSimpleK6S1P1I4/0",
    "convTransposeConvolutionCompareSimpleK5S1P2I3/0",
    "EmbeddingBag_1D_Float_End_Offset_Partial/0",
    "EmbeddingBag_2D_Float_End_Offset_Partial/0",
    "EmbeddingBagByteRowwiseOffsets_Float_End_Offset_Partial/0",
    "EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat_End_Offset_Partial/0",
    "EmbeddingBagByteRowwiseOffsets_Float16_AccumFloat16_End_Offset_Partial/0",
    "FusedRowwiseQuantizedSLWSAllLengthsOne_Float16_AccumFloat/0",
    "FusedRowwiseQuantizedSLWSAllLengthsOne_Float16_AccumFloat16/0",
    "FusedRowwiseQuantizedSLWSAllLengthsOne_Fused4Bit_Float16_AccumFloat16/0",
    "SLWSAllLengthsOne_Float16_AccumFloat/0",
    "TopK1int32/0",
    "mul_int32/0",
    "mul_int64/0",
    "add_int32/0",
    "add_int64/0",
    "floor/0",
    "Erf_FloatTy/0",
    "Erf_Int8QTy/0",
    "TestFP32Accumulator/0",
    "ROIAlign/0",
    "ROIAlignImplicit/0",
    "Sign_FloatTy/0",
    "Sign_Int8QTy/0",
    "TestFP32Accumulator/0",
    "RoiAlign/0",
    "RoiAlignWithAlignedCoordinates/0",
    "RoiAlignBatchIndexInBoxesTensor/0",
    "RoiAlignC2Batched/0",
    "RoiAlignRotatedBatchIndexInBoxesTensor/0",
    "FP16RoiAlign/0",
    "FP16RoiAlignWithAlignedCoordinates/0",
    "FP16RoiAlignRotatedBatchIndexInBoxesTensor/0",
    "FP16RoiAlignC2Batched/0",
    "FP16RoiAlignBatchIndexInBoxesTensor/0",
    "FP16RoiAlignBatchIndexInBoxesTensorCompareToInterpreter/0",
    "Asin_FloatTy/0",
    "Acos_FloatTy/0",
    "Atan_FloatTy/0",
    "Asin_Int8QTy/0",
    "Acos_Int8QTy/0",
    "Atan_Int8QTy/0",
    "BBoxTransform/0",
    "NonSquareDilationConvTranspose/0",
    "NonSquareDilationConv2D/0",
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
    "CollectRpnProposals/0",
};
