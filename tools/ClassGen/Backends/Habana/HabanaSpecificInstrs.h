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

#ifdef GLOW_WITH_HABANA
#ifndef TOOLS_CLASSGEN_BACKENDS_HABANA_HABANASPECIFICINSTRS_H
#define TOOLS_CLASSGEN_BACKENDS_HABANA_HABANASPECIFICINSTRS_H

BB.newBackendSpecificInstr("HabanaFullyConnected")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Weights", OperandKind::In)
    .addOperand("Bias", OperandKind::In)
    .addMember(MemberType::Boolean, "DoRelu")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src", "Weights"});

BB.newBackendSpecificInstr("HabanaConvolution")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Filter", OperandKind::In)
    .addOperand("Bias", OperandKind::In)
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .addMember(MemberType::Unsigned, "Group")
    .addMember(MemberType::Boolean, "DoRelu")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("HabanaConvolutionAdd")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addOperand("Filter", OperandKind::In)
    .addOperand("Bias", OperandKind::In)
    .addOperand("Addend", OperandKind::In)
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .addMember(MemberType::Unsigned, "Group")
    .addMember(MemberType::Boolean, "DoRelu")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.newBackendSpecificInstr("HabanaReshape")
    .addOperand("Dest", OperandKind::Out)
    .addOperand("Src", OperandKind::In)
    .addMember(MemberType::VectorSizeT, "Dims")
    .autoIRGen()
    .autoVerify(VerifyKind::SameElementType, {"Dest", "Src"});

BB.includeBackendSpecificVerification(
    "glow/HabanaSpecificInstrsVerification.h");

#endif // TOOLS_CLASSGEN_BACKENDS_HABANA_HABANASPECIFICINSTRS_H
#endif // GLOW_WITH_HABANA
