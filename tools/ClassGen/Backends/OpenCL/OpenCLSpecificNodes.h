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

#ifdef GLOW_WITH_OPENCL

BB.newNode("OCLConvolution")
    .addInput("Input")
    .addInput("Filter")
    .addInput("Bias")
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .addMember(MemberType::Unsigned, "Group")
    .addResultFromCtorArg()
    .setDocstring(
        "This is an OpenCL-specific convolution implementation where the "
        "filter, the bias and the input are in the HCHW format");

BB.newNode("OCLAvgPool")
    .addInput("Input")
    .addMember(MemberType::Unsigned, "Kernel")
    .addMember(MemberType::Unsigned, "Stride")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .addResultFromCtorArg()
    .setDocstring(
        "This is an OpenCL-specific Average Pool operation on the Input given "
        "provided Kernel, Stride, and Pads. The input and output are in NCHW "
        "format");

BB.newNode("OCLMaxPool")
    .addInput("Input")
    .addMember(MemberType::Unsigned, "Kernel")
    .addMember(MemberType::Unsigned, "Stride")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .addResultFromCtorArg()
    .setDocstring(
        "This is an OpenCL-specific Max Pool operation on the Input given "
        "provided "
        "Kernel, Stride, and Pads. The input and output are in NCHW format");

BB.includeBackendSpecificVerification("OpenCLSpecificNodesVerification.h");

#endif // GLOW_WITH_CPU
