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
#ifndef TOOLS_CLASSGEN_BACKENDS_HABANA_HABANASPECIFICNODES_H
#define TOOLS_CLASSGEN_BACKENDS_HABANA_HABANASPECIFICNODES_H

BB.newNode("HabanaFullyConnected")
    .addInput("Input")
    .addInput("Weights")
    .addInput("Bias")
    .addMember(MemberType::Boolean, "DoRelu")
    .addResultFromCtorArg()
    .setDocstring(
        "This is a Habana-specific FC that combines weights and bias.");

BB.newNode("HabanaConvolution")
    .addInput("Input")
    .addInput("Filter")
    .addInput("Bias")
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .addMember(MemberType::Unsigned, "Group")
    .addMember(MemberType::Boolean, "DoRelu")
    .addResultFromCtorArg()
    .setDocstring("This is a Habana-specific convolution node that is "
                  "identical to the normal ConvolutionNode. That node "
                  "and convolution + relu are replaced with this one "
                  "for backend-specific code generation");

BB.newNode("HabanaConvolutionAdd")
    .addInput("Addend")
    .addInput("Input")
    .addInput("Filter")
    .addInput("Bias")
    .addMember(MemberType::VectorUnsigned, "Kernels")
    .addMember(MemberType::VectorUnsigned, "Strides")
    .addMember(MemberType::VectorUnsigned, "Pads")
    .addMember(MemberType::Unsigned, "Group")
    .addMember(MemberType::Boolean, "DoRelu")
    .addResultFromCtorArg()
    .setDocstring("This is similar to HabanaConvolution but has an "
                  "extra addend input to allow for a tensor to be added "
                  "to the convolution output.");

BB.newNode("HabanaReshape")
    .addInput("Input")
    .addMember(MemberType::VectorSizeT, "Dims")
    .addResultFromCtorArg()
    .setDocstring("This is a Habana-specific Reshape node that exists only to "
                  "bypass generic IR generation of Reshape so that it can "
                  "be implemented using a Habana kernel.");

BB.includeBackendSpecificVerification("glow/HabanaSpecificNodesVerification.h");

#endif // TOOLS_CLASSGEN_BACKENDS_HABANA_HABANASPECIFICNODES_H
#endif // GLOW_WITH_HABANA
