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

#ifdef GLOW_WITH_NNPI
BB.includeHeader("glow/NNPISpecificTypes.h");

BB.newNode("NNPICustomDSP")
    .addMember(MemberType::VectorNodeValue, "Inputs")
    .addResultFromCtorArg()   // for now use single output
    .addInput("KernelParams") // paramsblob
    .addInput("WalkConfig")   // NNPIWalkConfig
    .addMember(MemberType::Unsigned, "PrivateAreaSize")
    .addMember(MemberType::String, "KernelName")
    .addMember(MemberType::Int64, "ICERefCallback") // NNPIDspIceRefCallback*
    .setDocstring("This is an experimental NNPI-specific node representing a "
                  "custom DSP op");

BB.newNode("NNPICustomIA")
    .addMember(MemberType::VectorNodeValue, "Inputs")
    .addResultFromCtorArg()   // for now use single output
    .addInput("KernelParams") // paramsblob
    .addMember(MemberType::String, "KernelName")
    .addMember(MemberType::String, "IAPath")
    .addMember(MemberType::Int64,
               "ICERefCallback") // NNPICustomIAIceRefCallback*
    .addMember(MemberType::Int64, "PointerToIALib")
    .addMember(MemberType::Unsigned, "SizeOfIALib")
    .setDocstring("This is an experimental NNPI-specific node representing a "
                  "custom IA op");

BB.newNode("NNPILookupTable")
    .addInput("Input")
    .addInput("LookupTable")
    .addMember(MemberType::Enum, "LookupType")
    .addMember(MemberType::Float, "LowerRange")
    .addMember(MemberType::Float, "UpperRange")
    .addMember(MemberType::Float, "UpperMulFactor")
    .addMember(MemberType::Float, "UpperOffset")
    .addMember(MemberType::Float, "LowerMulFactor")
    .addMember(MemberType::Float, "LowerOffset")
    .addMember(MemberType::Float, "Delta")
    .addMember(MemberType::Float, "Bias")
    .addResultFromCtorArg() // for now use single output
    .setDocstring("This is an experimental NNPI-specific node representing a "
                  "LUT op");

BB.includeBackendSpecificVerification("glow/NNPISpecificNodesVerification.h");
#endif // GLOW_WITH_NNPI
