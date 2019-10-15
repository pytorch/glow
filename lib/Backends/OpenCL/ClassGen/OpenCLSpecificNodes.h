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

#ifdef GLOW_WITH_OPENCL

BB.newNode("OCLBatchedReduceAdd")
    .addInput("Input")
    .addInput("DestSliceSizes")
    .addInput("SrcSliceSizes")
    .addMember(MemberType::Unsigned, "Axis")
    .addMember(MemberType::Unsigned, "AxisSrcSliceSize")
    .addResultFromCtorArg()
    .setDocstring(
        "This is an OpenCL-specific BatchedReduceAdd operation which has the "
        "slice sizes of the input and output as explicit inputs.");

BB.includeBackendSpecificVerification("glow/OpenCLSpecificNodesVerification.h");

#endif // GLOW_WITH_CPU
