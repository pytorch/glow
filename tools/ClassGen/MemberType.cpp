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
#include "MemberType.h"

MemberTypeInfo kTypeRefTypeInfo{MemberType::TypeRef, "TypeRef", "TypeRef",
                                "TypeRef"};
MemberTypeInfo kFloatTypeInfo{MemberType::Float, "float", "float", "float"};
MemberTypeInfo kUnsignedTypeInfo{MemberType::Unsigned, "unsigned_t",
                                 "unsigned_t", "unsigned_t"};
MemberTypeInfo kBooleanTypeInfo{MemberType::Boolean, "bool", "bool", "bool"};
MemberTypeInfo kStringTypeInfo{MemberType::String, "std::string", "std::string",
                               "std::string"};
MemberTypeInfo kVectorFloatTypeInfo{MemberType::VectorFloat,
                                    "llvm::ArrayRef<float>",
                                    "std::vector<float>", "std::vector<float>"};
MemberTypeInfo kVectorUnsignedTypeInfo{
    MemberType::VectorUnsigned, "llvm::ArrayRef<unsigned_t>",
    "std::vector<unsigned_t>", "std::vector<unsigned_t>"};
MemberTypeInfo kVectorSizeTTypeInfo{
    MemberType::VectorSizeT, "llvm::ArrayRef<size_t>", "std::vector<size_t>",
    "std::vector<size_t>"};
MemberTypeInfo kVectorNodeValueTypeInfo{
    MemberType::VectorNodeValue, "NodeValueArrayRef", "std::vector<NodeHandle>",
    "std::vector<NodeValue>"};
