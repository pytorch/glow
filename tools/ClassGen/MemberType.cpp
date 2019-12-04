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
#include "MemberType.h"

MemberTypeInfo kTypeRefTypeInfo{MemberType::TypeRef, "TypeRef", "TypeRef",
                                "TypeRef"};
MemberTypeInfo kFloatTypeInfo{MemberType::Float, "float", "float", "float"};
MemberTypeInfo kUnsignedTypeInfo{MemberType::Unsigned, "unsigned_t",
                                 "unsigned_t", "unsigned_t"};
MemberTypeInfo kBooleanTypeInfo{MemberType::Boolean, "bool", "bool", "bool"};
MemberTypeInfo kInt64TypeInfo{MemberType::Int64, "int64_t", "int64_t",
                              "int64_t"};
MemberTypeInfo kStringTypeInfo{MemberType::String, "std::string", "std::string",
                               "std::string"};
MemberTypeInfo kVectorFloatTypeInfo{MemberType::VectorFloat,
                                    "llvm::ArrayRef<float>",
                                    "std::vector<float>", "std::vector<float>"};
MemberTypeInfo kVectorUnsignedTypeInfo{
    MemberType::VectorUnsigned, "llvm::ArrayRef<unsigned_t>",
    "std::vector<unsigned_t>", "std::vector<unsigned_t>"};
MemberTypeInfo kVectorInt64TypeInfo{
    MemberType::VectorInt64, "llvm::ArrayRef<int64_t>", "std::vector<int64_t>",
    "std::vector<int64_t>"};
MemberTypeInfo kVectorSignedTypeInfo{MemberType::VectorSigned,
                                     "llvm::ArrayRef<int>", "std::vector<int>",
                                     "std::vector<int>"};
MemberTypeInfo kVectorSizeTTypeInfo{
    MemberType::VectorSizeT, "llvm::ArrayRef<size_t>", "std::vector<size_t>",
    "std::vector<size_t>"};
MemberTypeInfo kVectorDimTTypeInfo{MemberType::VectorDimT,
                                   "llvm::ArrayRef<dim_t>",
                                   "std::vector<dim_t>", "std::vector<dim_t>"};
MemberTypeInfo kVectorNodeValueTypeInfo{
    MemberType::VectorNodeValue, "NodeValueArrayRef", "std::vector<NodeHandle>",
    "std::vector<NodeValue>"};
// We currently use 'unsigned_t' to represent a enum in the generic API. It can
// carry fully the enum information even of the signess of enum is
// implementation defined.
// C++03: The underlying type of an enumeration is an integral type that can
// represent all the enumerator values defined in the enumeration. It is
// implementation-defined which integral type is used as the underlying type for
// an enumeration except that the underlying type shall not be larger than int
// unless the value of an enumerator cannot fit in an int or unsigned int.
MemberTypeInfo kEnumTypeInfo{MemberType::Unsigned, "unsigned_t", "unsigned_t",
                             "unsigned_t"};
