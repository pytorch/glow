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

#include "LLVMIRGen.h"
#include "AllocationsInfo.h"

#include "glow/IR/IR.h"

#include "gtest/gtest.h"

using namespace glow;

#ifndef GLOW_WITH_CPU
#error "This should be compiled with the CPU backend"
#endif

/// Check that get/setMainEntryName behaves in a sane way.
TEST(LLVMIRGen, getEntryName) {
  IRFunction irfunc;
  AllocationsInfo allocInfo;
  LLVMIRGen llvmIRGen(&irfunc, allocInfo, "name");
  EXPECT_EQ(llvmIRGen.getMainEntryName(), "name");

  llvmIRGen.setMainEntryName("customName");
  EXPECT_EQ(llvmIRGen.getMainEntryName(), "customName");

  // If we set an empty name we get "main".
  llvmIRGen.setMainEntryName("");
  EXPECT_EQ(llvmIRGen.getMainEntryName(), "main");
}
