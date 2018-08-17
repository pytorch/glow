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

#include "glow/Graph/Graph.h"
#include "glow/IR/IR.h"

#include "gtest/gtest.h"

using namespace glow;

#ifndef GLOW_WITH_CPU
#error "This shouldn't be compiled without the CPU backend"
#endif

/// Check that we get a non-empty entry name for various
/// situation.
TEST(LLVMIRGen, getEntryName) {
  Module mod;
  Function *func = mod.createFunction("funcname");
  IRFunction irfunc(func);
  AllocationsInfo allocInfo;
  LLVMIRGen llvmIRGen(&irfunc, allocInfo, "name");
  // FIXME: We actually use the name of the high level function
  // as soon as the name we set for LLVMIRGen is not empty.
  EXPECT_EQ(llvmIRGen.getMainEntryName(), "funcname");

  // FIXME: Disturbing enough the setter doesn't help with the situation.
  llvmIRGen.setMainEntryName("customName");
  EXPECT_EQ(llvmIRGen.getMainEntryName(), "funcname");

  // If we set an empty name we get "main".
  llvmIRGen.setMainEntryName("");
  EXPECT_EQ(llvmIRGen.getMainEntryName(), "main");

  // Now check that we get a reasonable naming when we are given paths
  // for the function names. This happens when using the BundleSaver.
  // In that context, the entry name is used to create files.
  // Therefore it should not include path delimiters.

  // FIXME: Given the behavior of setMainEntryName, the only way to
  // change the name is to change the name of the function.
  func = mod.createFunction("path/to/model_dir");
  IRFunction irfuncTmp(func);
  LLVMIRGen llvmIRGenTmp(&irfuncTmp, allocInfo, "name");

  // Check that we only get the last part of the path.
  EXPECT_EQ(llvmIRGenTmp.getMainEntryName(), "model_dir");

  // Now end the name with a delimiter.
  func = mod.createFunction("path/to/model_dir/");
  IRFunction irfuncTmp1(func);
  LLVMIRGen llvmIRGenTmp1(&irfuncTmp1, allocInfo, "name");

  // Check that we only get the last part of the path.
  EXPECT_EQ(llvmIRGenTmp1.getMainEntryName(), "model_dir");

  // Now end the name with several delimiters.
  func = mod.createFunction("path/to/model_dir////");
  IRFunction irfuncTmp2(func);
  LLVMIRGen llvmIRGenTmp2(&irfuncTmp2, allocInfo, "name");

  // Check that we only get the last part of the path.
  EXPECT_EQ(llvmIRGenTmp2.getMainEntryName(), "model_dir");

  // Now with several delimiters sparkled around.
  func = mod.createFunction("//path/to///model_dir////");
  IRFunction irfuncTmp3(func);
  LLVMIRGen llvmIRGenTmp3(&irfuncTmp3, allocInfo, "name");

  // Check that we only get the last part of the path.
  EXPECT_EQ(llvmIRGenTmp3.getMainEntryName(), "model_dir");

  // Empty name should return a non-null name.
  func = mod.createFunction("");
  IRFunction irfuncTmp4(func);
  LLVMIRGen llvmIRGenTmp4(&irfuncTmp4, allocInfo, "name");

  // Check that we only get the last part of the path.
  EXPECT_EQ(llvmIRGenTmp4.getMainEntryName(), "main");

  // Finally, only delimiters should return a non-null name.
  func = mod.createFunction("////");
  IRFunction irfuncTmp5(func);
  LLVMIRGen llvmIRGenTmp5(&irfuncTmp5, allocInfo, "name");

  // Check that we only get the last part of the path.
  EXPECT_EQ(llvmIRGenTmp5.getMainEntryName(), "main");
}
