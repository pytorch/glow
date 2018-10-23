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

#include "glow/Base/Image.h"

#include "llvm/Support/FileSystem.h"

#include "gtest/gtest.h"

#include <cstdio>
#include <utility>

using namespace glow;

TEST(Image, writePngImage) {
  auto range = std::make_pair(0.f, 1.f);
  Tensor localCopy;
  bool loadSuccess =
      !readPngImage(&localCopy, "tests/images/imagenet/cat_285.png", range);
  ASSERT_TRUE(loadSuccess);

  llvm::SmallVector<char, 10> resultPath;
  llvm::sys::fs::createTemporaryFile("prefix", "suffix", resultPath);
  std::string outfilename(resultPath.begin(), resultPath.end());

  bool storeSuccess = !writePngImage(&localCopy, outfilename.c_str(), range);
  ASSERT_TRUE(storeSuccess);

  Tensor secondLocalCopy;
  loadSuccess = !readPngImage(&secondLocalCopy, outfilename.c_str(), range);
  ASSERT_TRUE(loadSuccess);
  EXPECT_TRUE(secondLocalCopy.isEqual(localCopy, 0.01));

  // Delete the temporary file.
  std::remove(outfilename.c_str());
}
