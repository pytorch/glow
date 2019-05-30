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

#include "glow/Base/IO.h"

#include "llvm/Support/FileSystem.h"

#include <glog/logging.h>

#include <cstdio>
#include <sys/stat.h>
#include <sys/types.h>

namespace glow {

void writeToFile(const Tensor &T, llvm::StringRef filename) {
  FILE *fp = fopen(filename.data(), "wb");
  CHECK(fp) << "Failed to open file: " << filename.str();
  auto nitems = fwrite(&T.getType(), sizeof(Type), 1, fp);
  CHECK_EQ(nitems, 1);
  nitems = fwrite(T.getUnsafePtr(), T.getType().getElementSize(), T.size(), fp);
  CHECK_EQ(nitems, T.size());
  fclose(fp);
}

void readFromFile(Tensor &T, llvm::StringRef filename) {
  FILE *fp = fopen(filename.data(), "rb");
  CHECK(fp) << "Failed to open file: " << filename.str();
  Type type;
  auto nitems = fread(&type, sizeof(Type), 1, fp);
  CHECK_EQ(nitems, 1);
  T.reset(type);
  nitems = fread(T.getUnsafePtr(), T.getType().getElementSize(), T.size(), fp);
  CHECK_EQ(nitems, T.size());
  fclose(fp);
}

} // namespace glow
