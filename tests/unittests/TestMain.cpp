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

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Signals.h"
#include "gtest/gtest.h"

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  bool doArgsInit = true;
  for (size_t i = 0; i < argc; i++) {
    if (std::string(argv[i]) == "--test_skip_cmd_args") {
      doArgsInit = false;
      break;
    }
  }
  if (doArgsInit) {
    llvm::cl::ParseCommandLineOptions(argc, argv);
  }
  return RUN_ALL_TESTS();
}
