// Copyright 2018 Facebook Inc.  All Rights Reserved.

#include "llvm/Support/CommandLine.h"
#include "gtest/gtest.h"

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  return RUN_ALL_TESTS();
}
