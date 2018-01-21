// Copyright 2017 Facebook, Inc. All Rights Reserved.

#include "InferFuncBuilder.h"

#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " header.h impl.cpp\n";
    return -1;
  }

  std::cout << "Writing test inference functions to:\n\t" << argv[1] << "\n\t"
            << argv[2] << "\n";

  std::ofstream headerFile(argv[1]);
  std::ofstream cppFile(argv[2]);

  Builder BB(headerFile, cppFile);

  BB.newInferFunc("RELU");
  BB.newInferFunc("Sigmoid");
  BB.newInferFunc("Tanh");

  return 0;
}
