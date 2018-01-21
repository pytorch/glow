// Copyright 2017 Facebook, Inc. All Rights Reserved.

#ifndef GLOW_TESTS_INFERFUNCBUILDER_H
#define GLOW_TESTS_INFERFUNCBUILDER_H

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>

class InferFuncBuilder {
  /// The test operation name.
  std::string name_;
  /// Header file stream.
  std::ofstream &headerStream_;
  /// CPP file stream.
  std::ofstream &cppStream_;

public:
  InferFuncBuilder(std::ofstream &H, std::ofstream &C,
                       const std::string &name)
      : name_(name), headerStream_(H), cppStream_(C) {}

  ~InferFuncBuilder();

private:
  /// Emit the function declaration.
  void emitFunctionDecl(std::ostream &os) const;

  /// Emit the function definition.
  void emitFunctionDef(std::ostream &os) const;
};

class Builder {
  std::ofstream &hStream_;
  std::ofstream &cStream_;

public:
  /// Create a new top-level builder that holds the two output streams that
  /// point to the header file and the cpp file.
  Builder(std::ofstream &H, std::ofstream &C)
      : hStream_(H), cStream_(C) {
    hStream_ << "#include \"glow/ExecutionEngine/ExecutionEngine.h\"\n"
                "#include \"glow/Graph/Graph.h\"\n"
                "#include \"glow/IR/IR.h\"\n";
    cStream_ << "#include \"glow/ExecutionEngine/ExecutionEngine.h\"\n"
                "#include \"glow/Graph/Graph.h\"\n"
                "#include \"glow/IR/IR.h\"\n"
                "#include \"glow/IR/IRBuilder.h\"\n"
                "#include \"glow/IR/Instrs.h\"\n\n"
                "using namespace glow;\n"
                "using llvm::cast;\n";
  }

  /// Declare a new inference test function and generate code for it.
  InferFuncBuilder newInferFunc(const std::string &name) {
    return InferFuncBuilder(hStream_, cStream_, name);
  }
};

#endif // GLOW_TESTS_INFERFUNCBUILDER_H
