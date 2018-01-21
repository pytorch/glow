// Copyright 2017 Facebook Inc. All Rights Reserved.

#include "InferFuncBuilder.h"

#include "glow/Support/Support.h"

void InferFuncBuilder::emitFunctionDecl(std::ostream &os) const {
  os << "\nvoid infer" << glow::tosentence(name_) << "Net("
     << "glow::Tensor *inputs, glow::Tensor *out, glow::BackendKind kind);\n";
}

void InferFuncBuilder::emitFunctionDef(std::ostream &os) const {
  os << "\nvoid infer" << glow::tosentence(name_) << "Net(Tensor *inputs, "
     << "Tensor *out, BackendKind kind) {\n"
     << "  ExecutionEngine EE(kind);\n"
     << "  auto &G = EE.getGraph();\n"
     << "  auto *var = G.createVariable(inputs->getElementType(), "
     << "inputs->dims(), \"input\");\n"
     << "  auto *op = G.create" << name_ << "(\"" << glow::tolower(name_)
     << "\", var);\n"
     << "  auto result = G.createSave(\"ret\", op);\n"
     << "  EE.compile(CompilationMode::Infer);\n"
     << "  EE.run({var}, {inputs});\n"
     << "  out->copyFrom(&result->getVariable()->getPayload());\n"
     << "}\n";
}

InferFuncBuilder::~InferFuncBuilder() {
  emitFunctionDecl(headerStream_);
  emitFunctionDef(cppStream_);
}
