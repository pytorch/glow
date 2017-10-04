// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Graph/Nodes.h"
#include "glow/IR/Instrs.h"
#include "glow/IR/Type.h"

using namespace glow;

TypeRef ConvolutionNode::getOutputType(Module &M, TypeRef Ty, size_t pad,
                                       size_t kernel, size_t stride,
                                       size_t depth) {
  ShapeNHWC idim = ShapeNHWC(Ty->dims());

  // Calculate the size and allocate the output buffer.
  auto outSz =
      ConvolutionInst::calculateOutputDims(idim.h, idim.w, pad, kernel, stride);

  return M.uniqueType(ElemKind::FloatTy,
                      {idim.n, outSz.first, outSz.second, depth});
}

TypeRef PoolNode::getOutputType(Module &M, TypeRef Ty, size_t pad,
                                size_t kernel, size_t stride) {
  ShapeNHWC idim = ShapeNHWC(Ty->dims());

  // Calculate the size and allocate the output buffer.
  auto outSz =
      ConvolutionInst::calculateOutputDims(idim.h, idim.w, pad, kernel, stride);

  return M.uniqueType(ElemKind::FloatTy,
                      {idim.n, outSz.first, outSz.second, idim.c});
}

/// \returns the calculated size of the output tensor.
TypeRef FullyConnectedNode::getOutputType(Module &M, TypeRef Ty, size_t depth) {
  auto idim = flattenCdr(Ty->dims());
  return M.uniqueType(Ty->getElementType(), {idim.first, depth});
}

TypeRef ConcatNode::getOutputType(Module &M, llvm::ArrayRef<Node *> inputs,
                                  unsigned dimension) {
  auto inDim = inputs[0]->dims();
  for (auto in : inputs) {
    (void)in;
    assert(in->dims() == inDim && "Invalid input shape");
  }

  std::vector<size_t> shape(inDim.begin(), inDim.end());
  // We are stacking the tensors along a specific dimension. This means that
  // we increase the size of the tensor along this dimension.
  shape[dimension] *= inputs.size();
  return M.uniqueType(inputs[0]->getElementType(), shape);
}
