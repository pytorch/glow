#ifndef GLOW_VERIFICATION_VERIFICATION_H
#define GLOW_VERIFICATION_VERIFICATION_H

namespace glow {

/// Check that the type of the first operand matches the type of the second
/// operand.
template <class T> void checkSameType(T A, T B) {
  assert(A.getType() == B.getType() && "Invalid type");
}

/// Check that the shape of the first operand matches the shape of the second
/// operand.
template <class T> void checkSameShape(T A, T B) {
  assert(A.dims() == B.dims() && "Invalid shape");
}

template <class T> void checkType(T A, ElemKind expectedType) {
  assert(A.getElementType() == expectedType && "Invalid type");
}

template <class T>
void verifyConvolution(T src, T dest, T filter, T bias, size_t kernel,
                       size_t stride, size_t pad, size_t depth) {
  assert(src.getElementType() == dest.getElementType() && "Invalid Type");
  assert(src.getElementType() == filter.getElementType() && "Invalid Type");
  assert(src.getElementType() == bias.getElementType() && "Invalid Type");

  ShapeNHWC idim(src.getType()->dims());
  ShapeNHWC odim(dest.getType()->dims());
  (void)odim;

  assert(idim.w >= kernel && idim.h >= kernel &&
         "buffer too small for selected stride");

  auto outSz = calculateConvOutputDims(idim.h, idim.w, kernel, stride, pad);
  ShapeNHWC exp(idim.n, outSz.first, outSz.second, depth);
  (void)exp;
  assert(exp == odim && "Invalid output dimensions");

  auto filterDims = {depth, kernel, kernel, idim.c};
  assert(filter.getType()->dims().equals(filterDims) && "Invalid filter dims");
  (void)filterDims;

  auto biasDims = {depth};
  assert(bias.getType()->dims().equals(biasDims) && "Invalid bias dims");
  (void)biasDims;
}

template <class T>
void verifyPool(T src, T dest, size_t kernel, size_t stride, size_t pad) {
  ShapeNHWC idim = ShapeNHWC(src.getType()->dims());
  ShapeNHWC odim = ShapeNHWC(dest.getType()->dims());
  (void)odim;
  assert(idim.w >= kernel && idim.h >= kernel &&
         "buffer too small for selected stride");

  auto outSz = calculateConvOutputDims(idim.h, idim.w, kernel, stride, pad);
  ShapeNHWC exp(idim.n, outSz.first, outSz.second, idim.c);
  (void)exp;
  assert(exp == odim && "Unexpected output dimensions");
}

template <class T>
void verifyPoolMaxWithXY(T src, T dest, T srcXY, size_t kernel, size_t stride,
                         size_t pad) {
  ShapeNHWC idim = ShapeNHWC(src.getType()->dims());
  ShapeNHWC odim = ShapeNHWC(dest.getType()->dims());
  (void)odim;
  assert(idim.w >= kernel && idim.h >= kernel &&
         "buffer too small for selected stride");

  auto outSz = calculateConvOutputDims(idim.h, idim.w, kernel, stride, pad);
  ShapeNHWC exp(idim.n, outSz.first, outSz.second, idim.c);
  (void)exp;
  assert(exp == odim && "Unexpected output dimensions");

  // Allocate cache arrays that store the x and y coordinates of the incoming
  // gradient for each max element.
  auto E = {idim.n, outSz.first, outSz.second, idim.c, 2UL};
  assert(srcXY.getType()->dims().equals(E) && "Invalid srcXY dims");
  (void)E;
}

template <class T>
void verifyBatchNormalization(T src, T dest, T bias, T scale, T mean, T var,
                              size_t channel) {
  checkSameType(dest, src);

  // Figure out how many channels are in the tensor.
  size_t channels = src.dims()[channel];

  auto exp = {channels};
  (void)exp;
  assert(bias.getType()->dims().equals(exp) && "Invalid bias dim");
  assert(scale.getType()->dims().equals(exp) && "Invalid scale dim");
  assert(mean.getType()->dims().equals(exp) && "Invalid mean dim");
  assert(var.getType()->dims().equals(exp) && "Invalid var dim");
}

template <class T> void verifySigmoid(T src, T dest) {
  checkSameType(src, dest);
}

template <class T> void verifyTanh(T src, T dest) { checkSameType(src, dest); }

template <class T> void verifyArithmetic(T LHS, T RHS, T res) {
  checkSameShape(res, LHS);
  checkSameShape(LHS, RHS);
}

template <class T> void verifySoftMax(T src, T dest) {
  checkSameType(src, dest);
  assert(src.dims() == dest.dims() && "Invalid shape");
}

template <class T> void verifySoftMaxGrad(T src, T dest, T srcGrad) {
  checkSameType(dest, src);
  checkSameType(dest, srcGrad);
  auto destShape = dest.dims();
  assert(destShape == src.dims() && "Invalid shape");
  assert(destShape == srcGrad.dims() && "Invalid shape");
  (void)destShape;
}

template <class T> void verifyCrossEntropyLoss(T P, T CE, T labels) {
  assert(P.getElementType() == CE.getElementType());
  assert(P.dims()[0] == labels.dims()[0] && "Invalid shape");
}

template <class T> void verifyReshape(T src, T dest) {
  assert(dest.getType()->size() == src.getType()->size() &&
         "Reshape into a different size");
}

template <class T>
void verifyTranspose(T src, T dest, llvm::ArrayRef<unsigned> shuffle) {
  llvm::SmallVector<size_t, 6> shape;

  auto dims = src.dims();
  for (size_t i = 0; i < dims.size(); i++) {
    shape.push_back(dims[shuffle[i]]);
  }

  assert(dest.dims().equals(shape) && "Invalid transpose dims");
}

template <class T>
void verifyBroadcast(T src, T dest, llvm::ArrayRef<size_t> shape) {
  assert(src.dims().size() <= dest.dims().size() &&
         "Source being broadcasted must have <= number dims of result shape.");
  assert(dest.dims().equals(shape) &&
         "New broadcasted shape does not match shape to broadcast to.");
}

template <class T>
void verifyInsertTensor(T src, T dest, llvm::ArrayRef<size_t> offsets) {
  unsigned numDims = dest.dims().size();
  (void)numDims;
  assert(numDims == src.dims().size() && numDims == offsets.size() &&
         "Invalid number of dimensions");

  for (unsigned i = 0; i < numDims; i++) {
    assert(src.dims()[i] + offsets[i] <= dest.dims()[i] && "out of bounds");
  }
}

template <class T> void verifyBatchedAdd(T dest, T batch, T slice) {
  assert(batch.dims().drop_front() == slice.dims() && "Invalid shape");
  assert(batch.dims() == dest.dims() && "Invalid dest type");
  assert(batch.getType()->getElementType() ==
             slice.getType()->getElementType() &&
         "Mismatched element types");
}

template <class T> void verifyBatchedReduceAdd(T batch) {
  assert(batch.dims().size() > 1 && "Invalid shape");
}

template <class T> void verifyQuantizationProfile(T src, T compInfo) {
  // Make sure that input tensor is a floating point type.
  assert(src.getElementType() == ElemKind::FloatTy &&
         "Floating point type is expected");

  // Check computation info has proper size.
  assert(compInfo.dims().size() == 1 &&
         "Computation info should be 1 dimensional");
  assert(compInfo.dims()[0] == 2 &&
         "Computation info should contain Min and Max value only");
}

template <class T> void verifyQuantize(T src, T dest) {
  // Dest must be quantized.
  checkType(dest, ElemKind::Int8QTy);
  // Src must be float.
  checkType(src, ElemKind::FloatTy);
  checkSameShape(dest, src);
}

template <class T> void verifyDequantize(T src, T dest) {
  // Dest must be float.
  checkType(dest, ElemKind::FloatTy);
  // Src must be quantized.
  checkType(src, ElemKind::Int8QTy);
  checkSameShape(dest, src);
}

template <class T> void verifyRescaleQuantized(T src, T dest) {
  // Dest must be quantized.
  checkType(dest, ElemKind::Int8QTy);
  // Src must be quantized.
  checkType(src, ElemKind::Int8QTy);
  checkSameShape(dest, src);
}

template <class T> void verifyTopK(T src, T values, T indices) {
  assert(src.getElementType() == ElemKind::FloatTy);
  assert(values.getElementType() == ElemKind::FloatTy);
  assert(values.dims() == indices.dims());
}

template <class T> void verifyGather(T dest, T data, T indices) {
  assert(dest.getElementType() == data.getElementType());
  assert(indices.getElementType() == ElemKind::IndexTy);
  assert(dest.dims().size() == data.dims().size() + indices.dims().size() - 1);
}

template <class T> void verifyIntrinsic(T name) {
  assert(name.size() && "Name must not be empty");
}

template <class T> void verifySelect(T dest, T cond, T lhs, T rhs) {
  checkSameType(dest, cond);
  checkSameType(dest, lhs);
  checkSameType(dest, rhs);
}

template <class T> void verifySave(T src, T dest) { checkSameType(src, dest); }

template <class T>
void verifyConcat(llvm::ArrayRef<T> inputs, T dest, unsigned dimension) {
  for (size_t i = 1; i < inputs.size(); i++) {
    for (size_t j = 0; j < inputs[0].dims().size(); j++) {
      if (j == dimension) {
        continue;
      }
      assert(inputs[0].dims()[j] == inputs[i].dims()[j]);
    }
  }

  for (size_t i = 0; i < inputs.size(); i++) {
    checkType<NodeValue>(inputs[i], dest->getElementType());
    if (dest->getType()->isQuantizedType()) {
      assert(inputs[i]->getType()->getScale() == dest->getType()->getScale());
      assert(inputs[i]->getType()->getOffset() == dest->getType()->getOffset());
    }
  }
}

template <class T> void verifyFullyConnected(T src, T weights, T bias, T dest) {
  assert(src.dims()[0] == dest.dims()[0] &&
         flattenCdr(src.dims()).second == weights.dims()[0] &&
         "Mismatch on expected source dimensions");

  assert(bias.dims()[0] == weights.dims()[1] &&
         weights.dims()[1] == dest.dims()[1] &&
         "Inconsistent bias/weights/dest sizes.");
}

template <class T> void verifyLocalResponseNormalization(T src, T dest) {
  checkSameType(src, dest);
}

template <class T>
void verifyLocalResponseNormalization(T src, T dest, T scale) {
  checkSameType(dest, src);
  checkSameType(dest, scale);
}

template <class T> void verifyBatchedMatMul(T dest, T lhs, T rhs) {
  auto LDims = lhs.dims();
  auto RDims = rhs.dims();
  auto DDims = dest.dims();
  (void)DDims;
  assert(DDims.size() == 3);
  auto elem = dest.getType()->getElementType();
  (void)elem;
  assert(lhs.getType()->getElementType() == elem);
  assert(rhs.getType()->getElementType() == elem);

  size_t N, X, Y;
  std::tie(N, X, Y) = calculateMatMulOutputDims(LDims, RDims);

  assert(N == DDims[0] && "Invalid matrix dims");
  assert(X == DDims[1] && "Invalid matrix dims");
  assert(Y == DDims[2] && "Invalid matrix dims");

  (void)N;
  (void)X;
  (void)Y;
}

template <class T>
void verifySlice(T src, T dest, llvm::ArrayRef<size_t> offsets) {
  unsigned numDims = dest.dims().size();
  (void)numDims;
  assert(numDims == src.dims().size() && numDims == offsets.size() &&
         "Invalid number of dimensions");

  for (unsigned i = 0; i < numDims; i++) {
    assert(dest.dims()[i] + offsets[i] <= src.dims()[i] && "out of bounds");
  }
}

template <class T>
void verifyExtractTensor(T src, T dest, llvm::ArrayRef<size_t> offsets) {
  unsigned numDims = dest.dims().size();
  (void)numDims;
  assert(numDims == src.dims().size() && numDims == offsets.size() &&
         "Invalid number of dimensions");

  for (unsigned i = 0; i < numDims; i++) {
    assert(dest.dims()[i] + offsets[i] <= src.dims()[i] && "out of bounds");
  }
}

template <class T>
void verifySGD(T gradient, T gSum, T weight, float momentum) {
  if (momentum > 0.0) {
    assert(gradient.getType() == gSum.getType() && "Invalid gsum type");
  }

  assert(gradient.getType() == weight.getType() &&
         "Invalid weight or gradient type");
}

template <class T> void verifyRegression(T src, T dest, T expected) {
  checkSameType(src, dest);
  checkSameType(dest, expected);
}

template <class T> void verifyRelu(T src, T dest) { checkSameType(src, dest); }

template <class T> void verifyTensorView(T src, TypeRef type) {
  assert(src.getType()->size() == type->size() &&
         "TensorView view size should be the same as Src size");
  assert(src.getElementType() == type->getElementType() &&
         "TensorView view element type should be the same as Src type");
}

} // namespace glow

#endif // GLOW_VERIFICATION_VERIFICATION_H
