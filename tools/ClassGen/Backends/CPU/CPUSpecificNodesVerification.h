#ifdef GLOW_WITH_CPU

void CPUMaxSplatNode::verify() const {
  assert(getInput().getType() == getResult().getType() && "Invalid type");
  assert(getInput().dims() == getResult().dims() && "Invalid shape");
}

void CPUConvDKKC8Node::verify() const {
  ShapeNHWC idim(getInput().getType()->dims());
  ShapeNHWC odim(getResult().getType()->dims());
  auto outSz = calculateConvOutputDims(idim.h, idim.w, getKernel(), getStride(),
                                       getPad());
  ShapeNHWC exp(idim.n, outSz.first, outSz.second, getDepth());
  (void)exp;
  assert(exp == odim && "Invalid output dimensions");
}

#endif // GLOW_WITH_CPU
