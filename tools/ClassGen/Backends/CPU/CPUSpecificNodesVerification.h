#ifdef GLOW_WITH_CPU

void CPUMaxZeroNode::verify() const {
  assert(getInput().getType() == getResult().getType() && "Invalid type");
  assert(getInput().dims() == getResult().dims() && "Invalid shape");
}

#endif // GLOW_WITH_CPU
