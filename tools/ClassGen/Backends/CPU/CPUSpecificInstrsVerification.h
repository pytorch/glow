#ifdef GLOW_WITH_CPU

void CPUMaxSplatInst::verify() const {
  assert(getSrc()->getType() == getDest()->getType() && "Invalid type");
  assert(getSrc()->dims() == getDest()->dims() && "Invalid shape");
}

#endif // GLOW_WITH_CPU
