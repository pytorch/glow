#include "glow/IR/IR.h"
#include "glow/IR/Instrs.h"

using namespace glow;

void CPUBackend__MaxZeroInst::verify() const {
  assert(getSrc()->getType() == getDest()->getType() && "Invalid type");
  assert(getSrc()->dims() == getDest()->dims() && "Invalid shape");
}
