#include "../../InstrBuilder.h"

void addInstrsForCPUBackend(Builder &BB) {
  BB.newBackendSpecificInstr("CPUBackend__MaxZero")
      .addOperand("Dest", OperandKind::Out)
      .addOperand("Src", OperandKind::In);
}
