#include "../../NodeBuilder.h"

void addNodesForCPUBackend(Builder &BB) {
  BB.newNode("CPUBackend__MaxZero")
      .addInput("Input")
      .addResult("Input.getType()")
      .setDocstring("Intrinsic for a Max node with one ZeroNode input.");
}
