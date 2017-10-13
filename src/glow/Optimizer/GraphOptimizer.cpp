// Copyright 2017 Facebook Inc.  All Rights Reserved.

#include "glow/Graph/Graph.h"
#include "glow/Graph/Node.h"
#include "glow/Optimizer/Optimizer.h"
#include "glow/Support/Casting.h"

#include <unordered_map>
#include <unordered_set>

using namespace glow;

void glow::optimize(Graph &G, OptimizationMode mode) {
  if (mode == OptimizationMode::None) {
    return;
  }
}
