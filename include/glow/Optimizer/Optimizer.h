#ifndef GLOW_OPTIMIZER_OPTIMIZER_H
#define GLOW_OPTIMIZER_OPTIMIZER_H

namespace glow {

class Module;

enum class OptimizationMode {
  kNone,  // Don't optimize the module.
  kTrain, // Optimize the module but allow training.
  kInfer, // Optimize the module and break training.
};

void optimize(Module &M, OptimizationMode mode);

} // namespace glow

#endif // GLOW_OPTIMIZER_OPTIMIZER_H
