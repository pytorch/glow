#ifndef GLOW_OPTIMIZER_OPTIMIZER_H
#define GLOW_OPTIMIZER_OPTIMIZER_H

namespace glow {

class Module;

enum class OptimizationMode {
  None,  // Don't optimize the module.
  Train, // Optimize the module but allow training.
  Infer, // Optimize the module and break training.
};

void optimize(Module &M, OptimizationMode mode);

} // namespace glow

#endif // GLOW_OPTIMIZER_OPTIMIZER_H
